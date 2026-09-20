"""Exercise the real pinned upstream driver with generated media and no API calls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from streambudget.config import Config
from streambudget.replay import write_jsonl
from streambudget.screening import frame


def main():
    from io import BytesIO

    import av
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = Config()
    cfg.policy.mode = "fixed"
    cfg.policy.fixed_interval_s = .5
    config_path = args.out / "config.json"
    config_path.write_text(cfg.model_dump_json(indent=2))
    video = args.out / "fixture.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("mpeg4", rate=2)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for i in range(16):
            img = Image.open(BytesIO(frame([("red", "P", 90, 170)] if i >= 4 else [])))
            for pkt in stream.encode(av.VideoFrame.from_image(img)):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    questions = [
        {"video_id": "fixture", "qid": 1, "qtype": "Pro", "ask_sec": 0, "ref_sec": 2,
         "question": "Notify me when a red object appears.", "answer": "A red object appears."},
        {"video_id": "fixture", "qid": 2, "qtype": "RTP", "ask_sec": 4,
         "question": "What is visible?", "answer": "A red object."},
        {"video_id": "fixture", "qid": 3, "qtype": "HR", "ask_sec": 6,
         "question": "What appeared earlier?", "answer": "A red object."},
    ]
    write_jsonl(args.out / "question.en.jsonl", questions)
    env = dict(os.environ, STREAMBUDGET_CONFIG=str(config_path.resolve()), STREAMBUDGET_ALLOW_MOCK="1",
               STREAMBUDGET_ALLOW_NETWORK="0", STREAMBUDGET_OUTPUT=str((args.out / "runtime").resolve()))
    command = [sys.executable, str(args.upstream / "method/streammind/run_streammind.py"),
        "--agent", "streambudget.adapters.streamarena_native:NativeStreamBudgetAgent",
        "--dataset", str(args.out.resolve()), "--video-dir", str(args.out.resolve()),
        "--videos", "fixture", "--out", str(args.out / "records.jsonl"),
        "--fps", "2", "--answer-grace-sec", "2", "--disable-audio"]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=45)
    (args.out / "driver.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("Native driver failed; see local driver.log")
    rows = [json.loads(line) for line in (args.out / "records.jsonl").read_text().splitlines()]
    records = [r for r in rows if "question" in r]
    assert len(records) == 3 and all(r["model_answer"] for r in records)
    report = json.loads(next((args.out / "runtime").glob("*/report.json")).read_text())
    assert report["ledger"]["reported_usd"] == 0
    assert report["synthetic_backend"]
    receipt = {"native_driver_passed": True, "records": len(records), "synthetic_only": True,
               "paid_calls": 0, "ledger": report["ledger"]}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
