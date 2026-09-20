from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .bench.metrics import score_events, score_qa
from .config import load_config
from .demo import make_demo
from .replay import InputEvent, TaskEvent, prepare_video, read_jsonl, replay, write_jsonl


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="StreamBudget: API-first continuous perception research runtime")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Generate and run a synthetic, no-API integration fixture")
    d.add_argument("--out", type=Path, default=Path("runs/demo"))
    d.add_argument("--policy", choices=["adaptive", "fixed", "motion", "recent_only"], default="adaptive")
    d.add_argument("--timing", choices=["deterministic", "realtime"], default="deterministic")
    r = sub.add_parser("run", help="Run normalized events and tasks; does NOT accept labels")
    r.add_argument("--events", type=Path, required=True)
    r.add_argument("--tasks", type=Path)
    r.add_argument("--config", type=Path)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--timing", choices=["archive", "deterministic", "realtime"], default="deterministic")
    r.add_argument("--speed", type=float, default=1)
    r.add_argument("--allow-network", action="store_true", help="Allow inference calls that may incur charges")
    v = sub.add_parser("prepare-video", help="Use PyAV PTS to create an image manifest (decoding is measured)")
    v.add_argument("--video", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    v.add_argument("--source", default="camera")
    v.add_argument("--fps", type=float, default=2)
    v.add_argument("--max-seconds", type=float)
    s = sub.add_parser("score", help="Post-hoc scoring; labels are separate from runtime")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True)
    s.add_argument("--confidence", type=float, default=0)
    html = sub.add_parser("report", help="Create a local HTML run report")
    html.add_argument("--run", type=Path, required=True)
    doc = sub.add_parser("doctor", help="Validate configuration, dependencies and optionally one real VLM call")
    doc.add_argument("--config", type=Path)
    doc.add_argument("--probe", action="store_true")
    doc.add_argument("--allow-network", action="store_true")
    doc.add_argument("--out", type=Path, default=Path("runs/probe"))
    val = sub.add_parser("validate-data")
    val.add_argument("--events", type=Path, required=True)
    val.add_argument("--tasks", type=Path)
    srv = sub.add_parser("serve")
    srv.add_argument("--config", type=Path)
    srv.add_argument("--out", type=Path, required=True)
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)
    srv.add_argument("--allow-network", action="store_true")
    srv.add_argument("--insecure-local", action="store_true")
    bridge = sub.add_parser("bridge-video")
    bridge.add_argument("--input", required=True, help="Local video or trusted RTSP URL")
    bridge.add_argument("--server", default="http://127.0.0.1:8765")
    bridge.add_argument("--source", default="camera")
    bridge.add_argument("--fps", type=float, default=2)
    bridge.add_argument("--max-seconds", type=float)
    bridge.add_argument("--log", type=Path)
    convert = sub.add_parser("import-videomme")
    convert.add_argument("--annotations", type=Path, required=True)
    convert.add_argument("--video-dir", type=Path, required=True)
    convert.add_argument("--out", type=Path, required=True)
    convert.add_argument("--fps", type=float, default=1)
    convert.add_argument("--limit", type=int, default=3)
    convert.add_argument("--acknowledge-data-license", action="store_true")
    exp = sub.add_parser("export-videomme")
    exp.add_argument("--template", type=Path, required=True)
    exp.add_argument("--predictions", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    speech = sub.add_parser("transcribe")
    speech.add_argument("--audio", type=Path, required=True, help="PCM WAV chunk")
    speech.add_argument("--config", type=Path, required=True)
    speech.add_argument("--out", type=Path, required=True)
    speech.add_argument("--source", default="camera")
    speech.add_argument("--start", type=float, default=0)
    speech.add_argument("--allow-network", action="store_true")
    return p


def score(run_dir: Path, labels_path: Path, confidence: float = 0) -> dict:
    data = json.loads((run_dir / "run.json").read_text())
    labels = read_jsonl(labels_path)
    result = {"qa": score_qa(labels, data.get("predictions", [])),
              "events": score_events(labels, data.get("alerts_detail", []), data["camera_hours"],
                                      confidence_threshold=confidence),
              "synthetic": data.get("synthetic_backend"), "official_score": False,
              "timing": data["timing"], "cost": data["ledger"]}
    (run_dir / "scores.json").write_text(json.dumps(result, indent=2))
    return result


async def probe(config, out, allow_network):
    from io import BytesIO
    from PIL import Image, ImageDraw
    from .backend import ImageInput, Request
    from .runtime import Runtime, SYSTEM_PERCEPTION
    from .types import Perception
    r = Runtime(config, out, allow_network=allow_network)
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).rectangle((32, 32, 96, 96), fill="red")
    data = BytesIO()
    image.save(data, format="JPEG")
    try:
        result = await r.pool.call("perception", Request("perceive", SYSTEM_PERCEPTION,
            "Describe the supplied test image. No watches are registered.",
            [ImageInput("probe-frame", 0, data.getvalue())], {"watches": []}))
        observation = Perception.model_validate(result.json())
        return {"schema_passed": True, "caption": observation.caption, "ledger": r.ledger.summary(),
                "note": "Schema/transport probe only; inspect the caption to verify visual grounding."}
    finally:
        await r.close()


async def transcribe(args):
    from .runtime import Runtime
    r = Runtime(load_config(args.config), args.out, allow_network=args.allow_network)
    try:
        text, duration = await r.specialists.transcribe(args.audio.read_bytes())
        path = args.out / "asr-events.jsonl"
        write_jsonl(path, [{"source": args.source, "ts": args.start + duration, "start": args.start,
                           "kind": "asr", "text": text}])
        return {"events": str(path), "duration_s": duration, "cost": r.ledger.summary()}
    finally:
        await r.close()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            events, tasks, labels = make_demo(args.out / "input")
            config = load_config()
            config.policy.mode = args.policy
            data = asyncio.run(replay(events, tasks, config, args.out / "run", timing=args.timing))
            result = score(args.out / "run", labels)
            from .report import render
            result["report"] = str(render(args.out / "run"))
        elif args.command == "run":
            result = asyncio.run(replay(args.events, args.tasks, load_config(args.config), args.out,
                                       timing=args.timing, speed=args.speed, allow_network=args.allow_network))
        elif args.command == "prepare-video":
            result = {"events": str(prepare_video(args.video, args.out, source=args.source,
                        fps=args.fps, max_seconds=args.max_seconds))}
        elif args.command == "score":
            result = score(args.run, args.labels, args.confidence)
        elif args.command == "report":
            from .report import render
            result = {"report": str(render(args.run))}
        elif args.command == "doctor":
            config = load_config(args.config)
            versions = {}
            for name in ["streambudget", "httpx", "pydantic", "numpy", "Pillow", "av", "fastapi"]:
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = "not installed"
            result = {"python": sys.version, "packages": versions,
                      "roles": {role: {"kind": cfg.kind, "model": cfg.model,
                                       "key_present": bool(os.getenv(cfg.api_key_env))}
                                for role, cfg in config.models.items()}, "network_called": False}
            if args.probe:
                result["probe"] = asyncio.run(probe(config, args.out, args.allow_network))
                result["network_called"] = any(m.kind == "chat" for m in config.models.values())
        elif args.command == "validate-data":
            events = [InputEvent.model_validate(row) for row in read_jsonl(args.events)]
            tasks = [TaskEvent.model_validate(row) for row in read_jsonl(args.tasks)] if args.tasks else []
            result = {"valid": True, "events": len(events), "tasks": len(tasks)}
        elif args.command == "serve":
            from .adapters.server import serve
            serve(load_config(args.config), args.out, host=args.host, port=args.port,
                  allow_network=args.allow_network, insecure_local=args.insecure_local)
            return
        elif args.command == "bridge-video":
            from .adapters.bridge import bridge_video
            result = bridge_video(args.input, args.server, source=args.source, fps=args.fps,
                                  max_seconds=args.max_seconds, log_path=args.log)
        elif args.command == "import-videomme":
            from .bench.videomme import convert
            result = {"cases": [str(p) for p in convert(args.annotations, args.video_dir, args.out,
                fps=args.fps, limit=args.limit, acknowledge_license=args.acknowledge_data_license)]}
        elif args.command == "export-videomme":
            from .bench.videomme import export_official
            export_official(args.template, args.predictions, args.out)
            result = {"output": str(args.out)}
        elif args.command == "transcribe":
            result = asyncio.run(transcribe(args))
        else:
            raise ValueError("Unknown command")
        print(json.dumps(result, indent=2, default=str))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
