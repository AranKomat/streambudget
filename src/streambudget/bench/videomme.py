from __future__ import annotations

import json
from pathlib import Path

from ..replay import prepare_video, read_jsonl, write_jsonl


def convert(annotations: Path, video_dir: Path, output: Path, *, fps: float = 1,
            limit: int = 3, acknowledge_license: bool = False) -> list[Path]:
    """Convert user-provided official nested Video-MME JSON; no automatic dataset download.

    Default input filenames are videoID.mp4 or video_id.mp4. A `video_file` field
    can select another file beneath video_dir. Subtitle mode is deliberately not enabled.
    """
    if not acknowledge_license:
        raise ValueError("Confirm your rights to the input dataset with --acknowledge-data-license")
    rows = json.loads(annotations.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected the nested Video-MME JSON list (see docs/BENCHMARKS.md)")
    paths = []
    for item in rows[:limit]:
        id = str(item["video_id"])
        if not id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Unsafe video identifier")
        video_name = item.get("video_file", str(item.get("videoID", id)) + ".mp4")
        video = (video_dir / video_name).resolve()
        if not video.is_relative_to(video_dir.resolve()) or not video.is_file():
            raise ValueError(f"Video not found inside video-dir: {video_name}")
        case = output / id
        events = prepare_video(video, case, source=id, fps=fps)
        end = max(x["ts"] for x in read_jsonl(events))
        tasks, labels = [], []
        for q in item["questions"]:
            qid = str(q["question_id"])
            prompt = q["question"] + "\n" + "\n".join(q["options"]) + "\nReturn only A, B, C, or D."
            tasks.append({"type": "ask", "at": end, "id": qid, "source": id, "question": prompt})
            labels.append({"type": "qa", "id": qid, "answer": q["answer"],
                           "task_type": q.get("task_type"), "duration": item.get("duration"),
                           "domain": item.get("domain")})
        write_jsonl(case / "tasks.jsonl", tasks)
        write_jsonl(case / "labels.jsonl", labels)
        (case / "benchmark.json").write_text(json.dumps({"benchmark": "Video-MME", "subtitles": False,
            "schema": "official nested JSON", "fps": fps,
            "warning": "Custom agent protocol; use official scorer for a published official result."}, indent=2))
        paths.append(case)
    return paths


def export_official(template: Path, predictions: Path, output: Path):
    rows = json.loads(template.read_text(encoding="utf-8"))
    answers = {str(p["question_id"]): p.get("text", "") for p in read_jsonl(predictions)}
    for video in rows:
        for question in video["questions"]:
            question["response"] = answers.get(str(question["question_id"]), "")
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
