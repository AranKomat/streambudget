from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw

from .replay import write_jsonl


def make_demo(root: Path) -> tuple[Path, Path, Path]:
    """Original, deterministic synthetic imagery. No public benchmark data is bundled."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "frames").mkdir(exist_ok=True)
    events = []
    for t in range(60):
        image = Image.new("RGB", (320, 240), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.rectangle((60, 60, 260, 200), outline=(80, 80, 80), width=3)
        # Actual synthetic scene, not hidden labels or metadata fed to the model.
        if 10 <= t <= 18 or 35 <= t <= 44:
            draw.rectangle((120, 105, 175, 160), fill=(230, 25, 25))
        name = f"frames/{t:04d}.jpg"
        image.save(root / name, quality=90)
        events.append({"source": "camera", "ts": t, "kind": "frame", "media": name})
    # Independent sensor signals can wake the camera's observer.
    events.extend([
        {"source": "camera", "ts": 9, "kind": "sensor", "text": "Contact switch changed", "data": {"door_open": True}},
        {"source": "machine", "ts": 25, "kind": "sensor", "text": "Temperature sample", "data": {"temperature": 91.0}},
    ])
    events.sort(key=lambda e: e["ts"])
    tasks = [
        {"type": "watch", "at": 0, "id": "red_box_dwell", "source": "camera",
         "watch": {"goal": "A red box is present in the work area", "dwell_seconds": 3,
                   "repeat": True, "cooldown_seconds": 25, "max_observation_gap": 10}},
        {"type": "watch", "at": 0, "id": "temperature_high", "source": "machine",
         "watch": {"goal": "Temperature is above 85", "sensor_predicate": {
             "key": "temperature", "op": "gt", "value": 85.0}}},
        {"type": "ask", "at": 29, "id": "history_1", "source": "camera",
         "question": "What occupied the work area earlier? Inspect supporting evidence."},
        {"type": "ask", "at": 59, "id": "history_2", "source": "camera",
         "question": "Find visual evidence of the red box seen earlier."},
    ]
    # This file is read ONLY by scoring, never by Runtime, Agent, or replay().
    labels = [
        {"type": "event", "id": "e1", "watch_id": "red_box_dwell", "source": "camera", "start": 13, "end": 18},
        {"type": "event", "id": "e2", "watch_id": "red_box_dwell", "source": "camera", "start": 38, "end": 44},
        {"type": "event", "id": "e3", "watch_id": "temperature_high", "source": "machine", "start": 25, "end": 25},
        {"type": "qa", "id": "history_1", "answer": "A red box was observed in the work area."},
        {"type": "qa", "id": "history_2", "answer": "A red box was observed in the work area."},
    ]
    write_jsonl(root / "events.jsonl", events)
    write_jsonl(root / "tasks.jsonl", tasks)
    write_jsonl(root / "labels.jsonl", labels)
    (root / "DATA_LICENSE.json").write_text(json.dumps({"source": "original_synthetic_fixture",
        "license": "MIT", "purpose": "contract and integration tests, not model-quality evidence"}, indent=2))
    return root / "events.jsonl", root / "tasks.jsonl", root / "labels.jsonl"
