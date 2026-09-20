from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Trace:
    """Append-only operational trace. No prompts, API secrets, or images are recorded."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def emit(self, kind: str, **fields: Any) -> None:
        row = {"wall_time": time.time(), "monotonic": time.monotonic(), "kind": kind, **fields}
        with self._lock:
            self.counts[kind] = self.counts.get(kind, 0) + 1
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, allow_nan=False, default=str) + "\n")

    @contextmanager
    def span(self, stage: str, **fields: Any):
        start, cpu = time.monotonic(), time.thread_time()
        try:
            yield
        finally:
            self.emit("stage", stage=stage, wall_s=time.monotonic() - start,
                      thread_cpu_s=time.thread_time() - cpu, **fields)
