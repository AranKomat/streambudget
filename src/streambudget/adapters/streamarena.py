from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from ..config import load_config
from ..runtime import Runtime
from ..types import Watch


class StreamBudgetAgent:
    """Duck-typed adapter for StreamArena's released StreamingAgent lifecycle.

    Does not import or redistribute StreamArena code or data. Upstream loader
    compatibility and benchmark scores still require a live upstream run.
    """
    def __init__(self):
        self.pending: set[asyncio.Task] = set()
        self.runtime: Runtime | None = None
        self.driver = None
        self.origin: float | None = None

    async def start(self, driver):
        self.driver = driver
        config = load_config(os.getenv("STREAMBUDGET_CONFIG"))
        if all(m.kind == "mock" for m in config.models.values()) and os.getenv("STREAMBUDGET_ALLOW_MOCK") != "1":
            raise RuntimeError("Refusing a synthetic backend for a benchmark; set STREAMBUDGET_CONFIG")
        out = Path(os.getenv("STREAMBUDGET_OUTPUT", "runs/streamarena")) / uuid.uuid4().hex[:12]
        self.runtime = Runtime(config, out, allow_network=os.getenv("STREAMBUDGET_ALLOW_NETWORK") == "1",
                               on_alert=self._alert)
        # The native driver anchors video time after decoder warm-up, not at start().
        self.origin = None
        self.runtime.timeline_clock = self._video_now
        self.runtime.start()

    def _align_clock(self, video_ts: float):
        if self.origin is None:
            self.origin = time.monotonic() - video_ts
            self.runtime.trace.emit("adapter_clock_anchor", video_ts=video_ts)

    def _video_now(self):
        return time.monotonic() - self.origin if self.origin is not None else 0.0

    async def stop(self):
        for task in list(self.pending):
            task.cancel()
        await asyncio.gather(*list(self.pending), return_exceptions=True)
        self.pending.clear()
        if self.runtime:
            await self.runtime.close()
        self.runtime = None

    async def on_frame(self, frame):
        self._align_clock(float(frame.ts))
        await self.runtime.ingest_frame("video", float(frame.ts), base64.b64decode(frame.jpeg_b64, validate=True),
            available_at=max(float(frame.ts), self._video_now(), self.runtime.now))

    async def on_audio_observation(self, text: str, video_ts: float):
        self._align_clock(video_ts)
        await self.runtime.ingest_signal("video", video_ts, "asr", text,
            available_at=max(video_ts, self._video_now(), self.runtime.now))

    async def on_ask(self, qid: int, question: str, qtype: str, video_ts: float):
        self._align_clock(video_ts)
        self.runtime.advance(video_ts)
        if len(self.pending) >= self.runtime.config.scheduler.max_queue:
            self.runtime.trace.emit("query_dropped", question_id=qid, reason="adapter_limit")
            return
        task = asyncio.create_task(self._ask(qid, question, video_ts))
        self.pending.add(task)
        task.add_done_callback(self.pending.discard)
        task.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    async def _ask(self, qid, question, video_ts):
        try:
            answer = await self.runtime.ask(question, "video", as_of=video_ts, question_id=str(qid))
            row = asdict(answer)
            text = answer.text
        except asyncio.CancelledError:
            self.runtime.trace.emit("query_cancelled", question_id=qid)
            raise
        except Exception as exc:
            row = {"question_id": str(qid), "status": "error", "error_type": type(exc).__name__,
                   "text": "", "evidence_ids": [], "as_of": video_ts}
            text = ""
            self.runtime.trace.emit("query_failed", question_id=qid, error_type=type(exc).__name__)
        row["delivered_at"] = max(self.runtime.now, self._video_now())
        with (self.runtime.workdir / "predictions.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        await self.driver.commit_answer(qid, text)

    async def on_e_watch(self, qid: int, question: str, ref_ts: float, deadline_sec: float):
        # CRITICAL: ref_ts is evaluator knowledge of the target's future time.
        # Never put it in model context or use it to schedule observation/escalation.
        # The evaluator, not the agent, owns ground-truth-based acceptance deadlines.
        self.runtime.register_watch(Watch(id=f"q{qid}", source="video", goal=question,
                                          created_at=self.runtime.now, repeat=False))

    async def _alert(self, event):
        id = event["watch_id"]
        if id.startswith("q"):
            await self.driver.emit_speak(int(id[1:]), event["text"], event["delivered_at"])
