import asyncio
import base64
import json
from types import SimpleNamespace

from streambudget.adapters.streamarena import StreamBudgetAgent


class Driver:
    def __init__(self):
        self.answers = []

    async def commit_answer(self, qid, text):
        self.answers.append((qid, text))

    async def emit_speak(self, qid, text, video_ts):
        pass


def setup(monkeypatch, tmp_path):
    monkeypatch.setenv("STREAMBUDGET_ALLOW_MOCK", "1")
    monkeypatch.setenv("STREAMBUDGET_OUTPUT", str(tmp_path))
    monkeypatch.delenv("STREAMBUDGET_CONFIG", raising=False)


async def test_decoder_warmup_not_part_of_video_clock(tmp_path, monkeypatch, image_bytes):
    setup(monkeypatch, tmp_path)
    clock = [100.0]
    monkeypatch.setattr("streambudget.adapters.streamarena.time.monotonic", lambda: clock[0])
    agent = StreamBudgetAgent()
    await agent.start(Driver())
    clock[0] = 150.0
    await agent.on_frame(SimpleNamespace(ts=0, jpeg_b64=base64.b64encode(image_bytes(False)).decode()))
    assert agent.origin == 150.0
    assert agent.runtime.now == 0
    clock[0] = 152.5
    assert agent.runtime.timeline_clock() == 2.5
    await agent.stop()


async def test_error_is_recorded_not_silently_lost(tmp_path, monkeypatch):
    setup(monkeypatch, tmp_path)
    agent, driver = StreamBudgetAgent(), Driver()
    await agent.start(driver)
    folder = agent.runtime.workdir

    async def fail(*args, **kwargs):
        raise ValueError("fixture failure")

    monkeypatch.setattr(agent.runtime, "ask", fail)
    await agent.on_ask(3, "Question", "a", 5)
    await asyncio.gather(*list(agent.pending))
    row = json.loads((folder / "predictions.jsonl").read_text())
    assert row["status"] == "error" and row["error_type"] == "ValueError"
    assert driver.answers == [(3, "")]
    await agent.stop()
