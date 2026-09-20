from io import BytesIO
import wave
import json
import httpx
import pytest

from streambudget.config import Config, ModelConfig, Prices
from streambudget.runtime import Runtime


async def test_embeddings_http_and_metering(tmp_path):
    def handler(req):
        body = json.loads(req.content)
        assert req.url.path == "/v1/embeddings"
        assert body["input"] == "vehicle"
        return httpx.Response(200, json={"data": [{"embedding": [1, 0]}], "usage": {"prompt_tokens": 2}})
    cfg = Config(semantic_embeddings=True, models={"perception": ModelConfig(), "embedding": ModelConfig(
        kind="chat", base_url="https://embed.invalid/v1", model="embed-fixture",
        prices=Prices(input_per_million=1, output_per_million=0))})
    r = Runtime(cfg, tmp_path / "run", allow_network=True, transport=httpx.MockTransport(handler))
    try:
        vector, fp = await r.specialists.embed("vehicle")
        assert vector == [1, 0]
        assert r.ledger.requests == 1
        assert r.ledger.reported_usd == pytest.approx(.000002)
    finally:
        await r.close()


async def test_asr_wav_and_duration_price(tmp_path):
    b = BytesIO()
    with wave.open(b, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000)
    def handler(req):
        assert req.url.path == "/v1/audio/transcriptions"
        assert b"chunk.wav" in req.content
        return httpx.Response(200, json={"text": "test speech"})
    cfg = Config(models={"perception": ModelConfig(), "asr": ModelConfig(
        kind="chat", model="asr-fixture", base_url="https://asr.invalid/v1", audio_per_minute_usd=.06)})
    r = Runtime(cfg, tmp_path / "run", allow_network=True, transport=httpx.MockTransport(handler))
    try:
        text, duration = await r.specialists.transcribe(b.getvalue())
        assert text == "test speech" and duration == 1
        assert r.ledger.reported_usd == pytest.approx(.001)
    finally:
        await r.close()
