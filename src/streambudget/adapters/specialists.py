from __future__ import annotations

import hashlib
from io import BytesIO
import os
import time
import wave

import httpx

from ..backend import BackendError, ModelPool
from ..store import hash_embedding
from ..types import ContractError


class Specialists:
    """Optional standard embeddings and WAV transcription APIs. No native omni requirement.

    These requests share the runtime's cost ledger. No retries are hidden here.
    Bring your own compatible hosted endpoint or locally served specialist.
    """
    def __init__(self, pool: ModelPool):
        self.pool = pool

    def fingerprint(self) -> str:
        cfg = self.pool.config.models["embedding"]
        return hashlib.sha256(f"{cfg.base_url}|{cfg.model}|{cfg.revision}".encode()).hexdigest()

    async def embed(self, text: str) -> tuple[list[float], str]:
        if "embedding" not in self.pool.config.models:
            raise ContractError("No embedding endpoint configured")
        cfg = self.pool.config.models["embedding"]
        if cfg.kind == "mock":
            return hash_embedding(text), "lexical-hash-256-v1"
        quote = ((len(text) + 2) // 3) * cfg.prices.input_per_million / 1e6 if cfg.prices.known else None
        data = await self._call("embedding", "/embeddings", quote=quote,
                                json_body={"model": cfg.model, "input": text, "encoding_format": "float"})
        try:
            vector = [float(x) for x in data["data"][0]["embedding"]]
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            raise BackendError("Invalid embedding response") from exc
        return vector, self.fingerprint()

    async def transcribe(self, wav: bytes) -> tuple[str, float]:
        if "asr" not in self.pool.config.models:
            raise ContractError("No ASR endpoint configured")
        if len(wav) > 24_000_000:
            raise ContractError("Split audio into bounded WAV chunks (<24 MB)")
        with wave.open(BytesIO(wav)) as f:
            seconds = f.getnframes() / f.getframerate()
        cfg = self.pool.config.models["asr"]
        if cfg.kind == "mock":
            return "Synthetic transcript fixture", seconds
        quote = seconds / 60 * cfg.audio_per_minute_usd if cfg.audio_per_minute_usd is not None else None
        data = await self._call("asr", "/audio/transcriptions", quote=quote,
                                form={"model": cfg.model, "response_format": "json"},
                                files={"file": ("chunk.wav", wav, "audio/wav")}, media_cost=quote)
        if not isinstance(data.get("text"), str):
            raise BackendError("ASR response has no transcript text")
        return data["text"], seconds

    async def _call(self, role, path, *, quote, json_body=None, form=None, files=None, media_cost=None):
        if not self.pool.allow_network:
            raise BackendError("Network inference disabled; explicit --allow-network is required")
        cfg = self.pool.config.models[role]
        headers = {}
        key = os.getenv(cfg.api_key_env)
        if key:
            headers["Authorization"] = "Bearer " + key
        ticket = await self.pool.ledger.reserve(quote)
        usage, successful = None, False
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=cfg.timeout_s, transport=self.pool.transport,
                                         follow_redirects=False) as client:
                response = await client.post(cfg.base_url.rstrip("/") + path, headers=headers,
                                             json=json_body, data=form, files=files)
                if response.is_error or response.is_redirect:
                    raise BackendError(f"Specialist endpoint returned HTTP {response.status_code}")
                data = response.json()
                original = data.get("usage")
                if isinstance(original, dict) and "prompt_tokens" in original:
                    usage = {"prompt_tokens": original["prompt_tokens"], "completion_tokens": 0}
                successful = True
                return data
        finally:
            await self.pool.ledger.settle(ticket, role=role, prices=cfg.prices, usage=usage,
                status="ok" if successful else "specialist_error_billing_uncertain",
                measured_media_usd=media_cost if successful else None)
            self.pool.trace.emit("specialist_timing", role=role, wall_s=time.monotonic() - started,
                                 status="ok" if successful else "error")
