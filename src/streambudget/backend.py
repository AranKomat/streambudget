from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import httpx
import numpy as np
from PIL import Image

from .budget import Ledger
from .config import Config, ModelConfig
from .trace import Trace


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImageInput:
    evidence_id: str
    timestamp: float
    jpeg: bytes


@dataclass
class Request:
    operation: str
    system: str
    text: str
    images: list[ImageInput] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)  # fixture context; also included as JSON in prompts


@dataclass
class Result:
    text: str
    usage: dict | None
    latency_s: float
    request_id: str

    def json(self) -> dict:
        text = self.text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError("Model did not return one valid JSON object") from exc
        if not isinstance(value, dict):
            raise BackendError("Expected a JSON object")
        return value


class MockBackend:
    """Explicitly synthetic test double. Reads red pixels, never benchmark labels.

    This is NOT a general vision model; its output is only for plumbing tests.
    """

    def generate(self, request: Request) -> dict:
        ctx = request.context
        red = False
        if request.images:
            a = np.asarray(Image.open(BytesIO(request.images[-1].jpeg)).convert("RGB"), dtype=float)
            red = bool(((a[:, :, 0] > 180) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)).sum() > 30)
        if request.operation in {"perceive", "verify"}:
            return {"caption": "A red box occupies the work area." if red else "The work area is empty.",
                    "facts": {"red_box_present": str(red).lower()},
                    "checks": [{"watch_id": w["id"], "status": "yes" if red else "no",
                                "confidence": 0.99, "detail": "Synthetic red-pixel fixture"}
                               for w in ctx.get("watches", [])]}
        if request.operation == "plan":
            steps = ctx.get("steps", [])
            if not steps:
                return {"tool": "search", "arguments": {"query": "red box", "source": ctx["source"]}}
            if len(steps) == 1:
                hits = ctx.get("last_result", {}).get("evidence", [])
                t = hits[0]["end"] if hits else ctx["as_of"]
                return {"tool": "inspect", "arguments": {"source": ctx["source"],
                        "start": max(0, t - 2), "end": t, "frames": 2,
                        "question": "Describe the work area."}}
            evidence = ctx.get("allowed_evidence_ids", [])
            return {"tool": "answer", "arguments": {"text": "A red box was observed in the work area.",
                                                      "evidence_ids": evidence[-3:]}}
        if request.operation == "compact":
            return {"summary": " | ".join(x["text"] for x in ctx.get("evidence", []))[:3000]}
        if request.operation == "ocr":
            return {"text": "SYNTHETIC OCR FIXTURE"}
        return {"text": "Synthetic test response"}


class ModelPool:
    """HTTP Chat Completions adapter shared by hosted APIs, vLLM, and SGLang.

    Uses timestamped image sequences, not a model-specific native video payload.
    Therefore native-video token pruning is NOT enabled by this adapter.
    """

    def __init__(self, config: Config, ledger: Ledger, trace: Trace, *, allow_network: bool = False,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.config, self.ledger, self.trace = config, ledger, trace
        self.allow_network = allow_network
        self.transport = transport
        self.clients: dict[str, httpx.AsyncClient] = {}
        self.pending: dict[str, asyncio.Task] = {}
        self.mock = MockBackend()

    async def close(self):
        for task in list(self.pending.values()):
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.pending.values(), return_exceptions=True)
        for client in self.clients.values():
            await client.aclose()

    def _body(self, cfg: ModelConfig, request: Request) -> dict:
        if request.images and not cfg.supports_images:
            raise BackendError("Selected backend explicitly does not support images")
        content: list[dict] = [{"type": "text", "text": request.text}]
        for image in request.images:
            content.append({"type": "text", "text": f"Evidence {image.evidence_id}; t={image.timestamp:.6f}s"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(image.jpeg).decode(),
                "detail": cfg.image_detail}})
        body: dict = {"model": cfg.model,
                      "messages": [{"role": "system", "content": request.system},
                                   {"role": "user", "content": content}],
                      cfg.token_parameter: cfg.max_output_tokens, "stream": False}
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(cfg.extra_body)
        return body

    async def call(self, role: str, request: Request) -> Result:
        cfg = self.config.role(role)
        if cfg.kind == "chat" and not self.allow_network:
            raise BackendError("Network inference disabled. Run explicitly with --allow-network.")
        body = self._body(cfg, request)
        fingerprint = {"endpoint": cfg.base_url, "revision": cfg.revision, "body": body,
                       "operation": request.operation, "fixture_context": request.context}
        key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        # Coalesce only EXACT concurrent requests, not similar scenes or stale semantic state.
        if key in self.pending:
            self.trace.emit("model_coalesced", role=role, request_hash=key)
            return await asyncio.shield(self.pending[key])
        task = asyncio.create_task(self._execute(role, cfg, request, body, key))
        self.pending[key] = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self.pending.pop(key, None)
            else:
                task.add_done_callback(lambda _: self.pending.pop(key, None))

    async def _execute(self, role: str, cfg: ModelConfig, request: Request, body: dict, key: str) -> Result:
        import os
        text_chars = len(request.text) + len(request.system)
        input_estimate = (text_chars + 2) // 3 + len(request.images) * cfg.image_token_reserve
        quote = None
        if cfg.kind == "mock":
            quote = 0.0
        elif cfg.prices.known:
            quote = (input_estimate * cfg.prices.input_per_million
                     + cfg.max_output_tokens * cfg.prices.output_per_million) / 1e6
        for attempt in range(cfg.max_retries + 1):
            ticket = await self.ledger.reserve(quote)
            start = time.monotonic()
            usage = None
            status = "error"
            retry = False
            try:
                if cfg.kind == "mock":
                    value = self.mock.generate(request)
                    text = json.dumps(value)
                    status = "ok"
                    return Result(text, None, time.monotonic() - start, key)
                client = self.clients.get(role)
                if client is None:
                    client = httpx.AsyncClient(timeout=cfg.timeout_s, transport=self.transport,
                                               follow_redirects=False)
                    self.clients[role] = client
                headers = {"Content-Type": "application/json"}
                api_key = os.getenv(cfg.api_key_env)
                if api_key:
                    headers["Authorization"] = "Bearer " + api_key
                response = await client.post(cfg.base_url.rstrip("/") + "/chat/completions",
                                             headers=headers, json=body)
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    retry = True
                if response.is_error or response.is_redirect:
                    raise BackendError(f"Model endpoint returned HTTP {response.status_code}; body not logged")
                data = response.json()
                usage = data.get("usage")
                if not isinstance(usage, dict) or not all(k in usage for k in ("prompt_tokens", "completion_tokens")):
                    usage = None
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise BackendError("Model output truncated at token limit")
                message = choice["message"]
                if message.get("refusal"):
                    raise BackendError("Model refused the request")
                text = message.get("content")
                if not isinstance(text, str) or len(text) > 1000000:
                    raise BackendError("Missing, non-text, or oversized completion")
                status = "ok"
                return Result(text, usage, time.monotonic() - start, str(data.get("id", key)))
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                status = "transport_uncertain"
                retry = cfg.retry_timeouts
                if not retry or attempt == cfg.max_retries:
                    raise BackendError("Model transport failed; billing may be uncertain") from exc
            except asyncio.CancelledError:
                status = "cancelled_billing_uncertain"
                raise
            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raise BackendError("Invalid provider response schema") from exc
            except BackendError:
                if not retry or attempt == cfg.max_retries:
                    raise
            finally:
                provider_cost = None
                if cfg.base_url.rstrip("/") == "https://openrouter.ai/api/v1" and isinstance(usage, dict):
                    provider_cost = usage.get("cost")
                await self.ledger.settle(ticket, role=role, prices=cfg.prices, usage=usage,
                                         status=status, synthetic=cfg.kind == "mock",
                                         provider_cost_usd=provider_cost)
                self.trace.emit("model_timing", role=role, operation=request.operation,
                                model=cfg.model, revision=cfg.revision, request_hash=key,
                                frames=len(request.images), wall_s=time.monotonic() - start,
                                attempt=attempt, status=status)
            # Each retry reserves a NEW attempt and is accounted separately.
            await asyncio.sleep(min(0.25 * 2 ** attempt, 4))
        raise BackendError("No successful completion")
