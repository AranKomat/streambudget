"""Generic Chat Completions adapter with durable admission accounting, no implicit retries."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from concurrent.futures import Executor, Future
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from ..types import ContractError
from .contracts import Endpoint, GameConfig
from .ontology import canonical

T = TypeVar("T", bound=BaseModel)


class ModelError(RuntimeError):
    pass


class BudgetExhausted(ContractError):
    pass


@dataclass(frozen=True)
class ImageInput:
    id: str
    data: bytes
    historical: bool = False


@dataclass
class PendingCompletion:
    """Only HTTP/validation runs in the worker; ledger writes stay on its owner thread."""
    future: Future
    backend: object
    call: str
    endpoint: Endpoint
    role: str
    settled: bool = False

    def result(self):
        if self.settled:
            raise ContractError("Inference attempt already settled")
        response, value, elapsed, error = self.future.result()
        self.backend.ledger.finish(self.call, endpoint=self.endpoint, response=response,
                                   elapsed=elapsed, error=error)
        self.settled = True
        if error:
            raise ModelError(f"{self.role} failed ({error}); no retry dispatched; inspect model_calls ledger")
        return value


class Backend(Protocol):
    def complete(self, role: str, system: str, context: dict, images: list[ImageInput],
                 output: type[T]) -> T: ...


class Ledger:
    """Admission estimates are not a provider invoice cap. Unknown calls retain their holds."""
    def __init__(self, db: sqlite3.Connection, max_calls: int, max_usd: float):
        self.db, self.max_calls, self.max_usd = db, max_calls, max_usd
        db.executescript("""
          CREATE TABLE IF NOT EXISTS model_inputs(call_id TEXT PRIMARY KEY, context TEXT,
            system TEXT, images TEXT);
          CREATE TABLE IF NOT EXISTS model_calls(id TEXT PRIMARY KEY, role TEXT, model TEXT,
            request_hash TEXT, status TEXT, reserved REAL, cost REAL,
            usage TEXT, elapsed REAL, images INTEGER, context_chars INTEGER, response TEXT, error TEXT);
        """)
        # A reopened ledger preserves all charges; no fresh budget is silently granted.
        db.commit()

    def reserve(self, role: str, endpoint: Endpoint, request_hash: str, images: int, chars: int):
        count, spend = self.db.execute("SELECT COUNT(*),COALESCE(SUM(COALESCE(cost,reserved)),0) "
                                       "FROM model_calls").fetchone()
        reservation = endpoint.reservation_usd if endpoint.billing == "metered" else 0
        if count >= self.max_calls or spend + reservation > self.max_usd:
            raise BudgetExhausted("Run admission budget exhausted")
        id = "call_" + uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO model_calls VALUES(?,?,?,?,?,?,NULL,NULL,NULL,?,?,NULL,NULL)",
                (id, role, endpoint.model, request_hash, "pending", reservation, images, chars))
        return id

    def finish(self, id, *, endpoint: Endpoint, response=None, elapsed=0, error=None):
        usage = response.get("usage") if isinstance(response, dict) else None
        cost = 0.0 if endpoint.billing == "local" else None
        # Billing can be known even when JSON/schema validation subsequently fails.
        reported_tier = response.get("service_tier") if isinstance(response, dict) else None
        tier_matches = endpoint.service_tier is None or reported_tier == endpoint.service_tier
        model_matches = endpoint.openrouter is None or (isinstance(response, dict) and response.get("model") == endpoint.model)
        if endpoint.billing == "metered" and tier_matches and model_matches and isinstance(usage, dict):
            inp, out = usage.get("prompt_tokens"), usage.get("completion_tokens")
            detail = usage.get("prompt_tokens_details") or {}
            cached = detail.get("cached_tokens", 0) if isinstance(detail, dict) else None
            if (type(inp) is int and type(out) is int and type(cached) is int
                    and inp >= 0 and out >= 0 and 0 <= cached <= inp):
                rate = endpoint.cached_input_per_million
                if not cached or rate is not None:
                    cost = ((inp - cached) * endpoint.input_per_million +
                            cached * (rate or 0) + out * endpoint.output_per_million) / 1e6
            if endpoint.openrouter is not None:
                # OpenRouter's invoice includes provider/cache pricing dimensions.
                actual = usage.get("cost")
                cost = float(actual) if type(actual) in (int, float) and math.isfinite(actual) and actual >= 0 else None
        try:
            usage_json = canonical(usage) if usage is not None else None
            response_json = canonical(response) if response is not None else None
        except (ValueError, TypeError):
            usage_json = response_json = None
            cost, error = None, "InvalidResponseJSON"
        with self.db:
            self.db.execute("UPDATE model_calls SET status=?,cost=?,usage=?,elapsed=?,response=?,error=? WHERE id=?",
                ("failed" if error else "completed", cost,
                 usage_json, elapsed, response_json, error, id))

    def summary(self):
        rows = list(self.db.execute("SELECT * FROM model_calls"))
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
        missing_usage = 0
        tiers = {}
        for row in rows:
            usage = json.loads(row["usage"]) if row["usage"] else None
            if not isinstance(usage, dict):
                missing_usage += 1
            else:
                values = {"prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                        if isinstance(usage.get("prompt_tokens_details") or {}, dict) else None,
                    "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                        if isinstance(usage.get("completion_tokens_details") or {}, dict) else None}
                for key, value in values.items():
                    if type(value) is int and value >= 0:
                        tokens[key] += value
                if any(type(values[k]) is not int or values[k] < 0
                       for k in ("prompt_tokens", "completion_tokens")):
                    missing_usage += 1
            response = json.loads(row["response"]) if row["response"] else None
            tier = response.get("service_tier") if isinstance(response, dict) else None
            tier = tier if isinstance(tier, str) else "unreported"
            tiers[tier] = tiers.get(tier, 0) + 1
        return {"calls": len(rows), "failed_calls": sum(r["status"] != "completed" for r in rows),
                "known_provider_usd": sum(r["cost"] or 0 for r in rows),
                "unknown_charge_calls": sum(r["cost"] is None for r in rows),
                "outstanding_reserved_usd": sum(r["reserved"] for r in rows if r["cost"] is None),
                "model_wall_s": sum(r["elapsed"] or 0 for r in rows),
                "image_submissions": sum(r["images"] for r in rows),
                "reported_tokens": tokens, "calls_without_complete_token_usage": missing_usage,
                "reported_service_tiers": tiers,
                "local_compute_cost": None,
                "note": "Provider usage is observed when available. Local hosting, energy and storage are not free."}


class ChatBackend:
    def __init__(self, config: GameConfig, ledger: Ledger, *, allow_network=False,
                 allow_paid=False, transport=None):
        if not allow_network:
            raise ContractError("Network disabled. Pass --allow-network explicitly, including for localhost")
        if any(e.billing == "metered" for e in config.endpoints.values()) and not allow_paid:
            raise ContractError("Metered endpoints additionally require --allow-paid")
        for ep in config.endpoints.values():
            if ep.model.startswith("SET_"):
                raise ContractError("Configure an exact model identifier before live requests")
        self.config, self.ledger = config, ledger
        self.client = httpx.Client(transport=transport, follow_redirects=False, trust_env=False)

    def close(self):
        self.client.close()

    def _prepare(self, role, system, context, images, output):
        ep = self.config.endpoints[self.config.roles[role]]
        if len(images) > self.config.max_images:
            raise ContractError("Too many image attachments")
        text = canonical(context)
        if len(text) > self.config.max_context_chars:
            raise ContractError("Context character cap exceeded; this is not a token estimate")
        schema = output.model_json_schema()
        if role == "act":
            schema["properties"]["action_id"]["enum"] = [a.id for a in self.config.actions]
        elif role == "plan":
            offered = {e["id"] for e in context.get("world", {}).get("entities", [])}
            offered |= {e["id"] for r in context.get("retrieved", []) for e in r.get("entities", [])}
            if offered:
                schema["properties"]["focus_ids"]["items"]["enum"] = sorted(offered)
            else:
                schema["properties"]["focus_ids"]["maxItems"] = 0
        # Stable definitions precede dynamic state so compatible providers have an
        # opportunity for prefix reuse. This does NOT implement or guarantee KV reuse.
        stable_keys = ("goal", "ontology", "actions")
        stable = {k: context[k] for k in stable_keys if k in context}
        dynamic = {k: v for k, v in context.items() if k not in stable_keys}
        system_packet = (system + "\nStable task and runtime definitions:\n" + canonical(stable)
                         + "\nReturn only one JSON object matching:\n" + canonical(schema))
        user = [{"type": "text", "text": canonical(dynamic)}]
        for image in images:
            mime = "image/png" if image.data.startswith(b"\x89PNG") else "image/jpeg"
            user += [{"type": "text", "text": f"Evidence {image.id}; historical={image.historical}"},
                     {"type": "image_url", "image_url": {"url":
                      f"data:{mime};base64," + base64.b64encode(image.data).decode()}}]
        body = {"model": ep.model, "messages": [
            {"role": "system", "content": system_packet},
            {"role": "user", "content": user}], ep.token_parameter: min(ep.max_output_tokens, self.config.role_output_limits[role]),
            **ep.extra_body}
        if ep.service_tier is not None:
            body["service_tier"] = ep.service_tier
        if ep.reasoning_effort is not None:
            if ep.openrouter is not None:
                body["reasoning"] = {"effort": ep.reasoning_effort}
            else:
                body["reasoning_effort"] = ep.reasoning_effort
        if ep.openrouter is not None:
            body["provider"] = ep.openrouter.model_dump()
        if ep.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        elif ep.response_format == "json_schema":
            # Provider support varies; default json_object + strict local validation is more portable.
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": output.__name__, "schema": schema, "strict": False}}
        key = os.environ.get(ep.api_key_env, "") if ep.api_key_env else ""
        if ep.api_key_env and not key:
            raise ContractError("Configured API-key environment variable is unset")
        headers = {"Authorization": "Bearer " + key} if key else {}
        call = self.ledger.reserve(role, ep, hashlib.sha256(canonical(body).encode()).hexdigest(),
                                   len(images), len(text))
        with self.ledger.db:
            self.ledger.db.execute("INSERT INTO model_inputs VALUES(?,?,?,?)", (call, text, system,
                canonical([{"id": i.id, "historical": i.historical, "sha256": hashlib.sha256(i.data).hexdigest()}
                           for i in images])))
        return call, ep, body, headers

    def _request(self, ep, body, headers, output):
        start = time.monotonic()
        response = None
        try:
            r = self.client.post(ep.base_url.rstrip("/") + "/chat/completions", json=body,
                                 headers=headers, timeout=ep.timeout_s)
            r.raise_for_status()
            if len(r.content) > 2_000_000:
                raise ModelError("Oversized response")
            response = r.json()
            canonical(response)  # Reject non-JSON numeric values before accepting an action.
            if ep.openrouter is not None and response.get("model") != ep.model:
                raise ModelError("Served model does not match requested model")
            if ep.openrouter is not None and ep.service_tier is not None and response.get("service_tier") != ep.service_tier:
                raise ModelError("Served tier is unconfirmed or mismatched")
            choice = response["choices"][0]
            if choice.get("finish_reason") not in (None, "stop"):
                raise ModelError("Truncated, refused, or tool-only response")
            content = choice["message"].get("content")
            if not isinstance(content, str):
                raise ModelError("Expected textual JSON content")
            value = output.model_validate_json(content)
        except Exception as exc:
            # Never store raw exception strings which could contain provider credentials/URLs.
            error = type(exc).__name__
            if isinstance(exc, httpx.HTTPStatusError):
                error += ":HTTP_" + str(exc.response.status_code)
            return response, None, time.monotonic() - start, error
        return response, value, time.monotonic() - start, None

    def submit(self, executor: Executor, role, system, context, images, output):
        call, ep, body, headers = self._prepare(role, system, context, images, output)
        try:
            future = executor.submit(self._request, ep, body, headers, output)
        except Exception as exc:
            self.ledger.finish(call, endpoint=ep, error=type(exc).__name__)
            raise
        return PendingCompletion(future, self, call, ep, role)

    def complete(self, role, system, context, images, output):
        call, ep, body, headers = self._prepare(role, system, context, images, output)
        future = Future()
        future.set_result(self._request(ep, body, headers, output))
        return PendingCompletion(future, self, call, ep, role).result()
