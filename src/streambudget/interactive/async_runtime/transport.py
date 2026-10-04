"""Async vLLM-compatible transport. No database, memory, action or retry side effects.

The host owns request accounting and result acceptance. An HTTP cancellation does
not establish remote inference termination; the broker quarantines uncertain work.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass

import httpx

from ...types import ContractError
from ..ontology import canonical


@dataclass(frozen=True)
class RawCompletion:
    response: dict | None
    elapsed_s: float
    first_token_s: float | None = None
    first_content_s: float | None = None
    error: str | None = None
    termination_unknown: bool = False


def strict_json(text: str):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("Duplicate JSON key")
            result[k] = v
        return result

    return json.loads(
        text,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def request_body(endpoint, *, role, system, context, images, output, actions, output_limit, stream):
    schema = output.model_json_schema()
    if role == "act":
        schema["properties"]["action_id"]["enum"] = [a.id for a in actions]
    stable_keys = ("goal", "ontology", "actions")
    stable = {k: context[k] for k in stable_keys if k in context}
    dynamic = {k: v for k, v in context.items() if k not in stable_keys}
    text = (
        system + "\nStable definitions:\n" + canonical(stable) + "\nOutput JSON schema:\n" + canonical(schema)
    )
    user = [{"type": "text", "text": canonical(dynamic)}]
    for image in images:
        mime = "image/png" if image.data.startswith(b"\x89PNG") else "image/jpeg"
        user.extend(
            [
                {"type": "text", "text": f"Evidence {image.id}; historical={image.historical}"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64," + base64.b64encode(image.data).decode()},
                },
            ]
        )
    # Existing Endpoint validation allowlists decoding options. Check again at this boundary
    # in case an endpoint was created with model_construct or changed by a host plugin.
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "seed",
        "reasoning_effort",
        "chat_template_kwargs",
    }
    if set(endpoint.extra_body) - allowed:
        raise ContractError("Unsupported endpoint extra_body override")
    body = {
        "model": endpoint.model,
        "messages": [{"role": "system", "content": text}, {"role": "user", "content": user}],
        endpoint.token_parameter: min(output_limit, endpoint.max_output_tokens),
        "stream": stream,
        **endpoint.extra_body,
    }
    if getattr(endpoint, "reasoning_effort", None) is not None:
        body["reasoning_effort"] = endpoint.reasoning_effort
    if endpoint.response_format == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": output.__name__, "schema": schema, "strict": False},
        }
    elif endpoint.response_format == "json_object":
        body["response_format"] = {"type": "json_object"}
    if stream:
        body["stream_options"] = {"include_usage": True}
    canonical(body)  # Reject NaN and unsupported data before admission.
    return body


class VLLMTransport:
    """A client for the user's existing loopback/tunnel vLLM server, not an inference engine."""

    def __init__(self, *, allow_network=False, max_response_bytes=2_000_000, http_transport=None):
        if not allow_network:
            raise ContractError("Explicit --allow-network is required even for loopback vLLM")
        self.max_response_bytes = max_response_bytes
        self.client = httpx.AsyncClient(transport=http_transport, trust_env=False, follow_redirects=False)

    async def close(self):
        await self.client.aclose()

    async def invoke(self, endpoint, body) -> RawCompletion:
        # A deliberately narrower live boundary than the retained synchronous provider client.
        from urllib.parse import urlsplit

        u = urlsplit(endpoint.base_url)
        if endpoint.billing != "local" or u.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ContractError(
                "Background mode currently supports operator-owned loopback/tunneled vLLM only"
            )
        key = os.environ.get(endpoint.api_key_env, "") if endpoint.api_key_env else ""
        if endpoint.api_key_env and not key:
            raise ContractError("Configured credential environment variable is unset")
        headers = {"Authorization": "Bearer " + key} if key else {}
        start = time.monotonic()
        first = content_first = None
        response = None
        try:
            async with asyncio.timeout(endpoint.timeout_s):
                async with self.client.stream(
                    "POST",
                    endpoint.base_url.rstrip("/") + "/chat/completions",
                    json=body,
                    headers=headers,
                    timeout=endpoint.timeout_s,
                ) as r:
                    r.raise_for_status()
                    size = 0
                    if not body["stream"]:
                        pieces = []
                        async for chunk in r.aiter_bytes():
                            size += len(chunk)
                            if size > self.max_response_bytes:
                                raise ValueError("Oversized response")
                            pieces.append(chunk)
                        response = strict_json(b"".join(pieces).decode("utf-8"))
                    else:
                        chunks, reasoning, finish, model, usage = [], [], None, None, None
                        done = False
                        async for line in r.aiter_lines():
                            size += len(line.encode("utf-8"))
                            if size > self.max_response_bytes:
                                raise ValueError("Oversized SSE response")
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                done = True
                                break
                            if not payload:
                                continue
                            event = strict_json(payload)
                            if "error" in event:
                                raise ValueError("Server SSE error")
                            if event.get("model"):
                                if model and event["model"] != model:
                                    raise ValueError("Served model changed inside response")
                                model = event["model"]
                            if event.get("usage") is not None:
                                usage = event["usage"]
                            for c in event.get("choices", []):
                                if c.get("index", 0) != 0:
                                    raise ValueError("Multiple result choices not supported")
                                delta = c.get("delta") or {}
                                if delta.get("tool_calls") or delta.get("refusal"):
                                    raise ValueError("Non-action output")
                                think = delta.get("reasoning_content") or delta.get("reasoning")
                                text = delta.get("content")
                                if think or text:
                                    first = first if first is not None else time.monotonic() - start
                                if think:
                                    if not isinstance(think, str):
                                        raise ValueError("Invalid reasoning chunk")
                                    reasoning.append(think)
                                if text:
                                    if not isinstance(text, str):
                                        raise ValueError("Invalid content chunk")
                                    content_first = (
                                        content_first
                                        if content_first is not None
                                        else time.monotonic() - start
                                    )
                                    chunks.append(text)
                                if c.get("finish_reason") is not None:
                                    finish = c["finish_reason"]
                        # Never execute partial JSON, even if it happens to parse.
                        if not done or finish is None:
                            raise ConnectionError("Incomplete SSE response")
                        response = {
                            "model": model,
                            "choices": [
                                {
                                    "finish_reason": finish,
                                    "message": {
                                        "content": "".join(chunks),
                                        "reasoning_content": "".join(reasoning),
                                    },
                                }
                            ],
                            "usage": usage,
                        }
            return RawCompletion(response, time.monotonic() - start, first, content_first)
        except asyncio.CancelledError:
            # The runner must drain/quarantine this; remote cancellation is not assumed.
            raise
        except (httpx.TimeoutException, TimeoutError, httpx.TransportError, ConnectionError) as exc:
            return RawCompletion(
                response, time.monotonic() - start, first, content_first, type(exc).__name__, True
            )
        except Exception as exc:
            # A received server error/invalid stream can still leave remote work uncertain.
            return RawCompletion(
                response, time.monotonic() - start, first, content_first, type(exc).__name__, True
            )


def validate_completion(raw, output, *, expected_model, require_model=True):
    if raw.error:
        raise ValueError(raw.error)
    response = raw.response
    if not isinstance(response, dict):
        raise ValueError("Missing response object")
    if require_model and response.get("model") != expected_model:
        raise ValueError("Served model alias differs from configured served-model-name")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Exactly one response choice required")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Response is truncated/refused or otherwise not complete")
    message = choice.get("message") or {}
    if message.get("tool_calls") or message.get("refusal") or not isinstance(message.get("content"), str):
        raise ValueError("Expected completed textual structured output")
    body = strict_json(message["content"])
    canonical(body)
    return output.model_validate(body)
