import asyncio
import json

import httpx
import pytest

from streambudget.interactive.contracts import ActionChoice, Endpoint
from streambudget.interactive.async_runtime.contracts import SceneIndex
from streambudget.interactive.async_runtime.jobs import JobBroker
from streambudget.interactive.async_runtime.transport import (
    VLLMTransport,
    RawCompletion,
    validate_completion,
    request_body,
)
from streambudget.types import ContractError
from conftest import config, add_frame, basis


def response(model="fixture-model", content='{"action_id":"A"}', finish="stop"):
    return {
        "model": model,
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 7},
    }


class BlockedTransport:
    def __init__(self):
        self.events = {}
        self.active = 0
        self.peak = 0

    async def invoke(self, ep, body):
        name = body["messages"][0]["content"].split("\n")[0]
        ev = asyncio.Event()
        self.events[name] = ev
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await ev.wait()
            content = '{"action_id":"A"}' if name.startswith("ACT") else "{}"
            return RawCompletion(response(ep.model, content), 0.01)
        finally:
            self.active -= 1

    async def close(self):
        pass


def test_reserved_actor_slot_and_background_fairness(mem):
    async def test():
        m, s, media = mem
        f = add_frame(s, media, 0)
        t = BlockedTransport()
        c = config()
        broker = JobBroker(c, s.db, t)
        a = broker.submit("index", basis(m, f), "INDEX", {}, [], SceneIndex)
        enrich = broker.submit("enrich", basis(m, f), "ENRICH", {}, [], SceneIndex)
        broker.pump()
        await asyncio.sleep(0)
        assert a.status == "running" and enrich.status == "queued"
        actor = broker.submit("act", basis(m, f), "ACT1", {}, [], ActionChoice)
        broker.pump()
        await asyncio.sleep(0)
        assert actor.status == "running" and t.peak == 2
        t.events["INDEX"].set()
        await asyncio.sleep(0)
        broker.pump()
        await asyncio.sleep(0)
        assert enrich.status == "running"
        for e in t.events.values():
            e.set()
        await broker.close(1)

    asyncio.run(test())


def test_queue_capacity_preserves_actor_admission(mem):
    async def test():
        m, s, media = mem
        f = add_frame(s, media, 0)
        c = config(max_queue=2)
        b = JobBroker(c, s.db, BlockedTransport())
        b.submit("index", basis(m, f), "IDX1", {}, [], SceneIndex)
        y = b.submit("enrich", basis(m, f), "ENRICH", {}, [], SceneIndex)
        a = b.submit("act", basis(m, f), "ACT1", {}, [], ActionChoice)
        assert a.status == "queued" and y.status == "superseded_for_actor"
        b.cancel_queued()
        await b.close(1)
        assert b.ledger.summary()["calls"] == 0

    asyncio.run(test())


def test_latest_only_replaces_queued_not_running(mem):
    async def test():
        m, s, media = mem
        f = add_frame(s, media, 0)
        t = BlockedTransport()
        b = JobBroker(config(), s.db, t)
        a = b.submit("index", basis(m, f), "INDEX", {}, [], SceneIndex, key="scene")
        b.pump()
        await asyncio.sleep(0)
        x = b.submit("index", basis(m, f), "IDX2", {}, [], SceneIndex, key="scene")
        y = b.submit("index", basis(m, f), "IDX3", {}, [], SceneIndex, key="scene")
        assert a.status == "running" and x.status == "superseded" and y.status == "queued"
        t.events["INDEX"].set()
        b.cancel_queued()
        await b.close(1)
        assert b.ledger.summary()["calls"] == 1

    asyncio.run(test())


def test_snapshot_is_copied_before_dispatch(mem):
    async def test():
        m, s, media = mem
        f = add_frame(s, media, 0)
        b = JobBroker(config(), s.db, BlockedTransport())
        context = {"nested": {"value": "original"}}
        a = b.submit("index", basis(m, f), "INDEX", context, [], SceneIndex)
        context["nested"]["value"] = "mutated"
        assert a.context["nested"]["value"] == "original"
        b.cancel_queued()
        await b.close(1)

    asyncio.run(test())


def test_no_automatic_retry_and_failed_usage_retained(mem):
    class Broken:
        calls = 0

        async def invoke(self, ep, body):
            self.calls += 1
            return RawCompletion(None, 0.01, error="ReadTimeout", termination_unknown=True)

        async def close(self):
            pass

    async def test():
        m, s, media = mem
        f = add_frame(s, media, 0)
        t = Broken()
        b = JobBroker(config(), s.db, t)
        a = b.submit("index", basis(m, f), "INDEX", {}, [], SceneIndex)
        b.pump()
        await asyncio.sleep(0)
        b.pump()
        assert b.quarantined and a.status == "failed" and t.calls == 1
        with pytest.raises(ContractError):
            b.submit("act", basis(m, f), "ACT", {}, [], ActionChoice)
        await b.close(1)
        assert b.ledger.summary()["failed_calls"] == 1

    asyncio.run(test())


def test_total_call_budget_applies_across_roles(mem):
    class Immediate:
        async def invoke(self, ep, body):
            return RawCompletion(response(ep.model, "{}"), 0.01)

        async def close(self):
            pass

    async def test():
        m, s, media = mem
        f = add_frame(s, media, 0)
        c = config()
        c.game.max_calls = 1
        b = JobBroker(c, s.db, Immediate())
        b.submit("index", basis(m, f), "INDEX", {}, [], SceneIndex)
        second = b.submit("enrich", basis(m, f), "ENRICH", {}, [], SceneIndex)
        b.pump()
        await asyncio.sleep(0)
        b.pump()
        assert second.status == "rejected_budget"
        await b.close(1)
        assert b.ledger.summary()["calls"] == 1

    asyncio.run(test())


class Pieces(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        for i in range(0, len(self.content), 7):
            yield self.content[i : i + 7]
            await asyncio.sleep(0)


def sse(content='{"action_id":"A"}', finish="stop", done=True, model="fixture-model"):
    chunks = []
    for ch in content:
        chunks.append(
            "data: "
            + json.dumps({"model": model, "choices": [{"index": 0, "delta": {"content": ch}}]})
            + "\n\n"
        )
    chunks.append(
        "data: "
        + json.dumps({"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        + "\n\n"
    )
    chunks.append(
        "data: "
        + json.dumps({"model": model, "choices": [], "usage": {"prompt_tokens": 14, "completion_tokens": 9}})
        + "\n\n"
    )
    if done:
        chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode()


def test_streaming_transport_waits_for_complete_result_and_records_usage():
    async def test():
        ep = Endpoint(model="fixture-model")
        seen = []

        def handler(req):
            seen.append(json.loads(req.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Pieces(sse()))

        t = VLLMTransport(allow_network=True, http_transport=httpx.MockTransport(handler))
        body = request_body(
            ep,
            role="act",
            system="actor",
            context={},
            images=[],
            output=ActionChoice,
            actions=config().game.actions,
            output_limit=32,
            stream=True,
        )
        raw = await t.invoke(ep, body)
        value = validate_completion(raw, ActionChoice, expected_model=ep.model)
        assert value.action_id == "A" and raw.first_content_s is not None
        assert raw.response["usage"]["completion_tokens"] == 9
        assert seen[0]["stream_options"] == {"include_usage": True}
        await t.close()

    asyncio.run(test())


@pytest.mark.parametrize(
    "text",
    [
        '{"action_id":"A","action_id":"B"}',
        '{"action_id":"A","oops":1}',
        '{"action_id":7}',
        '{"action_id":"A"',
        "not json",
    ],
)
def test_malformed_action_rejected(text):
    with pytest.raises((ValueError, TypeError)):
        validate_completion(
            RawCompletion(response(content=text), 0.01), ActionChoice, expected_model="fixture-model"
        )


@pytest.mark.parametrize("finish", ["length", "tool_calls", "content_filter", None])
def test_unfinished_result_cannot_be_action(finish):
    with pytest.raises(ValueError):
        validate_completion(
            RawCompletion(response(finish=finish), 0.01), ActionChoice, expected_model="fixture-model"
        )


def test_wrong_served_model_rejected():
    with pytest.raises(ValueError):
        validate_completion(
            RawCompletion(response(model="other"), 0.01), ActionChoice, expected_model="fixture-model"
        )


def test_incomplete_sse_marks_remote_termination_unknown():
    async def test():
        t = VLLMTransport(
            allow_network=True,
            http_transport=httpx.MockTransport(
                lambda req: httpx.Response(200, stream=Pieces(sse(done=False)))
            ),
        )
        raw = await t.invoke(Endpoint(model="fixture-model"), {"stream": True})
        assert raw.termination_unknown and raw.error
        await t.close()

    asyncio.run(test())


def test_transport_needs_explicit_network_opt_in():
    with pytest.raises(ContractError):
        VLLMTransport()


def test_background_transport_rejects_metered_provider():
    async def test():
        t = VLLMTransport(
            allow_network=True, http_transport=httpx.MockTransport(lambda r: httpx.Response(200))
        )
        ep = Endpoint(
            model="m",
            base_url="https://example.com/v1",
            billing="metered",
            input_per_million=1,
            output_per_million=1,
        )
        with pytest.raises(ContractError):
            await t.invoke(ep, {})
        await t.close()

    asyncio.run(test())


def test_request_preserves_thinking_profiles_and_short_actor_cap():
    ep = Endpoint(
        model="m", reasoning_effort="medium", extra_body={"chat_template_kwargs": {"enable_thinking": True}}
    )
    b = request_body(
        ep,
        role="act",
        system="x",
        context={"goal": "g", "ontology": {}, "actions": []},
        images=[],
        output=ActionChoice,
        actions=config().game.actions,
        output_limit=32,
        stream=False,
    )
    assert b["max_tokens"] == 32 and b["reasoning_effort"] == "medium"
    assert b["chat_template_kwargs"]["enable_thinking"]
    assert b["stream"] is False


def test_hard_wall_deadline_not_just_socket_read_timeout():
    async def test():
        async def handler(req):
            await asyncio.sleep(0.1)
            return httpx.Response(200, json=response())

        t = VLLMTransport(allow_network=True, http_transport=httpx.MockTransport(handler))
        raw = await t.invoke(Endpoint(model="fixture-model", timeout_s=0.01), {"stream": False})
        assert raw.termination_unknown and raw.error == "TimeoutError"
        await t.close()

    asyncio.run(test())


def test_nonstream_transport_also_validates_schema():
    async def test():
        t = VLLMTransport(
            allow_network=True,
            http_transport=httpx.MockTransport(lambda req: httpx.Response(200, json=response())),
        )
        raw = await t.invoke(Endpoint(model="fixture-model"), {"stream": False})
        assert validate_completion(raw, ActionChoice, expected_model="fixture-model").action_id == "A"
        assert raw.first_content_s is None
        await t.close()

    asyncio.run(test())


def test_real_loopback_http_sse_without_any_model():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(payload)
            assert body["stream"] is True
            data = sse()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def test():
        transport = VLLMTransport(allow_network=True)
        try:
            ep = Endpoint(model="fixture-model", base_url=f"http://127.0.0.1:{server.server_port}/v1")
            raw = await transport.invoke(ep, {"stream": True})
            assert validate_completion(raw, ActionChoice, expected_model="fixture-model").action_id == "A"
        finally:
            await transport.close()

    try:
        asyncio.run(test())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
