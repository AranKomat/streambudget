import asyncio
import json
import httpx
import pytest

from streambudget.backend import ModelPool, Request, ImageInput, BackendError
from streambudget.budget import Ledger, BudgetExceeded
from streambudget.config import Config, ModelConfig, Prices, BudgetConfig
from streambudget.trace import Trace


def make_pool(tmp_path, handler, **kwargs):
    cfg = ModelConfig(kind="chat", model="fixture-vlm", base_url="https://test.invalid/v1",
                      max_retries=kwargs.pop("max_retries", 0),
                      prices=Prices(input_per_million=1, output_per_million=2,
                                    cached_input_per_million=.1), **kwargs)
    config = Config(models={"perception": cfg})
    trace = Trace(tmp_path / "trace.jsonl")
    ledger = Ledger(config.budget, trace)
    pool = ModelPool(config, ledger, trace, allow_network=True, transport=httpx.MockTransport(handler))
    return pool, ledger


def reply(content='{"text":"ok"}', usage=True):
    data = {"id": "fixture", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage:
        data["usage"] = {"prompt_tokens": 100, "completion_tokens": 20,
                         "prompt_tokens_details": {"cached_tokens": 30}}
    return httpx.Response(200, json=data)


async def test_payload_timestamps_usage_and_no_secret_logs(tmp_path, image_bytes, monkeypatch):
    monkeypatch.setenv("VLM_API_KEY", "SECRET_SENT_ONLY_AS_HEADER")
    captured = []
    def handler(req):
        captured.append(req)
        return reply()
    pool, ledger = make_pool(tmp_path, handler)
    result = await pool.call("perception", Request("answer", "system", "look",
        [ImageInput("frame-id", 12.5, image_bytes(True))]))
    body = json.loads(captured[0].content)
    content = body["messages"][1]["content"]
    assert "12.500000" in content[1]["text"]
    assert content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert ledger.reported_usd == pytest.approx((70 + 3 + 40) / 1e6)
    assert ledger.cached_input_tokens == 30
    assert result.json()["text"] == "ok"
    assert "SECRET" not in (tmp_path / "trace.jsonl").read_text()
    await pool.close()


async def test_opt_in_required_before_billing(tmp_path):
    pool, ledger = make_pool(tmp_path, lambda _: reply())
    pool.allow_network = False
    with pytest.raises(BackendError):
        await pool.call("perception", Request("answer", "s", "q"))
    assert ledger.requests == 0
    await pool.close()


async def test_retry_is_charged_separately(tmp_path):
    n = 0
    def handler(req):
        nonlocal n
        n += 1
        return httpx.Response(429, json={"error": "busy"}) if n == 1 else reply()
    pool, ledger = make_pool(tmp_path, handler, max_retries=1)
    await pool.call("perception", Request("answer", "s", "q"))
    assert ledger.requests == 2
    assert ledger.unknown_attempts == 1
    assert ledger.provisional_usd > 0
    await pool.close()


async def test_no_silent_retry_on_schema_rejection(tmp_path):
    pool, ledger = make_pool(tmp_path, lambda _: httpx.Response(400), max_retries=3)
    with pytest.raises(BackendError):
        await pool.call("perception", Request("answer", "s", "q"))
    assert ledger.requests == 1
    await pool.close()


async def test_timeout_billing_uncertain_not_free(tmp_path):
    def handler(req):
        raise httpx.ReadTimeout("timeout", request=req)
    pool, ledger = make_pool(tmp_path, handler, max_retries=3)
    with pytest.raises(BackendError):
        await pool.call("perception", Request("answer", "s", "q"))
    assert ledger.requests == 1
    assert ledger.provisional_usd > 0
    assert ledger.unknown_attempts == 1
    await pool.close()


async def test_exact_concurrent_requests_share_one_attempt(tmp_path):
    async def handler(req):
        await asyncio.sleep(.02)
        return reply()
    pool, ledger = make_pool(tmp_path, handler)
    req = Request("answer", "s", "q")
    await asyncio.gather(pool.call("perception", req), pool.call("perception", req))
    assert ledger.requests == 1
    await pool.close()


async def test_different_timestamps_do_not_reuse_semantic_answer(tmp_path, image_bytes):
    pool, ledger = make_pool(tmp_path, lambda _: reply())
    for t in [1, 2]:
        await pool.call("perception", Request("answer", "s", "q", [ImageInput("id", t, image_bytes())]))
    assert ledger.requests == 2
    await pool.close()


async def test_missing_usage_remains_estimated(tmp_path):
    pool, ledger = make_pool(tmp_path, lambda _: reply(usage=False))
    await pool.call("perception", Request("answer", "s", "q"))
    assert ledger.reported_usd == 0
    assert ledger.provisional_usd > 0
    assert ledger.unknown_attempts == 1
    await pool.close()


async def test_atomic_budget_reservation(tmp_path):
    trace = Trace(tmp_path / "trace")
    ledger = Ledger(BudgetConfig(max_requests=1), trace)
    out = await asyncio.gather(ledger.reserve(None), ledger.reserve(None), return_exceptions=True)
    assert sum(isinstance(x, BudgetExceeded) for x in out) == 1


async def test_unknown_prices_cannot_pass_dollar_budget(tmp_path):
    with pytest.raises(ValueError):
        Config(models={"perception": ModelConfig(kind="chat")}, budget=BudgetConfig(max_usd=1))


async def test_malformed_json_is_not_a_successful_tool_result(tmp_path):
    pool, ledger = make_pool(tmp_path, lambda _: reply("not JSON"))
    result = await pool.call("perception", Request("answer", "s", "q"))
    with pytest.raises(BackendError):
        result.json()
    assert ledger.requests == 1
    await pool.close()


async def test_truncated_output_rejected(tmp_path):
    def handler(req):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length",
                                    "message": {"content": '{"truncated":'}}]})
    pool, ledger = make_pool(tmp_path, handler)
    with pytest.raises(BackendError):
        await pool.call("perception", Request("answer", "s", "q"))
    await pool.close()


@pytest.mark.parametrize("usage", [
    {}, {"prompt_tokens": -1, "completion_tokens": 0},
    {"prompt_tokens": "100", "completion_tokens": 1},
    {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": {"cached_tokens": 101}},
    {"prompt_tokens": 100, "completion_tokens": 1, "prompt_tokens_details": "invalid"},
])
async def test_invalid_usage_stays_provisional_not_free(tmp_path, usage):
    def handler(req):
        response = reply().json()
        response["usage"] = usage
        return httpx.Response(200, json=response)
    pool, ledger = make_pool(tmp_path, handler)
    await pool.call("perception", Request("answer", "s", "q"))
    assert ledger.provisional_usd > 0
    assert ledger.summary()["complete_reported_usd"] is None
    assert ledger.reserved_usd == pytest.approx(0)
    await pool.close()


@pytest.mark.parametrize("quote", [-1, float("nan"), float("inf")])
async def test_invalid_reservation_rejected(tmp_path, quote):
    ledger = Ledger(BudgetConfig(), Trace(tmp_path / "trace"))
    with pytest.raises(ValueError):
        await ledger.reserve(quote)
    assert ledger.requests == 0
