import importlib.util
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from streambudget.config import BudgetConfig, Config, ModelConfig, Prices
from streambudget.budget import BudgetExceeded, Ledger
from streambudget.backend import ModelPool
from streambudget.screening import build_cases, fingerprint, packet, probe, score
from streambudget.trace import Trace
from streambudget.footage_screen import FRAME_INDICES, build_footage_cases
from PIL import Image

spec = importlib.util.spec_from_file_location("screen_runner", Path(__file__).parents[1] / "scripts/screen_models.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_balanced_frozen_packets_and_separate_labels():
    cases, labels = build_cases()
    assert len(cases) == len(labels) == 20
    assert fingerprint(cases) == fingerprint(build_cases()[0])
    assert all(sum(v["answer"] == key for v in labels.values()) == 5 for key in "ABCD")
    for case in cases:
        public = packet(case)
        assert "anchor_groups" not in json.dumps(public)
        assert case.request.context == {}
        assert [im.timestamp for im in case.request.images] == list(range(8))
        allowed = {im.evidence_id for im in case.request.images}
        assert all(set(group) <= allowed for group in labels[case.id]["anchor_groups"])


def test_scoring_unknown_ids_missing_evidence_and_wrong_answer():
    case, label = probe()
    good = {"answer": "C", "evidence_ids": ["probe-f0"], "reason": "Red object"}
    assert score(good, case, label)["supported_correct"]
    assert not score({**good, "evidence_ids": ["future"]}, case, label)["citations_valid"]
    assert not score({**good, "evidence_ids": []}, case, label)["anchor_coverage"]
    assert not score({**good, "answer": "B"}, case, label)["answer_correct"]
    assert not score({**good, "answer": ["C"]}, case, label)["schema_valid"]
    assert not score({**good, "evidence_ids": ["probe-f0", "probe-f0"]}, case, label)["citations_valid"]


def test_carryover_never_releases_unknown_holds():
    previous = {"status": "complete", "all_passed": True, "ledger": {
        "request_attempts": 3, "reported_usd": .2, "provisional_usd": .3,
        "reserved_usd": .4, "unpriced_attempts": 0}}
    budget = runner.remaining_budget(BudgetConfig(max_requests=63, max_usd=3), previous)
    assert budget.max_requests == 60
    assert budget.max_usd == pytest.approx(2.1)
    with pytest.raises(ValueError):
        runner.remaining_budget(budget, {**previous, "status": "running"})
    with pytest.raises(ValueError):
        runner.remaining_budget(budget, {**previous, "all_passed": False})


def config():
    m = ModelConfig(kind="chat", model="test", max_retries=0,
                    prices=Prices(input_per_million=1, output_per_million=1))
    return Config(models={"perception": m, "qwen": m.model_copy(update={"model": "q"}),
                          "glm": m.model_copy(update={"model": "g"})},
                  budget=BudgetConfig(max_requests=63, max_usd=3))


async def test_probes_failures_count_and_directory_cannot_redispatch(tmp_path):
    cfg = config()
    out = tmp_path / "run"
    runner.prepare(cfg, out)
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    with pytest.raises(ValueError, match="allow-network"):
        await runner.run_phase(cfg, out, "probe", transport=transport)
    result = await runner.run_phase(cfg, out, "probe", allow_network=True, transport=transport)
    assert len(calls) == len(result["rows"]) == 3
    assert result["ledger"]["unknown_usage_attempts"] == 3
    assert result["ledger"]["provisional_usd"] > 0
    assert not result["all_passed"]
    with pytest.raises(FileExistsError):
        await runner.run_phase(cfg, out, "probe", allow_network=True, transport=transport)
    with pytest.raises(ValueError, match="pass"):
        await runner.run_phase(cfg, out, "screen", allow_network=True, transport=transport)
    assert len(calls) == 3


async def test_success_probe_and_screen_share_cap_and_identical_packets(tmp_path):
    cfg = config()
    out = tmp_path / "run"
    runner.prepare(cfg, out)
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        return httpx.Response(200, json={"id": "fixture", "usage": {
            "prompt_tokens": 100, "completion_tokens": 20}, "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps({
                    "answer": "C", "reason": "Synthetic test", "evidence_ids": ["probe-f0"]})}}]})

    transport = httpx.MockTransport(handler)
    first = await runner.run_phase(cfg, out, "probe", allow_network=True, transport=transport)
    assert first["all_passed"]
    second = await runner.run_phase(cfg, out, "screen", allow_network=True, transport=transport)
    assert len(second["rows"]) == 60
    assert second["ledger"]["request_attempts"] == len(calls) == 63
    assert second["ledger"]["reported_usd"] == pytest.approx(63 * 120 / 1e6)
    assert all(not row["score"]["supported_correct"] for row in second["rows"])
    packets = {}
    for call in calls[3:]:
        packets.setdefault(call["model"], []).append(call["messages"])
    assert packets["test"] == packets["q"] == packets["g"]
    report = runner.summarize(out)
    assert all(m["scores"]["supported_correct"] == 0 for m in report["models"].values())
    assert all(m["planned_cases"] == 20 for m in report["models"].values())
    assert all(m["provider_reported_usd_including_probe"] is None for m in report["models"].values())


async def test_changed_config_rejected_before_dispatch(tmp_path):
    cfg = config()
    out = tmp_path / "run"
    runner.prepare(cfg, out)
    cfg.models["glm"].temperature = 1
    with pytest.raises(ValueError, match="changed"):
        await runner.run_phase(cfg, out, "probe", allow_network=True)


async def test_provider_reported_cost_controls_next_admission(tmp_path):
    ledger = Ledger(BudgetConfig(max_usd=.1), Trace(tmp_path / "trace"))
    ticket = await ledger.reserve(.001)
    await ledger.settle(ticket, role="test", prices=Prices(input_per_million=1, output_per_million=1),
                        usage={"prompt_tokens": 100, "completion_tokens": 20}, status="ok", provider_cost_usd=.09)
    assert ledger.reported_usd == .09
    assert ledger.input_tokens == 100
    with pytest.raises(BudgetExceeded):
        await ledger.reserve(.02)


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf"), True, "0.001"])
async def test_invalid_provider_cost_cannot_discount_ledger(tmp_path, invalid):
    ledger = Ledger(BudgetConfig(max_usd=1), Trace(tmp_path / "trace"))
    ticket = await ledger.reserve(.01)
    await ledger.settle(ticket, role="test", prices=Prices(input_per_million=1, output_per_million=1),
                        usage={"prompt_tokens": 100, "completion_tokens": 20}, status="ok",
                        provider_cost_usd=invalid)
    assert ledger.reported_usd == pytest.approx(.00012)


@pytest.mark.parametrize("endpoint,expected", [("https://openrouter.ai/api/v1", .02),
                                              ("https://example.test/v1", .00012)])
async def test_cost_field_only_trusted_for_supported_provider(tmp_path, endpoint, expected):
    cfg = config()
    cfg.models["perception"].base_url = endpoint
    trace = Trace(tmp_path / "trace")
    ledger = Ledger(cfg.budget, trace)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "cost": .02},
        "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}))
    pool = ModelPool(cfg, ledger, trace, allow_network=True, transport=transport)
    try:
        await pool.call("perception", probe()[0].request)
    finally:
        await pool.close()
    assert ledger.reported_usd == pytest.approx(expected)


def test_real_packet_separation_cutoffs_and_paths(tmp_path):
    for cid, indices in FRAME_INDICES.items():
        folder = tmp_path / cid / "prepared"
        folder.mkdir(parents=True)
        Image.new("RGB", (40, 80), "red").save(folder / "frame.jpg")
        (folder / "events.jsonl").write_text("\n".join(json.dumps({
            "ts": i / 2, "media": "frame.jpg"}) for i in range(max(indices) + 1)))
    cases, labels = build_footage_cases(tmp_path)
    assert len(cases) == len(labels) == 20
    for case in cases:
        assert len(case.request.images) == 8
        assert "anchor_groups" not in json.dumps(packet(case))
        assert "sweetlabs" not in json.dumps(case.request.context)
        assert f"{case.request.images[-1].timestamp:.6f}" in case.request.text
        assert all(i.timestamp <= case.request.images[-1].timestamp for i in case.request.images)
    bad = tmp_path / "c01/prepared/events.jsonl"
    rows = [json.loads(line) for line in bad.read_text().splitlines()]
    rows[0]["media"] = "../../../outside.jpg"
    bad.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="escapes"):
        build_footage_cases(tmp_path)


async def test_parallel_workers_and_verified_flex_route(tmp_path):
    cfg = config()
    cfg.models["perception"].extra_body = {"service_tier": "flex", "provider": {
        "only": ["google-ai-studio/flex"], "allow_fallbacks": False}}
    out = tmp_path / "run"
    runner.prepare(cfg, out)
    active = peak = 0

    async def handler(req):
        nonlocal active, peak
        body = json.loads(req.content)
        active += 1
        peak = max(active, peak)
        await asyncio.sleep(.01)
        active -= 1
        if body["model"] == "test":
            assert body["service_tier"] == "flex"
            assert body["provider"]["only"] == ["google-ai-studio/flex"]
        return httpx.Response(200, json={
            "provider": "Google AI Studio", "service_tier": "flex", "model": body["model"],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "answer": "C", "reason": "Test", "evidence_ids": ["probe-f0"]})}}]})

    transport = httpx.MockTransport(handler)
    p = await runner.run_phase(cfg, out, "probe", allow_network=True, transport=transport)
    assert p["all_passed"]
    result = await runner.run_phase(cfg, out, "screen", allow_network=True, transport=transport, concurrency=9)
    assert peak == 9
    assert result["ledger"]["request_attempts"] == 63
    assert all(r["routing"]["service_tier"] == "flex" for r in result["rows"])


async def test_wrong_tier_is_not_silently_accepted(tmp_path):
    cfg = config()
    cfg.models["perception"].extra_body = {"provider": {"only": ["google-ai-studio/flex"]}}
    out = tmp_path / "run"
    runner.prepare(cfg, out)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={
        "provider": "Google AI Studio", "service_tier": "default",
        "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "choices": [{
            "finish_reason": "stop", "message": {"content": json.dumps({
                "answer": "C", "reason": "Test", "evidence_ids": ["probe-f0"]})}}]}))
    result = await runner.run_phase(cfg, out, "probe", allow_network=True, transport=transport)
    assert not result["all_passed"]
    assert result["ledger"]["request_attempts"] == 3
    assert result["ledger"]["reported_usd"] > 0
    assert next(r for r in result["rows"] if r["role"] == "perception")["status"] == "error"
