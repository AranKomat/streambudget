import pytest
from streambudget.agent import Action, Tools
from streambudget.config import Config
from streambudget.runtime import Runtime
from streambudget.types import ContractError, Check, Watch, WatchState


def test_dwell_and_duplicate_suppression():
    s = WatchState(Watch(id="x", source="cam", goal="condition", dwell_seconds=3))
    yes = Check(watch_id="x", status="yes", confidence=.9)
    assert not s.observe(yes, 1)
    assert not s.observe(yes, 3)
    assert s.observe(yes, 4)
    assert not s.observe(yes, 5)


def test_unknown_and_stale_gaps_break_dwell():
    s = WatchState(Watch(id="x", source="cam", goal="x", dwell_seconds=3, max_observation_gap=2))
    yes = Check(watch_id="x", status="yes", confidence=.9)
    assert not s.observe(yes, 1)
    assert not s.observe(yes, 5)
    assert s.positive_since == 5
    assert not s.observe(Check(watch_id="x", status="unknown", confidence=.9), 6)
    assert s.positive_since is None


def test_old_jobs_cannot_rewind_state():
    s = WatchState(Watch(id="x", source="cam", goal="x", dwell_seconds=3))
    s.observe(Check(watch_id="x", status="no", confidence=1), 10)
    assert not s.observe(Check(watch_id="x", status="yes", confidence=1), 5)
    assert s.last_observed == 10
    assert s.positive_since is None


async def test_multiple_watches_share_visual_call(runtime, image_bytes):
    for i in range(4):
        runtime.register_watch(Watch(id=f"w{i}", source="cam", goal="Red box visible"))
    await runtime.ingest_frame("cam", 0, image_bytes(True))
    await runtime.drain()
    assert runtime.ledger.requests == 1
    assert len(runtime.alerts) == 4


async def test_stale_frame_is_not_reinterpreted_as_new_timer_evidence(runtime, image_bytes):
    runtime.register_watch(Watch(id="w", source="cam", goal="Red box visible", dwell_seconds=3))
    await runtime.ingest_frame("cam", 0, image_bytes(True))
    await runtime.drain()
    await runtime.tick(10)
    await runtime.drain()
    assert not runtime.alerts


async def test_sensor_rule_requires_no_llm(runtime):
    runtime.register_watch(Watch(id="temp", source="machine", goal="temperature high",
        sensor_predicate={"key": "temp", "op": "gt", "value": 80}))
    await runtime.ingest_signal("machine", 1, "sensor", "temp", {"temp": 70})
    await runtime.ingest_signal("machine", 2, "sensor", "temp", {"temp": 90})
    assert len(runtime.alerts) == 1
    assert runtime.ledger.requests == 0


async def test_future_query_is_refused(runtime):
    with pytest.raises(ContractError):
        await runtime.ask("What?", "cam", as_of=100)


async def test_inspect_uses_only_snapshot_frames(runtime, image_bytes):
    await runtime.ingest_frame("cam", 1, image_bytes(False), schedule=False)
    snap = runtime.snapshot()
    await runtime.ingest_frame("cam", 2, image_bytes(True), schedule=False)
    tools = Tools(runtime, snap)
    with pytest.raises(ContractError):
        await tools.run(Action(tool="inspect", arguments={"source": "cam", "start": 0, "end": 2}))
    result = await tools.run(Action(tool="inspect", arguments={"source": "cam", "start": 0, "end": 1}))
    assert result["sampled_timestamps"] == [1]
    assert "empty" in result["evidence"][0]["text"]


async def test_search_and_inspect_agent_loop(runtime, image_bytes):
    await runtime.ingest_frame("cam", 0, image_bytes(True))
    await runtime.drain()
    answer = await runtime.ask("What was here?", "cam")
    assert answer.status == "ok"
    assert [s["tool"] for s in answer.tool_steps] == ["search", "inspect"]
    assert answer.evidence_ids
    for id in answer.evidence_ids:
        runtime.store.get(id, runtime.snapshot(0))


async def test_compaction_preserves_source_provenance(runtime, image_bytes):
    await runtime.ingest_frame("cam", 0, image_bytes(True))
    await runtime.drain()
    snap = runtime.snapshot()
    captions = runtime.store.list(snap, kinds=["caption"])
    summary = await runtime.compact("cam", [e.id for e in captions], snap)
    assert summary.parents
    assert summary.input_end == 0


async def test_cancel_watch(runtime):
    runtime.register_watch(Watch(id="x", source="cam", goal="x"))
    runtime.cancel_watch("x")
    assert not runtime.active_watches("cam", 1)


async def test_new_watch_does_not_claim_past_monitoring(runtime):
    runtime.advance(10)
    runtime.register_watch(Watch(id="x", source="cam", goal="x", created_at=0))
    assert runtime.watches["x"].watch.created_at == 10


async def test_motion_baseline_does_not_inherit_adaptive_timer_rechecks(tmp_path, image_bytes):
    from streambudget.config import Config
    from streambudget.runtime import Runtime
    from streambudget.types import Watch, Check
    cfg = Config()
    cfg.policy.mode = "motion"
    cfg.policy.refresh_interval_s = 100
    r = Runtime(cfg, tmp_path / "runtime")
    r.register_watch(Watch(id="dwell", source="cam", goal="Red box", dwell_seconds=10))
    await r.ingest_frame("cam", 0, image_bytes(True))
    await r.drain()
    calls = r.ledger.requests
    await r.ingest_frame("cam", 2, image_bytes(True))
    await r.tick(2)
    await r.drain()
    assert r.ledger.requests == calls
    await r.close()
