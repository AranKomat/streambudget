import asyncio
import json
import sqlite3

import pytest

from streambudget.interactive.fixture import FixtureEnvironment
from streambudget.interactive.async_runtime.fixture import FixtureTransport
from streambudget.interactive.async_runtime.runner import AsyncGameRunner
from streambudget.interactive.async_runtime.contracts import OCRPacket
from streambudget.interactive.async_runtime.transport import RawCompletion
from streambudget.types import ContractError
from conftest import config


class ReadingFixture:
    model_id = "explicit-fixture-ocr"

    def __init__(self, delay=0.01, wrong_source=False, fail=False):
        self.delay, self.wrong_source, self.fail = delay, wrong_source, fail

    def read(self, image, evidence_id):
        import time
        from streambudget.interactive.text import TextReading, OCRReadError

        time.sleep(self.delay)
        if self.fail:
            raise OCRReadError("fixture truncation", {"raw_text": "partial", "generated_tokens": 128})
        return TextReading(
            "wrong" if self.wrong_source else evidence_id, "Visible text", "Visible text", 5, 3
        )


def test_local_ocr_is_source_bound_accounted_and_has_no_invented_geometry(tmp_path):
    out = tmp_path / "ocr"
    c = config()
    c.game.max_steps = 5
    meta = asyncio.run(
        AsyncGameRunner(
            c, FixtureEnvironment(), out, transport=FixtureTransport(), ocr_reader=ReadingFixture()
        ).run()
    )
    assert meta["clean_checkpoint"] and meta["ocr_stats"]["submitted"] > 0
    db = sqlite3.connect(out / "memory.sqlite")
    packets = db.execute("SELECT frame_id,payload FROM tm_ocr_packets").fetchall()
    assert packets
    for frame, raw in packets:
        packet = json.loads(raw)
        assert packet["source_frame_id"] == frame
        assert packet["lines"][0]["box"] is None
        assert packet["lines"][0]["confidence"] is None
    assert (
        db.execute("SELECT COUNT(*) FROM model_calls WHERE role='ocr' AND status='pending'").fetchone()[0]
        == 0
    )
    assert (
        db.execute(
            "SELECT COUNT(*) FROM model_inputs WHERE call_id IN (SELECT id FROM model_calls WHERE role='ocr')"
        ).fetchone()[0]
        > 0
    )
    db.close()


@pytest.mark.parametrize("changing_pixels", [False, True])
def test_repeated_local_ocr_deduplicates_only_unchanged_screen(tmp_path, changing_pixels):
    class StationaryEnvironment(FixtureEnvironment):
        def execute(self, action):
            receipt = super().execute(action)
            self.x, self.room = 12, 0
            return receipt

    out = tmp_path / "ocr-dedup"
    c = config()
    c.game.max_steps = 5
    c.game.ocr_refresh_s = 0.001
    env = FixtureEnvironment() if changing_pixels else StationaryEnvironment()
    meta = asyncio.run(AsyncGameRunner(
        c, env, out, transport=FixtureTransport(actor_delay=0.04),
        ocr_reader=ReadingFixture(delay=0.001)).run())
    db = sqlite3.connect(out / "memory.sqlite")
    reads = db.execute("SELECT response FROM model_calls WHERE role='ocr'").fetchall()
    packets = db.execute("SELECT payload FROM tm_ocr_packets").fetchall()
    assert len(reads) > 1 and meta["clean_checkpoint"]
    if changing_pixels:
        assert len(packets) > 1 and meta["ocr_stats"]["deduplicated"] == 0
        occurrences = [json.loads(p[0])["lines"][0]["occurrence"] for p in packets]
        assert len(set(occurrences)) == len(occurrences)
    else:
        assert len(packets) == 1
        assert meta["ocr_stats"]["deduplicated"] == len(reads) - 1
        assert sum(json.loads(r[0]).get("memory_deduplicated", False) for r in reads) == len(reads) - 1
        published = json.loads(packets[0][0])["source_frame_id"]
        assert published == db.execute("SELECT id FROM evidence WHERE kind='frame' ORDER BY seq LIMIT 1").fetchone()[0]
    db.close()


def test_screen_return_starts_new_ocr_occurrence_even_without_intermediate_read(tmp_path):
    runner = AsyncGameRunner(config(), FixtureEnvironment(), tmp_path / "return", transport=FixtureTransport())
    original = runner.current
    first = runner._ocr_screen_run
    from types import SimpleNamespace
    runner._observe_ocr_screen(SimpleNamespace(payload={"sha256": "different-screen"}))
    runner._observe_ocr_screen(original)
    assert runner._ocr_screen_run != first
    asyncio.run(runner.run())


def test_local_ocr_never_blocks_initial_actor(tmp_path):
    out = tmp_path / "overlap"
    c = config()
    c.game.max_steps = 2
    reader = ReadingFixture(delay=0.2)
    meta = asyncio.run(
        AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport(), ocr_reader=reader).run()
    )
    assert meta["completed_steps"] == 2 and meta["clean_checkpoint"]
    trace = [json.loads(line) for line in (out / "trace.jsonl").read_text().splitlines()]
    first_action = next(i for i, event in enumerate(trace) if event["kind"] == "action")
    first_ocr = next(i for i, event in enumerate(trace) if event["kind"] == "ocr_completed")
    assert first_action < first_ocr


@pytest.mark.parametrize("wrong_source,fail", [(True, False), (False, True)])
def test_local_ocr_failures_are_retained_and_never_published(tmp_path, wrong_source, fail):
    out = tmp_path / "bad-ocr"
    meta = asyncio.run(
        AsyncGameRunner(
            config(),
            FixtureEnvironment(),
            out,
            transport=FixtureTransport(),
            ocr_reader=ReadingFixture(wrong_source=wrong_source, fail=fail),
        ).run()
    )
    assert meta["status"] == "ocr_failed" and meta["ocr_stats"]["failed"] == 1
    db = sqlite3.connect(out / "memory.sqlite")
    assert db.execute("SELECT COUNT(*) FROM tm_ocr_packets").fetchone()[0] == 0
    row = db.execute("SELECT status,response FROM model_calls WHERE role='ocr'").fetchone()
    assert row[0] == "failed"
    if fail:
        assert json.loads(row[1])["raw_text"] == "partial"
    db.close()


def test_local_ocr_setup_failure_is_accounted_without_download(tmp_path):
    c = config(ocr_backend="hunyuan", ocr_model_paths={"hunyuan": str(tmp_path / "absent")})
    out = tmp_path / "missing"
    meta = asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport()).run())
    assert meta["status"] == "ocr_failed"
    db = sqlite3.connect(out / "memory.sqlite")
    assert db.execute("SELECT status FROM model_calls WHERE role='ocr_setup'").fetchone()[0] == "failed"
    db.close()


def test_unresolved_local_ocr_blocks_clean_checkpoint(tmp_path):
    import time

    c = config(background_drain_s=0.01)
    c.game.max_steps = 1
    out = tmp_path / "unresolved"
    meta = asyncio.run(
        AsyncGameRunner(
            c, FixtureEnvironment(), out, transport=FixtureTransport(), ocr_reader=ReadingFixture(delay=0.2)
        ).run()
    )
    assert not meta["clean_checkpoint"]
    db = sqlite3.connect(out / "memory.sqlite")
    assert db.execute("SELECT status FROM async_ocr_jobs").fetchone()[0] == "termination_unknown"
    assert db.execute("SELECT status FROM model_calls WHERE role='ocr'").fetchone()[0] == "pending"
    db.close()
    time.sleep(0.25)  # Let the non-writing fixture worker finish; no callbacks touch the closed DB.


def restore(path):
    d = json.loads(path.read_text())
    e = FixtureEnvironment()
    e.x, e.room, e.frame_number = d["x"], d["room"], d["frame_number"]
    return e


def test_actions_continue_while_memory_is_running(tmp_path):
    c = config()
    c.game.max_steps = 15
    t = FixtureTransport(index_delay=0.05, enrich_delay=0.25)
    meta = asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), tmp_path / "run", transport=t).run())
    assert meta["completed_steps"] == 15 and meta["clean_checkpoint"] and meta["game_success"] is None
    assert t.peak == 2 and t.closed
    trace = [json.loads(x) for x in (tmp_path / "run" / "trace.jsonl").read_text().splitlines()]
    assert sum(x["kind"] == "action" and x["pending_memory"] > 0 for x in trace) >= 3
    assert meta["memory_nodes"] > 0
    assert meta["accounting"]["roles"]["enrich"]["completed"] > 0


def test_initial_plan_is_not_blocked_by_memory_writer(tmp_path):
    t = FixtureTransport(index_delay=0.4, plan_delay=0.01)
    c = config()
    c.game.max_steps = 2
    meta = asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), tmp_path / "run", transport=t).run())
    roles = [x["role"] for x in t.requests]
    assert roles[0] == "plan"
    assert meta["completed_steps"] == 2
    db = sqlite3.connect(tmp_path / "run" / "memory.sqlite")
    first_action = db.execute("SELECT wall_s FROM async_actions ORDER BY wall_s LIMIT 1").fetchone()[0]
    index_end = db.execute(
        "SELECT finished_at FROM async_jobs WHERE role='index' AND status='completed' LIMIT 1"
    ).fetchone()[0]
    assert first_action < index_end
    db.close()


def test_clean_resume_preserves_memory_and_total_budget(tmp_path):
    c = config()
    c.game.max_steps = 6
    out = tmp_path / "run"
    a = asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport()).run())
    c.game.max_steps = 10
    b = asyncio.run(
        AsyncGameRunner(
            c, restore(out / "environment.state"), out, resume=True, transport=FixtureTransport()
        ).run()
    )
    assert b["completed_steps"] == 10 and b["memory_revision"] >= a["memory_revision"]
    assert b["accounting"]["ledger"]["calls"] > a["accounting"]["ledger"]["calls"]
    assert b["elapsed_wall_s"] > a["elapsed_wall_s"]


def test_resume_rejects_changed_goal_and_tampered_records(tmp_path):
    c = config()
    c.game.max_steps = 2
    out = tmp_path / "run"
    asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport()).run())
    bad = c.model_copy(deep=True)
    bad.game.goal = "other goal"
    with pytest.raises(ContractError):
        AsyncGameRunner(
            bad, restore(out / "environment.state"), out, resume=True, transport=FixtureTransport()
        )
    db = sqlite3.connect(out / "memory.sqlite")
    db.execute("UPDATE async_actions SET action='changed'")
    db.commit()
    db.close()
    with pytest.raises(ContractError):
        AsyncGameRunner(c, restore(out / "environment.state"), out, resume=True, transport=FixtureTransport())


def test_unknown_action_is_never_dispatched(tmp_path):
    class Bad(FixtureTransport):
        async def invoke(self, ep, body):
            raw = await super().invoke(ep, body)
            if self.decode(body)[0] == "act":
                raw.response["choices"][0]["message"]["content"] = '{"action_id":"UNAVAILABLE"}'
            return raw

    c = config()
    e = FixtureEnvironment()
    result = asyncio.run(AsyncGameRunner(c, e, tmp_path / "run", transport=Bad()).run())
    assert result["status"] == "actor_failed"
    assert e.frame_number == 0 and e.closed


def test_source_changed_during_actor_never_dispatches(tmp_path):
    e = FixtureEnvironment()

    class Stale(FixtureTransport):
        async def invoke(self, ep, body):
            raw = await super().invoke(ep, body)
            if self.decode(body)[0] == "act":
                e.x += 1
            return raw

    with pytest.raises(ContractError):
        asyncio.run(AsyncGameRunner(config(), e, tmp_path / "run", transport=Stale()).run())
    assert e.frame_number == 0
    meta = json.loads((tmp_path / "run" / "run.json").read_text())
    assert not meta["clean_checkpoint"]


def test_uncertain_native_action_is_journaled_not_retried(tmp_path):
    class BadEnvironment(FixtureEnvironment):
        def execute(self, action):
            self.frame_number += 1
            raise RuntimeError("native failure after an unknown partial action")

    e = BadEnvironment()
    with pytest.raises(RuntimeError):
        asyncio.run(AsyncGameRunner(config(), e, tmp_path / "run", transport=FixtureTransport()).run())
    db = sqlite3.connect(tmp_path / "run" / "memory.sqlite")
    assert db.execute("SELECT COUNT(*) FROM async_actions WHERE status='dispatched'").fetchone()[0] == 1
    assert not json.loads((tmp_path / "run" / "async_checkpoint.json").read_text())["clean"]
    db.close()


def test_external_ocr_bridge_without_ocr_engine(tmp_path):
    c = config()
    c.game.max_steps = 4
    runner = None

    def enqueue(notice):
        if runner is not None:
            packet = OCRPacket(
                worker_session="external-fixture",
                source_frame_id=notice.evidence_id,
                sequence=notice.frame_number,
                lines=[{"track_id": "t1", "occurrence": "o1", "text": "test only", "confidence": 1}],
            )
            assert runner.offer_ocr(packet)

    runner = AsyncGameRunner(
        c, FixtureEnvironment(), tmp_path / "run", transport=FixtureTransport(), on_frame=enqueue
    )
    meta = asyncio.run(runner.run())
    db = sqlite3.connect(tmp_path / "run" / "memory.sqlite")
    assert db.execute("SELECT COUNT(*) FROM tm_ocr_packets").fetchone()[0] > 0
    assert meta["ocr_rejected"] == 0
    db.close()


def test_ocr_unknown_frame_rejected_without_state_pollution(tmp_path):
    c = config()
    c.game.max_steps = 2
    runner = AsyncGameRunner(c, FixtureEnvironment(), tmp_path / "run", transport=FixtureTransport())
    runner.offer_ocr(
        OCRPacket(worker_session="external", source_frame_id="not-present", sequence=0, lines=[])
    )
    meta = asyncio.run(runner.run())
    assert meta["ocr_rejected"] == 1 and meta["completed_steps"] == 2


def test_stale_background_plan_does_not_overwrite_current_intent(tmp_path):
    c = config(require_initial_plan=False, plan_max_age_frames=0)
    c.game.max_steps = 4
    meta = asyncio.run(
        AsyncGameRunner(
            c,
            FixtureEnvironment(),
            tmp_path / "run",
            transport=FixtureTransport(plan_delay=0.1, index_delay=0.2),
        ).run()
    )
    db = sqlite3.connect(tmp_path / "run" / "memory.sqlite")
    assert (
        db.execute("SELECT COUNT(*) FROM async_acceptance WHERE status='discarded_stale_plan'").fetchone()[0]
        > 0
    )
    assert meta["completed_steps"] == 4
    db.close()


def test_request_timeout_quarantines_and_no_silent_fallback(tmp_path):
    class Timeout(FixtureTransport):
        async def invoke(self, ep, body):
            await asyncio.sleep(0.01)
            return RawCompletion(None, 0.01, error="TimeoutError", termination_unknown=True)

    c = config()
    e = FixtureEnvironment()
    meta = asyncio.run(AsyncGameRunner(c, e, tmp_path / "run", transport=Timeout()).run())
    assert meta["status"] == "transport_quarantined" and not meta["clean_checkpoint"]
    assert meta["completed_steps"] == 0


def test_source_newframes_not_waiting_for_prior_snapshot(mem, tmp_path):
    # All committed memory source times are earlier than their publication; old knowledge
    # remains reconstructible by knowledge_revision after multiple actor steps.
    c = config()
    c.game.max_steps = 10
    out = tmp_path / "run"
    meta = asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport()).run())
    db = sqlite3.connect(out / "memory.sqlite")
    db.row_factory = sqlite3.Row
    for row in db.execute("SELECT basis FROM async_jobs"):
        basis = json.loads(row["basis"])
        assert basis["source_frame_id"] in basis["offered_frames"]
        assert basis["knowledge_revision"] <= meta["memory_revision"]
    db.close()


def test_existing_output_directory_not_overwritten(tmp_path):
    out = tmp_path / "keep"
    out.mkdir()
    (out / "keep").write_text("stay")
    with pytest.raises(ContractError):
        AsyncGameRunner(config(), FixtureEnvironment(), out, transport=FixtureTransport())
    assert (out / "keep").read_text() == "stay"


def test_drain_timeout_records_uncertainty_and_disables_clean_resume(tmp_path):
    c = config(background_drain_s=0.03)
    c.game.max_steps = 2
    out = tmp_path / "run"
    meta = asyncio.run(
        AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport(index_delay=10)).run()
    )
    assert not meta["clean_checkpoint"] and meta["accounting"]["quarantined"]
    db = sqlite3.connect(out / "memory.sqlite")
    assert db.execute("SELECT COUNT(*) FROM model_calls WHERE status='failed'").fetchone()[0] > 0
    db.close()
