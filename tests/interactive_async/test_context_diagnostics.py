import asyncio
import sqlite3
import json

import pytest

from streambudget.interactive.async_runtime.context import build_context
from streambudget.interactive.async_runtime.contracts import MemoryDelta
from streambudget.interactive.async_runtime.memory import TemporalMemory
from streambudget.interactive.async_runtime.runner import AsyncGameRunner
from streambudget.interactive.async_runtime.fixture import FixtureTransport
from streambudget.interactive.async_runtime.report import make_report, summary
from streambudget.interactive.async_runtime.diagnostics import probe
from streambudget.interactive import cli
from streambudget.interactive.fixture import FixtureEnvironment
from streambudget.types import ContractError
from conftest import config, add_frame, basis


def test_index_prompt_states_normalized_geometry_and_keeps_pixel_boxes_invalid():
    from streambudget.interactive.async_runtime.prompts import INDEX
    from streambudget.interactive.async_runtime.contracts import SceneIndex

    assert 'normalized [left, top, right, bottom]' in INDEX
    with pytest.raises(ValueError):
        SceneIndex.model_validate({'mentions': [{'ref': 'new:x', 'kind': 'entity',
            'label': 'visible character', 'region': {'frame_id': 'frame',
            'box': [170, 100, 290, 300]}}]})


def test_context_preserves_focused_memory_and_exposes_provenance(mem):
    m, s, media = mem
    frame = add_frame(s, media, 0)
    result = m.commit(
        basis(m, frame),
        MemoryDelta.model_validate(
            {
                "mentions": [
                    {"ref": "new:p", "kind": "place", "label": "a room"},
                    {"ref": "new:x", "kind": "entity", "label": "a person"},
                ],
                "current_place": "new:p",
            }
        ),
        now=0,
    )
    focus = result["ids"]["new:x"]
    packet = build_context(m, config(), frame, intent="talk to this person", focus=[focus])
    assert focus in packet.entities and frame.id in packet.evidence
    assert packet.cutoff.revision == m.revision
    assert packet.value["world"]["current_place"]["id"] == result["ids"]["new:p"]


def test_context_does_not_silently_drop_mandatory_task(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    c = config(actor_context_chars=2000)
    c.game.goal = "A" * 6000
    with pytest.raises(ContractError):
        build_context(m, c, f, intent="important")


def test_evidence_can_be_retrieved_after_many_interpretations(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    result = m.commit(
        basis(m, f),
        MemoryDelta.model_validate(
            {
                "mentions": [{"ref": "new:a", "kind": "entity", "label": "first person"}],
                "facts": [
                    {
                        "subject": "new:a",
                        "key": "lesson",
                        "value": "remember the amber object",
                        "confidence": 1,
                    }
                ],
            }
        ),
        now=0,
    )
    id = result["ids"]["new:a"]
    for n in range(1, 151):
        f = add_frame(s, media, n)
        m.commit(
            basis(m, f, offered=[id]),
            MemoryDelta.model_validate(
                {"facts": [{"subject": id, "key": "position_hint", "value": str(n), "confidence": 1}]}
            ),
            now=n,
        )
    hits = m.search("amber", m.cutoff(f.seq))
    assert hits and hits[0]["evidence_id"] == result["evidence_id"]
    assert m.facts(id, m.cutoff(f.seq), frame_number=150)["position_hint"]["value"] == "150"


def test_report_escapes_model_html_and_keeps_db_readonly(tmp_path):
    c = config()
    c.game.max_steps = 3
    out = tmp_path / "run"
    asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport()).run())
    meta = json.loads((out / "run.json").read_text())
    meta["status"] = "<script>bad()</script>"
    (out / "run.json").write_text(json.dumps(meta))
    before = (out / "memory.sqlite").read_bytes()
    page = make_report(out).read_text()
    assert "<script>bad()" not in page and "&lt;script&gt;" in page
    assert "SOFTWARE FIXTURE" in page and "not a learned model" in page
    assert before == (out / "memory.sqlite").read_bytes()
    assert summary(out)["task_success"] is None


def test_readonly_memory_inspection_cannot_mutate(tmp_path):
    c = config()
    c.game.max_steps = 3
    out = tmp_path / "run"
    asyncio.run(AsyncGameRunner(c, FixtureEnvironment(), out, transport=FixtureTransport()).run())
    memory = TemporalMemory.open_readonly(out / "memory.sqlite")
    try:
        assert memory.revision > 0
        with pytest.raises(sqlite3.OperationalError):
            memory.db.execute("DELETE FROM tm_nodes")
    finally:
        memory.store.close()


def test_probe_counts_all_pairs_without_actuation(tmp_path):
    image = tmp_path / "frame.png"
    FixtureEnvironment().capture().image.save(image)
    c = config()
    out = tmp_path / "probe"
    transport = FixtureTransport()
    result = asyncio.run(probe(c, image, out, count=3, mode="mixed", transport=transport))
    assert result["actuation"] is False and result["status"] == "completed"
    assert result["metrics"]["ledger"]["calls"] == 6
    assert transport.peak == 2
    db = sqlite3.connect(out / "memory.sqlite")
    assert db.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='async_actions'").fetchone()[0] == 0
    db.close()


def test_cli_demo_and_memory_query(tmp_path, capsys):
    out = tmp_path / "run"
    assert cli.main(["background", "demo", "--out", str(out), "--steps", "3"]) == 0
    assert (out / "background-report.html").exists()
    assert cli.main(["background", "memory", "--run", str(out), "--kind", "visits"]) == 0
    assert "place" in capsys.readouterr().out


def test_metadata_slot_limits_reject_false_actor_reservation():
    with pytest.raises(ValueError):
        config(total_slots=2, background_slots=2)


def test_same_pixel_later_observation_not_same_evidence(mem):
    m, s, media = mem
    a = add_frame(s, media, 0)
    b = add_frame(s, media, 255)
    assert a.payload["sha256"] == b.payload["sha256"]
    assert a.id != b.id and a.seq != b.seq
