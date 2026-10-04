from dataclasses import replace
import threading

import pytest

from streambudget.types import ContractError
from streambudget.interactive.async_runtime.contracts import MemoryDelta, OCRPacket
from conftest import add_frame, basis


def seed(memory, store, media):
    f = add_frame(store, media, 0)
    result = memory.commit(
        basis(memory, f),
        MemoryDelta.model_validate(
            {
                "mentions": [
                    {"ref": "new:p", "kind": "place", "label": "Room"},
                    {"ref": "new:q", "kind": "place", "label": "Other room"},
                    {"ref": "new:person", "kind": "entity", "label": "Unknown person"},
                    {"ref": "new:portal", "kind": "surface", "label": "Opening"},
                    {"ref": "new:text", "kind": "surface", "label": "Text surface"},
                ],
                "current_place": "new:p",
            }
        ),
        now=0,
    )
    return f, result["ids"]


def test_older_results_enrich_history_without_rewinding_fields(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    old = add_frame(s, media, 1)
    old_basis = basis(m, old, offered=ids.values())
    fresh = add_frame(s, media, 2)
    m.commit(
        basis(m, fresh, offered=ids.values()),
        MemoryDelta.model_validate(
            {
                "facts": [
                    {"subject": ids["new:person"], "key": "position_hint", "value": "east", "confidence": 1}
                ]
            }
        ),
        now=1,
    )
    before = m.cutoff(fresh.seq)
    m.commit(
        old_basis,
        MemoryDelta.model_validate(
            {
                "facts": [
                    {"subject": ids["new:person"], "key": "position_hint", "value": "west", "confidence": 1},
                    {"subject": ids["new:person"], "key": "name", "value": "Ada", "confidence": 1},
                ]
            }
        ),
        now=2,
    )
    assert m.facts(ids["new:person"], m.cutoff(fresh.seq), frame_number=2)["position_hint"]["value"] == "east"
    assert m.facts(ids["new:person"], m.cutoff(fresh.seq), frame_number=2)["name"]["value"] == "Ada"
    assert "name" not in m.facts(ids["new:person"], before, frame_number=2)
    assert not m.search("Ada", before)
    assert m.search("Ada", m.cutoff(fresh.seq))


def test_duplicate_job_idempotent_but_conflict_rejected(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    b = basis(m, f)
    d = MemoryDelta()
    m.commit(b, d, now=0)
    rev = m.revision
    assert not m.commit(b, d, now=1)["changed"]
    assert m.revision == rev
    with pytest.raises(ContractError):
        m.commit(b, MemoryDelta(needs_planning=True), now=2)


def test_multiple_independent_patches_on_one_frame(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for i, key in enumerate(["name", "status"]):
        m.commit(
            basis(m, f, offered=ids.values()),
            MemoryDelta.model_validate(
                {"facts": [{"subject": ids["new:person"], "key": key, "value": str(i), "confidence": 1}]}
            ),
            now=i + 1,
        )
    assert len(m.facts(ids["new:person"], m.cutoff(f.seq), frame_number=0)) == 2


def test_same_frame_conflicting_facts_surface_uncertainty(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for i, value in enumerate(["Ada", "Grace"]):
        m.commit(
            basis(m, f, offered=ids.values()),
            MemoryDelta.model_validate(
                {"facts": [{"subject": ids["new:person"], "key": "name", "value": value, "confidence": 1}]}
            ),
            now=i + 1,
        )
    assert m.facts(ids["new:person"], m.cutoff(f.seq), frame_number=0)["name"]["conflict"]


def test_appearance_does_not_merge_ids(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    r = m.commit(
        basis(m, f),
        MemoryDelta.model_validate(
            {
                "mentions": [
                    {"ref": "new:a", "kind": "entity", "label": "same sprite"},
                    {"ref": "new:b", "kind": "entity", "label": "same sprite"},
                ]
            }
        ),
        now=0,
    )
    assert len(set(r["ids"].values())) == 2


def test_ambiguous_association_and_unoffered_ids_rejected(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    person = ids["new:person"]
    for offered, confidence in [((), 1), (tuple(ids.values()), 0.2)]:
        d = MemoryDelta.model_validate(
            {
                "mentions": [
                    {
                        "ref": person,
                        "kind": "entity",
                        "label": "someone",
                        "association": "reidentified",
                        "confidence": confidence,
                    }
                ]
            }
        )
        with pytest.raises(ContractError):
            m.commit(basis(m, f, offered=offered), d, now=1)


def test_new_identity_can_gain_name_and_alias_without_new_schema(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    version = m.ontology.current["version"]
    m.commit(
        basis(m, f, offered=ids.values()),
        MemoryDelta.model_validate(
            {
                "facts": [
                    {"subject": ids["new:person"], "key": "name", "value": "Ada", "confidence": 1},
                    {"subject": ids["new:person"], "key": "aliases", "value": ["Guide"], "confidence": 1},
                ]
            }
        ),
        now=1,
    )
    assert m.node(ids["new:person"], m.cutoff(f.seq))["label"] == "Unknown person"
    assert len(m.nodes(m.cutoff(f.seq))) == 5
    assert m.ontology.current["version"] == version


def test_bad_patch_is_atomic(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    rev = m.revision
    d = MemoryDelta.model_validate(
        {
            "mentions": [{"ref": "new:a", "kind": "entity", "label": "a"}],
            "facts": [{"subject": "new:a", "key": "nonexistent", "value": "x", "confidence": 1}],
        }
    )
    with pytest.raises(ContractError):
        m.commit(basis(m, f), d, now=0)
    assert m.revision == rev and not m.nodes(m.cutoff(f.seq))


def test_enrichment_cannot_mint_nodes_or_change_other_targets(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    b = basis(m, f, offered=ids.values(), targets=(ids["new:person"],))
    for d in [
        {"mentions": [{"ref": "new:x", "kind": "entity", "label": "x"}]},
        {"facts": [{"subject": ids["new:p"], "key": "name", "value": "wrong", "confidence": 1}]},
        {"current_place": ids["new:p"]},
    ]:
        with pytest.raises(ContractError):
            m.commit(b, MemoryDelta.model_validate(d), now=1)


def test_future_source_or_future_knowledge_never_leaks(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    cutoff = m.cutoff(f.seq)
    b = basis(m, f, offered=ids.values())
    future = add_frame(s, media, 10)
    r = m.commit(
        basis(m, future, offered=ids.values()),
        MemoryDelta.model_validate(
            {"facts": [{"subject": ids["new:person"], "key": "lesson", "value": "secret", "confidence": 1}]}
        ),
        now=1,
    )
    assert not m.search("secret", cutoff)
    with pytest.raises(ContractError):
        m.commit(replace(b, evidence_ids=(*b.evidence_ids, r["evidence_id"])), MemoryDelta(), now=2)


def test_volatile_facts_stale_names_retained(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    m.commit(
        basis(m, f, offered=ids.values()),
        MemoryDelta.model_validate(
            {
                "facts": [
                    {"subject": ids["new:person"], "key": k, "value": v, "confidence": 1}
                    for k, v in [("name", "Ada"), ("position_hint", "left")]
                ]
            }
        ),
        now=1,
    )
    facts = m.facts(ids["new:person"], m.cutoff(f.seq), frame_number=1000)
    assert facts["position_hint"]["stale"] and not facts["name"]["stale"]


def test_missing_delta_does_not_delete_previous_state(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    m.commit(
        basis(m, f, offered=ids.values()),
        MemoryDelta.model_validate(
            {"facts": [{"subject": ids["new:person"], "key": "visible", "value": True, "confidence": 1}]}
        ),
        now=1,
    )
    g = add_frame(s, media, 2)
    m.commit(basis(m, g, offered=ids.values()), MemoryDelta(), now=2)
    assert m.facts(ids["new:person"], m.cutoff(g.seq), frame_number=2)["visible"]["value"] is True


def test_explicit_relation_retraction_does_not_resurrect_old_location(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for n, destination, operation in [
        (1, "new:p", "assert"),
        (2, "new:q", "assert"),
        (3, "new:q", "retract"),
    ]:
        frame = add_frame(s, media, n)
        m.commit(
            basis(m, frame, offered=ids.values()),
            MemoryDelta.model_validate(
                {
                    "relations": [
                        {
                            "subject": ids["new:person"],
                            "predicate": "located_in",
                            "target": ids[destination],
                            "operation": operation,
                            "confidence": 1,
                        }
                    ]
                }
            ),
            now=n,
        )
    assert not m.relations(m.cutoff(frame.seq), ids["new:person"])


def test_visits_sort_by_observation_not_arrival_and_no_automatic_edges(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    frames = [add_frame(s, media, i) for i in (1, 2, 3)]
    bases = [basis(m, fr, offered=ids.values()) for fr in frames]
    for now, i, place in [(1, 2, "new:p"), (2, 0, "new:p"), (3, 1, "new:q")]:
        m.commit(bases[i], MemoryDelta(current_place=ids[place]), now=now)
    visits = m.visits(m.cutoff(frames[-1].seq))
    assert [v["place"] for v in visits] == [ids["new:p"], ids["new:q"], ids["new:p"]]
    assert not m.route(ids["new:p"], ids["new:q"], m.cutoff(frames[-1].seq))["found"]


def test_route_records_attempts_traversal_direction_and_blockage(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for n, status in [(1, "observed"), (2, "attempted"), (3, "traversed")]:
        receipt = s.add_raw(
            source="game", kind="action_receipt", start=n / 60, end=n / 60, available_at=n / 60, text="RIGHT"
        )
        fr = add_frame(s, media, n)
        m.commit(
            basis(m, fr, offered=ids.values(), parents=(receipt.id,)),
            MemoryDelta.model_validate(
                {
                    "routes": [
                        {
                            "origin": ids["new:p"],
                            "portal": ids["new:portal"],
                            "destination": ids["new:q"] if status == "traversed" else None,
                            "status": status,
                            "occurrence": "attempt1" if n > 1 else "observe1",
                            "confidence": 1,
                            "action_evidence": receipt.id if status == "traversed" else None,
                        }
                    ]
                }
            ),
            now=n,
        )
    cutoff = m.cutoff(fr.seq)
    edge = m.exits(ids["new:p"], cutoff)[0]
    assert edge["attempts"] == 1 and edge["traversals"] == 1
    assert m.route(ids["new:p"], ids["new:q"], cutoff)["found"]
    assert not m.route(ids["new:q"], ids["new:p"], cutoff)["found"]
    fr = add_frame(s, media, 4)
    m.commit(
        basis(m, fr, offered=ids.values()),
        MemoryDelta.model_validate(
            {
                "routes": [
                    {
                        "origin": ids["new:p"],
                        "portal": ids["new:portal"],
                        "status": "blocked",
                        "occurrence": "attempt2",
                        "confidence": 1,
                    }
                ]
            }
        ),
        now=4,
    )
    assert not m.route(ids["new:p"], ids["new:q"], m.cutoff(fr.seq))["found"]


def test_conversation_threads_persist_while_surface_changes(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for n, occ, text, surface in [
        (1, "one", "hello", "new:text"),
        (2, "one", "hello friend", "new:text"),
        (3, "two", "hello friend", "new:portal"),
    ]:
        fr = add_frame(s, media, n)
        m.commit(
            basis(m, fr, offered=ids.values()),
            MemoryDelta.model_validate(
                {
                    "utterances": [
                        {
                            "speaker": ids["new:person"],
                            "surface": ids[surface],
                            "text": text,
                            "occurrence": occ,
                            "attribution_confidence": 0.9,
                        }
                    ]
                }
            ),
            now=n,
        )
    messages = m.conversation(ids["new:person"], m.cutoff(fr.seq))
    assert len(messages) == 2 and len({r["thread_id"] for r in messages}) == 1
    assert messages[0]["text"] == "hello friend"
    assert len(m.conversation(ids["new:person"], m.cutoff(fr.seq), query="friend")) == 2


def test_unknown_speaker_not_assigned_to_nearest_entity(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    m.commit(
        basis(m, f, offered=ids.values()),
        MemoryDelta.model_validate(
            {"utterances": [{"surface": ids["new:text"], "text": "unknown", "occurrence": "one"}]}
        ),
        now=1,
    )
    assert m.conversation(None, m.cutoff(f.seq))[0]["speaker"] is None
    assert not m.conversation(ids["new:person"], m.cutoff(f.seq))


def test_events_query_by_entity_distinct_repeated_occurrences(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for n, occ in [(1, "a"), (2, "a"), (3, "b")]:
        fr = add_frame(s, media, n)
        m.commit(
            basis(m, fr, offered=ids.values()),
            MemoryDelta.model_validate(
                {
                    "events": [
                        {
                            "kind": "spoke",
                            "text": "same words",
                            "participants": [ids["new:person"]],
                            "occurrence": occ,
                        }
                    ]
                }
            ),
            now=n,
        )
    events = m.event_history(m.cutoff(fr.seq), entity=ids["new:person"])
    assert len(events) == 2
    assert not m.event_history(m.cutoff(fr.seq), entity=ids["new:q"])


def test_visual_memory_full_frame_by_default_crop_optional(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    refs = m.visual_refs(ids["new:person"], m.cutoff(f.seq))
    assert refs[0]["frame_id"] == f.id and refs[0]["box"] is None
    fr = add_frame(s, media, 1)
    m.commit(
        basis(m, fr, offered=ids.values()),
        MemoryDelta.model_validate(
            {
                "mentions": [
                    {
                        "ref": ids["new:person"],
                        "kind": "entity",
                        "label": "person",
                        "association": "continuity",
                        "confidence": 1,
                        "region": {"frame_id": fr.id, "box": [0.1, 0.1, 0.4, 0.5]},
                    }
                ]
            }
        ),
        now=1,
    )
    assert len(m.visual_refs(ids["new:person"], m.cutoff(fr.seq))) == 2


def test_tiers_demote_without_deleting_pinned_entries_remain(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    m.pin(
        ids["new:person"],
        reason="active_goal",
        active=True,
        now=1,
        epoch="test-episode",
        cutoff=m.cutoff(f.seq),
    )
    tiers = m.tiers(m.cutoff(f.seq), frame_number=0, hot=2, warm=3)
    assert len(tiers["hot"]) == 2 and len(tiers["warm_ids"]) == 1 and tiers["cold_count"] == 2
    assert ids["new:person"] in {n["id"] for n in tiers["hot"]}
    assert len(m.nodes(m.cutoff(f.seq))) == 5
    with pytest.raises(ContractError):
        m.tiers(m.cutoff(f.seq), frame_number=0, hot=1, warm=1)


def test_ocr_input_is_idempotent_and_late_result_not_current(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    g = add_frame(s, media, 5)

    def packet(fr, seq, text):
        return OCRPacket(
            worker_session="worker",
            source_frame_id=fr.id,
            sequence=seq,
            lines=[{"track_id": "t1", "occurrence": "o1", "text": text, "confidence": 0.9}],
        )

    p = packet(g, 2, "new")
    eid = m.ingest_ocr(p, now=1, epoch="test-episode")
    cutoff = m.cutoff(g.seq)
    assert m.ingest_ocr(p, now=2, epoch="test-episode") == eid
    m.ingest_ocr(packet(f, 1, "old"), now=3, epoch="test-episode")
    assert m.ocr_view(m.cutoff(g.seq), frame_number=5)[0]["text"] == "new"
    assert not m.ocr_view(type(cutoff)(0, g.seq), frame_number=5)
    with pytest.raises(ContractError):
        m.ingest_ocr(packet(g, 2, "conflict"), now=4, epoch="test-episode")


def test_thread_write_rejected(mem):
    m, s, media = mem
    f = add_frame(s, media, 0)
    b = basis(m, f)
    errors = []

    def work():
        try:
            m.commit(b, MemoryDelta(), now=0)
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert isinstance(errors[0], ContractError)


def test_low_confidence_route_does_not_reuse_old_traversal_as_current_authority(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    for n, confidence in [(1, 1.0), (2, 0.1)]:
        receipt = s.add_raw(
            source="game", kind="action_receipt", start=n / 60, end=n / 60, available_at=n / 60, text="RIGHT"
        )
        fr = add_frame(s, media, n)
        m.commit(
            basis(m, fr, offered=ids.values(), parents=(receipt.id,)),
            MemoryDelta.model_validate(
                {
                    "routes": [
                        {
                            "origin": ids["new:p"],
                            "portal": ids["new:portal"],
                            "destination": ids["new:q"],
                            "status": "traversed",
                            "occurrence": str(n),
                            "confidence": confidence,
                            "action_evidence": receipt.id,
                        }
                    ]
                }
            ),
            now=n,
        )
    assert m.exits(ids["new:p"], m.cutoff(fr.seq))[0]["status"] == "unknown"
    assert not m.route(ids["new:p"], ids["new:q"], m.cutoff(fr.seq))["found"]


def test_route_rejects_entity_instead_of_place(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    with pytest.raises(ContractError):
        m.route(ids["new:person"], ids["new:person"], m.cutoff(f.seq))


def test_pin_of_future_source_node_not_in_old_source_projection(mem):
    m, s, media = mem
    old = add_frame(s, media, 0)
    f = add_frame(s, media, 5)
    result = m.commit(
        basis(m, f),
        MemoryDelta.model_validate(
            {"mentions": [{"ref": "new:p", "kind": "entity", "label": "later person"}]}
        ),
        now=1,
    )
    id = result["ids"]["new:p"]
    m.pin(id, reason="active_goal", active=True, now=2, epoch="test-episode", cutoff=m.cutoff(f.seq))
    assert id in m.pins(m.cutoff(f.seq))
    assert id not in m.pins(m.cutoff(old.seq))


def test_external_ocr_surface_cannot_be_a_person(mem):
    m, s, media = mem
    f, ids = seed(*mem)
    packet = OCRPacket(
        worker_session="w",
        source_frame_id=f.id,
        sequence=1,
        lines=[
            {
                "surface_id": ids["new:person"],
                "track_id": "t",
                "occurrence": "o",
                "text": "hello",
                "confidence": 1,
            }
        ],
    )
    with pytest.raises(ContractError):
        m.ingest_ocr(packet, now=1, epoch="test-episode")
