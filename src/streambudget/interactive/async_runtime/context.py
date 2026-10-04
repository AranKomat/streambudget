"""Task-relevant context projection with explicit omissions and frozen knowledge cutoffs."""

from __future__ import annotations

from dataclasses import dataclass

from ...types import ContractError, Snapshot
from ..ontology import canonical


@dataclass
class ContextPacket:
    value: dict
    cutoff: object
    entities: tuple[str, ...]
    evidence: tuple[str, ...]


def evidence_ids(value):
    out = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in (
                "evidence_id",
                "evidence",
                "current_frame_id",
                "frame_id",
                "action_evidence",
            ) and isinstance(item, str):
                out.add(item)
            elif key in ("evidence_ids",) and isinstance(item, list):
                out.update(x for x in item if isinstance(x, str))
            out.update(evidence_ids(item))
    elif isinstance(value, list):
        for item in value:
            out.update(evidence_ids(item))
    return out


def entity_handles(value):
    out = set()
    if isinstance(value, str) and value.startswith("e_") and len(value) == 18:
        out.add(value)
    elif isinstance(value, dict):
        for item in value.values():
            out.update(entity_handles(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            out.update(entity_handles(item))
    return out


def build_context(memory, config, frame, *, intent, focus=(), role="act", retrieved=(), targets=()):
    cutoff = memory.cutoff(frame.seq)
    settings, game = config.background, config.game
    # A target was minted by an index of this very frame; no future handles may leak in.
    focus = tuple(dict.fromkeys([*focus, *targets]))
    existing = {n["id"] for n in memory.nodes(cutoff)}
    focus = tuple(x for x in focus if x in existing)
    tiers = memory.tiers(
        cutoff,
        frame_number=frame.payload["frame_number"],
        focus=focus,
        hot=settings.hot_entities,
        warm=settings.warm_entities,
    )
    place = memory.current_place(cutoff)
    relations = [r for r in memory.relations(cutoff) if r["subject"] in {n["id"] for n in tiers["hot"]}][:24]
    # Avoid dumping SQL implementation metadata into the prompt.
    rels = [
        {k: r[k] for k in ("subject", "predicate", "target", "confidence", "evidence", "frame_number")}
        for r in relations
    ]
    events = memory.event_history(cutoff, limit=8)
    conversations = []
    for id in focus[:3]:
        conversations.extend(memory.conversation(id, cutoff, limit=4))
    if role == "plan" and not conversations:
        conversations = memory.conversation(None, cutoff, limit=4)
    snap = Snapshot(frame.end, frame.seq)
    actions = list(reversed(memory.store.list(snap, source=game.source, kinds=["action_receipt"], limit=4)))
    action_views = [e.view() for e in actions]
    value = {
        "goal": game.goal,
        "ontology": memory.ontology.current,
        "actions": [a.model_dump() for a in game.actions],
        "intent": intent,
        "current_frame_id": frame.id,
        "source_frame_number": frame.payload["frame_number"],
        "knowledge_revision": cutoff.revision,
        "world": {
            "entities": tiers["hot"],
            "relations": rels,
            "current_place": place,
            "events": events,
            "conversations": conversations,
        },
        "recent_actions": action_views,
        "ocr": memory.ocr_view(
            cutoff, frame_number=frame.payload["frame_number"], max_age_frames=settings.ocr_max_age_frames
        ),
        "retrieved": list(retrieved),
        "targets": list(targets),
        "omissions": {
            "entities": 0,
            "events": 0,
            "conversations": 0,
            "relations": 0,
            "ocr": 0,
            "retrieved": 0,
            "warm_entities_not_in_prompt": len(tiers["warm_ids"]),
            "cold_entities_not_in_prompt": tiers["cold_count"],
        },
        "memory_note": "Beliefs with evidence, not hidden game state. Missing means unknown. Old positions may be stale.",
    }
    max_chars = {
        "act": settings.actor_context_chars,
        "plan": settings.planner_context_chars,
        "index": settings.semantic_context_chars,
        "enrich": settings.semantic_context_chars,
    }[role]
    mandatory = set(focus) | set(tiers["pinned"]) | ({place["id"]} if place and place["id"] else set())
    while len(canonical(value)) > max_chars:
        changed = False
        for container, key in (
            (value["world"], "events"),
            (value["world"], "conversations"),
            (value, "ocr"),
            (value, "retrieved"),
            (value["world"], "relations"),
        ):
            if container[key]:
                container[key].pop(0)
                value["omissions"][key] += 1
                changed = True
                break
        if changed:
            continue
        disposable = next(
            (
                i
                for i in range(len(value["world"]["entities"]) - 1, -1, -1)
                if value["world"]["entities"][i]["id"] not in mandatory
            ),
            None,
        )
        if disposable is not None:
            value["world"]["entities"].pop(disposable)
            value["omissions"]["entities"] += 1
            continue
        raise ContractError(
            "Essential task/schema/focused memory exceeds the configured context character cap"
        )
    exposed = entity_handles(value)
    # Only visible nodes become legal handles. Uncertain OCR strings cannot mint entities.
    exposed &= existing
    for id in exposed:
        memory.node(id, cutoff)
    evs = evidence_ids(value) | {frame.id}
    for id in evs:
        memory.store.get(id, snap)
    return ContextPacket(value, cutoff, tuple(sorted(exposed)), tuple(sorted(evs)))
