"""Generous but bounded context, explicit omissions, no claim of native KV persistence."""
from __future__ import annotations

from .ontology import canonical
from ..types import ContractError


def bounded_context(*, goal, intent, schema, actions, world, recent, current_frame_id,
                    max_chars, extra=None):
    base = {"goal": goal, "intent": intent, "ontology": schema, "actions": actions,
            "current_frame_id": current_frame_id,
            "world": world, "recent_evidence": recent, "retrieved": extra or [],
            "omissions": {"entities": 0, "recent_evidence": 0, "retrieved": 0,
                          "recent_conversations": 0, "relations": 0}}
    # Trimming never silently removes current task or action grammar. Explicit cap is
    # characters, not guessed tokenizer counts; provider usage is recorded separately.
    while len(canonical(base)) > max_chars:
        if base["recent_evidence"]:
            base["recent_evidence"].pop(0)
            base["omissions"]["recent_evidence"] += 1
        elif base["retrieved"]:
            base["retrieved"].pop()
            base["omissions"]["retrieved"] += 1
        elif base["world"].get("recent_conversations"):
            base["world"]["recent_conversations"].pop(0)
            base["omissions"]["recent_conversations"] += 1
        elif base["world"].get("relations"):
            base["world"]["relations"].pop()
            base["omissions"]["relations"] += 1
        elif base["world"].get("entities"):
            base["world"]["entities"].pop()
            base["omissions"]["entities"] += 1
        else:
            raise ContractError("Essential context does not fit. Increase budget rather than truncate the goal")
    return base
