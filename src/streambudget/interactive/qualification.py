"""Explicit non-actuating model probes and operator-only controller qualification."""
from __future__ import annotations

from pathlib import Path

from ..media import MediaStore
from ..store import EvidenceStore
from ..types import ContractError
from . import prompts
from .context import bounded_context
from .contracts import ActionChoice, ObservationPatch, Plan, SchemaPatch
from .models import ChatBackend, ImageInput, Ledger
from .ontology import Ontology
from .runner import inference_png, write_json

OUTPUTS = {"extract": ObservationPatch, "plan": Plan, "act": ActionChoice, "compile": SchemaPatch}
SYSTEMS = {"extract": prompts.EXTRACT, "plan": prompts.PLAN, "act": prompts.ACT, "compile": prompts.COMPILE}


def probe(config, image_path: Path, out: Path, *, role="extract", intent=None,
          allow_network=False, allow_paid=False, transport=None):
    """One explicitly authorized model request. Its result NEVER reaches an executor.

    Fresh output/ledger is required; this is its own operator-authorized budget,
    not a way to resume or reset an existing campaign's accounting.
    """
    if config.backend != "chat":
        raise ContractError("Probe requires a configured chat endpoint")
    if out.exists():
        raise ContractError("Probe destination exists")
    if role not in OUTPUTS:
        raise ContractError("Unknown model role")
    raw = image_path.read_bytes()
    image = MediaStore.decode(raw)
    out.mkdir(parents=True)
    store = EvidenceStore(out / "memory.sqlite", config.source)
    backend = None
    ledger = None
    try:
        media = MediaStore(out / "media")
        key, _ = media.put(raw)
        frame = store.add_raw(source=config.source, kind="frame", start=0, end=0,
                              available_at=0, payload={"media_key": key})
        ontology = Ontology(store.db)
        ledger = Ledger(store.db, config.max_calls, config.max_estimated_usd)
        context = bounded_context(goal=config.goal, intent=intent or "Describe the current observation.",
            schema=ontology.current, actions=[a.model_dump() for a in config.actions],
            world={"entities": [], "relations": [], "current_place": None, "recent_conversations": []},
            recent=[], current_frame_id=frame.id, max_chars=config.max_context_chars)
        write_json(out / "probe.json", {"status": "pending", "role": role, "actuation": False,
                                        "config": config.model_dump()})
        backend = ChatBackend(config, ledger, allow_network=allow_network,
                              allow_paid=allow_paid, transport=transport)
        result = backend.complete(role, SYSTEMS[role], context,
            [ImageInput(frame.id, inference_png(image, config.view_scale))], OUTPUTS[role])
        if role == "extract" and result.frame_id != frame.id:
            raise ContractError("Probe result refers to a different frame")
        if role == "act" and result.action_id not in {a.id for a in config.actions}:
            raise ContractError("Probe returned an action outside the manifest")
        record = {"status": "returned_for_human_review", "role": role,
                  "actuation": False, "result": result.model_dump(), "accounting": ledger.summary(),
                  "note": "Schema success is not visual grounding or policy qualification."}
        write_json(out / "probe.json", record)
        return record
    except BaseException as exc:
        write_json(out / "probe.json", {"status": "failed", "role": role, "actuation": False,
            "error_type": type(exc).__name__, "accounting": ledger.summary() if ledger else None,
            "note": "No retries or actions were dispatched. Unknown attempts retain their charge holds."})
        raise
    finally:
        try:
            if backend:
                backend.close()
        finally:
            store.close()
