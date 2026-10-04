"""Explicit non-actuating model probes and operator-only controller qualification."""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import statistics
import time

from ..media import MediaStore
from ..store import EvidenceStore
from ..types import ContractError
from . import prompts
from .context import bounded_context
from .contracts import ActionChoice, ObservationPatch, Plan, SchemaPatch
from .models import ChatBackend, ImageInput, Ledger
from .ontology import Ontology
from .runner import inference_png, png, write_json

OUTPUTS = {"extract": ObservationPatch, "plan": Plan, "act": ActionChoice, "compile": SchemaPatch}
SYSTEMS = {"extract": prompts.EXTRACT, "plan": prompts.PLAN, "act": prompts.ACT, "compile": prompts.COMPILE}


def action_latency(config, environment, out: Path, *, count=20, warmup=2,
                   intent="Choose one useful action for the currently visible screen.",
                   allow_network=False, transport=None):
    """Frozen-screen, local-only latency qualification; never execute selected actions.

    Captures are timed on the caller's host. Warmup calls remain in the ledger, and
    failures stop the experiment without retry. This is not a full agent-loop score.
    """
    if config.backend != "chat" or any(e.billing != "local" for e in config.endpoints.values()):
        raise ContractError("Latency qualification requires local chat endpoints only")
    if not allow_network:
        raise ContractError("Latency qualification requires --allow-network")
    if (type(count) is not int or not 1 <= count <= 100 or type(warmup) is not int
            or not 0 <= warmup <= 5 or count + warmup > config.max_calls):
        raise ContractError("Use count in 1..100, warmup in 0..5, and sufficient call admission")
    if out.exists():
        raise ContractError("Latency destination exists")
    out.mkdir(parents=True)
    store = EvidenceStore(out / "memory.sqlite", config.source)
    backend = ledger = None
    record = {"status": "running", "actuation": False, "count": count, "warmup": warmup,
        "config": config.model_dump(), "samples": [], "environment": getattr(environment, "descriptor", None),
        "rom_sha256": getattr(environment, "rom_sha256", None),
        "initialization": getattr(environment, "initialization", "unknown"),
        "initial_state_sha256": getattr(environment, "initial_state_sha256", None),
        "note": "Frozen-screen single action calls, not autonomous gameplay or full-loop latency. "
                "Repeated pixels are a cache-friendly condition. HTTP wall time is not GPU time."}
    write_json(out / "latency.json", record)
    try:
        media = MediaStore(out / "media")
        ontology = Ontology(store.db)
        ledger = Ledger(store.db, config.max_calls, config.max_estimated_usd)
        backend = ChatBackend(config, ledger, allow_network=True, transport=transport)
        allowed = {a.id for a in config.actions}
        for index in range(count + warmup):
            sample = {"index": index, "phase": "warmup" if index < warmup else "measured",
                      "status": "preparing"}
            record["samples"].append(sample)
            write_json(out / "latency.json", record)
            start = time.perf_counter()
            screen = environment.capture()
            captured = time.perf_counter()
            raw = png(screen.image)
            key, _ = media.put(raw)
            frame = store.add_raw(source=config.source, kind="frame", start=screen.frame_number / 60,
                end=screen.frame_number / 60, available_at=screen.frame_number / 60,
                payload={"media_key": key})
            context = bounded_context(goal=config.goal, intent=intent,
                schema=ontology.current, actions=[a.model_dump() for a in config.actions],
                world={"entities": [], "relations": [], "current_place": None, "recent_conversations": []},
                recent=[], current_frame_id=frame.id, max_chars=config.max_context_chars)
            images = [ImageInput(frame.id, inference_png(screen.image, config.view_scale))]
            prepared = time.perf_counter()
            sample.update(status="calling", frame_id=frame.id, frame_number=screen.frame_number,
                          png_sha256=hashlib.sha256(raw).hexdigest(), capture_s=captured - start,
                          evidence_and_image_preparation_s=prepared - captured)
            value = backend.complete("act", SYSTEMS["act"], context, images, ActionChoice)
            if value.action_id not in allowed:
                raise ContractError("Latency result returned an action outside the manifest")
            ended = time.perf_counter()
            sample.update(status="valid_not_executed", action_id=value.action_id,
                request_and_validation_s=ended - prepared, screenshot_to_usable_action_s=ended - start)
            write_json(out / "latency.json", record)
        timings = sorted(s["screenshot_to_usable_action_s"] for s in record["samples"]
                         if s["phase"] == "measured")
        record.update(status="completed", accounting=ledger.summary(),
            measured={"count": len(timings), "p50_s": statistics.median(timings),
                      "p95_s": timings[math.ceil(.95 * len(timings)) - 1],
                      "min_s": timings[0], "max_s": timings[-1],
                      "under_one_second": sum(t < 1 for t in timings)})
        write_json(out / "latency.json", record)
        return record
    except BaseException as exc:
        if record["samples"] and record["samples"][-1]["status"] != "valid_not_executed":
            record["samples"][-1].update(status="failed", error_type=type(exc).__name__)
        record.update(status="failed", error_type=type(exc).__name__,
                      accounting=ledger.summary() if ledger else None)
        write_json(out / "latency.json", record)
        raise
    finally:
        try:
            if backend:
                backend.close()
        finally:
            store.close()


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
