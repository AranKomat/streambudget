"""Bounded, non-actuating vLLM probes on an operator-provided still image."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
import uuid

from ...media import MediaStore
from ...store import EvidenceStore
from ...types import ContractError
from ..contracts import ActionChoice
from ..models import ImageInput
from .contracts import JobBasis, SceneIndex
from .context import build_context
from .jobs import JobBroker
from .memory import TemporalMemory
from .runner import png, atomic_json
from .transport import VLLMTransport
from . import prompts


async def probe(
    config, image: Path, out: Path, *, mode="mixed", count=3, allow_network=False, transport=None
):
    if not 1 <= count <= 20 or mode not in ("actor", "index", "mixed"):
        raise ContractError("Use count 1..20 and actor/index/mixed mode")
    out = out.resolve()
    if out.exists():
        raise ContractError("Use a fresh probe directory")
    if transport is None and (
        not allow_network or any(e.billing != "local" for e in config.game.endpoints.values())
    ):
        raise ContractError(
            "Probe requires explicit network permission and loopback/tunneled local endpoints"
        )
    if transport is None and any(e.model.startswith("SET_") for e in config.game.endpoints.values()):
        raise ContractError("Configure exact served model names before a probe")
    raw = image.read_bytes()
    im = MediaStore.decode(raw)
    out.mkdir(parents=True)
    store = EvidenceStore(out / "memory.sqlite", config.game.source)
    media = MediaStore(out / "media")
    broker = None
    start = time.monotonic()
    report = {
        "mode": "non_actuating_probe",
        "actuation": False,
        "status": "running",
        "requested_pairs_or_calls": count,
        "condition": "Repeated frozen image; cache-friendly. Not sustained gameplay, GPU throughput or task quality.",
        "config": config.model_dump(),
    }
    try:
        data = png(im)
        key, _ = media.put(data)
        frame = store.add_raw(
            source=config.game.source,
            kind="frame",
            start=0,
            end=0,
            available_at=0,
            payload={
                "frame_number": 0,
                "media_key": key,
                "sha256": hashlib.sha256(data).hexdigest(),
                "width": im.width,
                "height": im.height,
            },
        )
        if max(im.size) <= 320:
            from PIL import Image

            im = im.resize(
                (im.width * config.game.view_scale, im.height * config.game.view_scale),
                Image.Resampling.NEAREST,
            )
        else:
            im.thumbnail((1024, 1024))
        image_input = ImageInput(frame.id, png(im), False)
        memory = TemporalMemory(store)
        transport = transport or VLLMTransport(
            allow_network=allow_network, max_response_bytes=config.background.max_response_bytes
        )
        broker = JobBroker(config, store.db, transport, clock=lambda: time.monotonic() - start)
        for sample in range(count):
            roles = ["act", "index"] if mode == "mixed" else ["act" if mode == "actor" else "index"]
            tickets = []
            for role in roles:
                packet = build_context(
                    memory, config, frame, intent=config.background.initial_intent, role=role
                )
                basis = JobBasis(
                    "job_" + uuid.uuid4().hex,
                    frame.id,
                    frame.seq,
                    0,
                    packet.cutoff.revision,
                    memory.ontology.current["version"],
                    "probe",
                    0,
                    0,
                    0,
                    (frame.id,),
                    packet.entities,
                    packet.evidence,
                )
                tickets.append(
                    broker.submit(
                        role,
                        basis,
                        prompts.ACT if role == "act" else prompts.INDEX,
                        packet.value,
                        [image_input],
                        ActionChoice if role == "act" else SceneIndex,
                    )
                )
            while any(t.status in ("queued", "running") for t in tickets):
                broker.pump()
                await broker.wait_activity()
            if any(t.status != "completed" for t in tickets):
                report["status"] = "failed"
                break
        else:
            report["status"] = "completed"
    finally:
        if broker:
            await broker.close(config.background.background_drain_s)
            report["metrics"] = broker.metrics()
        report["wall_s"] = time.monotonic() - start
        atomic_json(out / "probe.json", report)
        store.close()
    return report
