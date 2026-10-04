"""Actor-first stepped execution with asynchronously formed spatiotemporal memory.

This is not independent-clock real-time game control. The emulator still advances
only on explicit actions; semantic work can overlap many such actions.
"""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image

from ...media import MediaStore
from ...store import EvidenceStore
from ...types import ContractError
from ..contracts import ActionChoice, Endpoint
from ..locking import RunLock
from ..models import ImageInput, BudgetExhausted
from ..text import GenerativeTextReader, RapidTextReader, TemporalTextTracker, TextReading, TextRegionCache
from ..ontology import canonical
from .contracts import JobBasis, MemoryCutoff, MemoryDelta, SceneIndex, IntentPlan, OCRPacket
from .context import build_context
from .jobs import JobBroker
from .memory import TemporalMemory
from .transport import VLLMTransport
from . import prompts


def png(image):
    b = BytesIO()
    image.convert("RGB").save(b, "PNG")
    return b.getvalue()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def config_fingerprint(config):
    body = config.model_dump()
    for key in ("max_steps", "max_calls", "max_estimated_usd", "wall_limit_s", "max_storage_mb"):
        body["game"].pop(key, None)
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def database_digest(db):
    h = hashlib.sha256()
    # Exact record digest for clean resume, not a claim of adversarial tamper protection.
    tables = [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND "
            "(name LIKE 'tm_%' OR name LIKE 'async_%' OR name IN ('world_schemas','schema_candidates','model_calls','model_inputs','evidence')) ORDER BY name"
        )
    ]
    for table in tables:
        h.update(table.encode())
        for row in db.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
            h.update(canonical(dict(row)).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class FrameNotice:
    evidence_id: str
    frame_number: int
    source_seq: int
    png_bytes: bytes
    # Passed only to a trusted host callback. It cannot authorize an action.


class AsyncGameRunner:
    def __init__(
        self,
        config,
        env,
        out: Path,
        *,
        allow_network=False,
        transport=None,
        on_frame=None,
        resume=False,
        ocr_reader=None,
    ):
        self.config, self.env, self.out = config, env, Path(out).resolve()
        self.on_frame = on_frame
        self.start_wall, self.previous_wall = time.monotonic(), 0.0
        self.epoch = uuid.uuid4().hex
        self.task_revision, self.scene_revision, self.intent_revision = 0, 0, 0
        self.steps, self.last_plan_step = 0, 0
        self.intent = config.background.initial_intent
        self.focus = []
        self.intent_valid = not config.background.require_initial_plan
        self.needs_plan, self.status = True, "running"
        self.last_frames = deque(maxlen=config.game.history_frames)
        self.important = deque(maxlen=config.background.max_important_frames)
        self.last_index_step, self.last_index_wall = -100000, -float("inf")
        self.last_index_frame = None
        self.important_frame_omissions = 0
        self._thumbnail = None
        self._last_change = 1.0
        self._plan_round = 0
        self._retrieved = []
        self._inspect = []
        self._ocr_queue = deque(maxlen=config.background.max_ocr_queue)
        self.ocr_rejected = 0
        self.background_rejected = 0
        self.actor_discarded = 0
        self.lock = self.store = self.broker = None
        self._closing = False
        self._loop = None
        self._ocr_reader = ocr_reader
        self._ocr_executor = None
        self._ocr_job = None
        self._ocr_enabled = ocr_reader is not None or config.background.ocr_backend != "none"
        self._ocr_session = "ocr_" + self.epoch
        self._ocr_sequence = 0
        self._ocr_previous_frame = None
        self._ocr_previous_sha = None
        self._ocr_occurrence = None
        self._ocr_cache = TextRegionCache(max_age=config.game.ocr_refresh_s, threshold=0.001)
        self._text_tracker = TemporalTextTracker()
        self.ocr_stats = {"submitted": 0, "coalesced": 0, "cache_hits": 0, "failed": 0}
        try:
            if resume:
                if not self.out.is_dir():
                    raise ContractError("Missing run directory")
                self.lock = RunLock(self.out / ".interactive.lock")
                cp = json.loads((self.out / "async_checkpoint.json").read_text())
                self.meta = json.loads((self.out / "run.json").read_text())
                if self.meta.get("mode") != "background" or not cp.get("clean"):
                    raise ContractError("Only clean background-mode checkpoints can resume")
                if cp["config_fingerprint"] != config_fingerprint(config):
                    raise ContractError("Resume allows total-budget increases only; settings changed")
                if getattr(env, "rom_sha256", None) != self.meta.get("rom_sha256"):
                    raise ContractError("ROM provenance changed")
                descriptor = getattr(env, "descriptor", {})
                old = self.meta.get("environment") or {}
                if any(descriptor.get(k) != old.get(k) for k in ("adapter", "version", "window")):
                    raise ContractError("Environment adapter/version differs")
                if (
                    hashlib.sha256((self.out / "environment.state").read_bytes()).hexdigest()
                    != cp["state_sha256"]
                ):
                    raise ContractError("Checkpoint state bytes changed")
                env.frame_number = cp["environment_frame"]
                if hashlib.sha256(png(env.capture().image)).hexdigest() != cp["screen_sha256"]:
                    raise ContractError("Loaded environment does not match the checkpoint")
                import sqlite3

                check = sqlite3.connect((self.out / "memory.sqlite").as_uri() + "?mode=ro", uri=True)
                check.row_factory = sqlite3.Row
                try:
                    if database_digest(check) != cp["database_sha256"]:
                        raise ContractError(
                            "Stored records changed after checkpoint; read-only preflight failed"
                        )
                finally:
                    check.close()
                self.steps, self.intent, self.focus = cp["steps"], cp["intent"], cp["focus"]
                self.task_revision = cp["task_revision"]
                self.scene_revision, self.intent_revision = cp["scene_revision"], cp["intent_revision"]
                self.last_plan_step, self.intent_valid = cp["last_plan_step"], cp["intent_valid"]
                self.needs_plan = not self.intent_valid
                self.previous_wall = self.meta["elapsed_wall_s"]
            else:
                if self.out.exists():
                    raise ContractError("Use a fresh output path or explicit clean resume")
                self.out.mkdir(parents=True)
                self.lock = RunLock(self.out / ".interactive.lock")
                self.meta = {
                    "mode": "background",
                    "format_version": 1,
                    "run_id": self.epoch,
                    "config": config.model_dump(),
                    "rom_sha256": getattr(env, "rom_sha256", None),
                    "environment": getattr(env, "descriptor", None),
                    "initialization": getattr(env, "initialization", "unknown"),
                    "initial_state_sha256": getattr(env, "initial_state_sha256", None),
                    "fixture": config.game.backend == "fixture" or bool(getattr(env, "synthetic", False)),
                    "game_success": None,
                    "timing": "stepped emulator; memory/planning requests run asynchronously",
                    "ocr_engine": "external host-owned worker; no OCR model installed or invoked here",
                }
            self.store = EvidenceStore(self.out / "memory.sqlite", config.game.source)
            self.media = MediaStore(self.out / "media")
            self.memory = TemporalMemory(
                self.store,
                max_entities=config.game.max_entities,
                fact_threshold=config.game.min_fact_confidence,
                association_threshold=config.game.min_association_confidence,
                volatile_age_frames=config.background.volatile_age_frames,
            )
            self.store.db.executescript("""
              CREATE TABLE IF NOT EXISTS async_actions(
                id TEXT PRIMARY KEY, epoch TEXT, job_id TEXT, source_frame TEXT, action TEXT,
                status TEXT, receipt TEXT, after_frame TEXT, wall_s REAL);
              CREATE TABLE IF NOT EXISTS async_acceptance(job_id TEXT PRIMARY KEY, status TEXT, reason TEXT, revision INTEGER);
              CREATE TABLE IF NOT EXISTS async_ocr_jobs(
                call_id TEXT PRIMARY KEY, source_frame TEXT, role TEXT, status TEXT);
            """)
            self.store.db.commit()
            if resume:
                if database_digest(self.store.db) != cp["database_sha256"]:
                    raise ContractError(
                        "Evidence/accounting/memory changed after checkpoint; review before continuing"
                    )
                if self.store.db.execute(
                    "SELECT COUNT(*) FROM async_jobs WHERE status IN ('queued','running')"
                ).fetchone()[0]:
                    raise ContractError("Unresolved inference prevents automatic resume")
                if self.store.db.execute(
                    "SELECT COUNT(*) FROM async_actions WHERE status!='completed'"
                ).fetchone()[0]:
                    raise ContractError("Unresolved action prevents automatic resume")
                self.current = self.store.get(cp["current_frame_id"])
                if self.current.payload["frame_number"] != env.frame_number:
                    raise ContractError("Checkpoint frame reference mismatch")
                self.last_frames.extend(
                    self.store.frames(
                        config.game.source,
                        0,
                        self.current.end,
                        self.store.snapshot(self.current.end),
                        config.game.history_frames,
                    )
                )
                self.last_notice = FrameNotice(
                    self.current.id, env.frame_number, self.current.seq, png(env.capture().image)
                )
            else:
                self.current = self.capture()
            if transport is None:
                if config.game.backend != "chat":
                    raise ContractError("Fixture runs need an explicit fixture transport")
                if any(ep.billing != "local" for ep in config.game.endpoints.values()):
                    raise ContractError(
                        "Background mode is qualified only for local/tunnel vLLM; no metered fallback"
                    )
                if any(ep.model.startswith("SET_") for ep in config.game.endpoints.values()):
                    raise ContractError("Configure exact served model names")
                transport = VLLMTransport(
                    allow_network=allow_network, max_response_bytes=config.background.max_response_bytes
                )
            self.broker = JobBroker(config, self.store.db, transport, clock=self.now)
            if self._ocr_reader and self.meta.get("ocr_model") not in (None, self._ocr_reader.model_id):
                raise ContractError("Injected OCR model changed on resume")
            if (
                resume
                and self.store.db.execute(
                    "SELECT COUNT(*) FROM async_ocr_jobs WHERE status IN ('running','termination_unknown')"
                ).fetchone()[0]
            ):
                raise ContractError("Unresolved OCR prevents automatic resume")
            self.meta["status"] = "running"
            atomic_json(self.out / "run.json", self.meta)
        except BaseException:
            with suppress(Exception):
                env.close()
            if self.store:
                self.store.close()
            if self.lock:
                self.lock.close()
            raise

    def now(self):
        return self.previous_wall + time.monotonic() - self.start_wall

    def log(self, kind, **value):
        with (self.out / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                canonical(
                    {"kind": kind, "wall_s": self.now(), "step": self.steps, "epoch": self.epoch, **value}
                )
                + "\n"
            )

    def capture(self):
        screen = self.env.capture()
        raw = png(screen.image)
        key, _ = self.media.put(raw)
        frame = self.store.add_raw(
            source=self.config.game.source,
            kind="frame",
            start=screen.frame_number / 60,
            end=screen.frame_number / 60,
            available_at=screen.frame_number / 60,
            payload={
                "media_key": key,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "frame_number": screen.frame_number,
                "width": screen.image.width,
                "height": screen.image.height,
            },
        )
        self.last_frames.append(frame)
        a = np.asarray(screen.image.resize((64, 64)).convert("RGB"), dtype=float) / 255
        self._last_change = float(np.abs(a - self._thumbnail).mean()) if self._thumbnail is not None else 1.0
        if self._thumbnail is not None and self._last_change >= self.config.background.scene_change:
            self.scene_revision += 1
            self.intent_valid, self.needs_plan = False, True
            self.log(
                "visual_boundary",
                source_frame=frame.id,
                note="Pixel-change heuristic, not semantic recognition",
            )
        self._thumbnail = a
        if self._last_change >= self.config.background.scene_change:
            if len(self.important) == self.important.maxlen:
                self.important_frame_omissions += 1
                self.log(
                    "keyframe_index_omitted",
                    frame_id=self.important[0],
                    reason="bounded important-frame queue; original evidence retained",
                )
            self.important.append(frame.id)
        self.last_notice = FrameNotice(frame.id, screen.frame_number, frame.seq, raw)
        if self.on_frame and self._loop is not None:
            # The callback should enqueue CPU work, not run OCR synchronously on this thread.
            self.on_frame(self.last_notice)
        return frame

    def image(self, frame, *, box=None, historical=False):
        key = frame.payload.get("media_key")
        path = (self.media.root / key).resolve() if key else None
        if not path or path.parent != self.media.root or not path.is_file():
            raise ContractError("Invalid source media path")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != key.split(".")[0]:
            raise ContractError("Retained media digest mismatch")
        im = MediaStore.decode(raw)
        if box is not None:
            a, b, c, d = box
            if not 0 <= a < c <= 1 or not 0 <= b < d <= 1:
                raise ContractError("Invalid crop bounds")
            im = im.crop(
                (
                    int(a * im.width),
                    int(b * im.height),
                    max(int(c * im.width), int(a * im.width) + 1),
                    max(int(d * im.height), int(b * im.height) + 1),
                )
            )
        if max(im.size) <= 320:
            im = im.resize(
                (im.width * self.config.game.view_scale, im.height * self.config.game.view_scale),
                Image.Resampling.NEAREST,
            )
        else:
            im.thumbnail((1024, 1024))
        # Root source ID is preserved; optional crop metadata is supplied in the context.
        return ImageInput(frame.id, png(im), historical)

    def offer_ocr(self, packet: OCRPacket):
        """Called on the loop thread. Other threads use loop.call_soon_threadsafe(...)."""
        self.memory._owner()
        if self._closing or len(self._ocr_queue) >= self._ocr_queue.maxlen:
            self.ocr_rejected += 1
            self.log("ocr_rejected", reason="closed_or_queue_full")
            return False
        # Copy at the boundary so a producer cannot mutate an enqueued observation.
        self._ocr_queue.append(OCRPacket.model_validate_json(packet.model_dump_json()))
        return True

    def _basis(self, frame, packet, images, *, targets=()):
        return JobBasis(
            "job_" + uuid.uuid4().hex,
            frame.id,
            frame.seq,
            frame.payload["frame_number"],
            packet.cutoff.revision,
            self.memory.ontology.current["version"],
            self.epoch,
            self.task_revision,
            self.scene_revision,
            self.intent_revision,
            tuple(dict.fromkeys(i.id for i in images)),
            packet.entities,
            packet.evidence,
            tuple(targets),
            self.steps,
        )

    def _submit(self, role, *, frame=None, targets=(), retrieved=(), anchors=(), key=None):
        frame = frame or self.current
        packet = build_context(
            self.memory,
            self.config,
            frame,
            intent=self.intent,
            focus=self.focus,
            role=role,
            retrieved=retrieved,
            targets=targets,
        )
        if role == "act":
            frames = list(self.last_frames)[-self.config.background.actor_image_history :]
            images = [self.image(f, historical=f.id != frame.id) for f in frames]
        elif role == "index":
            images = [self.image(frame, historical=frame.id != self.current.id)]
        else:
            images = [self.image(frame, historical=frame.id != self.current.id)]
        # Optional historical anchors retain geometry/source. No filename or model URL reader.
        for anchor in anchors[: max(0, self.config.game.max_images - len(images))]:
            root = self.store.get(anchor["frame_id"])
            if root.seq > frame.seq:
                raise ContractError("Inspection would reveal a future source")
            images.append(self.image(root, box=anchor.get("box"), historical=True))
        if anchors:
            packet.value["visual_anchors"] = list(anchors[: max(0, self.config.game.max_images - 1)])
        output, system = {
            "act": (ActionChoice, prompts.ACT),
            "plan": (IntentPlan, prompts.PLAN),
            "index": (SceneIndex, prompts.INDEX),
            "enrich": (MemoryDelta, prompts.ENRICH),
        }[role]
        basis = self._basis(frame, packet, images, targets=targets)
        ticket = self.broker.submit(role, basis, system, packet.value, images, output, key=key)
        self.log(
            "job_offered",
            job_id=ticket.id,
            role=role,
            source_frame=frame.id,
            knowledge_revision=basis.knowledge_revision,
            targets=list(targets),
        )
        return ticket

    def _schedule_index(self, force=False):
        cfg = self.config.background
        if self._closing or (self.current.id == self.last_index_frame and not self.important):
            return
        # Wait to FORM the next index packet until the previous index has committed.
        # Otherwise its frozen context may mint duplicate nodes because it cannot see
        # the previous index's IDs. Actor/enrichment/OCR still progress concurrently.
        if any(t.role == "index" for t in self.broker.tickets.values()):
            return
        due = (
            bool(self.important)
            or force
            or self.steps - self.last_index_step >= cfg.index_every_actions
            or self._last_change >= cfg.thumbnail_change
            or self.now() - self.last_index_wall >= cfg.index_refresh_s
        )
        if not due or self.now() - self.last_index_wall < cfg.index_min_interval_s:
            return
        # Queue replacement affects undispatched disposable snapshots only. Source frames remain on disk.
        frame = self.store.get(self.important.popleft()) if self.important else self.current
        important = frame.id != self.current.id or self._last_change >= cfg.scene_change
        self._submit("index", frame=frame, key=("boundary:" + frame.id) if important else "latest_index")
        self.last_index_frame, self.last_index_step, self.last_index_wall = frame.id, self.steps, self.now()

    def _schedule_plan(self):
        if self._closing or self.broker.pending_key("plan") or not self.needs_plan:
            return
        self._submit("plan", retrieved=self._retrieved, anchors=self._inspect, key="plan")

    def _accepted(self, ticket, status, reason="", revision=None):
        with self.store.db:
            self.store.db.execute(
                "INSERT OR REPLACE INTO async_acceptance VALUES(?,?,?,?)",
                (ticket.id, status, reason, revision),
            )
        self.log(
            "result_acceptance",
            job_id=ticket.id,
            role=ticket.role,
            status=status,
            reason=reason,
            revision=revision,
        )

    def _applicable(self, ticket):
        b, cfg = ticket.basis, self.config.background
        return (
            b.episode == self.epoch
            and b.task_revision == self.task_revision
            and b.ontology_version == self.memory.ontology.current["version"]
            and b.scene_revision == self.scene_revision
            and b.intent_revision == self.intent_revision
            and self.current.payload["frame_number"] - b.frame_number <= cfg.plan_max_age_frames
            and self.steps - b.action_step <= cfg.plan_max_age_actions
        )

    def _queries(self, ticket, plan):
        cutoff = MemoryCutoff(ticket.basis.knowledge_revision, ticket.basis.source_seq)
        offered = set(ticket.basis.offered_entities)
        allowed_evidence = set(ticket.basis.evidence_ids) | set(ticket.basis.offered_frames)
        results, anchors = [], []
        for q in plan.queries:
            for id in (q.entity_id, q.target_id):
                if id and id not in offered:
                    raise ContractError("Memory tool entity was not exposed to the planner")
            if q.evidence_id and q.evidence_id not in allowed_evidence:
                raise ContractError("Inspection evidence was not exposed")
            if q.kind == "search":
                value = self.memory.search(q.text, cutoff, limit=q.limit, entity=q.entity_id)
            elif q.kind == "conversation":
                value = self.memory.conversation(q.entity_id, cutoff, query=q.text, limit=q.limit)
            elif q.kind == "events":
                value = self.memory.event_history(cutoff, entity=q.entity_id, query=q.text, limit=q.limit)
            elif q.kind == "place":
                if not q.entity_id:
                    raise ContractError("Place query requires entity_id")
                value = self.memory.place_summary(q.entity_id, cutoff)
            elif q.kind == "route":
                if not q.entity_id or not q.target_id:
                    raise ContractError("Route requires origin and target")
                value = self.memory.route(q.entity_id, q.target_id, cutoff)
            elif q.kind == "inspect":
                if q.entity_id:
                    value = self.memory.visual_refs(
                        q.entity_id, cutoff, limit=min(self.config.background.max_views_per_entity, q.limit)
                    )
                    anchors.extend(value)
                elif q.evidence_id:
                    frame = self.store.get(q.evidence_id)
                    if frame.kind != "frame":
                        item = self.store.db.execute(
                            "SELECT basis FROM tm_patches WHERE evidence=?", (q.evidence_id,)
                        ).fetchone()
                        if not item:
                            raise ContractError("Inspect a source frame or indexed semantic evidence")
                        frame = self.store.get(json.loads(item["basis"])["source_frame_id"])
                    value = [{"frame_id": frame.id, "historical": True}]
                    anchors.extend(value)
                else:
                    raise ContractError("Inspection needs an exposed entity/evidence handle")
            results.append({"query": q.model_dump(), "result": value})
        return results, anchors

    def _consume(self, ticket):
        if ticket.status != "completed":
            self._accepted(ticket, ticket.status, ticket.error or "")
            if ticket.status == "rejected_budget":
                self.status = "budget_limit"
            elif ticket.role == "act" and ticket.status not in ("cancelled_stop", "superseded"):
                self.status = "actor_failed"
            elif ticket.status == "failed":
                self.background_rejected += 1
                if ticket.role == "plan":
                    self.intent_valid = False
                    self.status = "planner_failed"
            return
        try:
            if ticket.role == "act":
                return  # The action path validates freshness again immediately before dispatch.
            if ticket.role == "index":
                value = ticket.value
                old_place = self.memory.current_place(self.memory.cutoff(self.current.seq))
                applied = self.memory.commit(
                    ticket.basis,
                    MemoryDelta(
                        mentions=value.mentions,
                        current_place=value.current_place,
                        needs_planning=value.needs_planning,
                    ),
                    now=self.now(),
                )
                self._accepted(ticket, "memory_committed", revision=applied["revision"])
                new_place = self.memory.current_place(self.memory.cutoff(self.current.seq))
                if (
                    old_place
                    and new_place
                    and old_place.get("id")
                    and new_place.get("id")
                    and old_place["id"] != new_place["id"]
                ):
                    self.scene_revision += 1
                    self.intent_valid, self.needs_plan = False, True
                    self.log(
                        "place_boundary",
                        old_place=old_place["id"],
                        new_place=new_place["id"],
                        note="Newest observed place belief changed; not hidden game truth",
                    )
                # Fan-out jobs are formed AFTER the index mints shared handles. They do not
                # independently decide that three similar crops are the same entity.
                refs = [applied["ids"].get(r, r) for r in value.enrich]
                legal = set(applied["ids"].values()) | set(ticket.basis.offered_entities)
                if set(refs) - legal:
                    raise ContractError("Enrichment requested an unknown/unindexed handle")
                if not self._closing:
                    frame = self.store.get(ticket.basis.source_frame_id)
                    for ref in list(dict.fromkeys(refs))[: self.config.background.max_fanout]:
                        self._submit("enrich", frame=frame, targets=(ref,), key=None)
                if value.needs_planning and self._applicable(ticket):
                    self.needs_plan = True
            elif ticket.role == "enrich":
                applied = self.memory.commit(ticket.basis, ticket.value, now=self.now())
                self._accepted(ticket, "memory_committed", revision=applied["revision"])
                if ticket.value.needs_planning and self._applicable(ticket):
                    self.needs_plan = True
            elif ticket.role == "plan":
                if self._closing or not self._applicable(ticket):
                    self._accepted(ticket, "discarded_stale_plan")
                    self.needs_plan = True
                    self._retrieved, self._inspect, self._plan_round = [], [], 0
                    return
                if set(ticket.value.focus_ids) - set(ticket.basis.offered_entities):
                    raise ContractError("Planner invented an entity handle")
                if ticket.value.queries and self._plan_round < self.config.game.max_retrieval_rounds:
                    self._retrieved, self._inspect = self._queries(ticket, ticket.value)
                    self._plan_round += 1
                    self._accepted(ticket, "retrieval_completed")
                    self.needs_plan = True
                    return
                if ticket.value.queries:
                    self._accepted(ticket, "retrieval_budget_exhausted")
                    self.status = "blocked"
                    self.intent_valid = False
                    return
                from dataclasses import replace

                plan_event = MemoryDelta.model_validate(
                    {
                        "events": [
                            {
                                "kind": "planner_goal_claim"
                                if ticket.value.status == "goal_claimed"
                                else "intent_selected",
                                "text": ticket.value.intent,
                                "participants": ticket.value.focus_ids,
                                "occurrence": ticket.id,
                                "confidence": 0.5,
                            }
                        ]
                    }
                )
                self.memory.commit(
                    replace(ticket.basis, job_id=ticket.id + "_decision"), plan_event, now=self.now()
                )
                self.intent, self.focus = ticket.value.intent, ticket.value.focus_ids
                self.intent_revision += 1
                self.last_plan_step, self.needs_plan, self.intent_valid = self.steps, False, True
                self._retrieved, self._inspect, self._plan_round = [], [], 0
                self._accepted(ticket, "plan_accepted")
                if ticket.value.status == "blocked":
                    self.status = "blocked"
                elif ticket.value.status == "goal_claimed" and self.config.game.stop_on_goal_claim:
                    self.meta["unverified_goal_claim"] = self.intent
                    self.status = "goal_claimed"
        except ContractError as exc:
            # Retain rejected semantic evidence/attempts, do not silently retry or loosen gates.
            self.background_rejected += 1
            self._accepted(ticket, "rejected_semantics", str(exc))
            if ticket.role == "plan" or self.background_rejected >= 3:
                self.status = "semantic_review_required"

    def _drain_ocr(self):
        while self._ocr_queue:
            packet = self._ocr_queue.popleft()
            try:
                self.memory.ingest_ocr(packet, now=self.now(), epoch=self.epoch)
            except ContractError as exc:
                self.ocr_rejected += 1
                self.log("ocr_rejected", reason=str(exc))

    def _load_ocr_reader(self):
        settings = self.config.background
        if settings.ocr_backend == "rapid":
            return RapidTextReader(self.config.game.ocr_config_path, scale=self.config.game.ocr_scale)
        return GenerativeTextReader(
            settings.ocr_backend,
            settings.ocr_model_paths[settings.ocr_backend],
            device=settings.ocr_device,
            max_tokens=settings.ocr_max_tokens,
            transformers_path=settings.ocr_transformers_path if settings.ocr_backend == "hunyuan" else None,
        )

    @staticmethod
    def _invoke_ocr(operation, *args):
        start = time.monotonic()
        try:
            return operation(*args), time.monotonic() - start, None
        except Exception as exc:
            return None, time.monotonic() - start, exc

    def _schedule_ocr(self):
        if not self._ocr_enabled or self._closing:
            return
        if self._ocr_job:
            self.ocr_stats["coalesced"] += 1
            return
        if self._ocr_previous_frame == self.current.id:
            return
        image = MediaStore.decode(self.last_notice.png_bytes)
        loading = self._ocr_reader is None
        if not loading and not self._ocr_cache.needs_read("screen", image, self.now()):
            self.ocr_stats["cache_hits"] += 1
            return
        role = "ocr_setup" if loading else "ocr"
        ep = Endpoint(
            model=("ocr_setup:" + self.config.background.ocr_backend)
            if loading
            else self._ocr_reader.model_id
        )
        context = {
            "source_frame_id": self.current.id,
            "sha256": self.current.payload["sha256"],
            "backend": self.config.background.ocr_backend,
        }
        try:
            call = self.broker.ledger.reserve(
                role,
                ep,
                hashlib.sha256(canonical(context).encode()).hexdigest(),
                0 if loading else 1,
                len(canonical(context)),
            )
        except BudgetExhausted:
            self.status = "budget_limit"
            return
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO model_inputs VALUES(?,?,?,?)",
                (
                    call,
                    canonical(context),
                    "Local pixels-only OCR; no game context or labels",
                    canonical(
                        [{"id": self.current.id, "sha256": self.current.payload["sha256"]}]
                        if not loading
                        else []
                    ),
                ),
            )
            self.store.db.execute(
                "INSERT INTO async_ocr_jobs VALUES(?,?,?,?)", (call, self.current.id, role, "running")
            )
        if self._ocr_executor is None:
            self._ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        job = {
            "call": call,
            "ep": ep,
            "frame": self.current,
            "image": image,
            "start": self.now(),
            "loading": loading,
        }
        self._ocr_job = job
        try:
            job["future"] = self._ocr_executor.submit(
                self._invoke_ocr,
                self._load_ocr_reader if loading else self._ocr_reader.read,
                *(() if loading else (image, self.current.id)),
            )
        except Exception as exc:
            self._settle_ocr_error(job, exc)
            self._ocr_job = None
            self.status = "ocr_failed"
        self.ocr_stats["submitted"] += 1
        self.log("ocr_submitted", call_id=call, role=role, source_frame=self.current.id)

    def _settle_ocr_error(self, job, exc):
        self.broker.ledger.finish(
            job["call"],
            endpoint=job["ep"],
            response=getattr(exc, "response", None),
            elapsed=job.get("worker_wall_s", self.now() - job["start"]),
            error=type(exc).__name__,
        )
        with self.store.db:
            self.store.db.execute("UPDATE async_ocr_jobs SET status='failed' WHERE call_id=?", (job["call"],))
        self.ocr_stats["failed"] += 1
        self.log("ocr_failed", call_id=job["call"], error_type=type(exc).__name__)

    def _finish_local_ocr(self):
        job = self._ocr_job
        if job is None or not job["future"].done():
            return
        self._ocr_job = None
        try:
            result, job["worker_wall_s"], error = job["future"].result()
            if error is not None:
                raise error
            if job["loading"]:
                if self.meta.get("ocr_model") not in (None, result.model_id):
                    raise ContractError("OCR weights or inference configuration changed on resume")
                self._ocr_reader = result
                self.meta["ocr_model"] = result.model_id
                response = {"manifest": getattr(result, "manifest", {}), "model_id": result.model_id}
            else:
                frame = job["frame"]
                if isinstance(result, TextReading):
                    if result.evidence_id != frame.id:
                        raise ContractError("OCR reading refers to the wrong source")
                    if self._ocr_previous_sha != frame.payload["sha256"]:
                        self._ocr_occurrence = uuid.uuid4().hex
                    lines = (
                        [
                            {
                                "track_id": "screen-transcription",
                                "occurrence": self._ocr_occurrence,
                                "text": result.text,
                                "confidence": None,
                                "box": None,
                            }
                        ]
                        if result.text
                        else []
                    )
                    response = asdict(result)
                    response["usage"] = {
                        "prompt_tokens": result.input_tokens,
                        "completion_tokens": result.generated_tokens,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                else:
                    if any(o.evidence_id != frame.id for o in result):
                        raise ContractError("OCR observation refers to the wrong source")
                    self._text_tracker.update(result, frame.end)
                    lines = [
                        {
                            "track_id": track.id,
                            "occurrence": track.id,
                            "text": track.text,
                            "confidence": track.confidence,
                            "box": list(track.box),
                        }
                        for track in self._text_tracker.tracks.values()
                        if track.last_seen == frame.end
                    ]
                    response = {"observations": [asdict(o) for o in result]}
                packet = OCRPacket(
                    worker_session=self._ocr_session,
                    source_frame_id=frame.id,
                    sequence=self._ocr_sequence,
                    lines=lines,
                )
                self.memory.ingest_ocr(packet, now=self.now(), epoch=self.epoch)
                self._ocr_sequence += 1
                self._ocr_previous_frame, self._ocr_previous_sha = frame.id, frame.payload["sha256"]
                self._ocr_cache.record("screen", job["image"], job["start"])
            self.broker.ledger.finish(
                job["call"], endpoint=job["ep"], response=response, elapsed=job["worker_wall_s"]
            )
            with self.store.db:
                self.store.db.execute(
                    "UPDATE async_ocr_jobs SET status='completed' WHERE call_id=?", (job["call"],)
                )
            self.log(
                "ocr_completed",
                call_id=job["call"],
                source_frame=job["frame"].id,
                worker_wall_s=job["worker_wall_s"],
                delivery_delay_s=max(0, self.now() - job["start"] - job["worker_wall_s"]),
            )
        except Exception as exc:
            self._settle_ocr_error(job, exc)
            self.status = "ocr_failed"

    async def _close_local_ocr(self):
        clean = True
        try:
            if self._ocr_job:
                job = self._ocr_job
                try:
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.wrap_future(job["future"])),
                        timeout=self.config.background.background_drain_s,
                    )
                except TimeoutError:
                    clean = False
                    with self.store.db:
                        self.store.db.execute(
                            "UPDATE async_ocr_jobs SET status='termination_unknown' WHERE call_id=?",
                            (job["call"],),
                        )
                    self.log(
                        "ocr_unresolved",
                        call_id=job["call"],
                        note="Local thread may still run; no clean resume",
                    )
                except Exception:
                    pass  # The owner settles the original exception below.
                if clean:
                    self._finish_local_ocr()
        finally:
            if self._ocr_executor:
                self._ocr_executor.shutdown(wait=clean, cancel_futures=True)
        return clean

    def pump(self):
        self._finish_local_ocr()
        self._drain_ocr()
        for t in self.broker.pump():
            self._consume(t)
        if self.broker.quarantined:
            self.status = "transport_quarantined"

    async def _wait_ticket(self, ticket):
        while ticket.status in ("queued", "running") and self.status == "running":
            self.pump()
            if self.now() >= self.config.game.wall_limit_s:
                self.status = "wall_limit"
                break
            if ticket.status in ("queued", "running"):
                await self.broker.wait_activity()
        return ticket.value if ticket.status == "completed" else None

    def _execute(self, ticket):
        if self.status != "running" or self.now() >= self.config.game.wall_limit_s:
            return
        if (
            ticket.basis.task_revision != self.task_revision
            or ticket.basis.intent_revision != self.intent_revision
            or ticket.basis.scene_revision != self.scene_revision
        ):
            self.actor_discarded += 1
            self._accepted(ticket, "discarded_stale_actor_intent")
            return
        action = next((a for a in self.config.game.actions if a.id == ticket.value.action_id), None)
        if action is None:
            raise ContractError("Actor chose a capability that was not offered")
        fresh = self.env.capture()
        if (
            fresh.frame_number != ticket.basis.frame_number
            or hashlib.sha256(png(fresh.image)).hexdigest() != self.current.payload["sha256"]
        ):
            raise ContractError("Source changed before action; no stale-action fallback")
        id = "execution_" + uuid.uuid4().hex
        with self.store.db:
            self.store.db.execute(
                "INSERT INTO async_actions VALUES(?,?,?,?,?,?,NULL,NULL,?)",
                (
                    id,
                    self.epoch,
                    ticket.id,
                    self.current.id,
                    canonical(action.model_dump()),
                    "dispatched",
                    self.now(),
                ),
            )
        receipt = self.env.execute(action)  # The sole actuator owner. No automatic retries.
        with self.store.db:
            self.store.db.execute(
                "UPDATE async_actions SET status='completed',receipt=? WHERE id=?",
                (canonical(asdict(receipt)), id),
            )
        self.store.add_raw(
            source=self.config.game.source,
            kind="action_receipt",
            start=receipt.start_frame / 60,
            end=receipt.end_frame / 60,
            available_at=receipt.end_frame / 60,
            text=f"Executed {action.id}; effect is not independently verified",
            payload={"execution_id": id, **asdict(receipt)},
        )
        self.current = self.capture()
        with self.store.db:
            self.store.db.execute("UPDATE async_actions SET after_frame=? WHERE id=?", (self.current.id, id))
        self.steps += 1
        self._accepted(ticket, "action_dispatched")
        self.log(
            "action",
            execution_id=id,
            job_id=ticket.id,
            action=action.id,
            after_frame=self.current.id,
            knowledge_revision=ticket.basis.knowledge_revision,
            pending_memory=sum(t.role in ("index", "enrich") for t in self.broker.tickets.values()),
        )

    async def run(self):
        self._loop = asyncio.get_running_loop()
        clean = False
        try:
            if self.on_frame:
                self.on_frame(self.last_notice)
            self._schedule_ocr()
            self._schedule_index(force=True)
            self._schedule_plan()
            while self.steps < self.config.game.max_steps and self.status == "running":
                if self.now() >= self.config.game.wall_limit_s:
                    self.status = "wall_limit"
                    break
                self.pump()
                if self.status != "running":
                    break
                if self.steps - self.last_plan_step >= self.config.background.intent_max_actions:
                    self.intent_valid = False
                    self.needs_plan = True
                elif self.steps - self.last_plan_step >= self.config.game.plan_every:
                    self.needs_plan = True
                self._schedule_plan()
                self._schedule_index()
                self._schedule_ocr()
                if not self.intent_valid:
                    await self.broker.wait_activity()
                    continue
                ticket = self._submit("act")
                value = await self._wait_ticket(ticket)
                if value is not None and self.status == "running":
                    self._execute(ticket)
                if (
                    sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
                    > self.config.game.max_storage_mb * 1024**2
                ):
                    self.status = "storage_limit"
            if self.status == "running":
                self.status = "step_limit"
        except BaseException as exc:
            self.status = "failed"
            self.log("failure", error_type=type(exc).__name__)
            raise
        finally:
            self._closing = True
            try:
                ocr_clean = await self._close_local_ocr()
                self._drain_ocr()
                clean = await self.broker.close(
                    self.config.background.background_drain_s, consume=self._consume
                )
                unresolved = self.store.db.execute(
                    "SELECT COUNT(*) FROM async_actions WHERE status!='completed'"
                ).fetchone()[0]
                clean = ocr_clean and (
                    clean
                    and not unresolved
                    and self.status not in ("failed", "actor_failed", "transport_quarantined")
                )
                # Final stop does not run more actions or start more enrichment jobs.
                screen = self.env.capture()
                if (
                    screen.frame_number != self.current.payload["frame_number"]
                    or hashlib.sha256(png(screen.image)).hexdigest() != self.current.payload["sha256"]
                ):
                    clean = False
                if clean:
                    self.env.checkpoint(self.out / "environment.state")
                cutoff = self.memory.cutoff(self.current.seq)
                self.meta.update(
                    status=self.status,
                    completed_steps=self.steps,
                    elapsed_wall_s=self.now(),
                    accounting=self.broker.metrics(),
                    memory_revision=self.memory.revision,
                    memory_nodes=len(self.memory.nodes(cutoff)),
                    actor_discarded=self.actor_discarded,
                    semantic_rejections=self.background_rejected,
                    ocr_rejected=self.ocr_rejected,
                    ocr_engine=self.config.background.ocr_backend,
                    ocr_stats=self.ocr_stats,
                    important_frame_omissions=self.important_frame_omissions,
                    final_place=self.memory.current_place(cutoff),
                    clean_checkpoint=clean,
                    game_success=None,
                )
                atomic_json(self.out / "run.json", self.meta)
                cp = {
                    "clean": clean,
                    "config_fingerprint": config_fingerprint(self.config),
                    "steps": self.steps,
                    "intent": self.intent,
                    "focus": self.focus,
                    "current_frame_id": self.current.id,
                    "environment_frame": self.current.payload["frame_number"],
                    "screen_sha256": self.current.payload["sha256"],
                    "task_revision": self.task_revision,
                    "scene_revision": self.scene_revision,
                    "intent_revision": self.intent_revision,
                    "last_plan_step": self.last_plan_step,
                    "intent_valid": self.intent_valid,
                    "ontology_version": self.memory.ontology.current["version"],
                    "database_sha256": database_digest(self.store.db),
                    "state_sha256": hashlib.sha256((self.out / "environment.state").read_bytes()).hexdigest()
                    if clean
                    else None,
                }
                atomic_json(self.out / "async_checkpoint.json", cp)
            finally:
                try:
                    self.env.close()
                finally:
                    self.store.close()
                    self.lock.close()
        return self.meta
