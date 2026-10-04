"""One-writer stepped closed loop. No simulator truth enters model context."""
from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import uuid

from PIL import Image

from ..media import MediaStore
from ..store import EvidenceStore
from ..types import ContractError
from . import prompts
from .context import bounded_context
from .contracts import ActionChoice, Endpoint, GameConfig, ObservationPatch, Plan, SchemaPatch
from .fixture import FixtureBackend
from .models import BudgetExhausted, ChatBackend, ImageInput, Ledger
from .ontology import canonical
from .locking import RunLock
from .world import World
from .text import RapidTextReader, TemporalTextTracker, TextRegionCache, estimate_vertical_scroll


def png(image):
    b = BytesIO()
    image.convert("RGB").save(b, "PNG")
    return b.getvalue()


def inference_png(image, view_scale):
    # Probes and the live loop must qualify the same view; originals stay immutable.
    image = image.copy()
    if max(image.size) <= 320:
        image = image.resize((image.width * view_scale, image.height * view_scale), Image.Resampling.NEAREST)
    else:
        image.thumbnail((1024, 1024))
    return png(image)


def write_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def immutable_config(config):
    body = config.model_dump()
    for k in ("max_steps", "max_calls", "max_estimated_usd", "wall_limit_s"):
        body.pop(k)
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def attempt_digest(db):
    # Counts alone cannot detect a changed charge, request, status or action receipt.
    digest = hashlib.sha256()
    for table, key in (("model_calls", "id"), ("model_inputs", "call_id"), ("game_actions", "id")):
        digest.update((table + "\n").encode())
        for row in db.execute(f"SELECT * FROM {table} ORDER BY {key}"):
            digest.update((canonical(dict(row)) + "\n").encode())
    return digest.hexdigest()


class GameRunner:
    def __init__(self, config, env, out, **kwargs):
        self._lock = None
        try:
            self._initialize(config, env, out, **kwargs)
        except BaseException:
            for name in ("background", "ocr_worker"):
                worker = getattr(self, name, None)
                if worker:
                    worker.shutdown(wait=True)
            with suppress(Exception):
                env.close()
            with suppress(Exception):
                close = getattr(getattr(self, "backend", None), "close", None)
                if close:
                    close()
            try:
                if hasattr(self, "store"):
                    self.store.close()
            finally:
                if self._lock:
                    self._lock.close()
            raise

    def _initialize(self, config: GameConfig, env, out: Path, *, allow_network=False,
                 allow_paid=False, backend=None, resume=False, ocr_reader=None):
        self.config, self.env, self.out = config, env, out.resolve()
        self.start_wall = time.monotonic()
        self.completed_steps, self.last_plan_step = 0, -100000
        self.intent = "Observe the current situation and choose a useful next step."
        self.focus, self.last_frames = [], []
        self.needs_plan = True
        self.epoch = uuid.uuid4().hex
        self.status = "running"
        self.previous_wall = 0.0
        self.background, self.ocr_worker = None, None
        self.semantic_job, self.plan_job, self.ocr_job = None, None, None
        self.last_semantic_step, self.semantic_hash = -100000, None
        self.semantic_due = True
        # Cold start: acquire one grounded memory observation before proposing focus IDs.
        self.last_background_role = "plan"
        self.hot_text = None
        self.text_tracker = TemporalTextTracker(max_gap=5)
        self.text_cache = TextRegionCache(max_age=config.ocr_refresh_s, threshold=0.001)
        self.ocr_previous_image, self.ocr_previous_frame = None, None
        self.async_stats = {"semantic_submissions": 0, "semantic_coalesced_frames": 0,
            "ocr_submissions": 0, "ocr_coalesced_frames": 0, "ocr_frame_cache_hits": 0,
            "stale_plans": 0}
        if resume:
            self._lock = RunLock(self.out / ".interactive.lock")
            cpath = self.out / "checkpoint.json"
            if not cpath.is_file():
                raise ContractError("No clean checkpoint manifest")
            cp = json.loads(cpath.read_text())
            meta = json.loads((self.out / "run.json").read_text())
            if meta.get("rom_sha256") and meta["rom_sha256"] != getattr(env, "rom_sha256", None):
                raise ContractError("ROM digest differs from original run")
            if meta.get("environment"):
                current_env = getattr(env, "descriptor", None) or {}
                if any(meta["environment"].get(k) != current_env.get(k) for k in ("adapter", "version", "window")):
                    raise ContractError("Emulator version or window differs from original run")
            if meta["status"] not in ("step_limit", "wall_limit", "budget_limit", "goal_claimed", "blocked"):
                raise ContractError("Run did not stop cleanly; automatic crash recovery is not implemented")
            if cp["config_fingerprint"] != immutable_config(config):
                raise ContractError("Resume permits only explicit total-budget/step/wall cap changes")
            if hashlib.sha256((self.out / "environment.state").read_bytes()).hexdigest() != cp["state_sha256"]:
                raise ContractError("Checkpoint bytes changed")
            env.frame_number = cp["environment_frame"]
            if hashlib.sha256(png(env.capture().image)).hexdigest() != cp["screen_sha256"]:
                raise ContractError("Loaded emulator state does not match checkpoint screenshot")
            self.completed_steps = cp["completed_steps"]
            self.intent, self.focus = cp["intent"], cp["focus"]
            self.previous_wall = meta["elapsed_wall_s"]
            self.metadata = meta
            self.metadata.setdefault("resume_configurations", []).append(config.model_dump())
        else:
            if self.out.exists():
                raise ContractError("Output directory already exists; use a fresh path or an explicit clean resume")
            self.out.mkdir(parents=True)
            self._lock = RunLock(self.out / ".interactive.lock")
            self.metadata = {"format_version": 1, "implementation": "streambudget.interactive",
                "run_id": self.epoch, "backend": config.backend, "config": config.model_dump(),
                "initialization": getattr(env, "initialization", "unknown"),
                "initial_state_sha256": getattr(env, "initial_state_sha256", None),
                "rom_sha256": getattr(env, "rom_sha256", None),
                "environment": getattr(env, "descriptor", None),
                "fixture": config.backend == "fixture" or bool(getattr(env, "synthetic", False)), "status": "running",
                "game_success": None, "timing": "stepped; emulator frozen while model calls run",
                "source_time": "emulator_frame/60 nominal seconds; exact frame numbers also retained",
                "resume_configurations": []}
        self.store = EvidenceStore(self.out / "memory.sqlite", config.source)
        self.media = MediaStore(self.out / "media")
        self.world = World(self.store, max_entities=config.max_entities,
            fact_threshold=config.min_fact_confidence, association_threshold=config.min_association_confidence)
        self.ledger = Ledger(self.store.db, config.max_calls, config.max_estimated_usd)
        self.store.db.executescript("""
          CREATE TABLE IF NOT EXISTS game_actions(id TEXT PRIMARY KEY, epoch TEXT, source_frame TEXT,
            action TEXT, status TEXT, receipt TEXT);
          CREATE TABLE IF NOT EXISTS world_visuals(entity TEXT, evidence TEXT, root_frame TEXT);
        """)
        self.store.db.commit()
        if resume:
            pending = self.store.db.execute("SELECT COUNT(*) FROM model_calls WHERE status='pending'").fetchone()[0]
            pending += self.store.db.execute("SELECT COUNT(*) FROM game_actions WHERE status!='completed'").fetchone()[0]
            if pending:
                self.store.close()
                raise ContractError("Unresolved inference/action attempts prevent resume")
            if self.store.db.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0] != cp["ledger_calls"]:
                raise ContractError("Ledger changed after checkpoint; cannot resume automatically")
            if self.store.db.execute("SELECT COUNT(*) FROM game_actions").fetchone()[0] != cp["action_attempts"]:
                raise ContractError("Actions changed after checkpoint; cannot resume automatically")
            if cp.get("attempts_sha256") != attempt_digest(self.store.db):
                raise ContractError("Attempt records changed or checkpoint lacks their digest; review required")
            if self.world.ontology.current["version"] != cp["ontology_version"]:
                raise ContractError("Schema changed after checkpoint; create a reviewed continuation")
            self.current = self.store.get(cp["current_frame_id"])
            if (self.current.payload.get("frame_number") != cp["environment_frame"] or
                    self.current.payload.get("sha256") != cp["screen_sha256"]):
                raise ContractError("Checkpoint does not identify the retained current screen")
            self.last_frames = self.store.frames(config.source, 0, self.current.end,
                self.store.snapshot(self.current.end), config.history_frames)
        else:
            self.current = self.capture()
        self.backend = backend or (FixtureBackend() if config.backend == "fixture" else
            ChatBackend(config, self.ledger, allow_network=allow_network, allow_paid=allow_paid))
        self.ocr_reader = ocr_reader or (RapidTextReader(config.ocr_config_path, scale=config.ocr_scale)
                                       if config.ocr_config_path else None)
        if resume and self.metadata.get("ocr_model") != getattr(self.ocr_reader, "model_id", None):
            raise ContractError("OCR model/config changed since checkpoint")
        if config.async_perception:
            # One background Qwen request at a time, plus the actor. No unbounded queue.
            self.background = ThreadPoolExecutor(max_workers=1, thread_name_prefix="semantic")
            if self.ocr_reader:
                self.ocr_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
            self.metadata["timing"] = "stepped actions; OCR, semantics and planning run in bounded background workers"
        self.metadata["status"] = "running"
        write_json(self.out / "run.json", self.metadata)

    def log(self, kind, **payload):
        with (self.out / "trace.jsonl").open("a", encoding="utf-8") as f:
            f.write(canonical({"kind": kind, "wall_s": self.previous_wall + time.monotonic() - self.start_wall,
                "invocation_wall_s": time.monotonic() - self.start_wall, "epoch": self.epoch,
                "step": self.completed_steps, **payload}) + "\n")

    def capture(self):
        screen = self.env.capture()
        raw = png(screen.image)
        key, _ = self.media.put(raw)
        t = screen.frame_number / 60
        record = self.store.add_raw(source=self.config.source, kind="frame", start=t, end=t,
            available_at=t, payload={"media_key": key, "frame_number": screen.frame_number,
            "sha256": hashlib.sha256(raw).hexdigest(), "width": screen.image.width,
            "height": screen.image.height})
        self.last_frames.append(record)
        self.last_frames = self.last_frames[-self.config.history_frames:]
        return record

    def image(self, evidence, historical=False):
        key = evidence.payload.get("media_key")
        if not key:
            raise ContractError("Evidence is not a retained image")
        path = (self.media.root / key).resolve()
        if path.parent != self.media.root:
            raise ContractError("Invalid retained media key")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != key.split(".")[0]:
            raise ContractError("Retained image content hash mismatch")
        return ImageInput(evidence.id, inference_png(MediaStore.decode(raw), self.config.view_scale), historical)

    def packet(self, extra=None):
        snap = self.store.snapshot(self.current.end)
        world = self.world.view(snap, focus=self.focus, max_entities=self.config.max_entities_in_context)
        recent = list(reversed(self.store.list(snap, source=self.config.source,
            kinds=["world_observation", "action_receipt"], limit=8)))
        if self.config.baseline == "recent":
            world = {"entities": [], "relations": [], "current_place": None, "recent_conversations": []}
        packet = bounded_context(goal=self.config.goal, intent=self.intent,
            schema=self.world.ontology.current, actions=[a.model_dump() for a in self.config.actions],
            world=world, recent=[e.view() for e in recent], current_frame_id=self.current.id,
            max_chars=self.config.max_context_chars, extra=extra, hot_text=(
                {**self.hot_text, "source_age_s": self.current.end - self.hot_text["observed_at"],
                 "source_matches_current": self.hot_text["source_sha256"] == self.current.payload["sha256"],
                 "tracks": [dict(t) for t in self.hot_text["tracks"]]} if self.hot_text else None))
        if self.config.async_perception:
            packet["background_planning"] = bool(self.plan_job)
            if len(canonical(packet)) > self.config.max_context_chars:
                raise ContractError("Background status exceeds context cap")
        return snap, packet

    def _submit_model(self, role, system, context, images, output):
        submit = getattr(self.backend, "submit", None)
        if submit:
            return submit(self.background, role, system, context, images, output)
        if self.config.backend != "fixture":
            raise ContractError("Asynchronous backend must support owner-thread ledger admission")
        return self.background.submit(self.backend.complete, role, system, context, images, output)

    def _perception_inputs(self):
        snap, packet = self.packet()
        frames = list(self.last_frames)
        parents = [f.id for f in frames] + [e["id"] for e in packet["recent_evidence"]]
        for ent in packet["world"].get("entities", []):
            parents.append(ent["evidence_id"])
            parents.extend(f["evidence_id"] for f in ent["facts"].values())
        parents.extend(r["evidence"] for r in packet["world"].get("relations", []))
        parents.extend(r["evidence"] for r in packet["world"].get("recent_conversations", []))
        if packet.get("hot_text"):
            parents.append(packet["hot_text"]["evidence_id"])
        return dict(snap=snap, packet=packet, frame=self.current, frames=frames, parents=parents)

    def _apply_semantics(self, patch, job):
        if patch.frame_id != job["frame"].id:
            raise ContractError("Extractor must describe its supplied CURRENT frame")
        applied = self.world.apply(patch, job["snap"], offered_frames={f.id for f in job["frames"]},
            offered_entities={e["id"] for e in job["packet"]["world"].get("entities", [])},
            consumed_evidence=job["parents"])
        if applied.changed:
            self.save_regions(patch, applied, job["snap"])
        self.needs_plan |= applied.needs_planning
        self.log("semantic_completed", source_frame=job["frame"].id,
                 current_frame=self.current.id, historical=job["frame"].id != self.current.id,
                 evidence_id=applied.evidence_id)

    def _schedule_background(self):
        if self.ocr_reader and not self.ocr_job and self.current.id != self.ocr_previous_frame:
            image = MediaStore.decode(self.image_raw(self.current))
            now = time.monotonic()
            pixel_changed = self.hot_text is None or self.hot_text["source_sha256"] != self.current.payload["sha256"]
            if pixel_changed or self.text_cache.needs_read("screen", image, now):
                ep = Endpoint(model=self.ocr_reader.model_id, billing="local")
                call = self.ledger.reserve("ocr", ep, self.current.payload["sha256"], 1, 0)
                with self.store.db:
                    self.store.db.execute("INSERT INTO model_inputs VALUES(?,?,?,?)", (call, "{}",
                        "local OCR; text is untrusted evidence", canonical([{"id": self.current.id,
                         "sha256": self.current.payload["sha256"]}])))
                try:
                    future = self.ocr_worker.submit(self._ocr_read, image.copy(), self.current.id)
                except Exception as exc:
                    self.ledger.finish(call, endpoint=ep, error=type(exc).__name__)
                    raise
                self.ocr_job = dict(frame=self.current, image=image, start=now, call=call, endpoint=ep, future=future)
                self.async_stats["ocr_submissions"] += 1
                self.log("ocr_submitted", source_frame=self.current.id, call_id=call)
            else:
                self.async_stats["ocr_frame_cache_hits"] += 1
                self.log("ocr_frame_reused", source_frame=self.current.id,
                         prior_frame=self.ocr_previous_frame)
        elif self.ocr_job:
            self.async_stats["ocr_coalesced_frames"] += 1
        if self.semantic_job or self.plan_job:
            self.async_stats["semantic_coalesced_frames"] += 1
            return
        plan_due = self.needs_plan or self.completed_steps - self.last_plan_step >= self.config.plan_every
        extract_due = (self.semantic_due or self.completed_steps - self.last_semantic_step >= self.config.semantic_refresh_steps
                       ) and self.semantic_hash != self.current.payload["sha256"]
        if extract_due and self.store.db.execute("SELECT 1 FROM world_applied WHERE frame_id=?", (self.current.id,)).fetchone():
            extract_due = False
            self.semantic_hash = self.current.payload["sha256"]
            self.last_semantic_step = self.completed_steps
        if plan_due and (self.last_background_role != "plan" or not extract_due):
            snap, packet = self.packet()
            self.plan_job = dict(snap=snap, packet=packet, frame=self.current, intent=self.intent,
                                 round=0, step=self.completed_steps)
            self.plan_job["pending"] = self._submit_model("plan", prompts.PLAN, packet,
                                                         [self.image(self.current)], Plan)
            self.log("plan_submitted", source_frame=self.current.id)
            self.last_background_role = "plan"
            return
        if extract_due:
            job = self._perception_inputs()
            job["pending"] = self._submit_model("extract", prompts.EXTRACT, job["packet"],
                [self.image(f, f.id != self.current.id) for f in job["frames"]], ObservationPatch)
            self.semantic_job = job
            self.last_background_role = "extract"
            self.semantic_hash = self.current.payload["sha256"]
            self.last_semantic_step, self.semantic_due = self.completed_steps, False
            self.async_stats["semantic_submissions"] += 1
            self.log("semantic_submitted", source_frame=self.current.id)

    def image_raw(self, evidence):
        # image() validates the same immutable source before any worker sees its pixels.
        self.image(evidence)
        return (self.media.root / evidence.payload["media_key"]).read_bytes()

    def _ocr_read(self, image, frame_id):
        start = time.monotonic()
        observations = self.ocr_reader.read(image, frame_id)
        return observations, time.monotonic() - start

    def _finish_ocr(self, job, *, apply=True):
        try:
            observations, elapsed = job["future"].result()
            if any(o.evidence_id != job["frame"].id for o in observations):
                raise ContractError("OCR result does not identify its actual source frame")
            response = {"observations": [dataclasses.asdict(o) for o in observations]}
        except Exception as exc:
            self.ledger.finish(job["call"], endpoint=job["endpoint"], elapsed=time.monotonic()-job["start"],
                               error=type(exc).__name__)
            raise
        self.ledger.finish(job["call"], endpoint=job["endpoint"], response=response,
                           elapsed=elapsed)
        if not apply:
            return
        frame = job["frame"]
        scroll = {}
        if self.ocr_previous_image is not None:
            dy, residual = estimate_vertical_scroll(self.ocr_previous_image, job["image"])
            if dy is not None and residual < 2:
                scroll = {"screen": (0, dy/job["image"].height)}
        events = self.text_tracker.update(observations, frame.end, scroll=scroll)
        record = self.store.derive(source=frame.source, kind="ocr_observation",
            text="\n".join(o.text for o in observations), parents=[frame.id],
            snapshot=self.store.snapshot(self.current.end), payload={**response, "events": events,
                "observed_frame_id": frame.id, "observed_at": frame.end,
                "source_sha256": frame.payload["sha256"], "model": self.ocr_reader.model_id,
                "completed_wall": time.time(), "epistemic_status": "OCR_not_ground_truth"})
        self.hot_text = {"evidence_id": record.id, "source_frame": frame.id, "observed_at": frame.end,
            "source_sha256": frame.payload["sha256"], "tracks": [dict(id=t.id, text=t.text, box=t.box,
                confidence=t.confidence, last_seen=t.last_seen) for t in self.text_tracker.tracks.values()
                if t.last_seen == frame.end],
            "warning": "OCR is untrusted; tracks are heuristic, speaker and semantic identity remain unknown"}
        self.ocr_previous_frame, self.ocr_previous_image = frame.id, job["image"]
        self.text_cache.record("screen", job["image"], job["start"])
        self.semantic_due |= bool(events)
        self.log("ocr_completed", source_frame=frame.id, evidence_id=record.id, events=events,
                 elapsed_s=elapsed, delivery_delay_s=max(0, time.monotonic()-job["start"]-elapsed))

    def _finish_plan(self, job, plan):
        context, snap = job["packet"], job["snap"]
        offered = {e["id"] for e in context["world"].get("entities", [])}
        offered |= {e["id"] for r in context.get("retrieved", []) for e in r.get("entities", [])}
        if set(plan.focus_ids) - offered:
            raise ContractError("Planner focus contains unoffered entities")
        applicable = (job["frame"].payload["sha256"] == self.current.payload["sha256"]
                      and job["intent"] == self.intent
                      and self.completed_steps - job["step"] <= self.config.plan_every)
        if not applicable:
            self.async_stats["stale_plans"] += 1
            self.needs_plan = True
            self.log("plan_discarded", source_frame=job["frame"].id, reason="source_or_intent_changed",
                     result=plan.model_dump())
            return
        if (plan.search or plan.inspect_ids) and job["round"] < self.config.max_retrieval_rounds:
            results = self.store.search(plan.search, snap, source=self.config.source, limit=6) if plan.search else []
            extra = [self.world.retrieval_record(e, snap) for e in results]
            visible_ids = {job["frame"].id} | offered
            visible_ids |= {e["id"] for e in context["recent_evidence"] + context.get("retrieved", [])}
            visible_ids |= {p for e in context["recent_evidence"] + context.get("retrieved", []) for p in e.get("parents", [])}
            visible_ids |= {r["evidence"] for r in context["world"].get("relations", [])}
            visible_ids |= {e["evidence_id"] for e in context["world"].get("entities", [])}
            visible_ids |= {f["evidence_id"] for e in context["world"].get("entities", []) for f in e["facts"].values()}
            anchors = []
            for id in plan.inspect_ids:
                if id not in visible_ids:
                    raise ContractError("Inspection target was not exposed to the planner")
                e = self.resolve_image(id, snap, offered)
                if e:
                    anchors.append(self.image(e, historical=True))
            # Keep the original source-bound packet/snapshot throughout retrieval.
            packet = bounded_context(goal=context["goal"], intent=context["intent"], schema=context["ontology"],
                actions=context["actions"], world=context["world"], recent=context["recent_evidence"],
                current_frame_id=job["frame"].id, max_chars=self.config.max_context_chars, extra=extra,
                hot_text=context.get("hot_text"))
            job.update(packet=packet, round=job["round"]+1)
            job["pending"] = self._submit_model("plan", prompts.PLAN, packet,
                ([self.image(job["frame"])] + anchors)[:self.config.max_images], Plan)
            self.plan_job = job
            self.log("retrieval", round=job["round"], results=[r["id"] for r in extra], anchors=[a.id for a in anchors])
            return
        self.intent, self.focus = plan.intent, plan.focus_ids
        self.last_plan_step, self.needs_plan = self.completed_steps, False
        self.log("plan", source_frame=job["frame"].id, result=plan.model_dump())
        # A late model claim cannot terminate a newer episode.
        if job["frame"].id == self.current.id:
            if plan.status == "goal_claimed" and self.config.stop_on_goal_claim:
                self.status = "goal_claimed"
            elif plan.status == "blocked":
                self.status = "blocked"

    def _poll_background(self, *, wait=False, apply=True):
        errors = []
        for attr in ("ocr_job", "semantic_job", "plan_job"):
            job = getattr(self, attr)
            if not job:
                continue
            pending = job.get("pending", job.get("future"))
            future = getattr(pending, "future", pending)
            if not wait and not future.done():
                continue
            setattr(self, attr, None)
            try:
                if attr == "ocr_job":
                    self._finish_ocr(job, apply=apply)
                else:
                    value = pending.result()
                    if apply and attr == "semantic_job":
                        self._apply_semantics(value, job)
                    elif apply and attr == "plan_job" and not wait:
                        self._finish_plan(job, value)
                    elif attr == "plan_job":
                        self.log("plan_retained_at_stop", source_frame=job["frame"].id,
                                 result=value.model_dump(), note="not activated during shutdown")
            except Exception as exc:
                errors.append(exc)
                self.log("background_failure", role=attr, source_frame=job["frame"].id,
                         error_type=type(exc).__name__)
        if errors:
            raise errors[0]

    def save_regions(self, patch, applied, snap):
        for mention in patch.mentions:
            if not mention.region:
                continue
            region = mention.region
            frame = self.store.get(region.frame_id, snap)
            key = frame.payload["media_key"]
            path = (self.media.root / key).resolve()
            if path.parent != self.media.root:
                raise ContractError("Crop source escapes evidence store")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != key.split(".")[0]:
                raise ContractError("Crop source hash mismatch")
            im = MediaStore.decode(raw)
            x1, y1, x2, y2 = region.box
            bounds = (int(x1*im.width), int(y1*im.height), max(int(x2*im.width), int(x1*im.width)+1),
                      max(int(y2*im.height), int(y1*im.height)+1))
            key, _ = self.media.put(png(im.crop(bounds)))
            crop = self.store.derive(source=frame.source, kind="crop", text="Optional visual anchor",
                parents=[frame.id, applied.evidence_id], snapshot=self.store.snapshot(self.current.end),
                payload={"media_key": key, "box": region.box, "root_frame": frame.id,
                         "entity": applied.ids[mention.ref]})
            with self.store.db:
                self.store.db.execute("INSERT INTO world_visuals VALUES(?,?,?)",
                                      (applied.ids[mention.ref], crop.id, frame.id))

    def resolve_image(self, id, snap, offered_entities):
        if id.startswith("e_"):
            if id not in offered_entities:
                raise ContractError("Inspect request entity was not in context")
            for row in self.store.db.execute("SELECT evidence FROM world_visuals WHERE entity=? ORDER BY rowid DESC", (id,)):
                e = self.store.get(row[0])
                if e.visible(snap):
                    return e
            # Crops are optional. An entity can be re-inspected in its full source frame.
            mentions = self.store.db.execute(
                "SELECT evidence FROM world_mentions WHERE entity=? ORDER BY seq DESC", (id,)).fetchall()
            id = next((r[0] for r in mentions if self.store.get(r[0]).visible(snap)), None)
            if id is None:
                return None
        e = self.store.get(id, snap)
        anchor = e.payload.get("interpretation", {}).get("frame_id")
        if anchor:
            source_frame = self.store.get(anchor, snap)
            if source_frame.payload.get("media_key"):
                return source_frame
        if e.payload.get("media_key"):
            return e
        # Walk source lineage, not arbitrary filesystem paths or model-supplied URLs.
        stack, seen = list(e.parents), set()
        while stack and len(seen) < 200:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            x = self.store.get(p, snap)
            if x.payload.get("media_key"):
                return x
            stack.extend(x.parents)
        return None

    def plan(self):
        extra, anchors = [], []
        plan = None
        for round_no in range(self.config.max_retrieval_rounds + 1):
            snap, context = self.packet(extra)
            offered = {e["id"] for e in context["world"].get("entities", [])}
            offered |= {e["id"] for r in context.get("retrieved", []) for e in r.get("entities", [])}
            images = [self.image(self.current)] + anchors
            plan = self.backend.complete("plan", prompts.PLAN, context,
                                         images[:self.config.max_images], Plan)
            if set(plan.focus_ids) - offered:
                raise ContractError("Planner focus contains unoffered entities")
            if (not plan.search and not plan.inspect_ids) or round_no == self.config.max_retrieval_rounds:
                break
            results = self.store.search(plan.search, snap, source=self.config.source, limit=6) if plan.search else []
            extra = [self.world.retrieval_record(e, snap) for e in results]
            # Explicitly restrict inspect to evidence IDs already shown to the planner.
            visible_ids = {self.current.id} | {e["id"] for e in context["recent_evidence"]}
            visible_ids |= {e["id"] for e in context.get("retrieved", [])}
            visible_ids |= {p for e in [*context["recent_evidence"], *context.get("retrieved", [])]
                            for p in e.get("parents", [])}
            visible_ids |= offered
            visible_ids |= {r["evidence"] for r in context["world"].get("relations", [])}
            visible_ids |= {e["evidence_id"] for e in context["world"].get("entities", [])}
            for r in context["world"].get("entities", []):
                visible_ids |= {v["evidence_id"] for v in r["facts"].values()}
            anchors = []
            for id in plan.inspect_ids:
                if id not in visible_ids:
                    raise ContractError("Inspection target was not exposed to the planner")
                e = self.resolve_image(id, snap, offered)
                if e:
                    anchors.append(self.image(e, historical=True))
            self.log("retrieval", round=round_no, results=[r["id"] for r in extra],
                     anchors=[a.id for a in anchors])
        self.intent, self.focus = plan.intent, plan.focus_ids
        self.last_plan_step = self.completed_steps
        self.needs_plan = False
        self.log("plan", result=plan.model_dump())
        if plan.status == "goal_claimed" and self.config.stop_on_goal_claim:
            self.status = "goal_claimed"
        elif plan.status == "blocked":
            self.status = "blocked"

    def step(self):
        decision_start = time.monotonic()
        if self.config.async_perception:
            self._poll_background()
            if self.status != "running":
                return
            self._schedule_background()
        else:
            job = self._perception_inputs()
            already = self.store.db.execute("SELECT evidence_id FROM world_applied WHERE frame_id=?",
                                            (self.current.id,)).fetchone()
            if already:
                patch = ObservationPatch.model_validate(self.store.get(already[0]).payload["interpretation"])
            else:
                patch = self.backend.complete("extract", prompts.EXTRACT, job["packet"],
                    [self.image(f, f.id != self.current.id) for f in job["frames"]], ObservationPatch)
            self._apply_semantics(patch, job)
            if self.needs_plan or self.completed_steps - self.last_plan_step >= self.config.plan_every:
                self.plan()
        if self.status != "running":
            return
        _, context = self.packet()
        actor_start = time.monotonic()
        choice = self.backend.complete("act", prompts.ACT, context, [self.image(self.current)], ActionChoice)
        actor_elapsed = time.monotonic() - actor_start
        action = next((a for a in self.config.actions if a.id == choice.action_id), None)
        if action is None:
            raise ContractError("Action not in the operator-provided capability manifest")
        if time.monotonic() - self.start_wall > self.config.wall_limit_s:
            self.status = "wall_limit"
            return
        fresh = self.env.capture()
        if fresh.frame_number != self.current.payload["frame_number"] or \
           hashlib.sha256(png(fresh.image)).hexdigest() != self.current.payload["sha256"]:
            raise ContractError("World changed while deciding; refusing a stale action")
        action_id = "execution_" + uuid.uuid4().hex
        decision_elapsed = time.monotonic() - decision_start
        with self.store.db:
            self.store.db.execute("INSERT INTO game_actions VALUES(?,?,?,?,?,NULL)",
                (action_id, self.epoch, self.current.id, canonical(action.model_dump()), "dispatched"))
        receipt = self.env.execute(action)  # One attempt. Unknown failure is never retried.
        with self.store.db:
            self.store.db.execute("UPDATE game_actions SET status='completed',receipt=? WHERE id=?",
                                  (canonical(dataclasses.asdict(receipt)), action_id))
        self.store.add_raw(source=self.config.source, kind="action_receipt",
            start=receipt.start_frame/60, end=receipt.end_frame/60, available_at=receipt.end_frame/60,
            text=f"Executed {action.id}; physical/game outcome not yet verified",
            payload={"execution_id": action_id, **dataclasses.asdict(receipt)})
        before = self.current.id
        self.current = self.capture()
        self.completed_steps += 1
        self.log("action", execution_id=action_id, before=before, after=self.current.id,
                 action=action.model_dump(), receipt=dataclasses.asdict(receipt), intent=self.intent,
                 decision_wall_s=decision_elapsed, actor_request_wall_s=actor_elapsed)

    def run(self):
        try:
            if (self.config.compile_at_start and self.completed_steps == 0
                    and not self.metadata.get("initial_compilation_complete")):
                _, context = self.packet()
                try:
                    proposal = self.backend.complete("compile", prompts.COMPILE, context,
                                                      [self.image(self.current)], SchemaPatch)
                except BudgetExhausted:
                    self.status = "budget_limit"
                else:
                    if proposal.properties or proposal.relations:
                        id = self.world.ontology.propose(proposal)
                        self.world.ontology.approve(id, initial_prior=True,
                            review="Operator opted into one initial schema compilation; definitions are priors, not learned facts")
                        self.log("initial_schema", candidate=id)
                    self.metadata["initial_compilation_complete"] = True
            while self.completed_steps < self.config.max_steps and self.status == "running":
                if time.monotonic() - self.start_wall > self.config.wall_limit_s:
                    self.status = "wall_limit"
                    break
                size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
                if size > self.config.max_storage_mb * 1024**2:
                    raise ContractError("Storage budget exceeded; no automatic evidence deletion")
                try:
                    self.step()
                except BudgetExhausted:
                    self.status = "budget_limit"
                    break
            if self.status == "running":
                self.status = "step_limit"
            # Account for all dispatched work before creating a resumable checkpoint.
            # Draining is shutdown overhead, never reported as action-path latency.
            drain_start = time.monotonic()
            self._poll_background(wait=True)
            self.metadata["background_drain_s"] = time.monotonic() - drain_start
            screen = self.env.capture()
            if (screen.frame_number != self.current.payload["frame_number"] or
                    hashlib.sha256(png(screen.image)).hexdigest() != self.current.payload["sha256"]):
                raise ContractError("World changed before checkpoint; refusing a mismatched clean stop")
            self.env.checkpoint(self.out / "environment.state")
            write_json(self.out / "checkpoint.json", {
                "config_fingerprint": immutable_config(self.config), "completed_steps": self.completed_steps,
                "current_frame_id": self.current.id, "environment_frame": screen.frame_number,
                "screen_sha256": hashlib.sha256(png(screen.image)).hexdigest(),
                "state_sha256": hashlib.sha256((self.out / "environment.state").read_bytes()).hexdigest(),
                "intent": self.intent, "focus": self.focus,
                "ledger_calls": self.store.db.execute("SELECT COUNT(*) FROM model_calls").fetchone()[0],
                "action_attempts": self.store.db.execute("SELECT COUNT(*) FROM game_actions").fetchone()[0],
                "attempts_sha256": attempt_digest(self.store.db),
                "ontology_version": self.world.ontology.current["version"],
                "note": "Only clean checkpoints may be resumed; no crash-time replay of actions."})
        except BaseException as exc:
            self.status = "failed"
            self.log("failure", error_type=type(exc).__name__)
            raise
        finally:
            propagating_error = sys.exc_info()[0] is not None
            cleanup_error = None
            try:
                try:
                    self._poll_background(wait=True, apply=False)
                except Exception as exc:
                    cleanup_error = exc
                    self.status = "failed"
                finally:
                    for worker in (self.background, self.ocr_worker):
                        if worker:
                            worker.shutdown(wait=True)
                self.metadata.update(status=self.status, completed_steps=self.completed_steps,
                    elapsed_wall_s=self.previous_wall + time.monotonic()-self.start_wall,
                    current_frame_id=self.current.id, accounting=self.ledger.summary(),
                    ontology=self.world.ontology.current,
                    fixture_model_calls=getattr(self.backend, "calls", None),
                    async_perception=self.async_stats,
                    ocr_model=getattr(self.ocr_reader, "model_id", None),
                    ocr_stats=getattr(self.ocr_reader, "stats", None),
                    final_world=self.world.view(self.store.snapshot(self.current.end)))
                write_json(self.out / "run.json", self.metadata)
                if cleanup_error and not propagating_error:
                    raise cleanup_error
            finally:
                try:
                    self.env.close()
                finally:
                    try:
                        close = getattr(self.backend, "close", None)
                        if close:
                            close()
                    finally:
                        try:
                            self.store.close()
                        finally:
                            self._lock.close()
        return self.metadata
