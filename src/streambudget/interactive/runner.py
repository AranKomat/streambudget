"""One-writer stepped closed loop. No simulator truth enters model context."""
from __future__ import annotations

import dataclasses
from contextlib import suppress
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time
import uuid

from PIL import Image

from ..media import MediaStore
from ..store import EvidenceStore
from ..types import ContractError
from . import prompts
from .context import bounded_context
from .contracts import ActionChoice, GameConfig, ObservationPatch, Plan, SchemaPatch
from .fixture import FixtureBackend
from .models import BudgetExhausted, ChatBackend, ImageInput, Ledger
from .ontology import canonical
from .locking import RunLock
from .world import World


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
                 allow_paid=False, backend=None, resume=False):
        self.config, self.env, self.out = config, env, out.resolve()
        self.start_wall = time.monotonic()
        self.completed_steps, self.last_plan_step = 0, -100000
        self.intent = "Observe the current situation and choose a useful next step."
        self.focus, self.last_frames = [], []
        self.needs_plan = True
        self.epoch = uuid.uuid4().hex
        self.status = "running"
        self.previous_wall = 0.0
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
            max_chars=self.config.max_context_chars, extra=extra)
        return snap, packet

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
        snap, packet = self.packet()
        images = [self.image(f, f.id != self.current.id) for f in self.last_frames]
        already = self.store.db.execute("SELECT evidence_id FROM world_applied WHERE frame_id=?",
                                        (self.current.id,)).fetchone()
        if already:
            patch = ObservationPatch.model_validate(self.store.get(already[0]).payload["interpretation"])
        else:
            patch = self.backend.complete("extract", prompts.EXTRACT, packet, images, ObservationPatch)
        if patch.frame_id != self.current.id:
            raise ContractError("Extractor must describe the CURRENT frame, not a historical anchor")
        parents = [f.id for f in self.last_frames] + [e["id"] for e in packet["recent_evidence"]]
        for ent in packet["world"].get("entities", []):
            parents.append(ent["evidence_id"])
            parents.extend(f["evidence_id"] for f in ent["facts"].values())
        parents.extend(r["evidence"] for r in packet["world"].get("relations", []))
        parents.extend(r["evidence"] for r in packet["world"].get("recent_conversations", []))
        applied = self.world.apply(patch, snap, offered_frames={f.id for f in self.last_frames},
            offered_entities={e["id"] for e in packet["world"].get("entities", [])}, consumed_evidence=parents)
        if applied.changed:
            self.save_regions(patch, applied, snap)
        self.needs_plan |= patch.needs_planning
        if self.needs_plan or self.completed_steps - self.last_plan_step >= self.config.plan_every:
            self.plan()
        if self.status != "running":
            return
        _, context = self.packet()
        choice = self.backend.complete("act", prompts.ACT, context, [self.image(self.current)], ActionChoice)
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
                 action=action.model_dump(), receipt=dataclasses.asdict(receipt), intent=self.intent)

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
            try:
                self.metadata.update(status=self.status, completed_steps=self.completed_steps,
                    elapsed_wall_s=self.previous_wall + time.monotonic()-self.start_wall,
                    current_frame_id=self.current.id, accounting=self.ledger.summary(),
                    ontology=self.world.ontology.current,
                    fixture_model_calls=getattr(self.backend, "calls", None),
                    final_world=self.world.view(self.store.snapshot(self.current.end)))
                write_json(self.out / "run.json", self.metadata)
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
