"""Bounded cooperative request broker; one writer, transport-only async tasks.

A reserved *client dispatch* slot is NOT GPU partitioning or preemption. vLLM may
continuous-batch independent requests; speed/actor contention must be measured.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass

from ...types import ContractError
from ..models import Ledger, BudgetExhausted
from ..ontology import canonical
from .transport import RawCompletion, request_body, validate_completion


@dataclass
class Ticket:
    id: str
    role: str
    basis: object
    context: dict
    images: tuple
    output: type
    system: str
    key: str | None = None
    status: str = "queued"
    task: asyncio.Task | None = None
    call_id: str | None = None
    endpoint: object | None = None
    body: dict | None = None
    value: object | None = None
    error: str | None = None
    queued_at: float = 0
    started_at: float | None = None
    finished_at: float | None = None
    raw: RawCompletion | None = None


class JobBroker:
    def __init__(self, config, db, transport, *, clock=time.monotonic):
        self.config, self.db, self.transport, self.clock = config, db, transport, clock
        self.owner = threading.get_ident()
        self.ledger = Ledger(db, config.game.max_calls, config.game.max_estimated_usd)
        self.tickets = {}
        self.ready = []
        self.quarantined = False
        self.closed = False
        self.last_background_role = None
        db.executescript("""
          CREATE TABLE IF NOT EXISTS async_jobs(
            id TEXT PRIMARY KEY, role TEXT, status TEXT, basis TEXT, call_id TEXT,
            queued_at REAL, started_at REAL, finished_at REAL, first_token_s REAL,
            first_content_s REAL, response_wall_s REAL, error TEXT);
        """)
        db.commit()

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise ContractError("Broker methods must run on the writer thread")

    def submit(self, role, basis, system, context, images, output, *, key=None):
        self._owner()
        if self.closed or self.quarantined:
            raise ContractError("Broker closed/quarantined")
        if role not in ("act", "plan", "index", "enrich"):
            raise ContractError("Unknown worker role")
        # Freeze JSON and image list; workers cannot observe subsequent mutable context updates.
        context = json.loads(canonical(context))
        if len(images) > self.config.game.max_images:
            raise ContractError("Too many images")
        if key:
            for old in list(self.tickets.values()):
                if old.key == key and old.status == "queued":
                    self._terminal(old, "superseded")
        queued = sum(t.status == "queued" for t in self.tickets.values())
        if role == "act" and queued >= self.config.background.max_queue:
            victim = next(
                (
                    x
                    for x in reversed(list(self.tickets.values()))
                    if x.status == "queued" and x.role != "act"
                ),
                None,
            )
            if victim is not None:
                self._terminal(victim, "superseded_for_actor")
                queued -= 1
        ticket = Ticket(
            basis.job_id, role, basis, context, tuple(images), output, system, key=key, queued_at=self.clock()
        )
        if (
            ticket.id in self.tickets
            or self.db.execute("SELECT 1 FROM async_jobs WHERE id=?", (ticket.id,)).fetchone()
        ):
            raise ContractError("Job ID reuse")
        with self.db:
            self.db.execute(
                "INSERT INTO async_jobs(id,role,status,basis,queued_at) VALUES(?,?,?,?,?)",
                (ticket.id, role, "queued", canonical(asdict(basis)), ticket.queued_at),
            )
        self.tickets[ticket.id] = ticket
        if queued >= self.config.background.max_queue:
            self._terminal(ticket, "rejected_queue_full")
        return ticket

    def _terminal(self, t, status, *, error=None):
        t.status, t.error, t.finished_at = status, error, self.clock()
        with self.db:
            self.db.execute(
                "UPDATE async_jobs SET status=?,error=?,finished_at=? WHERE id=?",
                (status, error, t.finished_at, t.id),
            )
        self.ready.append(t)

    def _endpoint(self, role):
        alias = self.config.game.roles["extract" if role in ("index", "enrich") else role]
        return self.config.game.endpoints[alias]

    def _limit(self, role):
        settings = self.config.background
        return {
            "act": settings.actor_max_tokens,
            "plan": settings.plan_max_tokens,
            "index": settings.index_max_tokens,
            "enrich": settings.enrich_max_tokens,
        }[role]

    def _start(self, t):
        endpoint = self._endpoint(t.role)
        body = request_body(
            endpoint,
            role=t.role,
            system=t.system,
            context=t.context,
            images=t.images,
            output=t.output,
            actions=self.config.game.actions,
            output_limit=self._limit(t.role),
            stream=self.config.background.stream_responses,
        )
        t.endpoint, t.body = endpoint, body
        t.call_id = self.ledger.reserve(
            t.role,
            endpoint,
            hashlib.sha256(canonical(body).encode()).hexdigest(),
            len(t.images),
            len(canonical(t.context)),
        )
        with self.db:
            self.db.execute(
                "INSERT INTO model_inputs VALUES(?,?,?,?)",
                (
                    t.call_id,
                    canonical(t.context),
                    t.system,
                    canonical(
                        [
                            {
                                "id": i.id,
                                "historical": i.historical,
                                "sha256": hashlib.sha256(i.data).hexdigest(),
                            }
                            for i in t.images
                        ]
                    ),
                ),
            )
            self.db.execute(
                "UPDATE async_jobs SET status='running',call_id=?,started_at=? WHERE id=?",
                (t.call_id, self.clock(), t.id),
            )
        t.started_at, t.status = self.clock(), "running"
        # Transport has no reference to SQLite, memory, the emulator or this ticket.
        t.task = asyncio.create_task(self.transport.invoke(endpoint, body), name=t.id)

    def pump(self):
        """Finish completed requests, account on the writer, then admit bounded new work."""
        self._owner()
        for t in list(self.tickets.values()):
            if t.status != "running" or not t.task.done():
                continue
            try:
                raw = t.task.result()
            except BaseException as exc:
                raw = RawCompletion(
                    None, self.clock() - t.started_at, error=type(exc).__name__, termination_unknown=True
                )
            t.raw = raw
            if raw.termination_unknown:
                self.quarantined = True
            error = None
            try:
                t.value = validate_completion(
                    raw,
                    t.output,
                    expected_model=t.endpoint.model,
                    require_model=self.config.background.require_returned_model,
                )
                if t.role == "act" and t.value.action_id not in {a.id for a in self.config.game.actions}:
                    raise ContractError("Actor selected an unoffered action ID")
            except Exception as exc:
                error = type(exc).__name__ + (":" + raw.error if raw.error else "")
            self.ledger.finish(
                t.call_id, endpoint=t.endpoint, response=raw.response, elapsed=raw.elapsed_s, error=error
            )
            with self.db:
                self.db.execute(
                    "UPDATE async_jobs SET first_token_s=?,first_content_s=?,response_wall_s=? WHERE id=?",
                    (raw.first_token_s, raw.first_content_s, raw.elapsed_s, t.id),
                )
            self._terminal(t, "failed" if error else "completed", error=error)
        if self.quarantined:
            for t in list(self.tickets.values()):
                if t.status == "queued":
                    self._terminal(t, "cancelled_quarantine")
        elif not self.closed:
            priority = {"act": 0, "plan": 1, "index": 2, "enrich": 3}
            # Do not starve enrichments behind an endless sequence of fresh scene indices.
            if self.last_background_role in ("index", "plan"):
                priority.update(enrich=2, index=3)
            pending = sorted(
                (t for t in self.tickets.values() if t.status == "queued"),
                key=lambda t: (priority[t.role], t.queued_at, t.id),
            )
            for t in pending:
                active = [x for x in self.tickets.values() if x.status == "running"]
                if len(active) >= self.config.background.total_slots:
                    break
                if (
                    t.role != "act"
                    and sum(x.role != "act" for x in active) >= self.config.background.background_slots
                ):
                    continue
                try:
                    self._start(t)
                    if t.role != "act":
                        self.last_background_role = t.role
                except BudgetExhausted as exc:
                    self._terminal(t, "rejected_budget", error=type(exc).__name__)
                except Exception as exc:
                    # If reservation occurred but transport never began, preserve a failed attempt.
                    if t.call_id:
                        self.ledger.finish(t.call_id, endpoint=t.endpoint, error=type(exc).__name__)
                    self._terminal(t, "failed", error=type(exc).__name__)
        out, self.ready = self.ready, []
        for t in out:
            self.tickets.pop(t.id, None)  # Tickets held by callers remain valid; runtime RAM stays bounded.
        return out

    async def wait_activity(self, timeout=0.02):
        tasks = [t.task for t in self.tickets.values() if t.status == "running"]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        else:
            await asyncio.sleep(min(timeout, 0.005))

    def pending_key(self, key):
        return any(t.key == key for t in self.tickets.values())

    def cancel_queued(self, reason="cancelled_stop"):
        self._owner()
        for t in list(self.tickets.values()):
            if t.status == "queued":
                self._terminal(t, reason)

    async def close(self, timeout, *, consume=None):
        self._owner()
        self.closed = True
        self.cancel_queued()
        deadline = self.clock() + timeout
        while self.tickets:
            done = self.pump()
            if consume:
                for t in done:
                    consume(t)
            if not self.tickets:
                break
            if self.clock() >= deadline:
                self.quarantined = True
                for t in self.tickets.values():
                    if t.task:
                        t.task.cancel()
                await asyncio.gather(
                    *(t.task for t in self.tickets.values() if t.task), return_exceptions=True
                )
                done = self.pump()
                if consume:
                    for t in done:
                        consume(t)
                break
            await self.wait_activity()
        await self.transport.close()
        return not self.quarantined

    def metrics(self):
        rows = [dict(r) for r in self.db.execute("SELECT * FROM async_jobs")]
        roles = {}
        for r in rows:
            role = roles.setdefault(
                r["role"],
                {
                    "offered": 0,
                    "completed": 0,
                    "failed_or_dropped": 0,
                    "response_wall_s": [],
                    "queue_s": [],
                    "first_content_s": [],
                },
            )
            role["offered"] += 1
            role["completed"] += r["status"] == "completed"
            role["failed_or_dropped"] += r["status"] not in ("completed", "running", "queued")
            if r["response_wall_s"] is not None:
                role["response_wall_s"].append(r["response_wall_s"])
            if r["started_at"] is not None:
                role["queue_s"].append(r["started_at"] - r["queued_at"])
            if r["first_content_s"] is not None:
                role["first_content_s"].append(r["first_content_s"])
        return {
            "roles": roles,
            "quarantined": self.quarantined,
            "ledger": self.ledger.summary(),
            "note": "Model wall-time sums overlap; not episode duration, GPU time or measured batch speedup.",
        }
