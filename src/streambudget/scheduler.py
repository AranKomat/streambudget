from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Any

from .config import SchedulerConfig
from .trace import Trace


class QueueRejected(RuntimeError):
    pass


@dataclass(order=True)
class Job:
    priority: int
    order: int
    key: str = field(compare=False)
    factory: Callable[[], Awaitable[Any]] = field(compare=False)
    future: asyncio.Future = field(compare=False)
    deadline: float = field(compare=False)
    enqueued: float = field(compare=False)


class Scheduler:
    """Bounded priority queue; exact in-flight keys coalesce. Never pauses ingestion."""

    def __init__(self, config: SchedulerConfig, trace: Trace):
        self.config, self.trace = config, trace
        self.queue: asyncio.PriorityQueue[Job] = asyncio.PriorityQueue(config.max_queue)
        self.pending: dict[str, asyncio.Future] = {}
        self.workers: list[asyncio.Task] = []
        self.sequence = itertools.count()
        self.closed = False
        self.max_depth = 0

    def start(self):
        if not self.workers:
            self.workers = [asyncio.create_task(self._worker()) for _ in range(self.config.workers)]

    def submit(self, key: str, factory: Callable[[], Awaitable[Any]], *, priority: int = 10,
               deadline_s: float | None = None) -> asyncio.Future:
        if self.closed:
            raise QueueRejected("Scheduler closed")
        if key in self.pending:
            self.trace.emit("job_coalesced", key=key)
            return self.pending[key]
        if self.queue.full():
            self.trace.emit("job_dropped", key=key, reason="queue_full", priority=priority)
            raise QueueRejected("Inference queue full")
        future = asyncio.get_running_loop().create_future()
        # Observe unawaited errors; callers still receive them when they await the future.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        now = time.monotonic()
        job = Job(priority, next(self.sequence), key, factory, future,
                  now + (deadline_s or self.config.deadline_s), now)
        self.pending[key] = future
        self.queue.put_nowait(job)
        self.max_depth = max(self.max_depth, self.queue.qsize())
        self.trace.emit("job_enqueued", key=key, priority=priority, depth=self.queue.qsize())
        return future

    async def _worker(self):
        while True:
            job = await self.queue.get()
            start = time.monotonic()
            try:
                if start > job.deadline:
                    self.trace.emit("job_dropped", key=job.key, reason="expired_in_queue")
                    raise TimeoutError("Job deadline expired in queue")
                # Execution is also deadline bounded. Cancelling a client request does not
                # prove cancellation of remote billing; ModelPool records that uncertainty.
                result = await asyncio.wait_for(job.factory(), timeout=job.deadline - start)
                if not job.future.done():
                    job.future.set_result(result)
                self.trace.emit("job_done", key=job.key, queue_s=start - job.enqueued,
                                service_s=time.monotonic() - start)
            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                raise
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
                self.trace.emit("job_failed", key=job.key, error_type=type(exc).__name__,
                                queue_s=start - job.enqueued)
            finally:
                self.pending.pop(job.key, None)
                self.queue.task_done()

    async def drain(self):
        await asyncio.wait_for(self.queue.join(), self.config.drain_timeout_s)

    async def close(self):
        self.closed = True
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        while not self.queue.empty():
            job = self.queue.get_nowait()
            if not job.future.done():
                job.future.cancel()
            self.queue.task_done()
        self.pending.clear()
