import asyncio
import pytest
from streambudget.scheduler import Scheduler, QueueRejected
from streambudget.config import SchedulerConfig
from streambudget.trace import Trace


async def test_exact_coalescing_and_one_execution(tmp_path):
    s = Scheduler(SchedulerConfig(workers=1), Trace(tmp_path / "trace"))
    s.start()
    calls = []
    async def work():
        calls.append(1)
        await asyncio.sleep(.01)
        return 7
    a = s.submit("same", work)
    b = s.submit("same", work)
    assert a is b
    assert await a == 7
    await s.drain()
    assert calls == [1]
    await s.close()


async def test_bounded_queue_no_hidden_growth(tmp_path):
    s = Scheduler(SchedulerConfig(workers=1, max_queue=1), Trace(tmp_path / "trace"))
    async def work():
        return True
    a = s.submit("a", work)
    with pytest.raises(QueueRejected):
        s.submit("b", work)
    s.start()
    assert await a
    assert s.max_depth == 1
    await s.close()


async def test_priority_order(tmp_path):
    s = Scheduler(SchedulerConfig(workers=1), Trace(tmp_path / "trace"))
    order = []
    async def append(x):
        order.append(x)
    s.submit("background", lambda: append("background"), priority=30)
    s.submit("alert", lambda: append("alert"), priority=0)
    s.start()
    await s.drain()
    assert order == ["alert", "background"]
    await s.close()


async def test_expired_job_is_not_executed(tmp_path):
    s = Scheduler(SchedulerConfig(workers=1), Trace(tmp_path / "trace"))
    calls = []
    async def work():
        calls.append(1)
    f = s.submit("expire", work, deadline_s=.005)
    await asyncio.sleep(.02)
    s.start()
    with pytest.raises(TimeoutError):
        await f
    assert not calls
    await s.close()


async def test_worker_recovers_from_exception(tmp_path):
    s = Scheduler(SchedulerConfig(workers=1), Trace(tmp_path / "trace"))
    s.start()
    async def fail():
        raise ValueError("expected")
    async def good():
        return 42
    with pytest.raises(ValueError):
        await s.submit("bad", fail)
    assert await s.submit("good", good) == 42
    await s.close()


async def test_concurrency_limit(tmp_path):
    s = Scheduler(SchedulerConfig(workers=2), Trace(tmp_path / "trace"))
    current, peak = 0, 0
    async def work():
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(.01)
        current -= 1
    s.start()
    jobs = [s.submit(str(i), work) for i in range(8)]
    await asyncio.gather(*jobs)
    assert peak == 2
    await s.close()
