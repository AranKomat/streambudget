import pytest
import numpy as np

from streambudget.store import EvidenceStore
from streambudget.types import ContractError, NotAvailable, Snapshot, valid_time


@pytest.fixture
def store(tmp_path):
    s = EvidenceStore(tmp_path / "memory.sqlite")
    yield s
    s.close()


def raw(store, ts=1, text="old event", available=None, **kwargs):
    return store.add_raw(source="cam", kind="sensor", start=ts, end=ts,
                         available_at=ts if available is None else available, text=text, **kwargs)


@pytest.mark.parametrize("x", [-1, float("nan"), float("inf"), True])
def test_bad_times(x):
    with pytest.raises(ContractError):
        valid_time(x)


def test_late_arrival_does_not_rewrite_frozen_snapshot(store):
    e = raw(store)
    snap = store.snapshot(10)
    late = raw(store, 2, "late report", available=3)
    assert store.get(e.id, snap) == e
    with pytest.raises(NotAvailable):
        store.get(late.id, snap)


def test_availability_time_enforced_even_when_capture_is_old(store):
    e = raw(store, 1, available=50)
    with pytest.raises(NotAvailable):
        store.get(e.id, store.snapshot(10))


def test_future_summary_lineage_cannot_be_backdated(store):
    old, future = raw(store), raw(store, 100, "future event")
    summary = store.derive(source="cam", kind="summary", text="old-looking summary",
                           parents=[old.id, future.id], snapshot=store.snapshot(100))
    assert summary.input_end == 100
    with pytest.raises(NotAvailable):
        store.get(summary.id, store.snapshot(10))


def test_historical_materialization_is_allowed_from_old_inputs(store):
    e = raw(store)
    snap = store.snapshot(1)
    raw(store, 100, "future")
    d = store.derive(source="cam", kind="summary", text="derived now from old evidence",
                     parents=[e.id], snapshot=snap)
    assert d.seq > snap.max_source_seq
    assert store.get(d.id, snap).text.startswith("derived")


def test_future_parent_is_refused_before_derivation(store):
    old = raw(store)
    snap = store.snapshot(1)
    future = raw(store, 2)
    with pytest.raises(NotAvailable):
        store.derive(source="cam", kind="summary", text="x", parents=[old.id, future.id], snapshot=snap)


def test_parentless_derived_data_refused(store):
    with pytest.raises(ContractError):
        store.derive(source="cam", kind="summary", text="x", parents=[], snapshot=store.snapshot(1))


def test_ids_are_immutable_and_idempotent(store):
    a = raw(store, id="given")
    assert raw(store, id="given").seq == a.seq
    with pytest.raises(ContractError):
        raw(store, text="different", id="given")


def test_search_does_not_change_when_future_corpus_changes(store):
    raw(store, 1, "red box")
    raw(store, 2, "blue box blue blue")
    snap = store.snapshot(2)
    before = [x.id for x in store.search("red blue", snap)]
    for i in range(30):
        raw(store, 100+i, "red red red red")
    assert [x.id for x in store.search("red blue", snap)] == before


def test_search_filters_future_before_topk(store):
    old = raw(store, 1, "rare package")
    snap = store.snapshot(1)
    for i in range(20):
        raw(store, 100+i, "rare package")
    assert store.search("package", snap, limit=1)[0].id == old.id


def test_embedding_similarity_and_fingerprint(store):
    a, b = raw(store, text="car"), raw(store, 2, "flower")
    store.put_vector(a.id, [1, 0], "encoder-a")
    store.put_vector(b.id, [0, 1], "encoder-a")
    result = store.search("vehicle", store.snapshot(2), vector=[1, 0], vector_model="encoder-a")
    assert result[0].id == a.id
    assert not store.search("vehicle", store.snapshot(2), vector=[1, 0], vector_model="encoder-b")


def test_dimensions_and_finite_embeddings(store):
    a = raw(store)
    with pytest.raises(ContractError):
        store.put_vector(a.id, [float("nan")], "x")
    store.put_vector(a.id, [1, 0], "x")
    with pytest.raises(ContractError):
        store.search("x", store.snapshot(1), vector=[1], vector_model="x")


def test_interval_overlap_and_boundaries(store):
    a = store.add_raw(source="cam", kind="asr", start=5, end=8, available_at=8, text="speech")
    snap = store.snapshot(10)
    assert store.list(snap, start=6, end=9)[0].id == a.id
    assert not store.list(snap, start=9, end=10)
    with pytest.raises(NotAvailable):
        store.list(snap, start=6, end=20)


def test_namespace_isolation(tmp_path):
    path = tmp_path / "x.db"
    EvidenceStore(path, "tenant_a").close()
    with pytest.raises(ContractError):
        EvidenceStore(path, "tenant_b")


def test_sql_and_fts_injection_is_data(store):
    e = raw(store, text="safe package")
    store.search("\" OR 1=1; DROP TABLE evidence; --", store.snapshot(1))
    assert store.get(e.id).text == "safe package"
