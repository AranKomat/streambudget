from __future__ import annotations

import hashlib
from collections import Counter
import json
import math
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .types import ContractError, Evidence, NotAvailable, Snapshot, valid_time


def hash_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Dependency-free lexical feature hashing. NOT a learned semantic embedding."""
    vec = np.zeros(dimensions, dtype=np.float64)
    for token in re.findall(r"\w+", text.lower()):
        h = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big")
        vec[h % dimensions] += 1 if (h // dimensions) % 2 else -1
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm else vec.tolist()


class EvidenceStore:
    """Single-tenant durable evidence and FTS index with transitive source lineage.

    Ingestion sequence is frozen in Snapshot. A later derivation from the same
    old inputs is allowed, but a later-arriving source or future-consuming
    summary cannot enter that snapshot. All model inputs must be declared parents.
    """

    def __init__(self, path: Path, namespace: str = "local", use_hash_embeddings: bool = False):
        self.namespace = namespace
        self.use_hash_embeddings = use_hash_embeddings
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL, kind TEXT NOT NULL, start REAL NOT NULL, end REAL NOT NULL,
                available_at REAL NOT NULL, text TEXT NOT NULL, payload TEXT NOT NULL,
                parents TEXT NOT NULL, input_end REAL NOT NULL, input_available_at REAL NOT NULL,
                leaf_seq INTEGER NOT NULL, created_wall REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_time ON evidence(source, kind, end);
            CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(id UNINDEXED, text);
            CREATE TABLE IF NOT EXISTS vectors (
                evidence_id TEXT PRIMARY KEY REFERENCES evidence(id), model TEXT NOT NULL, vector TEXT NOT NULL
            );
        """)
        old = self.db.execute("SELECT value FROM meta WHERE key='namespace'").fetchone()
        if old and old[0] != namespace:
            raise ContractError("Database belongs to a different namespace")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES('namespace',?)", (namespace,))
        self.db.commit()

    def close(self):
        self.db.close()

    def snapshot(self, as_of: float) -> Snapshot:
        seq = self.db.execute("SELECT COALESCE(MAX(seq),0) FROM evidence").fetchone()[0]
        return Snapshot(valid_time(as_of), int(seq))

    @staticmethod
    def _row(row: sqlite3.Row) -> Evidence:
        d = dict(row)
        d["payload"] = json.loads(d["payload"])
        d["parents"] = tuple(json.loads(d["parents"]))
        return Evidence(**d)

    def get(self, id: str, snapshot: Snapshot | None = None) -> Evidence:
        row = self.db.execute("SELECT * FROM evidence WHERE id=?", (id,)).fetchone()
        if row is None:
            raise NotAvailable(f"Unknown evidence: {id}")
        e = self._row(row)
        if snapshot is not None and not e.visible(snapshot):
            raise NotAvailable(f"Evidence is not available in this snapshot: {id}")
        return e

    def add_raw(self, *, source: str, kind: str, start: float, end: float,
                available_at: float, text: str = "", payload: dict | None = None,
                id: str | None = None) -> Evidence:
        return self._insert(source, kind, start, end, available_at, text, payload or {}, (), id)

    def derive(self, *, source: str, kind: str, text: str, parents: list[str],
               snapshot: Snapshot, payload: dict | None = None) -> Evidence:
        if not parents:
            raise ContractError("Derived evidence requires input provenance")
        parents = sorted(set(parents))
        evidence = [self.get(p, snapshot) for p in parents]
        return self._insert(source, kind, min(e.start for e in evidence), max(e.end for e in evidence),
                            max(e.available_at for e in evidence), text, payload or {}, tuple(parents), None)

    def _insert(self, source: str, kind: str, start: float, end: float, available: float,
                text: str, payload: dict, parents: tuple[str, ...], id: str | None) -> Evidence:
        start, end, available = map(valid_time, (start, end, available))
        if end < start or available < end:
            raise ContractError("Evidence interval must satisfy 0 <= start <= end <= availability")
        if len(text) > 100000 or not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", source):
            raise ContractError("Invalid source identifier or oversized text")
        parent_rows = [self.get(p) for p in parents]
        input_end = max([end] + [p.input_end for p in parent_rows])
        input_available = max([available] + [p.input_available_at for p in parent_rows])
        leaf_seq = max([p.leaf_seq for p in parent_rows], default=0)
        canonical = json.dumps([source, kind, start, end, available, text, payload, parents],
                               sort_keys=True, allow_nan=False)
        id = id or uuid.uuid5(uuid.NAMESPACE_URL, self.namespace + canonical).hex
        existing = self.db.execute("SELECT * FROM evidence WHERE id=?", (id,)).fetchone()
        if existing:
            old = self._row(existing)
            if (old.source, old.kind, old.start, old.end, old.available_at, old.text, old.payload,
                old.parents) != (source, kind, start, end, available, text, payload, parents):
                raise ContractError("Evidence ID collision with different content")
            return old
        with self.db:
            cursor = self.db.execute("""INSERT INTO evidence
                (id,source,kind,start,end,available_at,text,payload,parents,input_end,input_available_at,
                 leaf_seq,created_wall) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (id, source, kind, start, end, available, text, json.dumps(payload, allow_nan=False),
                 json.dumps(parents), input_end, input_available, leaf_seq, time.time()))
            if not parents:
                self.db.execute("UPDATE evidence SET leaf_seq=? WHERE id=?", (cursor.lastrowid, id))
            self.db.execute("INSERT INTO evidence_fts(id,text) VALUES (?,?)", (id, text))
            if self.use_hash_embeddings and text:
                self.put_vector(id, hash_embedding(text), "lexical-hash-256-v1", commit=False)
        return self.get(id)

    def put_vector(self, id: str, vector: list[float], model: str, *, commit: bool = True):
        self.get(id)
        if not vector or any(not math.isfinite(v) for v in vector):
            raise ContractError("Invalid embedding")
        self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)", (id, model, json.dumps(vector)))
        if commit:
            self.db.commit()

    def list(self, snapshot: Snapshot, *, source: str | None = None, kinds: list[str] | None = None,
             start: float = 0, end: float | None = None, limit: int = 1000) -> list[Evidence]:
        valid_time(start)
        if limit < 1 or limit > 100000:
            raise ContractError("Invalid result limit")
        end = snapshot.as_of if end is None else valid_time(end)
        if end < start or end > snapshot.as_of:
            raise NotAvailable("Requested interval extends beyond snapshot or is reversed")
        where = ["input_end <= ?", "input_available_at <= ?", "leaf_seq <= ?", "end >= ?", "start <= ?"]
        params: list[Any] = [snapshot.as_of, snapshot.as_of, snapshot.max_source_seq, start, end]
        if source is not None:
            where.append("source=?")
            params.append(source)
        if kinds is not None:
            if not kinds:
                return []
            where.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            params.extend(kinds)
        rows = self.db.execute("SELECT * FROM evidence WHERE " + " AND ".join(where)
                               + " ORDER BY end DESC, seq DESC LIMIT ?", [*params, limit]).fetchall()
        return [self._row(r) for r in rows]

    def frames(self, source: str, start: float, end: float, snapshot: Snapshot,
               count: int) -> list[Evidence]:
        if count < 1 or count > 128:
            raise ContractError("Requested frame count out of bounds")
        items = list(reversed(self.list(snapshot, source=source, kinds=["frame"], start=start, end=end,
                                        limit=100000)))
        if len(items) <= count:
            return items
        indices = np.linspace(0, len(items) - 1, count).round().astype(int)
        return [items[i] for i in indices]

    def search(self, query: str, snapshot: Snapshot, *, source: str | None = None,
               limit: int = 8, start: float = 0, end: float | None = None,
               kinds: list[str] | None = None, vector: list[float] | None = None,
               vector_model: str | None = None) -> list[Evidence]:
        # Filter BEFORE ranking: inaccessible evidence cannot crowd out visible hits.
        candidates = self.list(snapshot, source=source, kinds=kinds, start=start, end=end, limit=100000)
        visible = {e.id: e for e in candidates if e.kind != "frame"}
        if not query.strip():
            return list(visible.values())[:limit]
        terms = re.findall(r"\w+", query.lower())[:64]
        scores: dict[str, float] = {}
        if terms:
            match = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
            rows = self.db.execute("SELECT id FROM evidence_fts WHERE evidence_fts MATCH ?", (match,)).fetchall()
            matched = {r[0] for r in rows if r[0] in visible}
            # FTS selects matches, but ranking statistics are computed ONLY on
            # the visible snapshot. Global FTS BM25 IDF would leak future corpus statistics.
            bags = {id: Counter(re.findall(r"\w+", e.text.lower())) for id, e in visible.items()}
            lengths = {id: sum(b.values()) for id, b in bags.items()}
            avg_len = sum(lengths.values()) / max(1, len(lengths)) or 1
            df = {t: sum(t in bag for bag in bags.values()) for t in set(terms)}
            bm25 = {}
            for id in matched:
                score = 0.0
                for t in set(terms):
                    tf = bags[id][t]
                    inverse = math.log(1 + (len(bags) - df[t] + .5) / (df[t] + .5))
                    score += inverse * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[id] / avg_len))
                bm25[id] = score
            ranked = sorted(matched, key=lambda id: (bm25[id], visible[id].end, id), reverse=True)
            for rank, id in enumerate(ranked):
                scores[id] = 1 / (61 + rank)
        if vector is None and self.use_hash_embeddings:
            vector, vector_model = hash_embedding(query), "lexical-hash-256-v1"
        if vector is not None:
            q = np.asarray(vector, dtype=float)
            ranked = []
            for row in self.db.execute("SELECT * FROM vectors WHERE model=?", (vector_model,)):
                if row["evidence_id"] not in visible:
                    continue
                v = np.asarray(json.loads(row["vector"]), dtype=float)
                if v.shape != q.shape:
                    raise ContractError("Embedding dimensions differ within the same model fingerprint")
                norm = float(np.linalg.norm(q) * np.linalg.norm(v))
                similarity = float(q @ v / norm) if norm else 0
                if similarity > 0:
                    ranked.append((similarity, row["evidence_id"]))
            for rank, (_, id) in enumerate(sorted(ranked, reverse=True)):
                scores[id] = scores.get(id, 0) + 1 / (61 + rank)
        ids = sorted(scores, key=lambda id: (scores[id], visible[id].end), reverse=True)[:limit]
        return [visible[id] for id in ids]
