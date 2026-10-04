"""Single-writer bitemporal semantic memory over StreamBudget's EvidenceStore.

Observation order decides current beliefs. Publication revision decides what an agent
could know. Neither a model confidence nor a stored relation is ground truth.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from dataclasses import asdict

from ...store import EvidenceStore
from ...types import ContractError, Snapshot
from ..contracts import Property, SchemaPatch
from ..ontology import Ontology, canonical
from .contracts import JobBasis, MemoryCutoff, MemoryDelta, OCRPacket


class TemporalMemory:
    def __init__(
        self,
        store: EvidenceStore,
        *,
        max_entities=3000,
        fact_threshold=0.5,
        association_threshold=0.85,
        volatile_age_frames=60,
    ):
        self.store, self.db = store, store.db
        self.owner = threading.get_ident()
        self.max_entities = max_entities
        self.fact_threshold = fact_threshold
        self.association_threshold = association_threshold
        self.volatile_age_frames = volatile_age_frames
        self.ontology = Ontology(self.db)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS tm_revisions(
            revision INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
            published_at REAL NOT NULL, epoch TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS tm_patches(
            job_id TEXT PRIMARY KEY, digest TEXT NOT NULL, evidence TEXT NOT NULL,
            source_seq INTEGER NOT NULL, frame_number INTEGER NOT NULL,
            revision INTEGER NOT NULL REFERENCES tm_revisions(revision),
            basis TEXT NOT NULL, ids TEXT NOT NULL, payload TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS tm_patch_time ON tm_patches(source_seq,revision);
          CREATE TABLE IF NOT EXISTS tm_nodes(
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT NOT NULL,
            created_patch TEXT NOT NULL REFERENCES tm_patches(job_id));
          CREATE TABLE IF NOT EXISTS tm_mentions(
            patch TEXT NOT NULL REFERENCES tm_patches(job_id), entity TEXT NOT NULL REFERENCES tm_nodes(id));
          CREATE INDEX IF NOT EXISTS tm_mention_entity ON tm_mentions(entity,patch);
          CREATE TABLE IF NOT EXISTS tm_facts(
            id INTEGER PRIMARY KEY, patch TEXT NOT NULL REFERENCES tm_patches(job_id),
            entity TEXT NOT NULL REFERENCES tm_nodes(id), key TEXT NOT NULL,
            value TEXT NOT NULL, confidence REAL NOT NULL, basis TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS tm_fact_entity ON tm_facts(entity,key,patch);
          CREATE TABLE IF NOT EXISTS tm_relations(
            id INTEGER PRIMARY KEY, patch TEXT NOT NULL REFERENCES tm_patches(job_id),
            subject TEXT NOT NULL REFERENCES tm_nodes(id), predicate TEXT NOT NULL,
            target TEXT NOT NULL REFERENCES tm_nodes(id), operation TEXT NOT NULL, confidence REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS tm_rel_subject ON tm_relations(subject,predicate,patch);
          CREATE TABLE IF NOT EXISTS tm_events(
            id INTEGER PRIMARY KEY, patch TEXT NOT NULL REFERENCES tm_patches(job_id),
            occurrence TEXT NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
            participants TEXT NOT NULL, confidence REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS tm_threads(id TEXT PRIMARY KEY, participants TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS tm_messages(
            id INTEGER PRIMARY KEY, patch TEXT NOT NULL REFERENCES tm_patches(job_id),
            thread TEXT NOT NULL REFERENCES tm_threads(id), occurrence TEXT NOT NULL,
            speaker TEXT, surface TEXT, text TEXT NOT NULL, attribution REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS tm_message_thread ON tm_messages(thread,occurrence,patch);
          CREATE TABLE IF NOT EXISTS tm_places(
            patch TEXT NOT NULL REFERENCES tm_patches(job_id), place TEXT NOT NULL REFERENCES tm_nodes(id));
          CREATE TABLE IF NOT EXISTS tm_routes(
            id INTEGER PRIMARY KEY, patch TEXT NOT NULL REFERENCES tm_patches(job_id),
            origin TEXT NOT NULL REFERENCES tm_nodes(id), portal TEXT NOT NULL REFERENCES tm_nodes(id),
            destination TEXT REFERENCES tm_nodes(id), status TEXT NOT NULL, occurrence TEXT NOT NULL,
            hint TEXT NOT NULL, action_evidence TEXT, confidence REAL NOT NULL);
          CREATE INDEX IF NOT EXISTS tm_route_origin ON tm_routes(origin,portal,patch);
          CREATE TABLE IF NOT EXISTS tm_views(
            id INTEGER PRIMARY KEY, patch TEXT NOT NULL REFERENCES tm_patches(job_id),
            entity TEXT NOT NULL REFERENCES tm_nodes(id), frame_id TEXT NOT NULL, box TEXT);
          CREATE TABLE IF NOT EXISTS tm_pins(
            revision INTEGER NOT NULL REFERENCES tm_revisions(revision), entity TEXT NOT NULL,
            reason TEXT NOT NULL, active INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS tm_ocr_packets(
            worker TEXT NOT NULL, sequence INTEGER NOT NULL, digest TEXT NOT NULL,
            frame_id TEXT NOT NULL, source_seq INTEGER NOT NULL, frame_number INTEGER NOT NULL,
            revision INTEGER NOT NULL REFERENCES tm_revisions(revision), evidence TEXT NOT NULL,
            payload TEXT NOT NULL, PRIMARY KEY(worker,sequence));
          CREATE TABLE IF NOT EXISTS tm_indexes(
            evidence TEXT PRIMARY KEY, revision INTEGER NOT NULL, source_seq INTEGER NOT NULL,
            text TEXT NOT NULL, entities TEXT NOT NULL, kind TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS tm_index_cutoff ON tm_indexes(revision,source_seq);
        """)
        self.db.commit()
        cur = self.ontology.current
        names = {p["name"] for p in cur["properties"]}
        additions = [
            p
            for p in (
                Property(
                    name="name",
                    definition="Evidence-backed display name; not identity proof.",
                    persistent=True,
                ),
                Property(
                    name="aliases",
                    value_type="text_list",
                    definition="Other observed names of this same hypothesized entity.",
                    persistent=True,
                ),
            )
            if p.name not in names
        ]
        if additions:
            patch = SchemaPatch(
                parent_version=cur["version"],
                properties=additions,
                reason="Initialize generic name/alias properties for this memory format.",
            )
            self.ontology.approve(
                self.ontology.propose(patch),
                initial_prior=True,
                review="Fixed generic memory schema initialization, not inferred world knowledge.",
            )

    @classmethod
    def open_readonly(cls, path, *, volatile_age_frames=60):
        """Read-only diagnostic handle; caller closes memory.store after use."""
        import sqlite3
        from pathlib import Path

        db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        store = EvidenceStore.__new__(EvidenceStore)
        store.db = db
        store.namespace = db.execute("SELECT value FROM meta WHERE key='namespace'").fetchone()[0]
        store.use_hash_embeddings = False
        obj = cls.__new__(cls)
        obj.store, obj.db, obj.owner = store, db, threading.get_ident()
        obj.max_entities, obj.fact_threshold, obj.association_threshold = 3000, 0.5, 0.85
        obj.volatile_age_frames = volatile_age_frames
        obj.ontology = Ontology.__new__(Ontology)
        obj.ontology.db = db
        return obj

    def _owner(self):
        if threading.get_ident() != self.owner:
            raise ContractError("Only the owning event-loop thread may read/write temporal memory")

    @property
    def revision(self):
        self._owner()
        return int(self.db.execute("SELECT COALESCE(MAX(revision),0) FROM tm_revisions").fetchone()[0])

    def cutoff(self, source_seq: int) -> MemoryCutoff:
        return MemoryCutoff(self.revision, source_seq)

    def _publish(self, kind, now, epoch):
        if not isinstance(now, (float, int)) or not 0 <= now < float("inf"):
            raise ContractError("Invalid publication time")
        previous = self.db.execute(
            "SELECT MAX(published_at) FROM tm_revisions WHERE epoch=?", (epoch,)
        ).fetchone()[0]
        if previous is not None and now < previous:
            raise ContractError("Publication time moved backwards within this clock epoch")
        return self.db.execute(
            "INSERT INTO tm_revisions(kind,published_at,epoch) VALUES(?,?,?)", (kind, float(now), epoch)
        ).lastrowid

    def _rows(self, table, cutoff, where="1", params=()):
        # table/where are internal literals, never model SQL.
        self._owner()
        return [
            dict(r)
            for r in self.db.execute(
                f"SELECT x.*,p.source_seq,p.frame_number,p.revision,p.evidence,p.basis AS job_basis "
                f"FROM {table} x JOIN tm_patches p ON x.patch=p.job_id "
                f"WHERE p.revision<=? AND p.source_seq<=? AND ({where}) "
                "ORDER BY p.source_seq DESC,p.revision DESC,x.rowid DESC",
                (cutoff.revision, cutoff.source_seq, *params),
            )
        ]

    def node(self, id, cutoff):
        row = self.db.execute(
            "SELECT n.*,p.revision,p.source_seq,p.evidence FROM tm_nodes n "
            "JOIN tm_patches p ON n.created_patch=p.job_id WHERE n.id=? AND p.revision<=? AND p.source_seq<=?",
            (id, cutoff.revision, cutoff.source_seq),
        ).fetchone()
        if row is None:
            raise ContractError("Unknown entity at the supplied knowledge/source cutoff")
        return dict(row)

    def nodes(self, cutoff):
        self._owner()
        return [
            dict(r)
            for r in self.db.execute(
                "SELECT n.*,p.source_seq,p.revision,p.evidence FROM tm_nodes n "
                "JOIN tm_patches p ON n.created_patch=p.job_id WHERE p.revision<=? AND p.source_seq<=?",
                (cutoff.revision, cutoff.source_seq),
            )
        ]

    def _check_basis(self, basis):
        self._owner()
        frame = self.store.get(basis.source_frame_id)
        if (
            frame.kind != "frame"
            or frame.seq != basis.source_seq
            or frame.payload.get("frame_number") != basis.frame_number
        ):
            raise ContractError("Job basis does not identify an exact source frame")
        if basis.source_frame_id not in basis.offered_frames or basis.knowledge_revision > self.revision:
            raise ContractError("Invalid job exposure/knowledge revision")
        schema = self.db.execute(
            "SELECT body FROM world_schemas WHERE version=?", (basis.ontology_version,)
        ).fetchone()
        if schema is None:
            raise ContractError("Job references unknown ontology revision")
        # Old additive schemas remain interpretable. They cannot invent newer properties.
        cutoff = MemoryCutoff(basis.knowledge_revision, basis.source_seq)
        for id in basis.offered_entities:
            self.node(id, cutoff)
        snap = Snapshot(frame.end, basis.source_seq)
        for id in {*basis.evidence_ids, *basis.offered_frames}:
            e = self.store.get(id, snap)
            idx = self.db.execute("SELECT revision FROM tm_indexes WHERE evidence=?", (id,)).fetchone()
            if idx and idx[0] > basis.knowledge_revision:
                raise ContractError("Job consumed unpublished semantic information")
            if e.source != frame.source:
                raise ContractError("Cross-source evidence requires an explicit adapter")
        return frame, json.loads(schema[0]), snap

    def commit(self, basis: JobBasis, delta: MemoryDelta, *, now: float):
        """Append old/new interpretations; independently project fields by source order.

        Same job + same payload is idempotent. Independent jobs may annotate one frame.
        Missing fields mean 'not updated', never disappearance/retraction.
        """
        frame, schema, snap = self._check_basis(basis)
        body = canonical(delta.model_dump())
        digest = hashlib.sha256((canonical(asdict(basis)) + body).encode()).hexdigest()
        previous = self.db.execute("SELECT * FROM tm_patches WHERE job_id=?", (basis.job_id,)).fetchone()
        if previous:
            if previous["digest"] != digest:
                raise ContractError("Conflicting completion for the same job")
            return {
                "evidence_id": previous["evidence"],
                "ids": json.loads(previous["ids"]),
                "revision": previous["revision"],
                "changed": False,
            }
        known = {
            n["id"]: n["kind"] for n in self.nodes(MemoryCutoff(basis.knowledge_revision, basis.source_seq))
        }
        ids, created, labels = {}, [], {}
        for m in delta.mentions:
            if m.ref in ids:
                raise ContractError("Duplicate mention handle")
            if m.ref.startswith("new:"):
                if basis.targets:
                    raise ContractError("Enrichment cannot invent nodes; return to the index for discovery")
                if m.association != "new":
                    raise ContractError("New local handle cannot claim continuity")
                # Deterministic per job; no semantic-label or crop-hash identity merging.
                id = "e_" + hashlib.sha256((basis.job_id + m.ref).encode()).hexdigest()[:16]
                created.append((id, m.kind, m.label))
            else:
                if m.ref not in basis.offered_entities or known.get(m.ref) != m.kind:
                    raise ContractError("Unoffered or kind-changing identity")
                if (
                    m.association not in ("continuity", "reidentified")
                    or m.confidence < self.association_threshold
                ):
                    raise ContractError("Identity is unresolved; keep a separate sighting")
                id = m.ref
            if m.region and m.region.frame_id not in basis.offered_frames:
                raise ContractError("Region must refer to an actually supplied frame")
            ids[m.ref], labels[id] = id, m.label
        if self.db.execute("SELECT COUNT(*) FROM tm_nodes").fetchone()[0] + len(created) > self.max_entities:
            raise ContractError("Entity capacity reached; no silent pruning or forced merging")
        local_kinds = {id: kind for id, kind, _ in created}

        def resolve(ref, kind=None):
            id = ids.get(ref, ref)
            if id not in local_kinds and id not in basis.offered_entities:
                raise ContractError("Reference was not exposed to this job")
            if kind and local_kinds.get(id, known.get(id)) != kind:
                raise ContractError("Reference has incorrect metamodel kind")
            return id

        props = {p["name"]: p for p in schema["properties"]}
        facts = []
        for f in delta.facts:
            if f.key not in props:
                raise ContractError("Property is absent from the job's frozen schema")
            # Current schema is additive; its value validator remains compatible.
            self.ontology.validate_value(f.key, f.value)
            facts.append((resolve(f.subject), f))
        predicates = {r["name"] for r in schema["relations"]}
        relations = []
        for r in delta.relations:
            if r.predicate not in predicates:
                raise ContractError("Unknown relation in the frozen schema")
            relations.append((resolve(r.subject), resolve(r.target), r))
        place = resolve(delta.current_place, "place") if delta.current_place else None
        evs = [(e, [resolve(p) for p in e.participants]) for e in delta.events]
        messages = []
        for u in delta.utterances:
            speaker = resolve(u.speaker) if u.speaker else None
            participants = sorted(set([resolve(p) for p in u.partners] + ([speaker] if speaker else [])))
            surface = resolve(u.surface, "surface") if u.surface else None
            # Unknown speakers get source/surface-scoped threads, not one global unknown-person DM.
            thread_basis = participants or ["unknown", frame.source, surface or basis.source_frame_id]
            thread = "thread_" + hashlib.sha256(canonical(thread_basis).encode()).hexdigest()[:20]
            messages.append((u, speaker, participants, surface, thread))
        routes = []
        for r in delta.routes:
            origin, portal = resolve(r.origin, "place"), resolve(r.portal)
            target = resolve(r.destination, "place") if r.destination else None
            if r.action_evidence:
                if r.action_evidence not in basis.evidence_ids:
                    raise ContractError("Route uses an unoffered action receipt")
                receipt = self.store.get(r.action_evidence, snap)
                if receipt.kind != "action_receipt":
                    raise ContractError("Route effect must reference an actual input receipt")
            routes.append((origin, portal, target, r))
        if basis.targets:
            # The host chooses the enrichment target; cross-entity relations may use offered context.
            if set(id for id, _ in facts) - set(basis.targets):
                raise ContractError("Enrichment facts must concern its assigned target")
            if place:
                raise ContractError("Entity enrichment cannot change the active place")
        search_text = "\n".join(
            [
                *labels.values(),
                *[e.text for e, _ in evs],
                *[u.text for u, *_ in messages],
                *[f"{f.key}: {canonical(f.value)}" for _, f in facts],
                *[r.direction_hint for *_, r in routes],
            ]
        )
        if len(search_text) > 80000:
            raise ContractError("Semantic record too large")
        parents = sorted(set([basis.source_frame_id, *basis.offered_frames, *basis.evidence_ids]))
        record = self.store.derive(
            source=frame.source,
            kind="async_memory",
            text=search_text,
            parents=parents,
            snapshot=snap,
            payload={
                "job_id": basis.job_id,
                "interpretation": delta.model_dump(),
                "source_frame_id": basis.source_frame_id,
                "epistemic_status": "model_assertion_not_independent_verification",
            },
        )
        # EvidenceStore commits source derivations itself. An orphan derivation on failure is
        # not exposed by this module: publication and every semantic table commit atomically below.
        with self.db:
            rev = self._publish("interpretation", now, basis.episode)
            self.db.execute(
                "INSERT INTO tm_patches VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    basis.job_id,
                    digest,
                    record.id,
                    frame.seq,
                    basis.frame_number,
                    rev,
                    canonical(asdict(basis)),
                    canonical(ids),
                    body,
                ),
            )
            for id, kind, label in created:
                self.db.execute("INSERT INTO tm_nodes VALUES(?,?,?,?)", (id, kind, label, basis.job_id))
            for m in delta.mentions:
                id = ids[m.ref]
                self.db.execute("INSERT INTO tm_mentions VALUES(?,?)", (basis.job_id, id))
                self.db.execute(
                    "INSERT INTO tm_views(patch,entity,frame_id,box) VALUES(?,?,?,?)",
                    (
                        basis.job_id,
                        id,
                        m.region.frame_id if m.region else basis.source_frame_id,
                        canonical(m.region.box) if m.region else None,
                    ),
                )
            for id, f in facts:
                if f.confidence >= self.fact_threshold:
                    self.db.execute(
                        "INSERT INTO tm_facts(patch,entity,key,value,confidence,basis) VALUES(?,?,?,?,?,?)",
                        (basis.job_id, id, f.key, canonical(f.value), f.confidence, f.basis),
                    )
            for a, b, r in relations:
                if r.confidence >= self.fact_threshold:
                    self.db.execute(
                        "INSERT INTO tm_relations(patch,subject,predicate,target,operation,confidence) VALUES(?,?,?,?,?,?)",
                        (basis.job_id, a, r.predicate, b, r.operation, r.confidence),
                    )
            for e, participants in evs:
                self.db.execute(
                    "INSERT INTO tm_events(patch,occurrence,kind,text,participants,confidence) VALUES(?,?,?,?,?,?)",
                    (basis.job_id, e.occurrence, e.kind, e.text, canonical(participants), e.confidence),
                )
            for u, speaker, participants, surface, thread in messages:
                self.db.execute(
                    "INSERT OR IGNORE INTO tm_threads VALUES(?,?)", (thread, canonical(participants))
                )
                self.db.execute(
                    "INSERT INTO tm_messages(patch,thread,occurrence,speaker,surface,text,attribution) VALUES(?,?,?,?,?,?,?)",
                    (basis.job_id, thread, u.occurrence, speaker, surface, u.text, u.attribution_confidence),
                )
            if place:
                self.db.execute("INSERT INTO tm_places VALUES(?,?)", (basis.job_id, place))
            for origin, portal, target, r in routes:
                self.db.execute(
                    "INSERT INTO tm_routes(patch,origin,portal,destination,status,occurrence,hint,action_evidence,confidence) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        basis.job_id,
                        origin,
                        portal,
                        target,
                        r.status,
                        r.occurrence,
                        r.direction_hint,
                        r.action_evidence,
                        r.confidence,
                    ),
                )
            involved = set(ids.values()) | {id for id, _ in facts}
            involved |= {p for _, ps in evs for p in ps}
            involved |= {p for _, _, ps, _, _ in messages for p in ps}
            involved |= {p for a, b, _ in relations for p in (a, b)}
            involved |= {p for a, b, c, _ in routes for p in (a, b, c) if p}
            self.db.execute(
                "INSERT INTO tm_indexes VALUES(?,?,?,?,?,?)",
                (record.id, rev, frame.seq, search_text, canonical(sorted(involved)), "memory"),
            )
        return {"evidence_id": record.id, "ids": ids, "revision": rev, "changed": True}

    def facts(self, entity, cutoff, *, frame_number):
        self.node(entity, cutoff)
        result = {}
        persistent = {p["name"] for p in self.ontology.current["properties"] if p.get("persistent")}
        for r in self._rows("tm_facts", cutoff, "x.entity=?", (entity,)):
            key = r["key"]
            if key in result:
                # Same observed instant with conflicting values is uncertainty, not arrival-order truth.
                previous = result[key]
                if r["source_seq"] == previous["source_seq"] and json.loads(r["value"]) != previous["value"]:
                    previous["conflict"] = True
                continue
            result[key] = {
                "value": json.loads(r["value"]),
                "confidence": r["confidence"],
                "basis": r["basis"],
                "evidence_id": r["evidence"],
                "observed_frame": r["frame_number"],
                "source_seq": r["source_seq"],
                "knowledge_revision": r["revision"],
                "conflict": False,
                "stale": key not in persistent
                and frame_number - r["frame_number"] > self.volatile_age_frames,
            }
        return result

    def relations(self, cutoff, entity=None):
        rows = self._rows("tm_relations", cutoff)
        latest, seen = [], set()
        for r in rows:
            # A newer retracted location does not resurrect a previous room.
            key = (
                (r["subject"], r["predicate"])
                if r["predicate"] == "located_in"
                else (r["subject"], r["predicate"], r["target"])
            )
            if key in seen:
                continue
            seen.add(key)
            if r["operation"] == "assert" and (entity is None or entity in (r["subject"], r["target"])):
                latest.append(r)
        return latest

    def current_place(self, cutoff):
        rows = self._rows("tm_places", cutoff)
        if not rows:
            return None
        first = rows[0]
        matches = {r["place"] for r in rows if r["source_seq"] == first["source_seq"]}
        if len(matches) != 1:
            return {
                "id": None,
                "ambiguous": True,
                "evidence_ids": [r["evidence"] for r in rows if r["source_seq"] == first["source_seq"]],
            }
        return {
            "id": first["place"],
            "observed_frame": first["frame_number"],
            "evidence_id": first["evidence"],
            "source_seq": first["source_seq"],
            "ambiguous": False,
        }

    def visits(self, cutoff, place=None):
        rows = list(reversed(self._rows("tm_places", cutoff)))
        by_seq = {}
        for r in rows:
            by_seq.setdefault(r["source_seq"], []).append(r)
        visits = []
        for source_seq, same_time in sorted(by_seq.items()):
            if len({r["place"] for r in same_time}) != 1:
                # Conflicting localization is a gap; do not assert continuity through it.
                visits.append(
                    {
                        "place": None,
                        "ambiguous": True,
                        "first_frame": same_time[0]["frame_number"],
                        "last_frame": same_time[0]["frame_number"],
                        "evidence_ids": [],
                    }
                )
                continue
            r = same_time[-1]
            if visits and visits[-1]["place"] == r["place"]:
                visits[-1]["last_frame"] = r["frame_number"]
                visits[-1]["evidence_ids"] = (visits[-1]["evidence_ids"] + [r["evidence"]])[-4:]
            else:
                visits.append(
                    {
                        "place": r["place"],
                        "first_frame": r["frame_number"],
                        "last_frame": r["frame_number"],
                        "evidence_ids": [r["evidence"]],
                        "note": "Observed visit span, not proof of continuous presence or traversable connection.",
                    }
                )
        return [v for v in visits if place is None or v["place"] == place]

    def event_history(self, cutoff, *, entity=None, query="", limit=12):
        rows = self._rows("tm_events", cutoff)
        seen, out = set(), []
        for r in rows:
            ps = json.loads(r["participants"])
            key = (r["kind"], r["occurrence"], tuple(ps))
            if key in seen:
                continue
            seen.add(key)
            if entity and entity not in ps:
                continue
            if query and not self._match(query, r["text"]):
                continue
            out.append(
                {
                    "event_id": "event_" + str(r["id"]),
                    "kind": r["kind"],
                    "occurrence": r["occurrence"],
                    "text": r["text"],
                    "participants": ps,
                    "observed_frame": r["frame_number"],
                    "knowledge_revision": r["revision"],
                    "confidence": r["confidence"],
                    "evidence_id": r["evidence"],
                }
            )
            if len(out) >= limit:
                break
        return list(reversed(out))

    def conversation(self, entity, cutoff, *, query="", limit=12):
        if entity:
            self.node(entity, cutoff)
        rows = self._rows("tm_messages", cutoff)
        seen, out = set(), []
        for r in rows:
            key = (r["thread"], r["surface"], r["occurrence"])
            if key in seen:
                continue
            seen.add(key)
            participants = json.loads(
                self.db.execute("SELECT participants FROM tm_threads WHERE id=?", (r["thread"],)).fetchone()[
                    0
                ]
            )
            if entity and entity not in participants:
                continue
            if query and not self._match(query, r["text"]):
                continue
            out.append(
                {
                    "thread_id": r["thread"],
                    "occurrence": r["occurrence"],
                    "speaker": r["speaker"],
                    "participants": participants,
                    "surface_id": r["surface"],
                    "text": r["text"],
                    "attribution_confidence": r["attribution"],
                    "observed_frame": r["frame_number"],
                    "evidence_id": r["evidence"],
                    "knowledge_revision": r["revision"],
                }
            )
            if len(out) >= limit:
                break
        return list(reversed(out))

    def exits(self, place, cutoff):
        self.node(place, cutoff)
        rows = self._rows("tm_routes", cutoff, "x.origin=?", (place,))
        grouped = {}
        for r in rows:
            g = grouped.setdefault(
                r["portal"],
                {
                    "portal_id": r["portal"],
                    "origin": place,
                    "status": r["status"] if r["confidence"] >= self.fact_threshold else "unknown",
                    "destination": r["destination"],
                    "direction_hint": r["hint"],
                    "observed_frame": r["frame_number"],
                    "evidence_id": r["evidence"],
                    "attempts": set(),
                    "traversals": set(),
                    "ever_traversed": False,
                },
            )
            if r["status"] in ("attempted", "blocked", "traversed"):
                g["attempts"].add(r["occurrence"])
            if r["status"] == "traversed" and r["confidence"] >= self.fact_threshold:
                g["traversals"].add(r["occurrence"])
                g["ever_traversed"] = True
                if g["destination"] is None:
                    g["last_known_destination"] = r["destination"]
        for g in grouped.values():
            g["attempts"], g["traversals"] = len(g["attempts"]), len(g["traversals"])
            g["note"] = "Evidence-backed model assertions, not current reachability guarantees."
        return list(grouped.values())

    def place_summary(self, place, cutoff):
        node = self.node(place, cutoff)
        if node["kind"] != "place":
            raise ContractError("Place query requires a place")
        visits = self.visits(cutoff, place)
        contained = [
            r
            for r in self.relations(cutoff, place)
            if r["target"] == place and r["predicate"] in ("located_in", "part_of")
        ]
        return {
            "place_id": place,
            "label": node["label"],
            "visit_count": len(visits),
            "visits": visits[-8:],
            "last_known_contents": contained[:40],
            "exits": self.exits(place, cutoff),
            "coverage_fraction": None,
            "note": "Only known exits and observed visits. Unknown is not unexplored coverage.",
        }

    def route(self, origin, destination, cutoff):
        if self.node(origin, cutoff)["kind"] != "place" or self.node(destination, cutoff)["kind"] != "place":
            raise ContractError("A route must connect places, not arbitrary entities")
        queue, seen = deque([(origin, [])]), {origin}
        while queue:
            here, path = queue.popleft()
            if here == destination:
                return {
                    "found": True,
                    "edges": path,
                    "authority": "historical_topological_hint_not_controller",
                }
            for edge in self.exits(here, cutoff):
                there = edge.get("destination")
                if edge["status"] != "traversed" or not edge["ever_traversed"] or not there or there in seen:
                    continue
                seen.add(there)
                queue.append((there, path + [edge]))
        return {"found": False, "edges": [], "authority": "unknown_not_proof_of_no_route"}

    @staticmethod
    def _match(query, text):
        terms = set(re.findall(r"\w+", query.casefold()))
        return len(terms & set(re.findall(r"\w+", text.casefold())))

    def search(self, query, cutoff, *, limit=8, entity=None):
        self._owner()
        # Filter by both cutoffs BEFORE relevance scores; no future corpus statistics.
        rows = self.db.execute(
            "SELECT * FROM tm_indexes WHERE revision<=? AND source_seq<=?",
            (cutoff.revision, cutoff.source_seq),
        ).fetchall()
        scored = []
        for r in rows:
            entities = json.loads(r["entities"])
            if entity and entity not in entities:
                continue
            score = self._match(query, r["text"])
            if query.strip() and not score:
                continue
            scored.append((score, r["source_seq"], r["revision"], dict(r)))
        result = []
        for _, _, _, r in sorted(scored, key=lambda x: x[:3], reverse=True)[:limit]:
            result.append(
                {
                    "evidence_id": r["evidence"],
                    "text": r["text"][:6000],
                    "kind": r["kind"],
                    "entities": [
                        {"id": id, "label": self.node(id, cutoff)["label"]}
                        for id in json.loads(r["entities"])
                    ],
                    "knowledge_revision": r["revision"],
                    "source_seq": r["source_seq"],
                }
            )
        return result

    def pin(self, entity, *, reason, active, now, epoch, cutoff, max_pins=16):
        self.node(entity, cutoff)
        if reason not in ("active_goal", "user_requested", "held", "route_landmark"):
            raise ContractError("Pinning is a host decision, not arbitrary model authority")
        pins = self.pins(cutoff)
        if active and entity not in pins and len(pins) >= max_pins:
            raise ContractError("Mandatory working set exceeds hot capacity")
        with self.db:
            rev = self._publish("pin", now, epoch)
            self.db.execute("INSERT INTO tm_pins VALUES(?,?,?,?)", (rev, entity, reason, int(active)))

    def pins(self, cutoff):
        out, seen = {}, set()
        visible = {n["id"] for n in self.nodes(cutoff)}
        for r in self.db.execute(
            "SELECT * FROM tm_pins WHERE revision<=? ORDER BY revision DESC", (cutoff.revision,)
        ):
            if r["entity"] not in visible or r["entity"] in seen:
                continue
            seen.add(r["entity"])
            if r["active"]:
                out[r["entity"]] = r["reason"]
        return out

    def tiers(self, cutoff, *, frame_number, focus=(), hot=16, warm=128):
        nodes = self.nodes(cutoff)
        pins = self.pins(cutoff)
        place = self.current_place(cutoff)
        current = place["id"] if place else None
        mandatory = set(pins) | set(focus) | ({current} if current else set())
        existing = {n["id"] for n in nodes}
        if set(focus) - existing:
            raise ContractError("Focus contains unknown nodes")
        if len(mandatory) > hot:
            raise ContractError("Focused/pinned working set cannot fit; adjust capacity explicitly")
        recency = {}
        for r in self._rows("tm_mentions", cutoff):
            recency.setdefault(r["entity"], r["source_seq"])
        nodes.sort(key=lambda n: (n["id"] not in mandatory, -recency.get(n["id"], n["source_seq"]), n["id"]))
        selected = []
        for n in nodes[:hot]:
            facts = self.facts(n["id"], cutoff, frame_number=frame_number)
            last = next(iter(self._rows("tm_mentions", cutoff, "x.entity=?", (n["id"],))), None)
            selected.append(
                {
                    "id": n["id"],
                    "kind": n["kind"],
                    "label": n["label"],
                    "facts": facts,
                    "last_seen_frame": last["frame_number"] if last else None,
                    "evidence_id": last["evidence"] if last else n["evidence"],
                }
            )
        return {
            "hot": selected,
            "warm_ids": [n["id"] for n in nodes[hot:warm]],
            "cold_count": max(0, len(nodes) - warm),
            "pinned": pins,
            "note": "Tiers affect context only. Evidence and nodes are not deleted.",
        }

    def visual_refs(self, entity, cutoff, *, limit=4):
        self.node(entity, cutoff)
        selected, seen = [], set()
        for r in self._rows("tm_views", cutoff, "x.entity=?", (entity,)):
            frame = self.store.get(r["frame_id"])
            key = (frame.payload.get("sha256", r["frame_id"]), r["box"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "frame_id": r["frame_id"],
                    "box": json.loads(r["box"]) if r["box"] else None,
                    "observed_frame": frame.payload.get("frame_number"),
                    "evidence_id": r["evidence"],
                    "historical": True,
                }
            )
            if len(selected) >= limit:
                break
        return selected

    def ingest_ocr(self, packet: OCRPacket, *, now, epoch):
        """Owner-thread bridge. Recognition, motion tracking and occurrence IDs stay external."""
        self._owner()
        frame = self.store.get(packet.source_frame_id)
        if frame.kind != "frame":
            raise ContractError("OCR source must be a retained frame")
        body = canonical(packet.model_dump())
        digest = hashlib.sha256(body.encode()).hexdigest()
        prev = self.db.execute(
            "SELECT digest,evidence FROM tm_ocr_packets WHERE worker=? AND sequence=?",
            (packet.worker_session, packet.sequence),
        ).fetchone()
        if prev:
            if prev["digest"] != digest:
                raise ContractError("Conflicting duplicate OCR packet")
            return prev["evidence"]
        cutoff = self.cutoff(frame.seq)
        for line in packet.lines:
            if line.surface_id and self.node(line.surface_id, cutoff)["kind"] != "surface":
                raise ContractError("OCR surface_id must reference a visible surface")
        record = self.store.derive(
            source=frame.source,
            kind="external_ocr",
            text="\n".join(line.text for line in packet.lines),
            parents=[frame.id],
            snapshot=Snapshot(frame.end, frame.seq),
            payload=packet.model_dump(),
        )
        with self.db:
            rev = self._publish("ocr", now, epoch)
            self.db.execute(
                "INSERT INTO tm_ocr_packets VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    packet.worker_session,
                    packet.sequence,
                    digest,
                    frame.id,
                    frame.seq,
                    frame.payload["frame_number"],
                    rev,
                    record.id,
                    body,
                ),
            )
            self.db.execute(
                "INSERT INTO tm_indexes VALUES(?,?,?,?,?,?)",
                (record.id, rev, frame.seq, record.text, "[]", "ocr"),
            )
        return record.id

    def ocr_view(self, cutoff, *, frame_number, max_age_frames=60):
        rows = self.db.execute(
            "SELECT * FROM tm_ocr_packets WHERE revision<=? AND source_seq<=? "
            "ORDER BY source_seq DESC,revision DESC",
            (cutoff.revision, cutoff.source_seq),
        ).fetchall()
        lines, seen = [], set()
        for row in rows:
            for line in json.loads(row["payload"])["lines"]:
                key = (row["worker"], line["track_id"])
                if key in seen:
                    continue
                seen.add(key)
                if line["ended"]:
                    continue
                lines.append(
                    {
                        **line,
                        "worker_session": row["worker"],
                        "evidence_id": row["evidence"],
                        "observed_frame": row["frame_number"],
                        "stale": frame_number - row["frame_number"] > max_age_frames,
                    }
                )
        return lines[:64]
