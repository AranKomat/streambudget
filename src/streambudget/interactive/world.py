"""Evidence-backed belief store. State estimates are not game truth or action authority."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from ..store import EvidenceStore
from ..types import ContractError, Snapshot
from .contracts import ObservationPatch
from .ontology import Ontology, canonical


@dataclass
class ApplyResult:
    evidence_id: str
    ids: dict[str, str]
    changed: bool
    needs_planning: bool


class World:
    def __init__(self, evidence: EvidenceStore, *, max_entities=3000, fact_threshold=0.5,
                 association_threshold=0.85):
        self.evidence, self.db = evidence, evidence.db
        self.max_entities, self.fact_threshold = max_entities, fact_threshold
        self.association_threshold = association_threshold
        self.ontology = Ontology(self.db)
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS world_entities(id TEXT PRIMARY KEY, kind TEXT NOT NULL,
            label TEXT NOT NULL, created_evidence TEXT NOT NULL, latest_evidence TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS world_mentions(seq INTEGER PRIMARY KEY, entity TEXT, evidence TEXT);
          CREATE TABLE IF NOT EXISTS world_facts(seq INTEGER PRIMARY KEY, entity TEXT NOT NULL,
            key TEXT NOT NULL, value TEXT NOT NULL, confidence REAL NOT NULL, basis TEXT NOT NULL,
            evidence TEXT NOT NULL, frame_seq INTEGER NOT NULL);
          CREATE INDEX IF NOT EXISTS fact_entity ON world_facts(entity,key,frame_seq);
          CREATE TABLE IF NOT EXISTS world_relations(seq INTEGER PRIMARY KEY, subject TEXT NOT NULL,
            predicate TEXT NOT NULL, target TEXT NOT NULL, confidence REAL NOT NULL, evidence TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS world_messages(seq INTEGER PRIMARY KEY, thread TEXT NOT NULL,
            speaker TEXT, text TEXT NOT NULL, evidence TEXT NOT NULL, attribution REAL NOT NULL);
          CREATE TABLE IF NOT EXISTS world_places(seq INTEGER PRIMARY KEY, place TEXT NOT NULL,
            evidence TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS world_applied(frame_id TEXT PRIMARY KEY, patch_hash TEXT NOT NULL,
            evidence_id TEXT NOT NULL, id_map TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS world_dialogue(key TEXT PRIMARY KEY, text TEXT NOT NULL,
            last_frame TEXT NOT NULL, occurrence TEXT NOT NULL DEFAULT '');
        """)
        # Explicit additive migration for early local 0.2 development databases.
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(world_dialogue)")}
        if "occurrence" not in columns:
            self.db.execute("ALTER TABLE world_dialogue ADD COLUMN occurrence TEXT NOT NULL DEFAULT ''")
        for table in ("world_mentions", "world_relations", "world_places", "world_messages"):
            columns = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if "frame_seq" not in columns:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN frame_seq INTEGER NOT NULL DEFAULT 0")
                self.db.execute(f"""UPDATE {table} SET frame_seq=COALESCE(
                    (SELECT e.seq FROM world_applied a JOIN evidence e ON e.id=a.frame_id
                     WHERE a.evidence_id={table}.evidence),0)""")
        self.db.commit()

    def _visible(self, id, snapshot):
        try:
            self.evidence.get(id, snapshot)
            return True
        except ContractError:
            return False

    def entity_ids(self):
        return {r[0] for r in self.db.execute("SELECT id FROM world_entities")}

    def apply(self, patch: ObservationPatch, snapshot: Snapshot, *, offered_frames: set[str],
              offered_entities: set[str], consumed_evidence: list[str]) -> ApplyResult:
        import hashlib
        if patch.frame_id not in offered_frames:
            raise ContractError("Observation refers to a frame not supplied to the model")
        frame = self.evidence.get(patch.frame_id, snapshot)
        if frame.kind != "frame":
            raise ContractError("Observation anchor is not a frame")
        body_hash = hashlib.sha256(canonical(patch.model_dump()).encode()).hexdigest()
        prev = self.db.execute("SELECT * FROM world_applied WHERE frame_id=?", (patch.frame_id,)).fetchone()
        if prev:
            if prev["patch_hash"] != body_hash:
                raise ContractError("Conflicting second interpretation; use a new explicit review, not overwrite")
            return ApplyResult(prev["evidence_id"], json.loads(prev["id_map"]), False, False)
        latest = self.db.execute("SELECT MAX(e.seq) FROM world_applied a JOIN evidence e ON e.id=a.frame_id").fetchone()[0]
        late = latest is not None and frame.seq < latest
        parents = sorted(set([patch.frame_id, *consumed_evidence]))
        for p in parents:
            self.evidence.get(p, snapshot)
        ids: dict[str, str] = {}
        existing = self.entity_ids()
        fresh = []
        for m in patch.mentions:
            if m.ref in ids:
                raise ContractError("Duplicate mention ref")
            if m.ref.startswith("new:"):
                if m.association != "new":
                    raise ContractError("New mentions cannot claim an existing identity")
                id = "e_" + uuid.uuid4().hex[:16]
                fresh.append((id, m))
            else:
                if m.ref not in existing or m.ref not in offered_entities:
                    raise ContractError("Entity association not grounded in offered context")
                if m.association not in ("continuity", "reidentified") or m.confidence < self.association_threshold:
                    raise ContractError("Uncertain identity: create a separate sighting, do not merge")
                row = self.db.execute("SELECT kind FROM world_entities WHERE id=?", (m.ref,)).fetchone()
                if row[0] != m.kind:
                    raise ContractError("Cannot change an instance's metamodel kind")
                id = m.ref
            if m.region and m.region.frame_id not in offered_frames:
                raise ContractError("Region must address its actual supplied source frame")
            ids[m.ref] = id
        if len(existing) + len(fresh) > self.max_entities:
            raise ContractError("Entity capacity reached; stop and review instead of dropping identities")

        def resolve(ref):
            if ref in ids:
                return ids[ref]
            if ref in existing and ref in offered_entities:
                return ref
            raise ContractError("Unknown/unoffered entity reference: " + str(ref))

        facts = [(resolve(f.subject), f) for f in patch.facts]
        for _, f in facts:
            self.ontology.validate_value(f.key, f.value)
        predicates = {r["name"] for r in self.ontology.current["relations"]}
        rels = []
        for r in patch.relations:
            if r.predicate not in predicates:
                raise ContractError("Unknown relation; schema evolution is separate from observations")
            rels.append((resolve(r.subject), resolve(r.target), r))
        for e in patch.events:
            for p in e.participants:
                resolve(p)
        messages = [(resolve(u.speaker) if u.speaker else None,
                     sorted(resolve(p) for p in u.partners),
                     resolve(u.surface) if u.surface else None, u) for u in patch.utterances]
        place = resolve(patch.current_place) if patch.current_place else None
        if place:
            kinds = {id: m.kind for id, m in fresh}
            if place not in kinds:
                kinds[place] = self.db.execute("SELECT kind FROM world_entities WHERE id=?", (place,)).fetchone()[0]
            if kinds[place] != "place":
                raise ContractError("current_place must reference a Place")

        # Validation above has no effects. One immutable interpretation captures all consumed inputs.
        search_text = "\n".join([patch.summary, *ids.values(), *[m.label for m in patch.mentions],
            *[e.text for e in patch.events], *[u.text for u in patch.utterances],
            *[f"{f.key}: {canonical(f.value)}" for f in patch.facts]])
        if len(search_text) > 90000:
            raise ContractError("Interpretation too large")
        record = self.evidence.derive(source=frame.source, kind="world_observation", text=search_text,
            parents=parents, snapshot=snapshot, payload={"interpretation": patch.model_dump(),
            "ontology_version": self.ontology.current["version"], "id_map": ids,
            "observed_frame_id": frame.id, "observed_frame_seq": frame.seq,
            "observed_at": frame.end,
            "epistemic_status": "model_interpretation_not_ground_truth"})
        # Compare dialogue to its predecessor in source order, not the last worker to finish.
        predecessor = self.db.execute("""SELECT a.evidence_id FROM world_applied a
            JOIN evidence e ON e.id=a.frame_id WHERE e.seq<? ORDER BY e.seq DESC LIMIT 1""",
            (frame.seq,)).fetchone()
        previous_dialogue = {}
        if predecessor:
            prior = self.evidence.get(predecessor[0]).payload
            mapping = prior["id_map"]
            for u in prior["interpretation"].get("utterances", []):
                speaker = mapping.get(u["speaker"], u["speaker"])
                partners = sorted(mapping.get(p, p) for p in u["partners"])
                surface = mapping.get(u["surface"], u["surface"])
                previous_dialogue[canonical([speaker, partners, surface])] = (u["text"], u["occurrence"])
        with self.db:
            for id, m in fresh:
                self.db.execute("INSERT INTO world_entities VALUES(?,?,?,?,?)",
                                (id, m.kind, m.label, record.id, record.id))
            for m in patch.mentions:
                self.db.execute("INSERT INTO world_mentions(entity,evidence,frame_seq) VALUES(?,?,?)",
                                (ids[m.ref], record.id, frame.seq))
                newest = self.db.execute("SELECT MAX(frame_seq) FROM world_mentions WHERE entity=?", (ids[m.ref],)).fetchone()[0]
                if frame.seq == newest:
                    self.db.execute("UPDATE world_entities SET latest_evidence=? WHERE id=?", (record.id, ids[m.ref]))
            for id, f in facts:
                if f.confidence < self.fact_threshold:
                    continue  # Full low-confidence interpretation remains in immutable evidence.
                self.db.execute("INSERT INTO world_facts(entity,key,value,confidence,basis,evidence,frame_seq) "
                    "VALUES(?,?,?,?,?,?,?)", (id, f.key, canonical(f.value), f.confidence, f.basis, record.id, frame.seq))
            for a, b, r in rels:
                if r.confidence < self.fact_threshold:
                    continue
                self.db.execute("INSERT INTO world_relations(subject,predicate,target,confidence,evidence,frame_seq) "
                    "VALUES(?,?,?,?,?,?)", (a, r.predicate, b, r.confidence, record.id, frame.seq))
            if place:
                self.db.execute("INSERT INTO world_places(place,evidence,frame_seq) VALUES(?,?,?)", (place, record.id, frame.seq))
            active = set()
            for speaker, partners, surface, u in messages:
                thread = canonical([speaker, partners, surface])
                active.add(thread)
                if previous_dialogue.get(thread) != (u.text, u.occurrence):
                    self.db.execute("INSERT INTO world_messages(thread,speaker,text,evidence,attribution,frame_seq) "
                        "VALUES(?,?,?,?,?,?)", (thread, speaker, u.text, record.id, u.attribution_confidence, frame.seq))
                if not late:
                    self.db.execute("INSERT OR REPLACE INTO world_dialogue VALUES(?,?,?,?)",
                                (thread, u.text, patch.frame_id, u.occurrence))
            for r in ([] if late else self.db.execute("SELECT key FROM world_dialogue").fetchall()):
                if r[0] not in active:
                    self.db.execute("DELETE FROM world_dialogue WHERE key=?", (r[0],))
            self.db.execute("INSERT INTO world_applied VALUES(?,?,?,?)",
                            (patch.frame_id, body_hash, record.id, canonical(ids)))
        return ApplyResult(record.id, ids, True, patch.needs_planning and not late)

    def observed_at(self, evidence_id):
        e = self.evidence.get(evidence_id)
        return e.payload.get("observed_at", e.end)

    def view(self, snapshot: Snapshot, *, focus=(), max_entities=40) -> dict[str, Any]:
        rows = [dict(r) for r in self.db.execute("SELECT * FROM world_entities")
                if self._visible(r["created_evidence"], snapshot)]
        for r in rows:
            visible_mentions = [self.evidence.get(m[0]) for m in self.db.execute(
                "SELECT evidence FROM world_mentions WHERE entity=? ORDER BY frame_seq DESC,seq DESC", (r["id"],))
                if self._visible(m[0], snapshot)]
            last = next(iter(visible_mentions), self.evidence.get(r["created_evidence"]))
            r["visible_last_time"], r["visible_last_evidence"] = self.observed_at(last.id), last.id
        place_row = next((dict(r) for r in self.db.execute("SELECT * FROM world_places ORDER BY frame_seq DESC,seq DESC")
                          if self._visible(r["evidence"], snapshot)), None)
        current_place = place_row["place"] if place_row else None
        rows.sort(key=lambda r: (0 if r["id"] == current_place else 1 if r["id"] in focus else 2,
                                 -r["visible_last_time"], r["id"]))
        selected, omitted = rows[:max_entities], len(rows[max_entities:])
        selected_ids = {r["id"] for r in selected}
        output = []
        for r in selected:
            facts = {}
            for f in self.db.execute("SELECT * FROM world_facts WHERE entity=? ORDER BY frame_seq DESC,seq DESC", (r["id"],)):
                if f["key"] not in facts and self._visible(f["evidence"], snapshot):
                    facts[f["key"]] = {"value": json.loads(f["value"]), "confidence": f["confidence"],
                        "basis": f["basis"], "evidence_id": f["evidence"],
                        "observed_at": self.observed_at(f["evidence"])}
            output.append({"id": r["id"], "kind": r["kind"], "label": r["label"], "facts": facts,
                           "last_mentioned_at": r["visible_last_time"], "evidence_id": r["visible_last_evidence"]})
        rels, seen = [], set()
        for r in self.db.execute("SELECT * FROM world_relations ORDER BY frame_seq DESC,seq DESC"):
            key = ((r["subject"], r["predicate"]) if r["predicate"] == "located_in"
                   else (r["subject"], r["predicate"], r["target"]))
            if r["subject"] in selected_ids and key not in seen and self._visible(r["evidence"], snapshot):
                rels.append({k: r[k] for k in ("subject", "predicate", "target", "confidence", "evidence")})
                seen.add(key)
        place = current_place
        messages = [dict(r) for r in self.db.execute("SELECT * FROM world_messages ORDER BY frame_seq DESC,seq DESC")
                    if self._visible(r["evidence"], snapshot)][:12]
        return {"entities": output, "relations": rels[:80], "current_place": place,
                "current_place_evidence": place_row["evidence"] if place_row else None,
                "current_place_observed_at": self.observed_at(place_row["evidence"]) if place_row else None,
                "recent_conversations": list(reversed(messages)), "omitted_entities": omitted,
                "warning": "All world facts are beliefs with provenance. Old positions/visibility may be stale."}

    def retrieval_record(self, evidence, snapshot):
        """Expose persisted entity handles with a retrieved episode, not just prose.

        This lets the planner focus an old entity next turn without inventing its ID.
        It does not make the old observation or position current.
        """
        if not evidence.visible(snapshot):
            raise ContractError("Retrieved evidence is outside the snapshot")
        view = evidence.view()
        body = evidence.payload.get("interpretation", {})
        mapping = evidence.payload.get("id_map", {})
        refs = set(mapping.values())
        for item in body.get("facts", []):
            refs.add(mapping.get(item["subject"], item["subject"]))
        for item in body.get("relations", []):
            refs.update(mapping.get(item[k], item[k]) for k in ("subject", "target"))
        entities = []
        for id in sorted(refs):
            row = self.db.execute("SELECT * FROM world_entities WHERE id=?", (id,)).fetchone()
            if row and self._visible(row["created_evidence"], snapshot):
                entities.append({"id": id, "kind": row["kind"], "label": row["label"]})
        view["entities"] = entities[:20]
        view["omitted_entity_handles"] = max(0, len(entities)-20)
        return view
