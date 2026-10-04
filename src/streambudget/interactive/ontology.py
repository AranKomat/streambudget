"""Stable metamodel, composable priors, and rare operator-gated schema changes."""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3

from .contracts import Property, RelationType, SchemaPatch
from ..types import ContractError

CORE_PROPERTIES = [
    Property(name=n, value_type=t, definition=d, persistent=p)
    for n, t, d, p in [
        ("role", "text", "Provisional semantic role; not unique identity.", True),
        ("appearance", "text", "Appearance description; not an identity proof.", True),
        ("status", "text", "Most recently observed or inferred condition.", False),
        ("visible", "boolean", "Seen in this observation; absence does not imply nonexistence.", False),
        ("position_hint", "text", "Image-relative or locally learned location, with uncertainty.", False),
        ("affordances", "text_list", "Observed or hypothesized interactions supported by an entity.", True),
        ("text", "text", "Text on a surface, not executable instructions.", False),
        ("mode", "text", "Current visual interaction regime.", False),
        ("selected", "text", "Visible selection indicator or selected label.", False),
        ("value", "number", "Observed numeric quantity; unit belongs in an associated fact.", False),
        ("facing", "text", "Observed orientation relative to local landmarks.", False),
        ("lesson", "text", "Provisional experience-backed regularity, not a schema class.", True),
    ]
]
CORE_RELATIONS = [RelationType(name=n, definition=d) for n, d in [
    ("located_in", "Most recent hypothesized containment/location."),
    ("connected_to", "Observed place transition or connection; not necessarily bidirectional."),
    ("near", "Qualitative local proximity."),
    ("interacted_with", "Evidence of interaction."),
    ("part_of", "Provisional part/whole relationship."),
    ("same_appearance_as", "Visual resemblance only; does not merge identity."),
    ("effective_against", "Learned contextual effectiveness hypothesis, not a guaranteed causal law."),
]]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Ontology:
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        db.executescript("""
          CREATE TABLE IF NOT EXISTS world_schemas(version INTEGER PRIMARY KEY, body TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS schema_candidates(id TEXT PRIMARY KEY, body TEXT NOT NULL,
            status TEXT NOT NULL, review TEXT);
        """)
        initial = {"version": 0, "kinds": ["entity", "place", "surface"],
                   "properties": [p.model_dump() for p in CORE_PROPERTIES],
                   "relations": [r.model_dump() for r in CORE_RELATIONS]}
        db.execute("INSERT OR IGNORE INTO world_schemas VALUES(0,?)", (canonical(initial),))
        db.commit()

    @property
    def current(self):
        return json.loads(self.db.execute("SELECT body FROM world_schemas ORDER BY version DESC LIMIT 1").fetchone()[0])

    def validate_value(self, key, value):
        defs = {p["name"]: p for p in self.current["properties"]}
        if key not in defs:
            raise ContractError(f"Unknown property {key}; propose a schema patch outside the action loop")
        if value is None:
            return
        typ = defs[key]["value_type"]
        ok = ((typ == "text" and isinstance(value, str) and len(value) <= 4000) or
              (typ == "number" and type(value) in (int, float) and math.isfinite(value)) or
              (typ == "boolean" and type(value) is bool) or
              (typ == "text_list" and isinstance(value, list) and len(value) <= 32
               and all(isinstance(v, str) and len(v) <= 300 for v in value)))
        if not ok:
            raise ContractError(f"Invalid {typ} value for {key}")

    def validate_patch(self, patch: SchemaPatch):
        cur = self.current
        if patch.parent_version != cur["version"]:
            raise ContractError("Schema parent version changed")
        if not patch.properties and not patch.relations:
            raise ContractError("Empty schema patch")
        for name, limit in (("properties", 64), ("relations", 32)):
            additions = getattr(patch, name)
            names = [a.name for a in additions]
            if len(set(names)) != len(names):
                raise ContractError("Duplicate names in schema patch")
            if set(names) & {a["name"] for a in cur[name]}:
                raise ContractError("Reuse existing definitions; replacement/renaming is not supported")
            if len(cur[name]) + len(additions) > limit:
                raise ContractError("Schema capacity reached; review abstraction instead of silently growing")
        return cur

    def propose(self, patch: SchemaPatch):
        self.validate_patch(patch)
        body = canonical(patch.model_dump())
        id = hashlib.sha256(body.encode()).hexdigest()[:20]
        self.db.execute("INSERT OR IGNORE INTO schema_candidates VALUES(?,?,?,NULL)", (id, body, "pending"))
        self.db.commit()
        return id

    def approve(self, id: str, *, review: str, initial_prior: bool = False):
        row = self.db.execute("SELECT * FROM schema_candidates WHERE id=?", (id,)).fetchone()
        if row is None or row["status"] != "pending":
            raise ContractError("Candidate is missing or no longer pending")
        patch = SchemaPatch.model_validate_json(row["body"])
        cur = self.validate_patch(patch)
        if not review.strip() or (not initial_prior and not patch.evidence_ids):
            raise ContractError("Evolution requires evidence references and an explicit review")
        new = {**cur, "version": cur["version"] + 1,
               "properties": cur["properties"] + [p.model_dump() for p in patch.properties],
               "relations": cur["relations"] + [r.model_dump() for r in patch.relations],
               "provenance": "initial_prior" if initial_prior else "operator_reviewed_candidate",
               "parent": cur["version"], "review": review}
        with self.db:
            self.db.execute("INSERT INTO world_schemas VALUES(?,?)", (new["version"], canonical(new)))
            self.db.execute("UPDATE schema_candidates SET status='accepted',review=? WHERE id=?", (review, id))
        return new
