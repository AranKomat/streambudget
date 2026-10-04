"""Small, fixed wire contracts. Domain facts are data, never executable plugins."""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True, validate_default=True)


class Region(Contract):
    # A retrieval annotation, not a certified object detector or identity claim.
    frame_id: str
    box: list[float] = Field(min_length=4, max_length=4)

    @field_validator("box")
    @classmethod
    def box_valid(cls, b):
        if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            raise ValueError("box must be normalized, ordered xyxy")
        return b


class Mention(Contract):
    ref: str = Field(pattern=r"^new:[A-Za-z0-9_-]{1,40}$|^e_[a-f0-9]{16}$")
    kind: Literal["entity", "place", "surface"]
    label: str = Field(min_length=1, max_length=240)
    association: Literal["new", "continuity", "reidentified", "uncertain"] = "new"
    confidence: float = Field(default=0.5, ge=0, le=1)
    region: Region | None = None


class Fact(Contract):
    subject: str
    key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    value: Any
    confidence: float = Field(default=0.5, ge=0, le=1)
    basis: Literal["visible", "inferred"] = "inferred"


class Relation(Contract):
    subject: str
    predicate: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    target: str
    confidence: float = Field(default=0.5, ge=0, le=1)


class Event(Contract):
    kind: str = Field(max_length=64)
    text: str = Field(min_length=1, max_length=2000)
    participants: list[str] = Field(default_factory=list, max_length=8)


class Utterance(Contract):
    # Attribution may be unknown. Never infer a speaker solely from crop similarity.
    speaker: str | None = None
    partners: list[str] = Field(default_factory=list, max_length=8)
    surface: str | None = None
    text: str = Field(min_length=1, max_length=4000)
    occurrence: str = Field(max_length=80)
    attribution_confidence: float = Field(default=0, ge=0, le=1)


class ObservationPatch(Contract):
    frame_id: str
    summary: str = Field(max_length=2000)
    mentions: list[Mention] = Field(default_factory=list, max_length=20)
    facts: list[Fact] = Field(default_factory=list, max_length=48)
    relations: list[Relation] = Field(default_factory=list, max_length=20)
    events: list[Event] = Field(default_factory=list, max_length=16)
    utterances: list[Utterance] = Field(default_factory=list, max_length=12)
    current_place: str | None = None
    needs_planning: bool = False


class ActionChoice(Contract):
    action_id: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,39}$")


class Plan(Contract):
    intent: str = Field(min_length=1, max_length=2000)
    focus_ids: list[str] = Field(default_factory=list, max_length=16)
    search: str | None = Field(default=None, max_length=300)
    inspect_ids: list[str] = Field(default_factory=list, max_length=4)
    status: Literal["continue", "blocked", "goal_claimed"] = "continue"


class Property(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    value_type: Literal["text", "number", "boolean", "text_list"] = "text"
    definition: str = Field(min_length=1, max_length=400)
    persistent: bool = False


class RelationType(Contract):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    definition: str = Field(min_length=1, max_length=400)


class SchemaPatch(Contract):
    parent_version: int = Field(ge=0)
    properties: list[Property] = Field(default_factory=list, max_length=16)
    relations: list[RelationType] = Field(default_factory=list, max_length=8)
    reason: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=16)


class ActionSpec(Contract):
    id: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,39}$")
    button: Literal["up", "down", "left", "right", "a", "b", "start", "select", "wait"]
    press_frames: int = Field(default=4, ge=1, le=24, strict=True)
    release_frames: int = Field(default=4, ge=1, le=24, strict=True)


class Endpoint(Contract):
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "SET_EXACT_MODEL_ID"
    api_key_env: str | None = None
    billing: Literal["local", "metered"] = "local"
    input_per_million: float | None = Field(default=None, ge=0)
    output_per_million: float | None = Field(default=None, ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)
    reservation_usd: float = Field(default=0.1, gt=0)
    timeout_s: float = Field(default=90, gt=0, le=600)
    max_output_tokens: int = Field(default=4096, ge=16, le=32768)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    response_format: Literal["json_object", "json_schema", "none"] = "json_object"
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def safe_endpoint(self):
        from urllib.parse import urlsplit
        u = urlsplit(self.base_url)
        if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment:
            raise ValueError("Use a plain HTTP(S) base URL without credentials/query/fragment")
        loopback = u.hostname in ("127.0.0.1", "localhost", "::1")
        if self.billing == "local" and not loopback:
            raise ValueError("local billing is restricted to loopback; tunnel a self-hosted remote server")
        if not loopback and u.scheme != "https":
            raise ValueError("Remote metered endpoints require HTTPS")
        if self.api_key_env and not re.fullmatch(r"[A-Z_][A-Z0-9_]*", self.api_key_env):
            raise ValueError("api_key_env must be an environment variable name, not a key")
        if self.billing == "metered" and (self.input_per_million is None or self.output_per_million is None):
            raise ValueError("Metered endpoints need configured current input/output rates")
        allowed = {"temperature", "top_p", "seed", "reasoning_effort", "chat_template_kwargs"}
        if set(self.extra_body) - allowed:
            raise ValueError("extra_body can only configure decoding/reasoning; not request routing or messages")
        return self


class GameConfig(Contract):
    source: str = Field(default="game", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    goal: str = Field(default="Make progress through the environment.", min_length=1, max_length=6000)
    backend: Literal["fixture", "chat"] = "chat"
    endpoints: dict[str, Endpoint] = Field(default_factory=lambda: {"main": Endpoint()})
    roles: dict[str, str] = Field(default_factory=lambda: {
        "extract": "main", "plan": "main", "act": "main", "compile": "main"})
    actions: list[ActionSpec] = Field(default_factory=lambda: [
        ActionSpec(id=b.upper(), button=b) for b in
        ("up", "down", "left", "right", "a", "b", "start", "select", "wait")])
    role_output_limits: dict[str, int] = Field(default_factory=lambda: {
        "extract": 4096, "plan": 2048, "act": 64, "compile": 2048})
    max_steps: int = Field(default=10, ge=1, le=100000)
    max_calls: int = Field(default=40, ge=1, le=1000000)
    max_estimated_usd: float = Field(default=5, gt=0)
    wall_limit_s: float = Field(default=900, gt=0, le=172800)
    plan_every: int = Field(default=8, ge=1, le=10000)
    history_frames: int = Field(default=3, ge=1, le=12)
    max_context_chars: int = Field(default=80000, ge=2000, le=500000)
    max_entities_in_context: int = Field(default=40, ge=1, le=500)
    max_images: int = Field(default=6, ge=1, le=16)
    view_scale: int = Field(default=3, ge=1, le=4)
    min_fact_confidence: float = Field(default=0.5, ge=0, le=1)
    min_association_confidence: float = Field(default=0.85, ge=0, le=1)
    max_entities: int = Field(default=3000, ge=10, le=100000)
    max_storage_mb: int = Field(default=512, ge=10, le=100000)
    max_retrieval_rounds: int = Field(default=2, ge=0, le=4)
    compile_at_start: bool = False
    baseline: Literal["world", "recent"] = "world"
    stop_on_goal_claim: bool = True
    # V0 is stepped. No claim of real-time control while remote requests run.
    timing: Literal["stepped"] = "stepped"

    @model_validator(mode="after")
    def references(self):
        if len({a.id for a in self.actions}) != len(self.actions):
            raise ValueError("Duplicate action IDs")
        if not self.actions or len(self.actions) > 32 or not any(a.button == "wait" for a in self.actions):
            raise ValueError("Provide 1–32 actions including WAIT")
        if set(self.role_output_limits) != {"extract", "plan", "act", "compile"} or any(
            type(v) is not int or not 16 <= v <= 32768 for v in self.role_output_limits.values()):
            raise ValueError("Invalid role output limits")
        if set(self.roles) != {"extract", "plan", "act", "compile"}:
            raise ValueError("Exactly extract/plan/act/compile roles are required")
        if any(v not in self.endpoints for v in self.roles.values()):
            raise ValueError("Unknown endpoint alias")
        return self
