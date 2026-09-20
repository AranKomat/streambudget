from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractError(ValueError):
    """Invalid time, evidence, or tool contract; never silently relax these checks."""


class NotAvailable(ContractError):
    pass


def valid_time(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ContractError("Time must be a finite, nonnegative number of seconds")
    return float(value)


@dataclass(frozen=True)
class Snapshot:
    as_of: float
    max_source_seq: int

    def __post_init__(self):
        valid_time(self.as_of)
        if self.max_source_seq < 0:
            raise ContractError("Negative source sequence")


@dataclass(frozen=True)
class Evidence:
    id: str
    seq: int
    source: str
    kind: str
    start: float
    end: float
    available_at: float
    text: str
    payload: dict[str, Any]
    parents: tuple[str, ...]
    input_end: float
    input_available_at: float
    leaf_seq: int
    created_wall: float

    def visible(self, snapshot: Snapshot) -> bool:
        return (self.input_end <= snapshot.as_of and self.input_available_at <= snapshot.as_of
                and self.leaf_seq <= snapshot.max_source_seq)

    def view(self) -> dict[str, Any]:
        # Do not expose filesystem paths or full base64 payloads to planners.
        return {"id": self.id, "source": self.source, "kind": self.kind,
                "start": self.start, "end": self.end, "text": self.text,
                "parents": list(self.parents)}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SensorPredicate(StrictModel):
    key: str
    op: Literal["eq", "ne", "gt", "ge", "lt", "le"] = "eq"
    value: str | float | bool
    kind: Literal["sensor", "detector", "audio_event"] = "sensor"


class Watch(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    source: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,100}$")
    goal: str = Field(min_length=1, max_length=4000)
    created_at: float = Field(default=0, ge=0)
    expires_at: float | None = Field(default=None, ge=0)
    dwell_seconds: float = Field(default=0, ge=0, le=86400)
    repeat: bool = False
    cooldown_seconds: float = Field(default=10, ge=0)
    min_confidence: float = Field(default=0.7, ge=0, le=1)
    max_observation_gap: float = Field(default=15, gt=0)
    sensor_predicate: SensorPredicate | None = None

    @model_validator(mode="after")
    def times(self):
        if self.expires_at is not None and self.expires_at < self.created_at:
            raise ValueError("Watch expires before creation")
        return self


class Check(StrictModel):
    watch_id: str
    status: Literal["yes", "no", "unknown"]
    confidence: float = Field(ge=0, le=1)
    detail: str = Field(default="", max_length=4000)


class Perception(StrictModel):
    caption: str = Field(max_length=8000)
    facts: dict[str, str] = Field(default_factory=dict)
    checks: list[Check] = Field(default_factory=list, max_length=128)


@dataclass
class WatchState:
    watch: Watch
    positive_since: float | None = None
    last_observed: float | None = None
    last_status: str = "unknown"
    last_emit: float | None = None
    done: bool = False

    def observe(self, check: Check, observed_at: float) -> bool:
        """Dwell is sampled continuity, not proof of unobserved behavior between frames."""
        w = self.watch
        if self.done or observed_at < w.created_at:
            return False
        if w.expires_at is not None and observed_at > w.expires_at:
            self.done = True
            return False
        # Slow concurrent jobs must not rewind monitor state.
        if self.last_observed is not None and observed_at <= self.last_observed:
            return False
        if self.last_observed is not None and observed_at - self.last_observed > w.max_observation_gap:
            self.positive_since = None
        self.last_observed = observed_at
        positive = check.status == "yes" and check.confidence >= w.min_confidence
        self.last_status = check.status if check.confidence >= w.min_confidence else "unknown"
        if not positive:
            self.positive_since = None
            return False
        if self.positive_since is None:
            self.positive_since = observed_at
        if observed_at - self.positive_since < w.dwell_seconds:
            return False
        if self.last_emit is not None and observed_at - self.last_emit < w.cooldown_seconds:
            return False
        self.last_emit = observed_at
        if not w.repeat:
            self.done = True
        return True


@dataclass
class Answer:
    question_id: str
    text: str
    evidence_ids: list[str]
    as_of: float
    status: str = "ok"
    elapsed_s: float = 0.0
    tool_steps: list[dict[str, Any]] = field(default_factory=list)
