"""Bounded data-only contracts; jobs own provenance, models never choose their clocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ..contracts import Contract, GameConfig, Mention, Fact, Relation, Utterance


class MemoryEvent(Contract):
    kind: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1200)
    participants: list[str] = Field(default_factory=list, max_length=8)
    occurrence: str = Field(min_length=1, max_length=100)
    # This is a semantic assertion. It is not independent outcome verification.
    confidence: float = Field(default=0.5, ge=0, le=1)


class RouteObservation(Contract):
    origin: str
    portal: str
    destination: str | None = None
    status: Literal["observed", "attempted", "blocked", "traversed"] = "observed"
    occurrence: str = Field(min_length=1, max_length=100)
    direction_hint: str = Field(default="", max_length=120)
    action_evidence: str | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def traversal(self):
        if self.status == "traversed" and (not self.destination or not self.action_evidence):
            raise ValueError("Traversal needs a destination and an actual offered action receipt")
        return self


class RelationChange(Relation):
    operation: Literal["assert", "retract"] = "assert"


class MemoryDelta(Contract):
    """Only changes; empty collections may be omitted, frame/job identifiers are host-bound."""

    mentions: list[Mention] = Field(default_factory=list, max_length=6)
    facts: list[Fact] = Field(default_factory=list, max_length=12)
    relations: list[RelationChange] = Field(default_factory=list, max_length=8)
    events: list[MemoryEvent] = Field(default_factory=list, max_length=6)
    utterances: list[Utterance] = Field(default_factory=list, max_length=4)
    routes: list[RouteObservation] = Field(default_factory=list, max_length=4)
    current_place: str | None = None
    needs_planning: bool = False


class SceneIndex(Contract):
    """A small index, not another mandatory full-scene caption before every action."""

    mentions: list[Mention] = Field(default_factory=list, max_length=4)
    current_place: str | None = None
    needs_planning: bool = False
    enrich: list[str] = Field(default_factory=list, max_length=3)


class MemoryQuery(Contract):
    kind: Literal["search", "conversation", "events", "place", "route", "inspect"] = "search"
    entity_id: str | None = None
    target_id: str | None = None
    text: str = Field(default="", max_length=300)
    evidence_id: str | None = None
    limit: int = Field(default=6, ge=1, le=12)


class IntentPlan(Contract):
    intent: str = Field(min_length=1, max_length=700)
    focus_ids: list[str] = Field(default_factory=list, max_length=8)
    queries: list[MemoryQuery] = Field(default_factory=list, max_length=3)
    status: Literal["continue", "blocked", "goal_claimed"] = "continue"


class OCRLine(Contract):
    """Bridge input, NOT an OCR implementation. IDs belong to the OCR worker's session."""

    track_id: str = Field(min_length=1, max_length=100)
    occurrence: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=4000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    surface_id: str | None = None
    box: list[float] | None = Field(default=None, min_length=4, max_length=4)
    ended: bool = False

    @field_validator("box")
    @classmethod
    def box_check(cls, b):
        if b is not None and not 0 <= b[0] < b[2] <= 1:
            raise ValueError("Invalid horizontal bounds")
        if b is not None and not 0 <= b[1] < b[3] <= 1:
            raise ValueError("Invalid vertical bounds")
        return b


class OCRPacket(Contract):
    worker_session: str = Field(min_length=1, max_length=100)
    source_frame_id: str
    sequence: int = Field(ge=0, strict=True)
    lines: list[OCRLine] = Field(default_factory=list, max_length=64)
    # An empty result is not evidence that all previously seen text disappeared.
    complete_surface_scan: bool = False


@dataclass(frozen=True)
class MemoryCutoff:
    revision: int
    source_seq: int


@dataclass(frozen=True)
class JobBasis:
    job_id: str
    source_frame_id: str
    source_seq: int
    frame_number: int
    knowledge_revision: int
    ontology_version: int
    episode: str
    task_revision: int
    scene_revision: int
    intent_revision: int
    offered_frames: tuple[str, ...]
    offered_entities: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    # Enrichment is bound to host-minted IDs from the committed index.
    targets: tuple[str, ...] = ()
    action_step: int = 0


class AsyncSettings(Contract):
    total_slots: int = Field(default=2, ge=2, le=8)
    background_slots: int = Field(default=1, ge=1, le=6)
    max_queue: int = Field(default=12, ge=2, le=100)
    index_max_tokens: int = Field(default=256, ge=64, le=1024)
    enrich_max_tokens: int = Field(default=256, ge=64, le=2048)
    plan_max_tokens: int = Field(default=1024, ge=64, le=8192)
    actor_max_tokens: int = Field(default=32, ge=16, le=128)
    max_fanout: int = Field(default=2, ge=0, le=3)
    index_every_actions: int = Field(default=4, ge=1, le=1000)
    index_min_interval_s: float = Field(default=0.25, ge=0, le=60)
    index_refresh_s: float = Field(default=12, gt=0, le=300)
    thumbnail_change: float = Field(default=0.08, ge=0, le=1)
    scene_change: float = Field(default=0.5, gt=0, le=1)
    plan_max_age_frames: int = Field(default=180, ge=0, le=10000)
    plan_max_age_actions: int = Field(default=6, ge=0, le=100)
    intent_max_actions: int = Field(default=12, ge=1, le=1000)
    initial_intent: str = Field(
        default="Make one useful, reversible step toward the goal using the current pixels. Wait if uncertain.",
        min_length=1,
        max_length=700,
    )
    require_initial_plan: bool = True
    background_drain_s: float = Field(default=95, gt=0, le=1000)
    hot_entities: int = Field(default=16, ge=1, le=200)
    warm_entities: int = Field(default=128, ge=1, le=2000)
    volatile_age_frames: int = Field(default=60, ge=0, le=10000)
    max_views_per_entity: int = Field(default=4, ge=1, le=16)
    max_important_frames: int = Field(default=8, ge=1, le=64)
    # Stable definitions are passed as a prefix, not claimed to be cached by the model.
    semantic_context_chars: int = Field(default=12000, ge=2000, le=200000)
    planner_context_chars: int = Field(default=16000, ge=2000, le=500000)
    actor_context_chars: int = Field(default=10000, ge=2000, le=200000)
    stream_responses: bool = True
    require_returned_model: bool = True
    max_response_bytes: int = Field(default=2_000_000, ge=1024, le=8_000_000)
    ocr_max_age_frames: int = Field(default=60, ge=0, le=10000)
    max_ocr_queue: int = Field(default=32, ge=1, le=1000)
    actor_image_history: int = Field(default=2, ge=1, le=8)
    ocr_backend: Literal["none", "hunyuan", "glm", "rapid"] = "none"
    ocr_model_paths: dict[Literal["hunyuan", "glm"], str] = Field(default_factory=dict)
    ocr_transformers_path: str | None = None
    ocr_device: str = Field(default="cuda:0", pattern=r"^cuda:[0-9]+$")
    ocr_max_tokens: int = Field(default=128, ge=16, le=512)

    @model_validator(mode="after")
    def capacity(self):
        if self.background_slots >= self.total_slots:
            raise ValueError("Reserve at least one dispatch slot for the actor")
        if self.warm_entities < self.hot_entities:
            raise ValueError("warm_entities must be at least hot_entities")
        if self.ocr_backend in ("hunyuan", "glm") and not self.ocr_model_paths.get(self.ocr_backend):
            raise ValueError("Generative OCR needs an explicitly provisioned local model path")
        return self


class AsyncConfig(Contract):
    game: GameConfig
    background: AsyncSettings = Field(default_factory=AsyncSettings)

    @model_validator(mode="after")
    def limits(self):
        if self.background.actor_image_history > self.game.history_frames:
            raise ValueError("Retain enough game.history_frames for actor_image_history")
        if self.background.actor_image_history > self.game.max_images:
            raise ValueError("actor_image_history exceeds image submission cap")
        if self.game.baseline != "world":
            raise ValueError("Use the retained stepped baselines for recent/world comparisons")
        if self.game.compile_at_start:
            raise ValueError("Schema compilation is a stopped-run operation in background mode")
        if self.background.ocr_backend == "rapid" and not self.game.ocr_config_path:
            raise ValueError("RapidOCR needs the existing game.ocr_config_path")
        if self.game.ocr_config_path and self.background.ocr_backend != "rapid":
            raise ValueError("Do not silently ignore or mix the old RapidOCR configuration")
        return self
