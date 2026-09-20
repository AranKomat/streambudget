from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import Field, model_validator

from .types import StrictModel


class Prices(StrictModel):
    input_per_million: float | None = Field(default=None, ge=0)
    output_per_million: float | None = Field(default=None, ge=0)
    cached_input_per_million: float | None = Field(default=None, ge=0)

    @property
    def known(self) -> bool:
        return self.input_per_million is not None and self.output_per_million is not None


class ModelConfig(StrictModel):
    kind: Literal["mock", "chat"] = "mock"
    model: str = "synthetic-pixel-fixture"
    revision: str = "unverified"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "VLM_API_KEY"
    supports_images: bool = True
    timeout_s: float = Field(default=60, gt=0, le=3600)
    max_retries: int = Field(default=1, ge=0, le=5)
    retry_timeouts: bool = False
    json_mode: bool = True
    max_output_tokens: int = Field(default=1200, gt=0, le=64000)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    image_detail: Literal["auto", "low", "high"] = "auto"
    image_token_reserve: int = Field(default=2048, ge=1)
    audio_per_minute_usd: float | None = Field(default=None, ge=0)
    temperature: float | None = 0
    prices: Prices = Field(default_factory=Prices)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    expected_provider_names: list[str] = Field(default_factory=list, max_length=32)
    allow_insecure_remote: bool = False

    @model_validator(mode="after")
    def safe_endpoint(self):
        u = urlparse(self.base_url)
        if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.query:
            raise ValueError("base_url must be an HTTP(S) origin/path without credentials or query")
        if u.scheme == "http" and u.hostname not in {"localhost", "127.0.0.1", "::1"}:
            if not self.allow_insecure_remote:
                raise ValueError("Remote HTTP requires allow_insecure_remote=true; prefer HTTPS")
        forbidden = {"model", "messages", "stream", "max_tokens", "max_completion_tokens"}
        if forbidden.intersection(self.extra_body):
            raise ValueError("extra_body cannot override model, messages, stream, or token caps")
        return self


class PolicyConfig(StrictModel):
    mode: Literal["adaptive", "fixed", "motion", "recent_only"] = "adaptive"
    min_interval_s: float = Field(default=0.5, ge=0)
    fixed_interval_s: float = Field(default=1, gt=0)
    refresh_interval_s: float = Field(default=8, gt=0)
    monitor_interval_s: float = Field(default=2, gt=0)
    motion_mean_threshold: float = Field(default=0.012, ge=0, le=1)
    motion_tile_threshold: float = Field(default=0.12, ge=0, le=1)
    audit_probability: float = Field(default=0.01, ge=0, le=1)
    random_seed: int = 7
    recent_frames: int = Field(default=4, ge=1, le=64)
    recent_window_s: float = Field(default=8, gt=0)
    max_image_side: int = Field(default=768, ge=64, le=2048)
    max_inspect_frames: int = Field(default=16, ge=1, le=128)
    max_agent_steps: int = Field(default=6, ge=1, le=30)
    escalation: bool = True
    compact_every: int = Field(default=0, ge=0, le=100)


class SchedulerConfig(StrictModel):
    workers: int = Field(default=2, ge=1, le=32)
    max_queue: int = Field(default=32, ge=1, le=10000)
    deadline_s: float = Field(default=30, gt=0)
    drain_timeout_s: float = Field(default=120, gt=0)


class BudgetConfig(StrictModel):
    max_requests: int = Field(default=300, ge=1)
    max_usd: float | None = Field(default=None, gt=0)


class Config(StrictModel):
    namespace: str = Field(default="local", pattern=r"^[A-Za-z0-9_.-]{1,80}$")
    models: dict[str, ModelConfig] = Field(default_factory=lambda: {"perception": ModelConfig()})
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    lexical_results: int = Field(default=8, ge=1, le=50)
    hash_embeddings: bool = False
    semantic_embeddings: bool = False
    max_watches: int = Field(default=32, ge=1, le=128)

    @model_validator(mode="after")
    def main_role(self):
        if "perception" not in self.models:
            raise ValueError("models.perception is required")
        if self.semantic_embeddings and "embedding" not in self.models:
            raise ValueError("semantic_embeddings requires a models.embedding endpoint")
        if self.budget.max_usd is not None:
            if any(m.kind == "chat" and not (m.audio_per_minute_usd is not None if role == "asr" else m.prices.known)
                   for role, m in self.models.items()):
                raise ValueError("A dollar budget requires explicit prices for every remote model")
        return self

    def role(self, name: str) -> ModelConfig:
        return self.models.get(name, self.models["perception"])

    def public_dict(self) -> dict:
        # Config stores env-variable names, never secret values.
        return self.model_dump()


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    raw = Path(path).read_text(encoding="utf-8")
    # Environment placeholders apply to config values, not runtime evidence.
    raw = os.path.expandvars(raw)
    if "${" in raw:
        raise ValueError("Unresolved ${ENV_VAR} in configuration")
    return Config.model_validate(yaml.safe_load(raw) or {})
