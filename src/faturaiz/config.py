"""Typed configuration loaded from versioned TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    synthetic_base_invoices: int
    synthetic_duplicate_rate: float


@dataclass(frozen=True)
class CandidateConfig:
    max_date_gap_days: int
    max_amount_relative_gap: float
    min_invoice_similarity: float


@dataclass(frozen=True)
class ModelConfig:
    min_validation_precision: float
    false_positive_review_cost_try: float


@dataclass(frozen=True)
class ServiceConfig:
    max_batch_size: int
    artifact_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class Settings:
    project: ProjectConfig
    candidates: CandidateConfig
    model: ModelConfig
    service: ServiceConfig


def load_settings(path: str | Path = "configs/default.toml") -> Settings:
    """Load and validate settings from TOML."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    project = ProjectConfig(**raw["project"])
    candidates = CandidateConfig(**raw["candidates"])
    model = ModelConfig(**raw["model"])
    service_raw = raw["service"]
    service = ServiceConfig(
        max_batch_size=int(service_raw["max_batch_size"]),
        artifact_path=Path(service_raw["artifact_path"]),
        manifest_path=Path(service_raw["manifest_path"]),
    )
    if not 0 < project.synthetic_duplicate_rate < 1:
        raise ValueError("synthetic_duplicate_rate must be between zero and one")
    if candidates.max_date_gap_days <= 0:
        raise ValueError("max_date_gap_days must be positive")
    if not 0 < model.min_validation_precision <= 1:
        raise ValueError("min_validation_precision must be in (0, 1]")
    return Settings(project, candidates, model, service)
