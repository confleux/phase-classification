"""Validation / evaluation configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from phase_classification.config.base import StrictModel

METRICS = (
    "accuracy",
    "accuracy_per_phase",
    "f1_macro",
    "f1_micro",
    "jaccard",
    "edit_score",
)
METRIC_STR = METRICS


class ValidationConfig(StrictModel):
    metrics: list[Literal[METRICS]] = Field(
        default_factory=lambda: ["accuracy", "accuracy_per_phase"]
    )
    checkpoint_path: Path | None = None
    top_k: int = Field(default=1, ge=1)
    batch_size: int | None = Field(default=None, ge=1)


ALL_METRICS = METRIC_STR
