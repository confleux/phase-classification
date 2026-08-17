"""Validation / evaluation configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from phase_classification.config.base import StrictModel

METRICS = (
    "jaccard",
    "f1_macro",
    "f1_per_class",
    "mean_accuracy",
    "accuracy_per_class",
    "edit_score",
)
METRIC_STR = METRICS


class ValidationConfig(StrictModel):
    metrics: list[Literal[METRICS]] = Field(
        default_factory=lambda: [
            "jaccard",
            "f1_macro",
            "f1_per_class",
            "mean_accuracy",
            "accuracy_per_class",
            "edit_score",
        ]
    )
    checkpoint_path: Path | None = None
    top_k: int = Field(default=1, ge=1)
    batch_size: int | None = Field(default=None, ge=1)


ALL_METRICS = METRIC_STR
