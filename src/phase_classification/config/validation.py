"""Validation / evaluation configuration."""

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


class ValidationConfig(StrictModel):
    metrics: list[Literal[METRICS]] = Field(default_factory=lambda: list(METRICS))
