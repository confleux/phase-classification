"""Training hyperparameter configuration."""

from typing import Literal

from pydantic import Field

from phase_classification.config.base import StrictModel
from phase_classification.heads import HEAD_TUPLE_INDEX

OPTIMIZERS = ("sgd", "adam", "adamw")

SCHEDULERS = ("none", "step", "plateau", "cosine")

LOSSES = ("cross_entropy", "focal")

CLASS_WEIGHTING = ("none", "inverse")

# Same head names as the label tuple order; usable in any order in configs.
HEADS = tuple(HEAD_TUPLE_INDEX)


class TrainingConfig(StrictModel):
    epochs: int = Field(default=50, ge=1)
    early_stopping_patience: int | None = Field(default=None, ge=1)
    optimizer: Literal[OPTIMIZERS] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)
    scheduler: Literal[SCHEDULERS] = "none"
    step_size: int = Field(default=10, ge=1)
    gamma: float = Field(default=0.1, ge=0.0, le=1.0)
    loss: Literal[LOSSES] = "cross_entropy"
    # "inverse": per-class weights proportional to inverse class frequency in
    # the train split (weighted CE / pos_weight in BCE). Computed at run start.
    class_weighting: Literal[CLASS_WEIGHTING] = "none"
    heads: list[Literal[HEADS]] = Field(default_factory=lambda: ["phase"])
    grad_clip_norm: float | None = Field(default=None, gt=0.0)
