"""Training hyperparameter configuration."""

from typing import Literal

from pydantic import Field

from phase_classification.config.base import StrictModel

OPTIMIZERS = ("sgd", "adam", "adamw")
OPTIMIZER_STR = ("sgd", "adam", "adamw")

SCHEDULERS = ("none", "step", "plateau", "cosine")
SCHEDULER_STR = ("none", "step", "plateau", "cosine")

LOSSES = ("cross_entropy", "focal")
LOSS_STR = ("cross_entropy", "focal")

CLASS_WEIGHTING = ("none", "inverse")
CLASS_WEIGHTING_STR = ("none", "inverse")

HEADS = ("phase", "tool", "verb", "target", "triplet")
HEAD_STR = ("phase", "tool", "verb", "target", "triplet")


class TrainingConfig(StrictModel):
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
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

    @property
    def effective_optimizer(self) -> str:
        return self.optimizer


# Re-export string tuples.
ALL_OPTIMIZERS = OPTIMIZER_STR
ALL_SCHEDULERS = SCHEDULER_STR
ALL_LOSSES = LOSS_STR
