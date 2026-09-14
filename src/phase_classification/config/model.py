"""Model configuration (loose coupling).

Only label-space facts are pinned here. The architecture itself is selected by
``name`` and configured through free-form ``params``, so swapping in a TCN, ViT,
ResNet+CVT, etc. requires no schema change -- a dedicated model registry maps
``(name, params)`` to a torch module in later phases.
"""

from typing import Any, Literal

from pydantic import Field, model_validator

from phase_classification.config.base import StrictModel
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX

HEAD_NAMES = tuple(HEAD_TUPLE_INDEX)
HEAD_LITERAL = Literal[HEAD_NAMES]


class ModelConfig(StrictModel):
    """Reference to an architecture registry entry plus arbitrary knobs.

    ``name`` selects the implementation (e.g. "resnet18"); the model
    registry resolves it with ``params`` passed through unchanged. ``num_classes``
    describes the output label space per head; override only if your dataset or
    variant differs from CholecT50.
    """

    name: str = "resnet18"
    params: dict[str, Any] = Field(default_factory=dict)
    num_classes: dict[HEAD_LITERAL, int] = Field(
        default_factory=lambda: dict(DEFAULT_NUM_CLASSES)
    )

    @model_validator(mode="after")
    def _validate_label_space(self) -> ModelConfig:
        unknown = set(self.num_classes) - set(HEAD_NAMES)
        if unknown:
            raise ValueError(
                f"num_classes keys must be in {HEAD_NAMES}, got {sorted(unknown)}."
            )
        if any(n < 1 for n in self.num_classes.values()):
            raise ValueError("num_classes values must be >= 1.")
        return self
