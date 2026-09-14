"""Runtime / hardware / reproducibility configuration."""

from typing import Literal

from pydantic import Field, model_validator

from phase_classification.config.base import StrictModel

DEVICES = ("auto", "cpu", "cuda")

PRECISIONS = ("fp32", "fp16")


class RuntimeConfig(StrictModel):
    device: Literal[DEVICES] = "auto"
    cuda_device: int | None = Field(default=None, ge=0)
    mixed_precision: Literal[PRECISIONS] = "fp32"
    seed: int = Field(default=42, ge=0)

    @model_validator(mode="after")
    def _cuda_requires_device(self) -> RuntimeConfig:
        if self.cuda_device is not None and self.device not in ("auto", "cuda"):
            raise ValueError("cuda_device requires device in ('auto', 'cuda').")
        return self
