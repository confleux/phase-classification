"""Shared base types and helpers for config models."""

from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base config model: extra fields are rejected instead of silently ignored."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
    )
