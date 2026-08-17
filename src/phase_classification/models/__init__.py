"""Model building: expose the registry as the public data-layer entrypoint."""

from phase_classification.models.registry import (
    ModelRegistryError,
    build_model,
)

__all__ = ["ModelRegistryError", "build_model"]
