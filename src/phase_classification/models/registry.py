"""Model registry: map ``(ModelConfig.name, params)`` to a torch module.

Config uses loose coupling: ``model.name`` selects an implementation here and
``model.params`` carries architecture-specific knobs unchanged.
"""

from typing import Any

from phase_classification.config.model import ModelConfig
from phase_classification.models.net import PhaseClassifier, ResNet18Classifier


class ModelRegistryError(Exception):
    """Raised when no model implementation matches the configured name."""


def _build_phase_classifier(cfg: ModelConfig, heads: list[str]) -> PhaseClassifier:
    params: dict[str, Any] = dict(cfg.params)
    in_channels = params.pop("in_channels", 3)
    hidden = params.pop("hidden", 64)
    features = params.pop("features", 256)
    if params:
        unknown = ", ".join(sorted(params))
        raise ModelRegistryError(f"Unsupported params for 'phase_classifier': {unknown}")
    return PhaseClassifier(
        num_classes=cfg.num_classes,
        heads=heads,
        in_channels=int(in_channels),
        hidden=int(hidden),
        features=int(features),
    )


def _build_resnet18(cfg: ModelConfig, heads: list[str]) -> ResNet18Classifier:
    params: dict[str, Any] = dict(cfg.params)
    in_channels = params.pop("in_channels", 3)
    pretrained = params.pop("pretrained", False)
    freeze_backbone = params.pop("freeze_backbone", False)
    features = params.pop("features", 512)
    if params:
        unknown = ", ".join(sorted(params))
        raise ModelRegistryError(f"Unsupported params for 'resnet18': {unknown}")
    return ResNet18Classifier(
        num_classes=cfg.num_classes,
        heads=heads,
        in_channels=int(in_channels),
        pretrained=bool(pretrained),
        freeze_backbone=bool(freeze_backbone),
        features=int(features),
    )


_BUILDERS = {
    "phase_classifier": _build_phase_classifier,
    # Backwards-friendly alias so the default config ("tcn") resolves.
    "tcn": _build_phase_classifier,
    "cnn": _build_phase_classifier,
    "resnet18": _build_resnet18,
}


def build_model(cfg: ModelConfig, heads: list[str] | None = None) -> Any:
    """Construct the torch module for the given model config.

    ``heads`` selects which heads get a final layer; when omitted the model
    builds all heads present in ``num_classes``.
    """
    if heads is None:
        heads = list(cfg.num_classes)
    builder = _BUILDERS.get(cfg.name)
    if builder is None:
        raise ModelRegistryError(
            f"No model builder for name '{cfg.name}'. "
            f"Available: {', '.join(sorted(_BUILDERS))}."
        )
    return builder(cfg, heads)
