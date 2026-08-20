"""Model registry: map ``(ModelConfig.name, params)`` to a torch module.

Config uses loose coupling: ``model.name`` selects an implementation here and
``model.params`` carries architecture-specific knobs unchanged.
"""

from typing import Any

from phase_classification.config.model import ModelConfig
from phase_classification.models.ms_tcn import SwinMSTCN
from phase_classification.models.resnet18 import ResNet18Classifier
from phase_classification.models.swin_classifier import SwinClassifier


class ModelRegistryError(Exception):
    """Raised when no model implementation matches the configured name."""


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


def _build_swin_mstcn(cfg: ModelConfig, heads: list[str]) -> SwinMSTCN:
    params: dict[str, Any] = dict(cfg.params)
    variant = params.pop("variant", "t")
    pretrained = params.pop("pretrained", False)
    freeze_backbone = params.pop("freeze_backbone", False)
    num_stages = params.pop("num_stages", 4)
    num_layers = params.pop("num_layers", 10)
    num_f_maps = params.pop("num_f_maps", 64)
    kernel_size = params.pop("kernel_size", 3)
    if params:
        unknown = ", ".join(sorted(params))
        raise ModelRegistryError(f"Unsupported params for 'swin-ms-tcn': {unknown}")
    return SwinMSTCN(
        num_classes=cfg.num_classes,
        heads=heads,
        variant=str(variant),
        pretrained=bool(pretrained),
        freeze_backbone=bool(freeze_backbone),
        num_stages=int(num_stages),
        num_layers=int(num_layers),
        num_f_maps=int(num_f_maps),
        kernel_size=int(kernel_size),
    )


def _build_swin_classifier(cfg: ModelConfig, heads: list[str]) -> SwinClassifier:
    params: dict[str, Any] = dict(cfg.params)
    variant = params.pop("variant", "t")
    pretrained = params.pop("pretrained", False)
    freeze_backbone = params.pop("freeze_backbone", False)
    features = params.pop("features", 512)
    if params:
        unknown = ", ".join(sorted(params))
        raise ModelRegistryError(f"Unsupported params for 'swin-cls': {unknown}")
    return SwinClassifier(
        num_classes=cfg.num_classes,
        heads=heads,
        variant=str(variant),
        pretrained=bool(pretrained),
        freeze_backbone=bool(freeze_backbone),
        features=int(features),
    )


_BUILDERS = {
    "resnet18": _build_resnet18,
    "swin-ms-tcn": _build_swin_mstcn,
    "ms-tcn": _build_swin_mstcn,
    "swin-cls": _build_swin_classifier,
    "swin-classifier": _build_swin_classifier,
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
