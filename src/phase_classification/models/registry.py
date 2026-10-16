"""Model registry: map ``(ModelConfig.name, params)`` to a torch module.

Config uses loose coupling: ``model.name`` selects an implementation here and
``model.params`` carries architecture-specific knobs unchanged. Each builder
declares the params it accepts (with defaults); unknown keys are rejected.
"""

from collections.abc import Mapping
from typing import Any

from phase_classification.config.model import ModelConfig
from phase_classification.models.ms_tcn import (
    CachedFeatureMSTCN,
    ConvNeXtMSTCN,
    SwinMSTCN,
)
from phase_classification.models.resnet18 import ResNet18Classifier
from phase_classification.models.swin_classifier import SwinClassifier


class ModelRegistryError(Exception):
    """Raised when no model implementation matches the configured name."""


_RESNET18_PARAMS: dict[str, Any] = {
    "in_channels": 3,
    "pretrained": False,
    "freeze_backbone": False,
    "features": 512,
}

_MSTCN_PARAMS: dict[str, Any] = {
    "variant": None,  # falls back to the class's default variant
    "pretrained": False,
    "freeze_backbone": False,
    "num_stages": 4,
    "num_layers": 10,
    "num_f_maps": 64,
    "kernel_size": 3,
}

_CACHED_MSTCN_PARAMS: dict[str, Any] = {
    "feature_dim": 256,
    "num_stages": 3,
    "num_layers": 8,
    "num_f_maps": 64,
    "kernel_size": 3,
}

_SWIN_CLASSIFIER_PARAMS: dict[str, Any] = {
    "variant": "t",
    "pretrained": False,
    "freeze_backbone": False,
    "features": 512,
}


def _pop_params(
    name: str, params: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    """Extract known keys (with spec defaults) from ``params``; reject unknown."""
    unknown = set(params) - set(spec)
    if unknown:
        raise ModelRegistryError(
            f"Unsupported params for '{name}': {', '.join(sorted(unknown))}. "
            f"Supported: {', '.join(sorted(spec))}."
        )
    return {key: params.get(key, default) for key, default in spec.items()}


def _build_resnet18(cfg: ModelConfig, heads: list[str]) -> ResNet18Classifier:
    p = _pop_params("resnet18", cfg.params, _RESNET18_PARAMS)
    return ResNet18Classifier(
        num_classes=cfg.num_classes,
        heads=heads,
        in_channels=int(p["in_channels"]),
        pretrained=bool(p["pretrained"]),
        freeze_backbone=bool(p["freeze_backbone"]),
        features=int(p["features"]),
    )


def _build_mstcn(
    cfg: ModelConfig, heads: list[str], cls: type, default_variant: str, name: str
) -> Any:
    p = _pop_params(name, cfg.params, _MSTCN_PARAMS)
    return cls(
        num_classes=cfg.num_classes,
        heads=heads,
        variant=str(p["variant"]) if p["variant"] is not None else default_variant,
        pretrained=bool(p["pretrained"]),
        freeze_backbone=bool(p["freeze_backbone"]),
        num_stages=int(p["num_stages"]),
        num_layers=int(p["num_layers"]),
        num_f_maps=int(p["num_f_maps"]),
        kernel_size=int(p["kernel_size"]),
    )


def _build_convnext_mstcn(cfg: ModelConfig, heads: list[str]) -> ConvNeXtMSTCN:
    return _build_mstcn(cfg, heads, ConvNeXtMSTCN, "tiny", "convnext-ms-tcn")


def _build_swin_mstcn(cfg: ModelConfig, heads: list[str]) -> SwinMSTCN:
    return _build_mstcn(cfg, heads, SwinMSTCN, "t", "swin-ms-tcn")


def _build_cached_feature_mstcn(cfg: ModelConfig, heads: list[str]) -> CachedFeatureMSTCN:
    p = _pop_params("surgclip-ms-tcn", cfg.params, _CACHED_MSTCN_PARAMS)
    return CachedFeatureMSTCN(
        feature_dim=int(p["feature_dim"]),
        num_classes=cfg.num_classes,
        heads=heads,
        num_stages=int(p["num_stages"]),
        num_layers=int(p["num_layers"]),
        num_f_maps=int(p["num_f_maps"]),
        kernel_size=int(p["kernel_size"]),
    )


def _build_swin_classifier(cfg: ModelConfig, heads: list[str]) -> SwinClassifier:
    p = _pop_params("swin-cls", cfg.params, _SWIN_CLASSIFIER_PARAMS)
    return SwinClassifier(
        num_classes=cfg.num_classes,
        heads=heads,
        variant=str(p["variant"]),
        pretrained=bool(p["pretrained"]),
        freeze_backbone=bool(p["freeze_backbone"]),
        features=int(p["features"]),
    )


_BUILDERS = {
    "resnet18": _build_resnet18,
    "convnext-ms-tcn": _build_convnext_mstcn,
    "ms-tcn": _build_convnext_mstcn,
    "swin-ms-tcn": _build_swin_mstcn,
    "surgclip-ms-tcn": _build_cached_feature_mstcn,
    "feature-ms-tcn": _build_cached_feature_mstcn,
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
