"""Swin Transformer feature extractor + per-head classifier."""

from collections.abc import Mapping

import torch

from phase_classification.models.backbones import (
    SwinFeatureExtractor,
    _HeadClassifierBase,
)


class SwinClassifier(_HeadClassifierBase):
    """Swin Transformer feature extractor + a linear classifier per label head.

    Accepts single frames ``(B, C, H, W)`` or temporal windows ``(B, L, C, H, W)``
    (mean-pooled over time, like :class:`ResNet18Classifier`), encodes each frame
    with :class:`SwinFeatureExtractor`, and emits per-head logits.
    """

    def __init__(
        self,
        num_classes: Mapping[str, int],
        heads: list[str],
        variant: str = "t",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        features: int = 512,
    ):
        encoder = SwinFeatureExtractor(
            variant=variant, pretrained=pretrained, freeze=freeze_backbone
        )
        super().__init__(
            num_classes, heads, backbone_dim=encoder.dim, features=features
        )
        self.encoder = encoder

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
