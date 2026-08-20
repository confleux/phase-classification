"""Swin Transformer feature extractor + per-head classifier."""

from collections.abc import Mapping

import torch
from torch import nn

from phase_classification.models.backbones import SwinFeatureExtractor


class SwinClassifier(nn.Module):
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
        super().__init__()
        self.heads = list(heads)
        self.encoder = SwinFeatureExtractor(
            variant=variant, pretrained=pretrained, freeze=freeze_backbone
        )
        self.project = nn.Linear(self.encoder.dim, features)
        self.heads_block: nn.ModuleDict = nn.ModuleDict(
            {
                name: nn.Linear(features, num_classes[name])
                for name in self.heads
                if name in num_classes
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        if x.ndim == 5:  # (B, L, C, H, W) temporal window
            b, length = x.shape[0], x.shape[1]
            feats = self.encoder(x.flatten(0, 1)).view(b, length, -1).mean(dim=1)
        else:  # (B, C, H, W) single frame
            feats = self.encoder(x)
        feats = self.project(feats)
        return {name: head(feats) for name, head in self.heads_block.items()}