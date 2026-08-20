"""Shared backbone modules used across models."""

from typing import ClassVar

import torch
from torch import nn


class SwinFeatureExtractor(nn.Module):
    """Swin Transformer per-frame feature extractor.

    Applies a torchvision Swin backbone (head removed) to each frame and returns
    a ``D``-dimensional descriptor via global average pooling. Works on the
    dataset's native frame size (e.g. 256x448).
    """

    _VARIANTS: ClassVar[dict[str, int]] = {
        "t": 768,
        "s": 768,
        "b": 1024,
    }

    def __init__(self, variant: str = "t", pretrained: bool = False, freeze: bool = False):
        super().__init__()
        from torchvision.models import (
            Swin_B_Weights,
            Swin_S_Weights,
            Swin_T_Weights,
            swin_b,
            swin_s,
            swin_t,
        )

        builders = {
            "t": (swin_t, Swin_T_Weights),
            "s": (swin_s, Swin_S_Weights),
            "b": (swin_b, Swin_B_Weights),
        }
        if variant not in builders:
            raise ValueError(f"Unknown swin variant '{variant}', choose from {sorted(builders)}.")
        builder, weights = builders[variant]
        self.net = builder(weights=weights.DEFAULT if pretrained else None)
        self.net.head = nn.Identity()
        self.dim = self._VARIANTS[variant]
        if freeze:
            for param in self.net.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net.features(x)  # (B, H/32, W/32, D) channels-last
        feats = feats.permute(0, 3, 1, 2)  # (B, D, H/32, W/32)
        return torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)  # (B, D)