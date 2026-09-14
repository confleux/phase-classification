"""Shared backbone modules used across models."""

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn


class _FeatureExtractor(nn.Module):
    """Base per-frame feature extractor over a torchvision backbone.

    Subclasses provide a ``_builders`` classmethod of ``variant -> (builder,
    weights)`` and a ``_channels_last`` flag. The backbone's classification head
    is replaced with an identity and each frame is reduced to a ``D``-dimensional
    descriptor via global average pooling.
    """

    _VARIANTS: ClassVar[dict[str, int]]
    _channels_last: ClassVar[bool] = False

    def __init__(self, variant: str, pretrained: bool = False, freeze: bool = False):
        super().__init__()
        builders = self._builders()
        if variant not in builders:
            raise ValueError(
                f"Unknown {self._name} variant '{variant}', "
                f"choose from {sorted(builders)}."
            )
        builder, weights = builders[variant]
        self.net = builder(weights=weights.DEFAULT if pretrained else None)
        self.net.head = nn.Identity()
        self.dim = self._VARIANTS[variant]
        if freeze:
            for param in self.net.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.net.features(x)  # (B, ..., D) or (B, D, ...)
        if self._channels_last:
            feats = feats.permute(0, 3, 1, 2)  # (B, D, H/32, W/32)
        return torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(
            1
        )  # (B, D)


class _HeadClassifierBase(nn.Module):
    """Shared per-head classifier over a frame encoder.

    Builds one linear head per configured label head and emits per-head logits.
    Accepts single frames ``(B, C, H, W)`` or temporal windows ``(B, L, C, H, W)``
    (mean-pooled over time). Subclasses provide :meth:`_encode` returning a
    ``(B, backbone_dim)`` descriptor per frame.
    """

    def __init__(
        self,
        num_classes: Mapping[str, int],
        heads: list[str],
        backbone_dim: int,
        features: int,
    ):
        super().__init__()
        self.heads = list(heads)
        self.project = nn.Linear(backbone_dim, features)
        self.heads_block: nn.ModuleDict = nn.ModuleDict(
            {
                name: nn.Linear(features, num_classes[name])
                for name in self.heads
                if name in num_classes
            }
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        if x.ndim == 5:  # (B, L, C, H, W) temporal window
            b, length = x.shape[0], x.shape[1]
            feats = self._encode(x.flatten(0, 1)).view(b, length, -1).mean(dim=1)
        else:  # (B, C, H, W) single frame
            feats = self._encode(x)
        feats = self.project(feats)
        return {name: head(feats) for name, head in self.heads_block.items()}


class SwinFeatureExtractor(_FeatureExtractor):
    """Swin Transformer per-frame feature extractor.

    Applies a torchvision Swin backbone (head removed) to each frame and returns
    a ``D``-dimensional descriptor via global average pooling. Works on the
    dataset's native frame size (e.g. 256x448).
    """

    _name = "swin"
    _channels_last = True
    _VARIANTS: ClassVar[dict[str, int]] = {
        "t": 768,
        "s": 768,
        "b": 1024,
    }

    @classmethod
    def _builders(cls) -> dict[str, tuple]:
        from torchvision.models import (
            Swin_B_Weights,
            Swin_S_Weights,
            Swin_T_Weights,
            swin_b,
            swin_s,
            swin_t,
        )

        return {
            "t": (swin_t, Swin_T_Weights),
            "s": (swin_s, Swin_S_Weights),
            "b": (swin_b, Swin_B_Weights),
        }


class ConvNeXtFeatureExtractor(_FeatureExtractor):
    """ConvNeXt per-frame feature extractor.

    Applies a torchvision ConvNeXt backbone (head removed) to each frame and
    returns a ``D``-dimensional descriptor via global average pooling. Works on
    the dataset's native frame size (e.g. 256x448). Defaults to ``tiny``.
    """

    _name = "convnext"
    _VARIANTS: ClassVar[dict[str, int]] = {
        "tiny": 768,
        "small": 768,
        "base": 1024,
        "large": 1536,
    }

    @classmethod
    def _builders(cls) -> dict[str, tuple]:
        from torchvision.models import (
            ConvNeXt_Base_Weights,
            ConvNeXt_Large_Weights,
            ConvNeXt_Small_Weights,
            ConvNeXt_Tiny_Weights,
            convnext_base,
            convnext_large,
            convnext_small,
            convnext_tiny,
        )

        return {
            "tiny": (convnext_tiny, ConvNeXt_Tiny_Weights),
            "small": (convnext_small, ConvNeXt_Small_Weights),
            "base": (convnext_base, ConvNeXt_Base_Weights),
            "large": (convnext_large, ConvNeXt_Large_Weights),
        }
