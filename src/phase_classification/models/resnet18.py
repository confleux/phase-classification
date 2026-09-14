"""ResNet18 backbone + per-head classifier."""

from collections.abc import Mapping

import torch
from torch import nn

from phase_classification.models.backbones import _HeadClassifierBase


class ResNet18Classifier(_HeadClassifierBase):
    """ResNet18 backbone + a linear classifier head per configured label head.

    Supports optionally loading ImageNet-pretrained weights and freezing the
    backbone so only the projection to ``features`` and the classification heads
    are trainable. Accepts single frames ``(B, C, H, W)`` or temporal windows
    ``(B, L, C, H, W)`` (mean-pooled over time).
    """

    def __init__(
        self,
        num_classes: Mapping[str, int],
        heads: list[str],
        in_channels: int = 3,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        features: int = 512,
    ):
        from torchvision.models import ResNet18_Weights, resnet18

        super().__init__(num_classes, heads, backbone_dim=512, features=features)
        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
