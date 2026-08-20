"""ResNet18 backbone + per-head classifier."""

from collections.abc import Mapping

import torch
from torch import nn


class ResNet18Classifier(nn.Module):
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
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.heads = list(heads)
        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT if pretrained else None
        )
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        self.backbone.fc = nn.Identity()
        self.features = features
        self.project = nn.Linear(512, features)
        self.heads_block: nn.ModuleDict = nn.ModuleDict(
            {
                name: nn.Linear(features, num_classes[name])
                for name in self.heads
                if name in num_classes
            }
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        if x.ndim == 5:  # (B, L, C, H, W) temporal window
            b, length = x.shape[0], x.shape[1]
            x = x.flatten(0, 1)
            feats = self.project(self.backbone(x))
            feats = feats.view(b, length, -1).mean(dim=1)
        else:  # (B, C, H, W) single frame
            feats = self.project(self.backbone(x))
        return {name: head(feats) for name, head in self.heads_block.items()}