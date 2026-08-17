"""Default classifier architecture for CholecT50 phase / multi-head prediction.

The network is deliberately small so it trains quickly on a single GPU (and is
sensible on CPU for smoke tests). It accepts either single frames
``(B, C, H, W)`` (``sequence_length == 1``) or temporal windows
``(B, L, C, H, W)``, runs a small convolutional feature extractor, aggregates
over the temporal dimension, and emits per-head logits.

Head selection is passed in from the config so only the requested heads build
a final linear layer (the rest are ignored at the loss/metric level).
"""

from collections.abc import Mapping

import torch
from torch import nn


class ConvFeatureExtractor(nn.Module):
    """Stack of conv-bn-relu blocks + adaptive pooling to a flat vector."""

    def __init__(self, in_channels: int = 3, hidden: int = 64, out_features: int = 256):
        super().__init__()
        block = []

        def add_block(ch_in: int, ch_out: int, pool: bool) -> None:
            block.extend(
                [
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                ]
            )
            if pool:
                block.append(nn.AdaptiveAvgPool2d((1, 1)))

        add_block(in_channels, hidden, False)
        add_block(hidden, hidden * 2, False)
        add_block(hidden * 2, hidden * 4, True)
        self.layers = nn.Sequential(*block)
        self.project = nn.Linear(hidden * 4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = torch.flatten(x, 1)
        return self.project(x)


class PhaseClassifier(nn.Module):
    """Multi-head classifier over CholecT50 label space.

    ``num_classes`` maps a head name to the number of output logits for that
    head. ``heads`` selects which heads get a final linear layer.
    """

    def __init__(
        self,
        num_classes: Mapping[str, int],
        heads: list[str],
        in_channels: int = 3,
        hidden: int = 64,
        features: int = 256,
    ):
        super().__init__()
        self.heads = list(heads)
        self.feature_extractor = ConvFeatureExtractor(
            in_channels=in_channels, hidden=hidden, out_features=features
        )
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
            x = x.flatten(0, 1)
            feats = self.feature_extractor(x)
            feats = feats.view(b, length, -1).mean(dim=1)
        else:  # (B, C, H, W) single frame
            feats = self.feature_extractor(x)
        return {name: head(feats) for name, head in self.heads_block.items()}


class ResNet18Classifier(nn.Module):
    """ResNet18 backbone + a linear classifier head per configured label head.

    Supports optionally loading ImageNet-pretrained weights and freezing the
    backbone so only the projection to ``features`` and the classification heads
    are trainable.
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
