"""ResNet18-based classifier for CholecT50 phase / multi-head prediction.

It accepts either single frames ``(B, C, H, W)`` (``sequence_length == 1``) or
temporal windows ``(B, L, C, H, W)``, runs the ResNet18 backbone, aggregates over
the temporal dimension, and emits per-head logits.

Head selection is passed in from the config so only the requested heads build a
final linear layer (the rest are ignored at the loss/metric level).
"""

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn


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


class TemporalResidualBlock(nn.Module):
    """Dilated residual 1D temporal conv (BN + ReLU + residual)."""

    def __init__(self, channels: int, kernel: int, dilation: int):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SingleStageTCN(nn.Module):
    """One MS-TCN stage: input conv + dilated residual layers + per-head out convs.

    Operates on ``(B, in_channels, L)`` and emits a dict of ``(B, K_head, L)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_f_maps: int,
        num_layers: int,
        kernel: int,
        head_sizes: dict[str, int],
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                TemporalResidualBlock(num_f_maps, kernel, dilation=2**i)
                for i in range(num_layers)
            ]
        )
        self.out_convs = nn.ModuleDict(
            {name: nn.Conv1d(num_f_maps, size, 1) for name, size in head_sizes.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.in_conv(x)
        for layer in self.layers:
            out = layer(out)
        return {name: conv(out) for name, conv in self.out_convs.items()}


class SwinMSTCN(nn.Module):
    """Swin Transformer frame encoder + multi-stage MS-TCN temporal head.

    Input ``(B, L, C, H, W)`` (or ``(B, C, H, W)``) -> per-frame Swin features ->
    ``num_stages`` of dilated temporal convs, each refining the previous stage's
    softmax (concatenated with the frame features). Returns per-head, per-frame
    logits ``(B, L, K)`` of the final stage, and keeps every stage's outputs in
    :attr:`stage_logits` for the MS-TCN (per-stage CE + smoothing) loss.
    """

    is_mstcn = True

    def __init__(
        self,
        num_classes: Mapping[str, int],
        heads: list[str],
        variant: str = "t",
        pretrained: bool = False,
        freeze_backbone: bool = False,
        num_stages: int = 4,
        num_layers: int = 10,
        num_f_maps: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.heads = list(heads)
        self.head_sizes = {h: num_classes[h] for h in self.heads if h in num_classes}
        self.encoder = SwinFeatureExtractor(
            variant=variant, pretrained=pretrained, freeze=freeze_backbone
        )
        self.in_channels = self.encoder.dim
        self.stages = nn.ModuleList()
        for _ in range(num_stages):
            self.stages.append(
                SingleStageTCN(
                    self.in_channels,
                    num_f_maps,
                    num_layers,
                    kernel_size,
                    self.head_sizes,
                )
            )
            self.in_channels = self.encoder.dim + sum(self.head_sizes.values())
        self.stage_logits: dict[str, list[torch.Tensor]] | None = None

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        x = frames.float()
        if x.ndim == 5:  # (B, L, C, H, W) temporal window
            b, length = x.shape[0], x.shape[1]
            feats = self.encoder(x.flatten(0, 1)).view(b, length, -1)
        else:  # (B, C, H, W) degenerate single-frame sequence
            feats = self.encoder(x).unsqueeze(1)
            length = 1
            b = feats.shape[0]

        frame_feats = feats.transpose(1, 2)  # (B, D, L)
        stage_logits: dict[str, list[torch.Tensor]] = {h: [] for h in self.heads}
        current = frame_feats
        prev_softmax: dict[str, torch.Tensor] | None = None
        for stage in self.stages:
            if prev_softmax is not None:
                extras = torch.cat([prev_softmax[h] for h in self.heads], dim=1)
                current = torch.cat([frame_feats, extras], dim=1)
            out = stage(current)
            for h in self.heads:
                stage_logits[h].append(out[h])  # (B, K, L)
            prev_softmax = {h: torch.softmax(out[h], dim=1) for h in self.heads}

        self.stage_logits = {
            h: [logits.transpose(1, 2) for logits in stage_logits[h]] for h in self.heads
        }
        return {h: self.stage_logits[h][-1] for h in self.heads}


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
