"""Feature extractor + MS-TCN temporal head (ConvNeXt or Swin encoder)."""

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn

from phase_classification.models.backbones import (
    ConvNeXtFeatureExtractor,
    SwinFeatureExtractor,
)


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


class _MSTCNBase(nn.Module):
    """Shared frame-encoder + multi-stage MS-TCN temporal head.

    Input ``(B, L, C, H, W)`` (or ``(B, C, H, W)``) -> per-frame encoder features ->
    ``num_stages`` of dilated temporal convs, each refining the previous stage's
    softmax (concatenated with the frame features). Returns per-head, per-frame
    logits ``(B, L, K)`` of the final stage, and keeps every stage's outputs in
    :attr:`stage_logits` for the MS-TCN (per-stage CE + smoothing) loss.
    """

    is_mstcn = True
    _encoder_cls: ClassVar[type[nn.Module]]
    _default_variant: ClassVar[str]

    def __init__(
        self,
        num_classes: Mapping[str, int],
        heads: list[str],
        variant: str | None = None,
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
        self.encoder = self._encoder_cls(
            variant=variant or self._default_variant,
            pretrained=pretrained,
            freeze=freeze_backbone,
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
            h: [logits.transpose(1, 2) for logits in stage_logits[h]]
            for h in self.heads
        }
        return {h: self.stage_logits[h][-1] for h in self.heads}


class CachedFeatureMSTCN(nn.Module):
    """MS-TCN temporal head for precomputed per-frame feature sequences.

    Input is ``(B, L, D)`` rather than decoded images. The module exposes the
    same output and ``stage_logits`` interface as the image-backed MS-TCNs, so
    it can use the shared :class:`Trainer` and :class:`MSTCNLoss`.
    """

    is_mstcn = True

    def __init__(
        self,
        feature_dim: int,
        num_classes: Mapping[str, int],
        heads: list[str],
        num_stages: int = 3,
        num_layers: int = 8,
        num_f_maps: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        if feature_dim < 1 or num_stages < 1:
            raise ValueError("feature_dim and num_stages must be >= 1")
        self.heads = list(heads)
        self.head_sizes = {head: num_classes[head] for head in self.heads}
        self.stages = nn.ModuleList()
        in_channels = feature_dim
        for _ in range(num_stages):
            self.stages.append(
                SingleStageTCN(
                    in_channels,
                    num_f_maps,
                    num_layers,
                    kernel_size,
                    self.head_sizes,
                )
            )
            in_channels = feature_dim + sum(self.head_sizes.values())
        self.stage_logits: dict[str, list[torch.Tensor]] | None = None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("CachedFeatureMSTCN expects features shaped (B, L, D).")
        frame_features = features.float().transpose(1, 2)
        current = frame_features
        stage_logits: dict[str, list[torch.Tensor]] = {head: [] for head in self.heads}
        previous: dict[str, torch.Tensor] | None = None
        for stage in self.stages:
            if previous is not None:
                refined = torch.cat([previous[head] for head in self.heads], dim=1)
                current = torch.cat([frame_features, refined], dim=1)
            output = stage(current)
            for head in self.heads:
                stage_logits[head].append(output[head].transpose(1, 2))
            previous = {head: torch.softmax(output[head], dim=1) for head in self.heads}
        self.stage_logits = stage_logits
        return {head: stage_logits[head][-1] for head in self.heads}


class DualDilatedResidualBlock(nn.Module):
    """MS-TCN++ prediction-generation layer with two receptive-field scales."""

    def __init__(
        self,
        channels: int,
        dilation_small: int,
        dilation_large: int,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding_small = (kernel_size - 1) * dilation_small // 2
        padding_large = (kernel_size - 1) * dilation_large // 2
        self.conv_small = nn.Conv1d(
            channels, channels, kernel_size, padding=padding_small, dilation=dilation_small
        )
        self.conv_large = nn.Conv1d(
            channels, channels, kernel_size, padding=padding_large, dilation=dilation_large
        )
        self.project = nn.Conv1d(2 * channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        small = torch.relu(self.conv_small(x))
        large = torch.relu(self.conv_large(x))
        combined = torch.relu(torch.cat((small, large), dim=1))
        return x + self.dropout(self.project(combined))


class RefinementResidualBlock(nn.Module):
    """MS-TCN++ dilated residual block used by refinement stages."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.dilated = nn.Conv1d(
            channels, channels, kernel_size, padding=padding, dilation=dilation
        )
        self.project = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.project(torch.relu(self.dilated(x)))
        return x + self.dropout(residual)


class _MSCTCNPlusPlusStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_f_maps: int,
        layers: int,
        kernel_size: int,
        dropout: float,
        dual_dilated: bool,
    ):
        super().__init__()
        self.input_projection = nn.Conv1d(in_channels, num_f_maps, kernel_size=1)
        if dual_dilated:
            self.layers = nn.ModuleList(
                [
                    DualDilatedResidualBlock(
                        num_f_maps,
                        dilation_small=2**index,
                        dilation_large=2 ** (layers - index - 1),
                        kernel_size=kernel_size,
                        dropout=dropout,
                    )
                    for index in range(layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    RefinementResidualBlock(
                        num_f_maps,
                        dilation=2**index,
                        kernel_size=kernel_size,
                        dropout=dropout,
                    )
                    for index in range(layers)
                ]
            )
        self.output_projection = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_projection(x)
        for layer in self.layers:
            out = layer(out)
        return self.output_projection(out)


class CachedFeatureMSTCNPlusPlus(nn.Module):
    """MS-TCN++ for cached feature sequences, with one prediction stage.

    The prediction-generation stage uses dual-dilated layers. Refinement stages
    receive only the preceding stage's class probabilities, as in MS-TCN++.
    ``stage_logits`` follows the shared ``MSTCNLoss`` interface.
    """

    is_mstcn = True

    def __init__(
        self,
        feature_dim: int,
        num_classes: Mapping[str, int],
        heads: list[str],
        num_refinement_stages: int = 3,
        generation_layers: int = 11,
        refinement_layers: int = 10,
        num_f_maps: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.5,
        smoothing_weight: float = 0.15,
    ):
        super().__init__()
        if heads != ["phase"]:
            raise ValueError("CachedFeatureMSTCNPlusPlus currently supports only the phase head.")
        if min(feature_dim, generation_layers, refinement_layers, num_f_maps) < 1:
            raise ValueError("Feature dimension, layers, and feature maps must be positive.")
        if num_refinement_stages < 0:
            raise ValueError("num_refinement_stages must be >= 0.")
        self.heads = list(heads)
        self.num_classes = num_classes["phase"]
        self.smoothing_on_log_probs = True
        self.smoothing_weight = smoothing_weight
        self.generation = _MSCTCNPlusPlusStage(
            feature_dim,
            self.num_classes,
            num_f_maps,
            generation_layers,
            kernel_size,
            dropout,
            dual_dilated=True,
        )
        self.refinement_stages = nn.ModuleList(
            [
                _MSCTCNPlusPlusStage(
                    self.num_classes,
                    self.num_classes,
                    num_f_maps,
                    refinement_layers,
                    kernel_size,
                    dropout,
                    dual_dilated=False,
                )
                for _ in range(num_refinement_stages)
            ]
        )
        self.stage_logits: dict[str, list[torch.Tensor]] | None = None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("CachedFeatureMSTCNPlusPlus expects features shaped (B, L, D).")
        current = self.generation(features.float().transpose(1, 2))
        stage_outputs = [current.transpose(1, 2)]
        for stage in self.refinement_stages:
            current = stage(torch.softmax(current, dim=1))
            stage_outputs.append(current.transpose(1, 2))
        self.stage_logits = {"phase": stage_outputs}
        return {"phase": stage_outputs[-1]}


class ConvNeXtMSTCN(_MSTCNBase):
    """ConvNeXt frame encoder + multi-stage MS-TCN temporal head."""

    _encoder_cls = ConvNeXtFeatureExtractor
    _default_variant = "tiny"


class SwinMSTCN(_MSTCNBase):
    """Swin Transformer frame encoder + multi-stage MS-TCN temporal head."""

    _encoder_cls = SwinFeatureExtractor
    _default_variant = "t"
