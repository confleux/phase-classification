"""ConvNeXt feature extractor + MS-TCN temporal head."""

from collections.abc import Mapping

import torch
from torch import nn

from phase_classification.models.backbones import ConvNeXtFeatureExtractor


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


class ConvNeXtMSTCN(nn.Module):
    """ConvNeXt frame encoder + multi-stage MS-TCN temporal head.

    Input ``(B, L, C, H, W)`` (or ``(B, C, H, W)``) -> per-frame ConvNeXt features ->
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
        variant: str = "tiny",
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
        self.encoder = ConvNeXtFeatureExtractor(
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