"""Lightweight residual temporal adapters for cached frame embeddings."""

import torch
from torch import nn


class ResidualTemporalAdapter(nn.Module):
    """Apply a bottleneck temporal convolution while preserving input features.

    Input and output are ``(batch, time, feature_dim)``. The adapter starts as
    an identity mapping, so adding it to an existing classifier does not
    initially perturb the pretrained/cached feature representation.
    """

    def __init__(
        self,
        feature_dim: int,
        bottleneck_dim: int = 64,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or bottleneck_dim < 1:
            raise ValueError("feature_dim and bottleneck_dim must be positive.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.down = nn.Linear(feature_dim, bottleneck_dim)
        self.temporal = nn.Conv1d(
            bottleneck_dim,
            bottleneck_dim,
            kernel_size,
            padding=kernel_size // 2,
            groups=bottleneck_dim,
        )
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, feature_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                "ResidualTemporalAdapter expects features shaped "
                f"(batch, time, {self.feature_dim}); got {tuple(features.shape)}."
            )
        hidden = self.activation(self.down(self.norm(features)))
        hidden = self.temporal(hidden.transpose(1, 2)).transpose(1, 2)
        return features + self.up(self.activation(hidden))
