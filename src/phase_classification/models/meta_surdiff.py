"""Conditional label diffusion and meta-weight network for Meta-SurDiff."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    exponent = -math.log(10_000.0) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    ) / max(half - 1, 1)
    angles = timesteps.float().unsqueeze(1) * exponent.exp().unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if width % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalLabelDiffusion(nn.Module):
    """Diffusion model that refines coarse phase distributions.

    Labels are diffused around the coarse phase probabilities ``z``. The
    denoiser predicts forward-process noise from ``(y_t, z, timestep)`` and
    reconstructs phase probabilities by the reverse DDPM process. Inputs have
    shape ``(B, T, C)``; no future video feature is added by this module.
    """

    def __init__(
        self,
        num_classes: int = 7,
        timesteps: int = 100,
        hidden_dim: int = 512,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        if num_classes < 2 or timesteps < 2 or hidden_dim < 2:
            raise ValueError("num_classes, timesteps, and hidden_dim must be >= 2")
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_projection = nn.Linear(2 * num_classes, hidden_dim)
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.noise_projection = nn.Linear(hidden_dim, num_classes)

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def forward(
        self,
        noisy_labels: torch.Tensor,
        coarse_probs: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_labels.shape != coarse_probs.shape:
            raise ValueError("noisy labels and coarse probabilities must have equal shapes")
        batch_size = noisy_labels.shape[0]
        time = self.time_projection(
            sinusoidal_timestep_embedding(timesteps, self.hidden1.in_features)
        ).view(batch_size, 1, -1)
        hidden = self.input_projection(torch.cat((noisy_labels, coarse_probs), dim=-1))
        hidden = F.softplus(hidden * time)
        hidden = F.softplus(self.hidden1(hidden))
        hidden = F.softplus(self.hidden2(hidden))
        return self.noise_projection(hidden)

    def per_frame_loss(
        self,
        targets: torch.Tensor,
        coarse_probs: torch.Tensor,
        coarse_logits: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the paper's per-frame noise-prediction + coarse CE objective."""
        if targets.ndim != 2 or coarse_probs.ndim != 3:
            raise ValueError("targets must be (B,T), probabilities must be (B,T,C)")
        batch_size, length, classes = coarse_probs.shape
        if classes != self.num_classes or targets.shape != (batch_size, length):
            raise ValueError("batch, sequence length, or class dimension mismatch")
        if coarse_logits.shape != coarse_probs.shape:
            raise ValueError("coarse logits and probabilities must have equal shapes")

        if timesteps is None:
            timesteps = torch.randint(self.timesteps, (batch_size,), device=targets.device)
        clean_labels = F.one_hot(targets, self.num_classes).to(coarse_probs.dtype)
        if noise is None:
            noise = torch.randn_like(clean_labels)
        alpha_bar = self.alpha_bars[timesteps].view(batch_size, 1, 1)
        noisy_labels = (
            alpha_bar.sqrt() * clean_labels
            + (1.0 - alpha_bar.sqrt()) * coarse_probs
            + (1.0 - alpha_bar).sqrt() * noise
        )
        predicted_noise = self(noisy_labels, coarse_probs, timesteps)
        noise_loss = (predicted_noise - noise).square().mean(dim=-1)
        coarse_loss = F.cross_entropy(
            coarse_logits.transpose(1, 2), targets, reduction="none"
        )
        losses = noise_loss + coarse_loss
        return losses, {
            "noise_loss": noise_loss,
            "coarse_loss": coarse_loss,
        }

    @torch.inference_mode()
    def sample_probabilities(
        self,
        coarse_probs: torch.Tensor,
        sampling_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one reverse-diffusion sample with a deterministic DDIM schedule."""
        if coarse_probs.ndim != 3 or coarse_probs.shape[-1] != self.num_classes:
            raise ValueError("coarse_probs must have shape (B,T,num_classes)")
        steps = self.timesteps if sampling_steps is None else sampling_steps
        if not 1 <= steps <= self.timesteps:
            raise ValueError("sampling_steps must be in [1, timesteps]")
        was_training = self.training
        self.eval()
        coarse_probs = coarse_probs.float()
        batch_size = coarse_probs.shape[0]
        timesteps = torch.linspace(
            self.timesteps - 1, 0, steps, device=coarse_probs.device
        ).round().long().unique_consecutive()
        # In shifted coordinates u = y - z, the endpoint is standard normal.
        state = torch.randn(
            coarse_probs.shape,
            device=coarse_probs.device,
            dtype=coarse_probs.dtype,
            generator=generator,
        )
        for index, timestep in enumerate(timesteps):
            t = int(timestep)
            timestep_batch = torch.full((batch_size,), t, device=coarse_probs.device)
            alpha_bar = self.alpha_bars[t]
            shifted = state
            noisy_labels = coarse_probs + shifted
            predicted_noise = self(noisy_labels, coarse_probs, timestep_batch)
            clean_shifted = (
                shifted - (1.0 - alpha_bar).sqrt() * predicted_noise
            ) / alpha_bar.sqrt()
            if index + 1 == len(timesteps):
                state = clean_shifted
            else:
                previous_timestep = int(timesteps[index + 1])
                previous_alpha_bar = self.alpha_bars[previous_timestep]
                state = (
                    previous_alpha_bar.sqrt() * clean_shifted
                    + (1.0 - previous_alpha_bar).sqrt() * predicted_noise
                )
        probabilities = (coarse_probs + state).softmax(dim=-1)
        if was_training:
            self.train()
        return probabilities


class MetaWeightNet(nn.Module):
    """Two-layer positive frame-loss reweighter used by Meta-SurDiff."""

    def __init__(self, hidden_dim: int = 100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, per_frame_loss: torch.Tensor) -> torch.Tensor:
        weights = self.network(per_frame_loss.reshape(-1, 1)).reshape_as(per_frame_loss)
        return weights / weights.mean().clamp_min(1e-8)


class MetaSurDiffSystem(nn.Module):
    """MS-TCN coarse predictor plus conditional label-diffusion refiner."""

    def __init__(
        self,
        coarse_model: nn.Module,
        diffusion: ConditionalLabelDiffusion,
        phase_head: str = "phase",
    ):
        super().__init__()
        self.coarse_model = coarse_model
        self.diffusion = diffusion
        self.phase_head = phase_head

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        sampling_steps: int | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        coarse_logits = self.coarse_model(features)[self.phase_head]
        coarse_probs = coarse_logits.softmax(dim=-1)
        if targets is not None:
            losses, _ = self.diffusion.per_frame_loss(
                targets,
                coarse_probs,
                coarse_logits,
                timesteps=timesteps,
                noise=noise,
            )
            return losses
        refined_probs = self.diffusion.sample_probabilities(
            coarse_probs, sampling_steps=sampling_steps
        )
        return {
            "coarse_logits": coarse_logits,
            "coarse_probs": coarse_probs,
            "refined_probs": refined_probs,
        }
