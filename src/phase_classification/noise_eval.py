"""Measure CholecT50 inference robustness to increasing Gaussian image noise.

Run against a checkpoint produced by ``phase-cls train``::

    uv run python -m phase_classification.noise_eval \
        configs/resnet18.yaml outputs/checkpoints/best.pt

The test videos are scored clean and at each requested RGB noise standard
deviation. Noise is added in image space (before ImageNet normalization), and
the same per-frame noise pattern is reused at every severity for fair comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from phase_classification.config.schema import ExperimentConfig
from phase_classification.data.dataloaders import (
    DataLoaders,
    build_dataloaders,
    build_test_loaders,
)
from phase_classification.engine import resolve_device
from phase_classification.heads import head_index
from phase_classification.loader import load_config
from phase_classification.metrics import MetricAccumulator
from phase_classification.models.registry import build_model

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GaussianNoiseDataset(Dataset):
    """Apply reproducible Gaussian RGB noise without changing the labels."""

    def __init__(
        self,
        dataset: Dataset,
        noise_std: float,
        seed: int,
        normalize: bool,
    ) -> None:
        if noise_std < 0:
            raise ValueError("noise_std must be non-negative.")
        self.dataset = dataset
        self.noise_std = noise_std
        self.seed = seed
        self.normalize = normalize
        self.mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        images, labels = self.dataset[index]
        if self.noise_std == 0:
            return images, labels

        if images.ndim not in (3, 4) or images.shape[-3] != 3:
            raise ValueError(
                "Expected RGB image tensors shaped (C,H,W) or (T,C,H,W); "
                f"received {tuple(images.shape)}."
            )

        frame_ids = self._frame_ids(index, images)
        noise = torch.stack(
            [
                torch.randn(
                    images.shape[-3:],
                    generator=torch.Generator().manual_seed(self.seed + frame_id),
                    dtype=images.dtype,
                )
                for frame_id in frame_ids
            ]
        )
        if images.ndim == 3:
            noise = noise.squeeze(0)
        mean = self.mean.to(dtype=images.dtype)
        std = self.std.to(dtype=images.dtype)
        if self.normalize:
            pixels = images * std + mean
        else:
            pixels = images
        pixels = (pixels + self.noise_std * noise).clamp_(0, 1)
        if self.normalize:
            pixels = (pixels - mean) / std
        return pixels, labels

    def _frame_ids(self, index: int, images: torch.Tensor) -> list[int]:
        """Use video frame numbers so overlapping temporal windows share noise."""
        frames = getattr(self.dataset, "frames", None)
        count = images.shape[0] if images.ndim == 4 else 1
        if frames is None:
            return [index + offset for offset in range(count)]
        try:
            return [int(frames[index + offset]) for offset in range(count)]
        except (IndexError, TypeError, ValueError):
            return [index + offset for offset in range(count)]


def _checkpoint_state(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} does not contain a model state dict.")
    return checkpoint


@torch.inference_mode()
def _evaluate_noise_level(
    cfg: ExperimentConfig,
    dl: DataLoaders,
    model: torch.nn.Module,
    device: torch.device,
    noise_std: float,
    seed: int,
) -> dict[str, float]:
    per_video = [
        GaussianNoiseDataset(
            video,
            noise_std=noise_std,
            seed=seed + video_index * 1_000_003,
            normalize=cfg.dataset.normalize,
        )
        for video_index, video in enumerate(dl.test)
    ]
    loaders = build_test_loaders(
        DataLoaders(train=dl.train, val=dl.val, test=per_video), cfg
    )
    accumulators: list[MetricAccumulator] = []
    amp = cfg.runtime.mixed_precision == "fp16" and device.type == "cuda"

    for loader in tqdm(loaders, desc=f"noise std={noise_std:g}", unit="video"):
        accumulator = MetricAccumulator(
            cfg.training.heads,
            list(cfg.validation.metrics),
            cfg.model.num_classes,
        )
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
            logits_np = {name: value.detach().cpu().numpy() for name, value in logits.items()}
            targets = {
                head: labels[head_index(head)].cpu().numpy()
                for head in cfg.training.heads
            }
            accumulator.add_batch(logits_np, targets)
        accumulators.append(accumulator)

    if not accumulators:
        raise ValueError("The configured CholecT50 test split contains no videos.")
    return MetricAccumulator.merged(accumulators).compute()


def evaluate_noise_robustness(
    cfg: ExperimentConfig,
    checkpoint_path: Path,
    noise_stds: list[float],
) -> dict[str, dict[str, float]]:
    """Load a trained model and score clean/noisy held-out test videos."""
    if any(std < 0 for std in noise_stds):
        raise ValueError("All noise standard deviations must be non-negative.")
    levels = [0.0, *sorted(set(noise_stds) - {0.0})]

    dl = build_dataloaders(cfg)
    device = resolve_device(cfg.runtime.device, cfg.runtime.cuda_device)
    model = build_model(cfg.model, heads=cfg.training.heads).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint_path, device))
    model.eval()

    results = {
        f"{level:g}": _evaluate_noise_level(
            cfg, dl, model, device, level, cfg.runtime.seed
        )
        for level in levels
    }
    clean_metrics = results["0"]
    results["delta_vs_clean"] = {
        level: {
            metric: value - clean_metrics[metric]
            for metric, value in metrics.items()
        }
        for level, metrics in results.items()
        if level != "0"
    }
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score a trained CholecT50 model under increasing Gaussian image noise."
    )
    parser.add_argument("config", type=Path, help="Training YAML config for this checkpoint.")
    parser.add_argument("checkpoint", type=Path, help="Path to best.pt or a state-dict checkpoint.")
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        metavar="STD",
        help="RGB-space Gaussian noise stds in [0,1] pixel units (default: 0.05 0.10 0.20).",
    )
    parser.add_argument("--output", type=Path, help="Optional path to save the JSON scores.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    results = evaluate_noise_robustness(cfg, args.checkpoint, args.noise_stds)
    report = json.dumps(results, indent=2, sort_keys=True)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"Saved noise robustness scores to {args.output}")


if __name__ == "__main__":
    main()
