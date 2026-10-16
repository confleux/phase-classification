"""Compare raw SurgCLIP phase scores with diffusion-refined scores.

Run with::

    uv run python -m phase_classification.surgclip_diffusion \\
        --dataset-dir /data/cholect50 --test-fold 1 \\
        --scores-cache outputs/cache/surgclip-fold1.pt \\
        --embeddings-file outputs/surgclip-diffusion/fold1/surgclip_embeddings.pt

The SurgCLIP video/text encoders stay frozen. A temporal denoiser is trained
on phase-label residuals, conditioned on the per-frame SurgCLIP scores and
saved visual embeddings. The original scores are skip-connected to the final
prediction, so diffusion predicts a correction rather than regenerating labels
from noise. The validation split selects the best refiner; the test split is
evaluated once for both raw and refined scores.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.heads import DEFAULT_NUM_CLASSES
from phase_classification.metrics import MetricAccumulator

PHASE_PROMPTS = [
    "Prepares for surgery by inserting trocars into the patient's abdominal cavity",
    "Employs grasper and hook during calot triangle dissection, manipulating gallbladder to reveal hepatic triangle, cystic duct and cystic artery",
    "Utilizes clipper to secure cystic duct and artery, followed by precise dissection using scissors",
    "Utilizes a hook to dissect the connective tissue during the dissection phase, separating gallbladder from the liver",
    "Secures the removed gallbladder in the specimen bag during the packaging phase of the procedure",
    "Employs suction and irrigation techniques to maintain a clear surgical field during the clean and coagulation phase, simultaneously coagulating bleeding vessels",
    "Handles the specimen bag during the retraction",
]
NUM_CLASSES = DEFAULT_NUM_CLASSES["phase"]


@dataclass
class VideoScores:
    """SurgCLIP embeddings, logits, and phase IDs for one video."""

    video_id: str
    embeddings: torch.Tensor  # (frames, embed_dim), pooled CLIP-space features
    scores: torch.Tensor  # (frames, classes), raw SurgCLIP similarities
    targets: torch.Tensor  # (frames,), phase class IDs


def _patch_surgclip_tokenizer(surgclip: Any) -> None:
    """Fall back to Transformers' BertTokenizer for older surgclip releases."""
    original = surgclip.load_tokenizer

    def compatible(config):
        try:
            return original(config)
        except AttributeError as exc:  # pragma: no cover - older tokenizer release
            print(f"SurgCLIP tokenizer compatibility fallback: {exc}")
            from transformers import BertTokenizer

            return BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)

    surgclip.load_tokenizer = compatible


def _phase_id(label_rows: list[list[float]], video_id: str, frame: str) -> int:
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for row in label_rows:
        if len(row) <= 14:
            continue
        phase = int(row[14])
        if phase >= 0:
            if phase >= NUM_CLASSES:
                raise ValueError(
                    f"Invalid phase ID {phase} in {video_id}, frame {frame}."
                )
            counts[phase] += 1
    if counts.sum() == 0:
        raise ValueError(f"No phase annotation for {video_id}, frame {frame}.")
    return int(counts.argmax())


def _window_start(anchor: int, total: int, length: int, mode: str) -> int:
    if length >= total:
        return 0
    if mode == "centered":
        start = max(0, anchor - length // 2)
        if start + length > total:
            start = total - length
        return start
    if mode == "online":
        return max(0, anchor + 1 - length)
    raise ValueError(f"Unknown clip mode {mode!r}.")


class _ClipWindowReader:
    """Sliding frame-window reader that mirrors SurgCLIP's video preprocessor."""

    def __init__(self, paths: list[Path], preprocess, length: int, mode: str):
        if not paths:
            raise ValueError("Cannot build a clip from an empty video.")
        self.paths = paths
        self.preprocess = preprocess
        self.length = length
        self.mode = mode
        self.buffer: deque[tuple[int, torch.Tensor]] = deque()
        self.next_index = 0

    def _extend(self, last_index: int) -> None:
        while self.next_index <= last_index:
            with Image.open(self.paths[self.next_index]) as image:
                tensor = self.preprocess(image.convert("RGB"))
            self.buffer.append((self.next_index, tensor))
            self.next_index += 1

    def window(self, anchor: int) -> torch.Tensor:
        start = _window_start(anchor, len(self.paths), self.length, self.mode)
        end = min(start + self.length, len(self.paths))
        if self.mode == "online":
            end = anchor + 1
        self._extend(end - 1)
        while self.buffer and self.buffer[0][0] < start:
            self.buffer.popleft()
        frames = [tensor for idx, tensor in self.buffer if start <= idx < end]
        while len(frames) < self.length:
            frames.append(frames[-1])
        return torch.stack(frames)


@torch.inference_mode()
def _score_video(
    dataset,
    model: nn.Module,
    text_features: torch.Tensor,
    temperature: torch.Tensor,
    preprocess,
    device: torch.device,
    num_frames: int,
    clip_mode: str,
    batch_size: int,
    amp: bool,
) -> VideoScores:
    video_id = Path(dataset.img_dir).name
    annotation_frames = sorted(dataset.frames, key=int)
    image_paths = sorted(
        Path(dataset.img_dir).glob("*.png"), key=lambda path: int(path.stem)
    )
    image_indices = {path.stem: i for i, path in enumerate(image_paths)}
    try:
        anchors = [image_indices[str(frame).zfill(6)] for frame in annotation_frames]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Missing CholecT50 frame {exc.args[0]}.png in {dataset.img_dir}."
        ) from exc

    targets = torch.tensor(
        [
            _phase_id(dataset.label_data[frame], video_id, str(frame))
            for frame in annotation_frames
        ],
        dtype=torch.long,
    )
    reader = _ClipWindowReader(image_paths, preprocess, num_frames, clip_mode)
    embeddings: list[torch.Tensor] = []
    output: list[torch.Tensor] = []
    clips: list[torch.Tensor] = []

    for anchor in tqdm(anchors, desc=video_id, leave=False, unit="frame"):
        clips.append(reader.window(anchor))
        if len(clips) == batch_size:
            clip_embeddings, clip_scores = _score_clip_batch(
                model,
                torch.stack(clips),
                text_features,
                temperature,
                device,
                amp,
            )
            embeddings.append(clip_embeddings.cpu())
            output.append(clip_scores.cpu())
            clips.clear()
    if clips:
        clip_embeddings, clip_scores = _score_clip_batch(
            model, torch.stack(clips), text_features, temperature, device, amp
        )
        embeddings.append(clip_embeddings.cpu())
        output.append(clip_scores.cpu())
    return VideoScores(video_id, torch.cat(embeddings), torch.cat(output), targets)


def _score_clip_batch(
    model: nn.Module,
    clips: torch.Tensor,
    text_features: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    clips = clips.to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        _, pooled_visual = model.encode_vision(clips)
        visual = model.vision_proj(pooled_visual)
    visual = F.normalize(visual.float(), dim=-1)
    if visual.ndim == 3:
        # Match SurgCLIP.get_sim: normalize each temporal feature, calculate
        # similarities, then average over clip time. The mean projected vector
        # has the same similarity to text prototypes as that temporal average.
        embeddings = visual.mean(dim=1)
    else:
        embeddings = visual
    logits = embeddings @ text_features.T / temperature
    return embeddings, logits


@torch.inference_mode()
def extract_scores(
    split,
    model: nn.Module,
    text_features: torch.Tensor,
    temperature: torch.Tensor,
    preprocess,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, list[VideoScores]]:
    datasets = {
        "train": list(split.train_dataset.datasets),
        "validation": list(split.val_dataset.datasets),
        "test": list(split.test_dataset),
    }
    result: dict[str, list[VideoScores]] = {}
    for name, videos in datasets.items():
        result[name] = [
            _score_video(
                ds, model, text_features, temperature, preprocess, device,
                args.num_frames, args.clip_mode, args.extract_batch_size, args.amp,
            )
            for ds in tqdm(videos, desc=f"SurgCLIP {name}", unit="video")
        ]
    return result


def _cache_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cache_version": 2,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_variant": args.variant,
        "test_fold": args.test_fold,
        "model_name": args.model_name,
        "num_frames": args.num_frames,
        "clip_mode": args.clip_mode,
        "prompts": PHASE_PROMPTS,
    }


def _sinusoidal_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    scale = math.log(10_000) / max(half - 1, 1)
    frequencies = torch.exp(
        -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
    )
    angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
    if width % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class _TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, time_width: int):
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.time_projection = nn.Linear(time_width, width)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(x))
        h = h + self.time_projection(time_embedding).unsqueeze(-1)
        h = self.conv2(F.gelu(h))
        return (x + h) / math.sqrt(2.0)


class ConditionalTemporalDiffusion(nn.Module):
    """Diffuse residual corrections conditioned on SurgCLIP scores/features."""

    def __init__(
        self,
        condition_dim: int,
        num_classes: int = NUM_CLASSES,
        width: int = 96,
        layers: int = 6,
        diffusion_steps: int = 1000,
        correction_scale: float = 1.0,
    ):
        super().__init__()
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2")
        time_width = width * 2
        self.num_classes = num_classes
        self.diffusion_steps = diffusion_steps
        self.correction_scale = correction_scale
        self.time_mlp = nn.Sequential(
            nn.Linear(time_width, time_width), nn.SiLU(),
            nn.Linear(time_width, time_width),
        )
        self.input_projection = nn.Conv1d(num_classes + condition_dim, width, 1)
        self.blocks = nn.ModuleList(
            _TemporalResidualBlock(width, 2 ** (i % 5), time_width)
            for i in range(layers)
        )
        self.output_projection = nn.Sequential(
            nn.GELU(), nn.Conv1d(width, num_classes, 1)
        )
        betas = torch.linspace(1e-4, 2e-2, diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def forward(
        self,
        noisy_labels: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat((noisy_labels, condition), dim=-1).transpose(1, 2)
        h = self.input_projection(x)
        time_embedding = self.time_mlp(
            _sinusoidal_embedding(timesteps, self.time_mlp[0].in_features)
        )
        for block in self.blocks:
            h = block(h, time_embedding)
        return self.output_projection(h).transpose(1, 2)

    def training_loss(
        self,
        labels: torch.Tensor,
        condition: torch.Tensor,
        valid_mask: torch.Tensor,
        auxiliary_ce_weight: float = 1.0,
    ) -> torch.Tensor:
        batch_size = labels.shape[0]
        baseline_probs = condition[..., : self.num_classes]
        timesteps = torch.randint(
            self.diffusion_steps, (batch_size,), device=labels.device
        )
        clean_labels = F.one_hot(labels, self.num_classes).float()
        clean = clean_labels - baseline_probs
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[timesteps].view(batch_size, 1, 1)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self(noisy, condition, timesteps)
        predicted_residual = (
            noisy - (1.0 - alpha_bar).sqrt() * predicted_noise
        ) / alpha_bar.sqrt()
        mask = valid_mask.unsqueeze(-1).float()
        squared_error = (predicted_noise - noise).square() * mask
        diffusion_loss = squared_error.sum() / (
            mask.sum() * self.num_classes
        ).clamp(min=1.0)
        corrected_logits = baseline_probs.clamp(min=1e-6).log() + (
            self.correction_scale * predicted_residual.clamp(-1.0, 1.0)
        )
        cross_entropy = F.cross_entropy(
            corrected_logits.transpose(1, 2), labels, reduction="none"
        )
        auxiliary_loss = (cross_entropy * valid_mask.float()).sum() / valid_mask.sum().clamp(
            min=1
        )
        return diffusion_loss + auxiliary_ce_weight * auxiliary_loss

    @torch.inference_mode()
    def sample_probabilities(
        self,
        condition: torch.Tensor,
        sample_steps: int = 25,
        num_samples: int = 1,
        seed: int = 0,
    ) -> torch.Tensor:
        """DDIM-sample score residuals and apply their mean as a correction."""
        if not 1 <= sample_steps <= self.diffusion_steps:
            raise ValueError("sample_steps must be in [1, diffusion_steps]")
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1")
        was_training = self.training
        self.eval()
        condition = condition.float()
        batch_size, length, _ = condition.shape
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        timestep_values = torch.linspace(
            self.diffusion_steps - 1,
            0,
            sample_steps,
            device=condition.device,
        ).round().long().unique_consecutive()
        baseline_probs = condition[..., : self.num_classes].clamp(min=1e-6)
        residual_samples = []
        for _ in range(num_samples):
            x = torch.randn(
                batch_size,
                length,
                self.num_classes,
                device=condition.device,
                generator=generator,
            )
            x0 = x
            for i, timestep in enumerate(timestep_values):
                ts = torch.full(
                    (batch_size,), int(timestep), device=condition.device,
                    dtype=torch.long,
                )
                alpha_bar = self.alpha_bars[timestep]
                epsilon = self(x, condition, ts)
                x0 = (x - (1.0 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()
                if i + 1 < len(timestep_values):
                    previous = timestep_values[i + 1]
                    alpha_previous = self.alpha_bars[previous]
                    x = alpha_previous.sqrt() * x0 + (1.0 - alpha_previous).sqrt() * epsilon
                else:
                    x = x0
            residual_samples.append(x0.clamp(-1.0, 1.0))
        if was_training:
            self.train()
        residual = torch.stack(residual_samples).mean(dim=0)
        corrected_logits = baseline_probs.log() + self.correction_scale * residual
        return corrected_logits.softmax(dim=-1)


def _make_training_batch(
    videos: list[VideoScores],
    batch_size: int,
    crop_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embedding_dim = videos[0].embeddings.shape[-1]
    conditions = torch.zeros(batch_size, crop_length, NUM_CLASSES + embedding_dim)
    labels = torch.zeros(batch_size, crop_length, dtype=torch.long)
    mask = torch.zeros(batch_size, crop_length, dtype=torch.bool)
    for i in range(batch_size):
        video = videos[int(np.random.randint(len(videos)))]
        length = len(video.targets)
        start = int(np.random.randint(max(length - crop_length + 1, 1)))
        stop = min(start + crop_length, length)
        actual_length = stop - start
        score_condition = video.scores[start:stop].softmax(dim=-1)
        embedding_condition = F.normalize(video.embeddings[start:stop], dim=-1)
        condition = torch.cat((score_condition, embedding_condition), dim=-1)
        target = video.targets[start:stop]
        conditions[i, :actual_length] = condition
        labels[i, :actual_length] = target
        # Repeat the final input outside the loss mask to avoid padding edges.
        if actual_length < crop_length:
            conditions[i, actual_length:] = condition[-1]
            labels[i, actual_length:] = target[-1]
        mask[i, :actual_length] = True
    return conditions.to(device), labels.to(device), mask.to(device)


def _evaluate(
    videos: list[VideoScores],
    refiner: ConditionalTemporalDiffusion | None,
    device: torch.device,
    sample_steps: int,
    num_samples: int,
    seed: int = 0,
) -> tuple[dict[str, float], dict[str, float], dict[str, np.ndarray]]:
    metric_names = ["f1_macro", "mean_accuracy", "edit_score"]
    accumulators: list[MetricAccumulator] = []
    per_video_f1: list[float] = []
    per_video_accuracy: list[float] = []
    predictions: dict[str, np.ndarray] = {}
    for video_index, video in enumerate(
        tqdm(videos, desc="evaluate", leave=False, unit="video")
    ):
        score_condition = video.scores.softmax(dim=-1)
        embedding_condition = F.normalize(video.embeddings, dim=-1)
        condition = torch.cat((score_condition, embedding_condition), dim=-1)
        condition = condition.unsqueeze(0).to(device)
        if refiner is None:
            logits = video.scores.numpy()
        else:
            probs = refiner.sample_probabilities(
                condition,
                sample_steps=sample_steps,
                num_samples=num_samples,
                seed=seed + video_index,
            )[0]
            logits = probs.cpu().numpy()
        targets = video.targets.numpy()
        accumulator = MetricAccumulator(
            ["phase"], metric_names, {"phase": NUM_CLASSES}
        )
        accumulator.add_batch(
            {"phase": logits},
            {"phase": np.eye(NUM_CLASSES, dtype=np.float32)[targets]},
        )
        metrics = accumulator.compute()
        accumulators.append(accumulator)
        per_video_f1.append(metrics.get("phase_f1_macro", 0.0))
        pred = logits.argmax(axis=-1)
        per_video_accuracy.append(float(np.mean(pred == targets)))
        predictions[video.video_id] = pred.astype(np.int64)

    pooled = MetricAccumulator.merged(accumulators).compute()
    video_average = {
        "mean_video_macro_f1": float(np.mean(per_video_f1)),
        "mean_video_accuracy": float(np.mean(per_video_accuracy)),
    }
    return pooled, video_average, predictions


def _fit_refiner(
    train_videos: list[VideoScores],
    val_videos: list[VideoScores],
    args: argparse.Namespace,
    device: torch.device,
    on_epoch: Callable[[int, float, dict[str, float], dict[str, float]], None]
    | None = None,
) -> tuple[ConditionalTemporalDiffusion, dict[str, float]]:
    if not train_videos or not val_videos:
        raise ValueError("Both training and validation splits must contain videos.")
    model = ConditionalTemporalDiffusion(
        condition_dim=NUM_CLASSES + train_videos[0].embeddings.shape[-1],
        width=args.width,
        layers=args.layers,
        diffusion_steps=args.diffusion_steps,
        correction_scale=args.correction_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, float] = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for _ in tqdm(
            range(args.steps_per_epoch), desc=f"train {epoch}/{args.epochs}",
            leave=False, unit="step",
        ):
            condition, targets, mask = _make_training_batch(
                train_videos, args.batch_size, args.crop_length, device
            )
            loss = model.training_loss(
                targets,
                condition,
                mask,
                auxiliary_ce_weight=args.auxiliary_ce_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        validation, video_average, _ = _evaluate(
            val_videos, model, device, args.sample_steps,
            num_samples=1, seed=args.seed,
        )
        score = validation.get("phase_f1_macro", -float("inf"))
        print(
            f"epoch {epoch}/{args.epochs}: noise_loss={np.mean(losses):.5f} "
            f"val_macro_f1={score:.4f} "
            f"val_mean_video_f1={video_average['mean_video_macro_f1']:.4f}"
        )
        if on_epoch is not None:
            on_epoch(epoch, float(np.mean(losses)), validation, video_average)
        if score > best_score:
            best_score = score
            best_validation = {**validation, **video_average}
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Training did not produce a valid diffusion checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_validation


def _serialize_scores(splits: dict[str, list[VideoScores]]) -> dict[str, list[dict[str, Any]]]:
    return {
        name: [
            {
                "video_id": video.video_id,
                "embeddings": video.embeddings.cpu(),
                "scores": video.scores.cpu(),
                "targets": video.targets.cpu(),
            }
            for video in videos
        ]
        for name, videos in splits.items()
    }


def _deserialize_scores(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[VideoScores]]:
    return {
        name: [
            VideoScores(
                str(video["video_id"]),
                video["embeddings"].float(),
                video["scores"].float(),
                video["targets"].long(),
            )
            for video in videos
        ]
        for name, videos in data.items()
    }


def _attach_saved_embeddings(
    splits: dict[str, list[VideoScores]],
    embeddings_path: Path,
    expected_metadata: dict[str, Any],
) -> None:
    """Load the separately saved embeddings and align them by split/video."""
    artifact = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    if artifact.get("metadata") != expected_metadata:
        raise ValueError(
            "Embedding artifact metadata does not match this run's dataset, "
            "fold, model, or clip settings."
        )
    saved_splits = artifact.get("splits", {})
    if set(saved_splits) != set(splits):
        raise ValueError("Embedding artifact is missing one or more dataset splits.")
    for split_name, videos in splits.items():
        by_video = {
            str(item["video_id"]): item for item in saved_splits[split_name]
        }
        if set(by_video) != {video.video_id for video in videos}:
            raise ValueError(f"Embedding artifact video IDs differ in {split_name}.")
        for video in videos:
            item = by_video[video.video_id]
            embedding = item["embeddings"].float()
            target = item["targets"].long()
            if len(embedding) != len(video.targets) or not torch.equal(
                target, video.targets
            ):
                raise ValueError(
                    f"Saved embeddings/targets are misaligned for {video.video_id}."
                )
            video.embeddings = embedding
    print(f"Loaded SurgCLIP embeddings from {embeddings_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="CholecT50 root with videos/ and labels/")
    parser.add_argument("--variant", default="cholect50-crossval")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--model-name", default="SurgCLIP-B")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--clip-mode", choices=("centered", "online"), default="centered")
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="Use fp16 for SurgCLIP extraction on CUDA")
    parser.add_argument("--scores-cache", default=None, help="Optional reusable SurgCLIP score cache")
    parser.add_argument(
        "--embeddings-file",
        default=None,
        help="Reuse a saved surgclip_embeddings.pt artifact",
    )
    parser.add_argument("--output-dir", default="outputs/surgclip-diffusion")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-length", type=int, default=256)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--test-samples", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    parser.add_argument("--auxiliary-ce-weight", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.num_frames < 1 or args.extract_batch_size < 1:
        raise ValueError("num-frames and extract-batch-size must be positive")
    if args.test_fold not in range(1, 6):
        raise ValueError("test-fold must be between 1 and 5")
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.crop_length < 1:
        raise ValueError("epochs, steps-per-epoch, and crop-length must be positive")
    if args.sample_steps > args.diffusion_steps:
        raise ValueError("sample-steps cannot exceed diffusion-steps")
    if args.correction_scale < 0 or args.auxiliary_ce_weight < 0:
        raise ValueError("correction-scale and auxiliary-ce-weight must be nonnegative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"Device: {device}; dataset={args.variant}; held-out fold={args.test_fold}")

    split = CholecT50(
        dataset_dir=args.dataset_dir,
        dataset_variant=args.variant,
        test_fold=args.test_fold,
        augmentation_list=["original"],
        normalize=False,
        sequence_length=1,
    )
    cache_path = Path(args.scores_cache) if args.scores_cache else None
    if cache_path and cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        metadata = _cache_metadata(args)
        if cache.get("metadata") != metadata:
            raise ValueError("Score cache metadata mismatch; use a fresh cache path.")
        splits = _deserialize_scores(cache["splits"])
        print(f"Loaded SurgCLIP scores from {cache_path}")
    else:
        import surgclip

        _patch_surgclip_tokenizer(surgclip)
        model, preprocess, tokenizer = surgclip.load(
            args.model_name,
            device=device,
            num_frames=args.num_frames,
            temporal_modeling=args.num_frames > 1,
        )
        token_ids = surgclip.tokenize(PHASE_PROMPTS, tokenizer, device=device)
        with torch.inference_mode():
            _, pooled_text = model.encode_text(token_ids)
            text_features = F.normalize(model.text_proj(pooled_text.float()), dim=-1)
            temperature = model.temp.detach().float().clamp(min=1e-6)
            splits = extract_scores(
                split, model, text_features, temperature, preprocess, device, args
            )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"metadata": _cache_metadata(args), "splits": _serialize_scores(splits)},
                cache_path,
            )
            print(f"Saved SurgCLIP score cache to {cache_path}")
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.embeddings_file:
        _attach_saved_embeddings(
            splits, Path(args.embeddings_file), _cache_metadata(args)
        )

    for name, videos in splits.items():
        print(f"{name}: {len(videos)} videos, {sum(len(v.targets) for v in videos)} frames")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "surgclip_embeddings.pt"
    torch.save(
        {
            "metadata": _cache_metadata(args),
            "embedding_type": "temporal_mean_of_l2_normalized_projected_visual_features",
            "embedding_dim": int(splits["train"][0].embeddings.shape[-1]),
            "splits": {
                name: [
                    {
                        "video_id": video.video_id,
                        "embeddings": video.embeddings,
                        "targets": video.targets,
                    }
                    for video in videos
                ]
                for name, videos in splits.items()
            },
        },
        embeddings_path,
    )
    print(f"Saved SurgCLIP embeddings to {embeddings_path}")

    baseline_metrics, baseline_video_metrics, baseline_predictions = _evaluate(
        splits["test"], None, device, args.sample_steps, num_samples=1,
        seed=args.seed,
    )
    refiner, validation_metrics = _fit_refiner(
        splits["train"], splits["validation"], args, device
    )
    refined_metrics, refined_video_metrics, refined_predictions = _evaluate(
        splits["test"], refiner, device, args.sample_steps, args.test_samples,
        seed=args.seed + 10_000,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": refiner.state_dict(),
            "config": {
                "width": args.width,
                "layers": args.layers,
                "diffusion_steps": args.diffusion_steps,
                "condition_dim": NUM_CLASSES + splits["train"][0].embeddings.shape[-1],
                "correction_scale": args.correction_scale,
            },
            "metadata": _cache_metadata(args),
        },
        output_dir / "best_diffusion_refiner.pt",
    )
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        **{
            f"{video}_baseline": pred
            for video, pred in baseline_predictions.items()
        },
        **{
            f"{video}_diffusion": pred
            for video, pred in refined_predictions.items()
        },
    )
    result = {
        "experiment": "SurgCLIP vs SurgCLIP + conditional temporal diffusion",
        "metadata": _cache_metadata(args),
        "config": vars(args),
        "validation_diffusion_metrics": validation_metrics,
        "test": {
            "surgclip": {
                "pooled_frame_metrics": baseline_metrics,
                "mean_video_metrics": baseline_video_metrics,
            },
            "surgclip_plus_diffusion": {
                "pooled_frame_metrics": refined_metrics,
                "mean_video_metrics": refined_video_metrics,
            },
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    print("\nHeld-out test results")
    for variant, metrics in (
        ("SurgCLIP", baseline_metrics),
        ("SurgCLIP + diffusion", refined_metrics),
    ):
        video_metrics = (
            baseline_video_metrics if variant == "SurgCLIP" else refined_video_metrics
        )
        print(f"{variant}:")
        for key, value in sorted({**metrics, **video_metrics}.items()):
            print(f"  {key}: {value:.4f}")
    print(f"Saved results under {output_dir}")


if __name__ == "__main__":
    main()
