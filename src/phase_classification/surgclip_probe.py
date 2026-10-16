"""Linear and temporal probing of frozen SurgCLIP features on CholecT50.

This is a CholecT50 adaptation of the SurgLaVi paper's video linear-probing
and MS-TCN protocols. It consumes either the flat feature cache written by
``experiments/surgclip_lstm.py`` or the per-video ``surgclip_embeddings.pt``
artifact written by ``phase_classification.surgclip_diffusion``.

Examples::

    uv run python -m phase_classification.surgclip_probe \\
        --dataset-dir /data/cholect50 \\
        --embeddings-file outputs/surgclip-diffusion/fold1/surgclip_embeddings.pt \\
        --experiment both

    uv run python -m phase_classification.surgclip_probe \\
        --dataset-dir /data/cholect50 --embeddings-file /path/to/features.pt \\
        --experiment linear --linear-fractions 1.0

The cached visual encoder features remain frozen. The linear probe trains only
one affine classifier. The MS-TCN trains a temporal convolutional head on one
whole video per batch. The cache's existing clip window, temporal pooling, and
feature dimension are used as-is; this is not an exact reproduction of the
paper's concatenated 1024-D, 32-frame linear-probe representation.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.heads import DEFAULT_NUM_CLASSES
from phase_classification.metrics import MetricAccumulator

NUM_PHASES = DEFAULT_NUM_CLASSES["phase"]


@dataclass
class VideoFeatures:
    """Features and phase targets in chronological order for one video."""

    video_id: str
    features: torch.Tensor  # (T, D)
    targets: torch.Tensor  # (T,)


def _phase_id(rows: list[list[float]], video_id: str, frame: str) -> int:
    counts = np.zeros(NUM_PHASES, dtype=np.int64)
    for row in rows:
        if len(row) > 14 and row[14] >= 0:
            phase = int(row[14])
            if phase >= NUM_PHASES:
                raise ValueError(f"Invalid phase id {phase} at {video_id}/{frame}.")
            counts[phase] += 1
    if not counts.any():
        raise ValueError(f"No phase annotation at {video_id}/{frame}.")
    return int(counts.argmax())


def _all_video_datasets(split: CholecT50) -> dict[str, list[Any]]:
    return {
        "train": list(split.train_dataset.datasets),
        "validation": list(split.val_dataset.datasets),
        "test": list(split.test_dataset),
    }


def _check_metadata(cache: dict[str, Any], args: argparse.Namespace) -> None:
    metadata = cache.get("metadata", {})
    variant = metadata.get("dataset_variant", cache.get("variant"))
    fold = metadata.get("test_fold", cache.get("test_fold"))
    if variant is not None and variant != args.variant:
        raise ValueError(
            f"Cache dataset variant is {variant!r}, but requested {args.variant!r}."
        )
    if fold is not None and int(fold) != args.test_fold:
        raise ValueError(
            f"Cache test fold is {fold}, but requested fold {args.test_fold}."
        )


def _cache_features_by_video(cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Normalize the two cache layouts used by the repository to video -> X."""
    if isinstance(cache.get("splits"), dict):
        result: dict[str, torch.Tensor] = {}
        for videos in cache["splits"].values():
            for item in videos:
                video_id = str(item["video_id"])
                if video_id in result:
                    raise ValueError(f"Video {video_id} occurs in multiple cache splits.")
                result[video_id] = torch.as_tensor(item["embeddings"]).float()
        return result

    if "features" not in cache or "paths" not in cache:
        raise ValueError(
            "Unsupported embeddings file. Expected either {'features', 'paths'} "
            "or the per-video {'splits': ...} SurgCLIP artifact."
        )
    features = torch.as_tensor(cache["features"]).float()
    paths = cache["paths"]
    if features.ndim != 2 or len(features) != len(paths):
        raise ValueError("Flat cache must have features [N, D] and one path per row.")
    result = {}
    indices: dict[str, list[int]] = {}
    for idx, raw_path in enumerate(paths):
        path = Path(str(raw_path))
        indices.setdefault(path.parent.name, []).append(idx)
    for video_id, video_indices in indices.items():
        result[video_id] = features[video_indices]
    return result


def _make_video_features(
    datasets: dict[str, list[Any]], cache: dict[str, Any], args: argparse.Namespace
) -> dict[str, list[VideoFeatures]]:
    _check_metadata(cache, args)
    cached = _cache_features_by_video(cache)

    saved_targets: dict[str, torch.Tensor] = {}
    if isinstance(cache.get("splits"), dict):
        for videos in cache["splits"].values():
            for item in videos:
                if "targets" in item:
                    saved_targets[str(item["video_id"])] = torch.as_tensor(
                        item["targets"], dtype=torch.long
                    )

    output: dict[str, list[VideoFeatures]] = {}
    all_dataset_ids = {
        str(Path(ds.img_dir).name)
        for video_datasets in datasets.values()
        for ds in video_datasets
    }
    missing_videos = all_dataset_ids - cached.keys()
    if missing_videos:
        raise ValueError(f"Feature cache is missing videos: {sorted(missing_videos)[:10]}")

    for split_name, video_datasets in datasets.items():
        output[split_name] = []
        for ds in video_datasets:
            video_id = str(Path(ds.img_dir).name)
            frames = sorted(ds.frames, key=int)
            targets = torch.tensor(
                [_phase_id(ds.label_data[frame], video_id, str(frame)) for frame in frames],
                dtype=torch.long,
            )
            features = cached[video_id]
            if features.ndim != 2 or len(features) != len(frames):
                raise ValueError(
                    f"{video_id}: cache has {len(features)} feature rows, "
                    f"but CholecT50 has {len(frames)} annotated frames."
                )
            if video_id in saved_targets and not torch.equal(saved_targets[video_id], targets):
                raise ValueError(f"Cached targets do not align with CholecT50 labels for {video_id}.")
            output[split_name].append(VideoFeatures(video_id, features, targets))
    return output


def _metric_accumulator() -> MetricAccumulator:
    return MetricAccumulator(
        ["phase"],
        ["f1_macro", "mean_accuracy", "edit_score"],
        {"phase": NUM_PHASES},
    )


def _metrics_for_predictions(
    videos: list[VideoFeatures], predictions: list[np.ndarray]
) -> dict[str, float]:
    if len(videos) != len(predictions):
        raise ValueError("Need one prediction sequence per video.")
    per_video_acc = []
    per_video_f1 = []
    accumulators = []
    segment_f1: dict[str, list[float]] = {"10": [], "25": [], "50": []}
    for video, predicted in zip(videos, predictions):
        target = video.targets.numpy()
        predicted = np.asarray(predicted, dtype=np.int64)
        if predicted.shape != target.shape:
            raise ValueError(f"Prediction/target length mismatch for {video.video_id}.")
        acc = _metric_accumulator()
        logits = np.eye(NUM_PHASES, dtype=np.float32)[predicted]
        labels = np.eye(NUM_PHASES, dtype=np.int32)[target]
        acc.add_batch({"phase": logits}, {"phase": labels})
        accumulators.append(acc)
        per_video_acc.append(float(np.mean(predicted == target)))
        per_video_f1.append(_macro_f1(target, predicted))
        for threshold in (10, 25, 50):
            segment_f1[str(threshold)].append(_segmental_f1(target, predicted, threshold / 100))

    merged = MetricAccumulator.merged(accumulators).compute()
    result = {key: float(value) for key, value in merged.items()}
    result["video_mean_accuracy"] = float(np.mean(per_video_acc))
    result["video_mean_f1_macro"] = float(np.mean(per_video_f1))
    for threshold, values in segment_f1.items():
        result[f"segment_f1@{threshold}"] = float(np.mean(values))
    return result


def _macro_f1(target: np.ndarray, predicted: np.ndarray) -> float:
    values = []
    for phase in range(NUM_PHASES):
        tp = np.sum((target == phase) & (predicted == phase))
        fp = np.sum((target != phase) & (predicted == phase))
        fn = np.sum((target == phase) & (predicted != phase))
        denom = 2 * tp + fp + fn
        values.append(float(2 * tp / denom) if denom else 0.0)
    return float(np.mean(values))


def _segments(sequence: np.ndarray) -> list[tuple[int, int, int]]:
    if len(sequence) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1]) + 1]
    ends = np.r_[starts[1:], len(sequence)]
    return [(int(sequence[start]), int(start), int(end)) for start, end in zip(starts, ends)]


def _segmental_f1(target: np.ndarray, predicted: np.ndarray, threshold: float) -> float:
    """Greedy segment F1 at an IoU threshold, following temporal-segmentation practice."""
    true_segments = _segments(target)
    pred_segments = _segments(predicted)
    if not pred_segments and not true_segments:
        return 1.0
    matched: set[int] = set()
    true_positive = 0
    false_positive = 0
    for pred_label, pred_start, pred_end in pred_segments:
        best_iou = 0.0
        best_index = -1
        for idx, (true_label, true_start, true_end) in enumerate(true_segments):
            intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
            union = max(pred_end, true_end) - min(pred_start, true_start)
            iou = intersection / union if union else 0.0
            if true_label == pred_label and iou > best_iou:
                best_iou, best_index = iou, idx
        if best_iou >= threshold and best_index not in matched:
            true_positive += 1
            matched.add(best_index)
        else:
            false_positive += 1
    false_negative = len(true_segments) - len(matched)
    denom = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denom) if denom else 0.0


def _evaluate_linear(
    model: nn.Module, videos: list[VideoFeatures], device: torch.device
) -> tuple[dict[str, float], list[np.ndarray]]:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for video in videos:
            logits = model(video.features.to(device))
            predictions.append(logits.argmax(dim=-1).cpu().numpy())
    return _metrics_for_predictions(videos, predictions), predictions


def _fit_linear_candidate(
    train_videos: list[VideoFeatures],
    val_videos: list[VideoFeatures],
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[nn.Linear, float]:
    torch.manual_seed(seed)
    features = torch.cat([video.features for video in train_videos])
    targets = torch.cat([video.targets for video in train_videos])
    loader = DataLoader(
        TensorDataset(features, targets), batch_size=batch_size, shuffle=True
    )
    model = nn.Linear(features.shape[-1], NUM_PHASES).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    best_f1 = -1.0
    best_state = None
    val_x = torch.cat([video.features for video in val_videos]).to(device)
    val_y = torch.cat([video.targets for video in val_videos]).numpy()
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_pred = model(val_x).argmax(dim=-1).cpu().numpy()
        score = _macro_f1(val_y, val_pred)
        if score > best_f1:
            best_f1 = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Linear probe training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_f1


def run_linear_probe(
    splits: dict[str, list[VideoFeatures]], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    """Video few-/full-shot linear probe with validation-tuned SGD settings."""
    rng = np.random.default_rng(args.seed)
    fractions = [float(value) for value in args.linear_fractions.split(",") if value.strip()]
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError("--linear-fractions must contain values in (0, 1].")
    output: dict[str, Any] = {"protocol": "frozen-feature video-level linear probe", "fractions": {}}
    candidates = [
        (lr, wd, momentum)
        for lr in (1e-3, 1e-4)
        for wd in (0.0, 0.01)
        for momentum in (0.9, 0.95)
    ]
    for fraction in fractions:
        repeats = 1 if math.isclose(fraction, 1.0) else args.linear_repeats
        runs = []
        for repeat in range(repeats):
            count = max(1, math.ceil(len(splits["train"]) * fraction))
            selected = rng.choice(len(splits["train"]), size=count, replace=False)
            train_videos = [splits["train"][int(index)] for index in selected]
            best: tuple[float, nn.Linear, tuple[float, float, float]] | None = None
            for learning_rate, weight_decay, momentum in candidates:
                model, val_f1 = _fit_linear_candidate(
                    train_videos,
                    splits["validation"],
                    device,
                    learning_rate,
                    weight_decay,
                    momentum,
                    args.linear_epochs,
                    args.linear_batch_size,
                    args.seed + repeat,
                )
                if best is None or val_f1 > best[0]:
                    best = (val_f1, model, (learning_rate, weight_decay, momentum))
            if best is None:
                raise RuntimeError("No linear-probe hyperparameter candidate was trained.")
            val_metrics, _ = _evaluate_linear(best[1], splits["validation"], device)
            test_metrics, _ = _evaluate_linear(best[1], splits["test"], device)
            run = {
                "repeat": repeat,
                "train_video_ids": [video.video_id for video in train_videos],
                "selected_hyperparameters": {
                    "learning_rate": best[2][0],
                    "weight_decay": best[2][1],
                    "momentum": best[2][2],
                },
                "best_validation_macro_f1": float(best[0]),
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            runs.append(run)
            fraction_dir = Path(args.output_dir) / f"linear_{int(fraction * 100):03d}pct"
            fraction_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best[1].cpu().state_dict(),
                    "feature_dim": best[1].in_features,
                    "num_classes": NUM_PHASES,
                    "train_video_ids": run["train_video_ids"],
                    "selected_hyperparameters": run["selected_hyperparameters"],
                },
                fraction_dir / f"repeat_{repeat + 1}.pt",
            )
        output["fractions"][str(fraction)] = {
            "runs": runs,
            "mean_test_video_accuracy": float(
                np.mean([run["test_metrics"]["video_mean_accuracy"] for run in runs])
            ),
            "mean_test_video_f1_macro": float(
                np.mean([run["test_metrics"]["video_mean_f1_macro"] for run in runs])
            ),
        }
    return output


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SingleMSTCNStage(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_layers: int, num_f_maps: int):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, num_f_maps, kernel_size=1)
        self.layers = nn.Sequential(
            *(TemporalResidualBlock(num_f_maps, 2**index) for index in range(num_layers))
        )
        self.out_conv = nn.Conv1d(num_f_maps, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.layers(self.in_conv(x)))


class FeatureMSTCN(nn.Module):
    """Standard multi-stage TCN operating on frozen feature sequences."""

    def __init__(self, feature_dim: int, num_classes: int, stages: int, layers: int, maps: int):
        super().__init__()
        self.stages = nn.ModuleList(
            [SingleMSTCNStage(feature_dim, num_classes, layers, maps)]
            + [SingleMSTCNStage(num_classes, num_classes, layers, maps) for _ in range(stages - 1)]
        )
        self.stage_logits: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, D); stages consume channel-first sequences.
        frame_features = features.transpose(1, 2)
        stage_logits = []
        stage_input = frame_features
        for index, stage in enumerate(self.stages):
            logits = stage(stage_input)
            stage_logits.append(logits)
            if index + 1 < len(self.stages):
                stage_input = torch.softmax(logits, dim=1)
        self.stage_logits = stage_logits
        return stage_logits[-1].transpose(1, 2)


def _mstcn_loss(model: FeatureMSTCN, target: torch.Tensor, tau: float) -> torch.Tensor:
    loss = torch.zeros((), device=target.device, dtype=torch.float32)
    for logits in model.stage_logits:
        loss = loss + F.cross_entropy(logits, target)
        probabilities = torch.softmax(logits, dim=1)
        if probabilities.shape[-1] > 1:
            difference = (probabilities[:, :, 1:] - probabilities[:, :, :-1]).square()
            loss = loss + torch.clamp(difference, max=tau).mean()
    return loss


def _run_mstcn_epoch(
    model: FeatureMSTCN,
    videos: list[VideoFeatures],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    tau: float,
) -> tuple[float, dict[str, float], list[np.ndarray]]:
    train = optimizer is not None
    model.train(train)
    order = np.random.permutation(len(videos)) if train else np.arange(len(videos))
    total_loss = 0.0
    predictions: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in videos]
    with torch.set_grad_enabled(train):
        for index in order:
            video = videos[int(index)]
            features = video.features.unsqueeze(0).to(device)
            targets = video.targets.unsqueeze(0).to(device)
            logits = model(features)
            loss = _mstcn_loss(model, targets, tau)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach())
            predictions[int(index)] = logits[0].argmax(dim=-1).detach().cpu().numpy()
    metrics = _metrics_for_predictions(videos, predictions)
    return total_loss / max(len(videos), 1), metrics, predictions


def run_mstcn(
    splits: dict[str, list[VideoFeatures]], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    feature_dim = int(splits["train"][0].features.shape[-1])
    model = FeatureMSTCN(
        feature_dim,
        NUM_PHASES,
        args.mstcn_stages,
        args.mstcn_layers,
        args.mstcn_maps,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.mstcn_lr, weight_decay=args.mstcn_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.mstcn_epochs)
    best_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.mstcn_epochs + 1):
        train_loss, train_metrics, _ = _run_mstcn_epoch(
            model, splits["train"], device, optimizer, args.smoothing_tau
        )
        val_loss, val_metrics, _ = _run_mstcn_epoch(
            model, splits["validation"], device, None, args.smoothing_tau
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_metrics": train_metrics,
                "validation_loss": val_loss,
                "validation_metrics": val_metrics,
            }
        )
        print(
            f"MS-TCN epoch {epoch}/{args.mstcn_epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_video_f1={val_metrics['video_mean_f1_macro']:.4f}"
        )
        score = val_metrics["video_mean_f1_macro"]
        if score > best_f1:
            best_f1 = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("MS-TCN training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_loss, test_metrics, _ = _run_mstcn_epoch(
        model, splits["test"], device, None, args.smoothing_tau
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "feature_dim": feature_dim,
            "num_classes": NUM_PHASES,
            "stages": args.mstcn_stages,
            "layers": args.mstcn_layers,
            "maps": args.mstcn_maps,
            "best_validation_video_f1_macro": best_f1,
        },
        output_dir / "mstcn_best.pt",
    )
    return {
        "protocol": "frozen-feature full-video MS-TCN temporal probe",
        "settings": {
            "stages": args.mstcn_stages,
            "layers_per_stage": args.mstcn_layers,
            "feature_maps": args.mstcn_maps,
            "optimizer": "AdamW",
            "learning_rate": args.mstcn_lr,
            "weight_decay": args.mstcn_weight_decay,
            "epochs": args.mstcn_epochs,
            "one_full_video_per_batch": True,
        },
        "best_validation_video_f1_macro": best_f1,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="CholecT50 root with videos/ and labels/")
    parser.add_argument("--embeddings-file", required=True, help="SurgCLIP embedding .pt artifact")
    parser.add_argument("--variant", default="cholect50-crossval")
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--experiment", choices=("linear", "mstcn", "both"), default="both")
    parser.add_argument("--output-dir", default="outputs/surgclip-probes")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--linear-fractions", default="0.1,0.5,1.0")
    parser.add_argument("--linear-repeats", type=int, default=5)
    parser.add_argument("--linear-epochs", type=int, default=50)
    parser.add_argument("--linear-batch-size", type=int, default=256)
    parser.add_argument("--mstcn-epochs", type=int, default=100)
    parser.add_argument("--mstcn-stages", type=int, default=3)
    parser.add_argument("--mstcn-layers", type=int, default=8)
    parser.add_argument("--mstcn-maps", type=int, default=64)
    parser.add_argument("--mstcn-lr", type=float, default=0.01)
    parser.add_argument("--mstcn-weight-decay", type=float, default=0.01)
    parser.add_argument("--smoothing-tau", type=float, default=4.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.linear_repeats < 1 or args.linear_epochs < 1 or args.mstcn_epochs < 1:
        raise ValueError("Repeat and epoch counts must be positive.")
    if min(args.mstcn_stages, args.mstcn_layers, args.mstcn_maps) < 1:
        raise ValueError("MS-TCN stages, layers, and feature maps must be positive.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    cache_path = Path(args.embeddings_file)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    split = CholecT50(
        dataset_dir=args.dataset_dir,
        dataset_variant=args.variant,
        test_fold=args.test_fold,
        augmentation_list=["original"],
        normalize=False,
        sequence_length=1,
    )
    datasets = _all_video_datasets(split)
    features = _make_video_features(datasets, cache, args)
    feature_dim = int(features["train"][0].features.shape[-1])
    print(
        f"Device={device}; feature_dim={feature_dim}; "
        f"train/val/test videos={len(features['train'])}/{len(features['validation'])}/"
        f"{len(features['test'])}"
    )
    print(
        "Protocol note: using the supplied cached features as-is; these may differ "
        "from the paper's 1024-D concatenated, 32-frame mean-pooled linear-probe features."
    )

    results: dict[str, Any] = {
        "dataset": args.variant,
        "test_fold": args.test_fold,
        "cache": str(cache_path),
        "feature_dim": feature_dim,
        "cache_metadata": cache.get("metadata", {}),
        "paper_protocol_note": (
            "CholecT50 adaptation using cached features. The paper probes concatenated "
            "768-D backbone and 256-D projection features, with 32-frame mean pooling."
        ),
    }
    if args.experiment in ("linear", "both"):
        results["linear_probe"] = run_linear_probe(features, args, device)
    if args.experiment in ("mstcn", "both"):
        results["mstcn"] = run_mstcn(features, args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as stream:
        json.dump(results, stream, indent=2)
    print(f"Saved probe metrics: {metrics_path}")


if __name__ == "__main__":
    main()
