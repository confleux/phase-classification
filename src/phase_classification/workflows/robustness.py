#!/usr/bin/env python3
"""Robustness of a trained model under synthetic input degradations.

Applies several types of synthetic artifacts to the CholecT50 test split at
increasing strength levels, runs pure inference, and reports how each head
metric degrades with the artifact intensity (the "metric vs artifact strength"
curve). This answers the question *why the model fails on bad frames* by
showing where each degradation breaks the model and which classes suffer first.

Degradations are implemented in raw pixel space with Numpy + OpenCV only
(no torchvision augmentation), injected *before* the loader's Resize/Normalize
pipeline, so the model sees exactly the same preprocessing as at training time.

Usage:
    uv run phase-cls-robustness [--config CONFIG] [--checkpoint CKPT]
                         [--artifacts gaussian_blur,noise] [--levels 6]
                         [--max-videos N] [--batch-size 32] [--output DIR]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # headless-safe; works on remote kernels without a display
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.heads import SINGLE_LABEL_HEADS, head_index
from phase_classification.loader import load_config
from phase_classification.metrics import MetricAccumulator
from phase_classification.models import build_model
from phase_classification.workflows import PROJECT_ROOT

# Canonical metric set for this experiment (aggregates for curves, per-class for
# the per-class accuracy heatmap). Matches the default `validation.metrics`.
METRIC_NAMES = [
    "jaccard",
    "f1_macro",
    "mean_accuracy",
    "edit_score",
    "accuracy_per_class",
    "f1_per_class",
]

# Aggregate metrics drawn on the degradation curves.
CURVE_METRICS = ["jaccard", "f1_macro", "mean_accuracy", "edit_score"]


# --- Synthetic degradations (Numpy + OpenCV, uint8 pixel space) -------------
def _gaussian_blur(img: np.ndarray, strength: float, rng) -> np.ndarray:
    sigma = 0.5 + 7.5 * strength
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)


def _noise(img: np.ndarray, strength: float, rng) -> np.ndarray:
    sigma = 80.0 * strength
    if sigma <= 0.0:
        return img
    out = img.astype(np.float32) + rng.normal(0.0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _darkening(img: np.ndarray, strength: float, rng) -> np.ndarray:
    factor = 1.0 - 0.9 * strength
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)


def _smoke(img: np.ndarray, strength: float, rng) -> np.ndarray:
    alpha = 0.9 * strength
    if alpha <= 0.0:
        return img
    h, w = img.shape[:2]
    # Low-frequency hazy overlay: coarse blotches upsampled + blurred.
    patch = rng.normal(190.0, 45.0, (max(h // 8, 1), max(w // 8, 1)))
    overlay = cv2.resize(patch.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    overlay = np.stack([overlay] * 3, axis=-1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=16)
    return cv2.addWeighted(overlay, alpha, img.astype(np.float32), 1.0 - alpha, 0).astype(
        np.uint8
    )


def _sharpness(img: np.ndarray, strength: float, rng) -> np.ndarray:
    amount = -0.8 * strength  # negative unsharp: soften edges, keep luminance
    if amount == 0.0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3).astype(np.float32)
    out = img.astype(np.float32) + (img.astype(np.float32) - blur) * amount
    return np.clip(out, 0, 255).astype(np.uint8)


# name -> {label, param_label, param(strength)->physical value, apply(img, s, rng)}
DEGRADATIONS: dict[str, dict] = {
    "gaussian_blur": {
        "label": "Gaussian blur",
        "param_label": "sigma",
        "param": lambda s: round(0.5 + 7.5 * s, 3),
        "apply": _gaussian_blur,
    },
    "noise": {
        "label": "Gaussian noise",
        "param_label": "std (uint8)",
        "param": lambda s: round(80.0 * s, 3),
        "apply": _noise,
    },
    "darkening": {
        "label": "Darkening",
        "param_label": "brightness x",
        "param": lambda s: round(1.0 - 0.9 * s, 3),
        "apply": _darkening,
    },
    "smoke": {
        "label": "Smoke / haze",
        "param_label": "opacity",
        "param": lambda s: round(0.9 * s, 3),
        "apply": _smoke,
    },
    "sharpness": {
        "label": "Sharpness loss",
        "param_label": "edge amount",
        "param": lambda s: round(-0.8 * s, 3),
        "apply": _sharpness,
    },
}


class Degrade:
    """PIL->PIL transform applying one artifact at a given strength.

    Operates in uint8 pixel space with Numpy + OpenCV and is slotted *before*
    the loader's Resize/ToTensor/Normalize pipeline so the artifact acts on the
    raw source frame, exactly like real sensor/optics degradation.
    """

    def __init__(self, name: str, strength: float, rng: np.random.Generator):
        self.apply = DEGRADATIONS[name]["apply"]
        self.strength = strength
        self._rng = rng

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        rgb = np.asarray(pil_img.convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        degraded = self.apply(bgr, self.strength, self._rng)
        return Image.fromarray(np.ascontiguousarray(degraded[:, :, ::-1]))


# --- Checkpoint / model helpers ---------------------------------------------
def load_checkpoint_state(path: Path, model) -> None:
    """Load weights into ``model`` from a checkpoint of any common format.

    Handles:
      - a raw ``state_dict`` (``OrderedDict`` of tensors),
      - a full checkpoint ``dict`` wrapping the state under a common key
        (``state_dict`` / ``model`` / ``model_state_dict`` / ``weights``),
      - a pickled full ``nn.Module`` (loaded directly).

    ``weights_only=False`` is used so pickled full models (which reference
    custom classes) can be unpickled. Only load files you trust.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, torch.nn.Module):
        model.load_state_dict(obj.state_dict())
        print(f"Loaded pickled model ({type(obj).__name__}) from {path}")
        return

    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "weights"):
            if key in obj and isinstance(obj[key], dict):
                model.load_state_dict(obj[key])
                print(f"Loaded state_dict under key '{key}' from {path}")
                return
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            model.load_state_dict(obj)
            print(f"Loaded raw state_dict from {path}")
            return

    raise ValueError(
        f"Unrecognized checkpoint format in {path}. Expected a state_dict, "
        "a dict with a 'state_dict'/'model' key, or a pickled nn.Module."
    )


@torch.no_grad()
def evaluate_video(
    video_ds,
    model,
    device,
    heads: list[str],
    seq_len: int,
    batch_size: int,
    num_classes: dict[str, int],
    metric_names: list[str],
) -> MetricAccumulator:
    """Run inference over one video and accumulate pred/target per head.

    Temporal models (seq_len >= 2) use a causal sliding window and take the
    prediction of the last frame of each window (same semantics as
    ``visualise_phase.py`` and the test loop); per-frame models batch frames.
    """
    acc = MetricAccumulator(heads, metric_names, num_classes)
    preds: dict[str, list] = {h: [] for h in heads}
    gts: dict[str, list] = {h: [] for h in heads}

    if seq_len > 1:
        for i in range(len(video_ds)):
            images, labels = video_ds[i]  # (L, C, H, W) window, (5, L) labels
            images = images.unsqueeze(0).to(device)
            logits = model(images)
            for h in heads:
                lo = logits[h].squeeze(0)  # (L, K)
                p = torch.softmax(lo, dim=-1).cpu().numpy()
                preds[h].append(p[-1].argmax(-1))  # prediction of the last frame
                gts[h].append(labels[head_index(h)][-1].argmax(-1).cpu().numpy())
    else:
        for start in range(0, len(video_ds), batch_size):
            end = min(start + batch_size, len(video_ds))
            batch_imgs, batch_labels = [], []
            for i in range(start, end):
                img, lab = video_ds[i]
                batch_imgs.append(img)
                batch_labels.append(lab)
            images = torch.stack(batch_imgs).to(device)
            logits = model(images)
            for h in heads:
                p = torch.softmax(logits[h], dim=-1).cpu().numpy()
                preds[h].append(p.argmax(-1))
                gts[h].append(
                    torch.stack([l[head_index(h)] for l in batch_labels])
                    .argmax(-1)
                    .cpu()
                    .numpy()
                )

    for h in heads:
        acc._append(h, np.concatenate(preds[h]), np.concatenate(gts[h]))
    return acc


def _evaluate_videos(
    video_ds_list, model, device, heads, seq_len, batch_size, num_classes, metric_names
) -> dict[str, float]:
    """Evaluate all test videos and merge accumulators (official test semantics)."""
    accs = [
        evaluate_video(
            ds, model, device, heads, seq_len, batch_size, num_classes, metric_names
        )
        for ds in video_ds_list
    ]
    if not accs:
        return {}
    return MetricAccumulator.merged(accs).compute()


def _evaluate_degraded(
    video_ds_list,
    orig_transforms,
    artifact: str,
    strength: float,
    seed: int,
    model,
    device,
    heads,
    seq_len,
    batch_size,
    num_classes,
    metric_names,
) -> dict[str, float]:
    """Wrap each video's transform with the artifact, then evaluate."""
    rng = np.random.default_rng(seed)
    for ds, orig in zip(video_ds_list, orig_transforms):
        ds.transform = transforms.Compose([Degrade(artifact, strength, rng), orig])
    return _evaluate_videos(
        video_ds_list, model, device, heads, seq_len, batch_size, num_classes, metric_names
    )


# --- Reporting helpers -------------------------------------------------------
def _primary_key(heads: list[str]) -> str:
    return f"{heads[0]}_f1_macro"


def _breakdown_strength(rows: list[dict], primary: str) -> float | None:
    """Smallest strength where the primary metric drops >10% vs the clean row."""
    clean = next(r for r in rows if r["artifact"] == "clean")[primary]
    threshold = 0.9 * clean
    for row in rows:
        if row["artifact"] == "clean":
            continue
        if row[primary] < threshold:
            return row["strength"]
    return None


def _write_csv(rows: list[dict], path: Path) -> None:
    columns = ["artifact", "level", "strength", "param_value"]
    metric_keys = sorted({k for r in rows for k in r if k not in set(columns)})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns + metric_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns + metric_keys})
    print("Saved:", path)


def _plot_curves(rows: list[dict], heads: list[str], output_dir: Path) -> None:
    """One figure per head: metric vs strength, one subplot per artifact."""
    for head in heads:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.ravel()
        clean = next(r for r in rows if r["artifact"] == "clean")
        for ax, artifact in zip(axes, DEGRADATIONS):
            spec = DEGRADATIONS[artifact]
            level_rows = [r for r in rows if r["artifact"] == artifact]
            for metric in CURVE_METRICS:
                key = f"{head}_{metric}"
                ax.plot(
                    [r["strength"] for r in level_rows],
                    [r[key] for r in level_rows],
                    marker="o",
                    label=metric,
                )
            for metric in CURVE_METRICS:
                ax.axhline(clean[f"{head}_{metric}"], color="gray", ls="--", lw=0.8)
            ax.set_title(
                f"{spec['label']}\n{spec['param_label']} "
                f"{spec['param'](0.0):g} -> {spec['param'](1.0):g}"
            )
            ax.set_xlabel("strength")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
        fig.suptitle(f"Robustness curves — head '{head}'", fontsize=14)
        plt.tight_layout()
        path = output_dir / f"degradation_curves_{head}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", path)


def _plot_per_class(rows: list[dict], heads: list[str], num_classes, output_dir: Path) -> None:
    """Heatmap of per-class accuracy vs strength, one subplot per artifact."""
    for head in heads:
        if head not in SINGLE_LABEL_HEADS:
            continue
        n = int(num_classes[head])
        fig, axes = plt.subplots(1, len(DEGRADATIONS), figsize=(18, 4), sharey=True)
        for ax, artifact in zip(axes, DEGRADATIONS):
            spec = DEGRADATIONS[artifact]
            level_rows = [r for r in rows if r["artifact"] == artifact]
            strengths = [r["strength"] for r in level_rows]
            mat = np.zeros((n, len(level_rows)))
            for j, r in enumerate(level_rows):
                for i in range(n):
                    mat[i, j] = r.get(f"{head}_accuracy_class_{i}", 0.0)
            im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(len(strengths)))
            ax.set_xticklabels([f"{s:.2f}" for s in strengths], rotation=45, fontsize=7)
            ax.set_yticks(range(n))
            ax.set_title(spec["label"], fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel(f"{head} class")
            ax.set_xlabel("strength", fontsize=8)
        fig.colorbar(im, ax=axes, shrink=0.8, label="class accuracy")
        fig.suptitle(f"Per-class accuracy vs artifact strength — head '{head}'", fontsize=14)
        plt.tight_layout()
        path = output_dir / f"per_class_accuracy_{head}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", path)


# --- CLI ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="YAML config used for training",
    )
    p.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"),
        help="Path to the trained checkpoint (.pt/.pth)",
    )
    p.add_argument(
        "--artifacts",
        default=",".join(DEGRADATIONS),
        help=f"Comma-separated artifacts to sweep (default: all of "
        f"{', '.join(DEGRADATIONS)})",
    )
    p.add_argument(
        "--levels",
        type=int,
        default=6,
        help="Number of strength levels per artifact, including 0 = clean (default: 6)",
    )
    p.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Cap the number of test videos evaluated (default: all)",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "robustness"),
        help="Output directory for plots and CSV",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [a.strip() for a in args.artifacts.split(",") if a.strip()]
    unknown = set(artifacts) - set(DEGRADATIONS)
    if unknown:
        raise ValueError(
            f"Unknown artifact(s): {sorted(unknown)}. "
            f"Available: {', '.join(DEGRADATIONS)}."
        )
    if args.levels < 2:
        raise ValueError("--levels must be >= 2 (level 0 is the clean baseline)")

    heads = list(cfg.training.heads)
    num_classes = dict(cfg.model.num_classes)
    seq_len = cfg.dataset.sequence_length
    primary = _primary_key(heads)

    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name} | params: {cfg.model.params}")
    print(f"Heads: {heads} | sequence_length: {seq_len}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Artifacts: {artifacts} | levels: {args.levels} | output: {output_dir}")

    # --- Load model ---------------------------------------------------------
    device = resolve_device(cfg.runtime.device, cfg.runtime.cuda_device)
    print("Device:", device)
    model = build_model(cfg.model, heads=heads)
    load_checkpoint_state(Path(args.checkpoint), model)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint with {sum(p.numel() for p in model.parameters()):,} params")

    # --- Load test split ----------------------------------------------------
    dataset = CholecT50(
        dataset_dir=str(cfg.dataset.root_dir),
        dataset_variant=cfg.dataset.dataset_variant,
        test_fold=cfg.dataset.test_fold,
        augmentation_list=["original"],  # no augmentation at inference time
        normalize=cfg.dataset.normalize,
        sequence_length=seq_len,
    )
    video_ds_list = dataset.test_dataset
    if args.max_videos is not None:
        video_ds_list = video_ds_list[: args.max_videos]
    if not video_ds_list:
        raise ValueError("No test videos found in the test split.")
    orig_transforms = [ds.transform for ds in video_ds_list]
    print(
        f"Evaluating {len(video_ds_list)} test video(s): "
        f"{[Path(ds.img_dir).name for ds in video_ds_list]}"
    )

    # --- Baseline (clean) ---------------------------------------------------
    rows = []
    baseline = _evaluate_videos(
        video_ds_list, model, device, heads, seq_len, args.batch_size, num_classes,
        METRIC_NAMES,
    )
    rows.append(
        {"artifact": "clean", "level": 0, "strength": 0.0, "param_value": 0.0, **baseline}
    )
    print(f"Clean baseline {primary}={baseline[primary]:.4f}")

    # --- Degradation sweep ---------------------------------------------------
    for artifact in artifacts:
        for level in range(1, args.levels):
            strength = level / (args.levels - 1)
            metrics = _evaluate_degraded(
                video_ds_list,
                orig_transforms,
                artifact,
                strength,
                seed=cfg.runtime.seed,
                model=model,
                device=device,
                heads=heads,
                seq_len=seq_len,
                batch_size=args.batch_size,
                num_classes=num_classes,
                metric_names=METRIC_NAMES,
            )
            rows.append(
                {
                    "artifact": artifact,
                    "level": level,
                    "strength": strength,
                    "param_value": DEGRADATIONS[artifact]["param"](strength),
                    **metrics,
                }
            )
            print(
                f"  {artifact:>13s} strength={strength:.2f} "
                f"{primary}={metrics.get(primary, float('nan')):.4f}"
            )

    # --- Outputs ------------------------------------------------------------
    _write_csv(rows, output_dir / "summary.csv")
    _plot_curves(rows, heads, output_dir)
    _plot_per_class(rows, heads, num_classes, output_dir)

    print("\nBreakdown strength (primary metric < 90% of clean):")
    for artifact in artifacts:
        artifact_rows = [r for r in rows if r["artifact"] in ("clean", artifact)]
        bs = _breakdown_strength(artifact_rows, primary)
        value = artifact_rows[-1][primary]
        if bs is None:
            print(f"  {artifact:>13s}: no breakdown up to max strength "
                  f"(end {primary}={value:.4f})")
        else:
            br = next(r for r in artifact_rows if r["artifact"] == artifact
                      and r["strength"] == bs)
            print(f"  {artifact:>13s}: breakdown at strength={bs:.2f} "
                  f"(param={br['param_value']:g}, {primary}={br[primary]:.4f})")


if __name__ == "__main__":
    main()
