#!/usr/bin/env python3
"""Visualize phase predictions of a trained CholecT50 model on one video.

Usage:
    python visualise_phase.py [--config CONFIG] [--checkpoint CKPT]
                              [--video VID06] [--output DIR] [--head phase]

Loads a trained model (state_dict), runs inference on one video from the
CholecT50 test split, and writes a timeline plot, an annotated-frame grid,
and a per-frame predictions CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless-safe; works on remote kernels without a display
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

# --- Make the project's src/ package importable regardless of cwd -----------
def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the folder containing src/phase_classification."""
    for parent in [start, *start.parents]:
        if (parent / "src" / "phase_classification").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate the phase_classification repo root "
        "(no src/phase_classification found above cwd)."
    )


REPO_ROOT = _find_repo_root(Path.cwd())
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.loader import load_config
from phase_classification.losses import head_index
from phase_classification.models import build_model

# CholecT50 phase names (index == class id). 7 phases for the standard variant.
PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="YAML config used for training",
    )
    p.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs" / "checkpoints" / "best.pt"),
        help="Path to the trained state_dict checkpoint",
    )
    p.add_argument("--video", default="VID06", help="Video ID to visualize (e.g. VID06)")
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "phase-viz"),
        help="Output directory for visualizations",
    )
    p.add_argument("--head", default="phase", help="Head to visualize (default: phase)")
    return p.parse_args()


@torch.no_grad()
def predict_video(video_ds, model, device, head: str, seq_len: int, batch_size: int = 32):
    """Return (pred_class, pred_probs, gt_class) arrays over the video frames."""
    preds, probs, gts = [], [], []

    if seq_len > 1:
        # Temporal model: feed the whole video as one windowed batch.
        images, labels = video_ds[0]  # (L, C, H, W) window
        images = images.unsqueeze(0).to(device)
        logits = model(images)[head]  # (1, K, L)
        logits = logits.squeeze(0).transpose(0, 1)  # (L, K)
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = p.argmax(-1)
        probs = p
        gts = labels[head_index(head)].argmax(-1).cpu().numpy()
    else:
        # Per-frame model: batch frames.
        for start in range(0, len(video_ds), batch_size):
            end = min(start + batch_size, len(video_ds))
            batch_imgs, batch_labels = [], []
            for i in range(start, end):
                img, lab = video_ds[i]
                batch_imgs.append(img)
                batch_labels.append(lab)
            images = torch.stack(batch_imgs).to(device)
            logits = model(images)[head]  # (B, K)
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            preds.append(p.argmax(-1))
            probs.append(p)
            gts.append(
                torch.stack([l[head_index(head)] for l in batch_labels])
                .argmax(-1)
                .cpu()
                .numpy()
            )
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
        gts = np.concatenate(gts)

    return preds, probs, gts


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    head = args.head
    assert head in cfg.training.heads, f"'{head}' not in training heads {cfg.training.heads}"

    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name} | params: {cfg.model.params}")
    print(f"Heads: {cfg.training.heads} | sequence_length: {cfg.dataset.sequence_length}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Video: {args.video} | Output: {output_dir}")

    # --- Load model ---------------------------------------------------------
    device = resolve_device(cfg.runtime.device, cfg.runtime.cuda_device)
    print("Device:", device)
    model = build_model(cfg.model, heads=cfg.training.heads)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint with {sum(p.numel() for p in model.parameters()):,} params")

    # --- Load video dataset -------------------------------------------------
    dataset = CholecT50(
        dataset_dir=str(cfg.dataset.root_dir),
        dataset_variant=cfg.dataset.dataset_variant,
        test_fold=cfg.dataset.test_fold,
        augmentation_list=["original"],  # no augmentation at inference time
        normalize=cfg.dataset.normalize,
        sequence_length=cfg.dataset.sequence_length,
    )
    video_ds = None
    for candidate in dataset.test_dataset:
        if getattr(candidate, "img_dir", "").endswith(args.video):
            video_ds = candidate
            break
    if video_ds is None:
        raise ValueError(
            f"Video {args.video} not found in the test split. "
            f"Available: {[Path(d.img_dir).name for d in dataset.test_dataset]}"
        )
    print(
        f"Loaded {args.video}: {len(video_ds)} samples "
        f"(sequence_length={cfg.dataset.sequence_length})"
    )

    # --- Predict ------------------------------------------------------------
    preds, probs, gts = predict_video(
        video_ds, model, device, head, cfg.dataset.sequence_length
    )
    n_frames = len(preds)
    print(f"Predicted {n_frames} frames")

    # --- Visualize ----------------------------------------------------------
    n_phases = len(PHASE_NAMES)
    cmap = plt.get_cmap("tab10", n_phases)
    colors = [cmap(i) for i in range(n_phases)]
    legend_handles = [Patch(color=colors[i], label=PHASE_NAMES[i]) for i in range(n_phases)]

    # Timeline plot.
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 2]})
    axes[0].imshow(preds[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=n_phases - 1)
    axes[0].set_yticks([])
    axes[0].set_title(f"Predicted phase — {args.video}")
    axes[1].imshow(gts[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=n_phases - 1)
    axes[1].set_yticks([])
    axes[1].set_title("Ground-truth phase")
    im = axes[2].imshow(probs.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[2].set_yticks(range(n_phases), PHASE_NAMES, fontsize=8)
    axes[2].set_ylabel("Phase")
    axes[2].set_xlabel("Frame")
    fig.colorbar(im, ax=axes[2], label="softmax prob")
    fig.legend(handles=legend_handles, loc="upper center", ncol=n_phases,
               bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)
    plt.tight_layout()
    timeline_path = output_dir / f"{args.video}_timeline.png"
    fig.savefig(timeline_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", timeline_path)

    # Annotated frames grid.
    n_cols, n_rows = 4, 3
    sample_idx = np.linspace(0, n_frames - 1, n_cols * n_rows).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))
    for ax, idx in zip(axes.ravel(), sample_idx):
        frame_key = video_ds.frames[idx] if hasattr(video_ds, "frames") else idx
        basename = f"{str(frame_key).zfill(6)}.png"
        img = Image.open(Path(video_ds.img_dir) / basename)
        ax.imshow(img)
        pred_name = PHASE_NAMES[preds[idx]]
        gt_name = PHASE_NAMES[gts[idx]]
        conf = probs[idx, preds[idx]]
        color = "lime" if pred_name == gt_name else "red"
        ax.set_title(
            f"#{idx} pred: {pred_name} ({conf:.2f})\ngt: {gt_name}",
            fontsize=9,
            color=color,
        )
        ax.axis("off")
    fig.suptitle(f"Annotated frames — {args.video}", fontsize=14)
    plt.tight_layout()
    grid_path = output_dir / f"{args.video}_frames.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", grid_path)

    # Per-frame predictions CSV.
    csv_path = output_dir / f"{args.video}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["frame", "pred_class", "pred_phase", "gt_class", "gt_phase", "confidence"]
        )
        for i in range(n_frames):
            writer.writerow(
                [
                    i,
                    int(preds[i]),
                    PHASE_NAMES[preds[i]],
                    int(gts[i]),
                    PHASE_NAMES[gts[i]],
                    round(float(probs[i, preds[i]]), 4),
                ]
            )
    print("Saved:", csv_path)

    print("\nAll outputs written to:", output_dir)


if __name__ == "__main__":
    main()
