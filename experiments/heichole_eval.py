#!/usr/bin/env python3
"""Cross-dataset inference: evaluate a CholecT50-trained model on HeiChole.

Usage:
    python heichole_eval.py --root /data/heichole \
        --config ../configs/ms-tcn.yaml \
        --checkpoint ../outputs/checkpoints/best.pt \
        [--split test] [--video Hei-Chole25] [--sample-every 25]

Loads one of the repo's trained models (``--config`` selects the architecture
through the model registry, ``--checkpoint`` provides the weights), streams the
HeiChole videos frame-by-frame with OpenCV, and scores the ``phase`` head
against the official per-frame phase annotations.

HeiChole facts (Wagner et al., 2021, "Comparative validation of ML algorithms
for surgical workflow and skill analysis with the HeiChole benchmark"):
  * 33 laparoscopic cholecystectomy videos (Hei-Chole1..33) from three centers;
    videos 1-24 are the official train split, 25-33 the test split. Frames
    outside the abdomen are censored as white frames.
  * Framewise phase annotation of SEVEN surgical phases whose IDs 0-6 are
    compatible with the CholecT80/CholecT50 taxonomy (the same 7 classes our
    ``phase`` head predicts), stored per video in
    ``Hei-CholeX_Annotation_Phase.csv`` as ``Frame,Phase`` rows (one row per
    video frame at the video's native fps, frame numbers 0-indexed).

Preprocessing matches the CholecT50 test transform (Resize to 256x448 +
ImageNet normalization), so the checkpoint sees input from the training
distribution. Per-frame models (resnet18, swin-cls) score single frames;
MS-TCN models score causal windows of ``sequence_length`` frames and take the
window's last frame prediction (as in ``experiments/visualise_phase.py``).

``--sample-every`` trades speed for coverage: with ``25`` (the default) only
every 25th video frame (approx. 1 fps) is scored, aligned to its annotation
row; use ``1`` for full framewise scoring (much slower).

Metrics are computed with ``phase_classification.metrics.MetricAccumulator``
(per-video ``edit_score``, pooled confusion-matrix metrics, and the merged mean
edit score over videos), the same machinery used for CholecT50 testing.
Results are written to ``--output`` as ``metrics.json`` plus one
``<video>_predictions.csv`` per video.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


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

from phase_classification.engine import resolve_device
from phase_classification.heads import DEFAULT_NUM_CLASSES
from phase_classification.loader import load_config
from phase_classification.metrics import MetricAccumulator
from phase_classification.models import build_model

# HeiChole phase IDs 0-6, aligned with the CholecT50 ``phase`` head order.
HEICHOLES_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

# Normalized (lowercased, non-alpha stripped) HeiChole phase name -> ID. The
# released CSVs use integer IDs, but accept names too for robustness.
_PHASE_NAME_TO_ID = {
    "preparation": 0,
    "calottriangledissection": 1,
    "calotdissection": 1,
    "clippingandcutting": 2,
    "clippingcutting": 2,
    "gallbladderdissection": 3,
    "galbladderdissection": 3,
    "gallbladderpackaging": 4,
    "galbladderpackaging": 4,
    "cleaningandcoagulation": 5,
    "cleaningcoagulation": 5,
    "gallbladderretraction": 6,
    "galbladderretraction": 6,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        required=True,
        help="HeiChole dataset root (contains Train/ and/or Test/ with Videos/ + Annotations/).",
    )
    p.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "default.yaml"),
        help="YAML config used to train the checkpoint (selects the architecture).",
    )
    p.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "outputs" / "checkpoints" / "best.pt"),
        help="Path to the trained model weights (.pt/.pth).",
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "test", "all"],
        help="Which HeiChole split to evaluate (test = Hei-Chole25..33).",
    )
    p.add_argument(
        "--video",
        default=None,
        help="Restrict to a single video (e.g. Hei-Chole25); otherwise all videos in the split.",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=25,
        help="Score every Nth video frame (25 ~= 1 fps). Use 1 for full framewise.",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Per-frame model batch size.")
    p.add_argument("--head", default="phase", help="Head to evaluate (default: phase).")
    p.add_argument(
        "--metrics",
        default="jaccard,f1_macro,mean_accuracy,accuracy_per_class,edit_score",
        help="Comma-separated metric names.",
    )
    p.add_argument("--device", default=None, help="Override config device (auto|cpu|cuda).")
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "heichole-eval"),
        help="Directory for metrics.json and per-video prediction CSVs.",
    )
    return p.parse_args()


def discover_videos(root: Path, split: str) -> list[tuple[str, Path, Path]]:
    """Return ``(video_id, video_path, phase_csv_path)`` triples.

    Handles the OPARA/heidata layout (``<split>/Videos/*.mp4`` +
    ``<split>/Annotations/*_Annotation_Phase.csv``) and a flat layout
    (``<root>/videos/`` + ``<root>/annotations/``).
    """
    pairs: list[tuple[str, Path, Path]] = []
    split_dirs = {
        "train": ["Train", "train"],
        "test": ["Test", "test"],
        "all": ["Train", "train", "Test", "test"],
    }[split]
    seen: set[tuple[str, Path]] = set()

    candidates: list[tuple[Path, Path]] = []
    for name in split_dirs:
        base = root / name
        if base.is_dir():
            candidates.append((base / "Videos", base / "Annotations"))
            candidates.append((base / "videos", base / "annotations"))
    # Flat layout (videos and annotations at the root level).
    candidates.append((root / "Videos", root / "Annotations"))
    candidates.append((root / "videos", root / "annotations"))

    for videos_dir, ann_dir in candidates:
        if not videos_dir.is_dir() or not ann_dir.is_dir():
            continue
        for mp4 in sorted(videos_dir.glob("*.mp4")):
            vid = mp4.stem
            ann = ann_dir / f"{vid}_Annotation_Phase.csv"
            if ann.is_file() and (vid, videos_dir) not in seen:
                seen.add((vid, videos_dir))
                pairs.append((vid, mp4, ann))

    if not pairs:
        raise SystemExit(
            f"No HeiChole videos found under {root} for split '{split}' "
            "(expected <split>/Videos/*.mp4 and <split>/Annotations/*_Annotation_Phase.csv)."
        )
    return pairs


def _parse_phase_value(value: str) -> int:
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    if re.fullmatch(r"p[0-6]", key):
        return int(key[1])
    phase_id = _PHASE_NAME_TO_ID.get(key)
    if phase_id is None:
        raise ValueError(f"Unrecognized HeiChole phase value: {value!r}")
    return phase_id


def load_phase_annotations(csv_path: Path) -> dict[int, int]:
    """Map ``frame -> phase_id`` from a ``Hei-CholeX_Annotation_Phase.csv``."""
    mapping: dict[int, int] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = iter(reader)
        header = next(rows, None)
        # Skip a header row if present (columns are non-numeric strings).
        if header is None or not header[0].strip().lstrip("-").isdigit():
            col_phase = 1
            if header is not None:
                for i, name in enumerate(header):
                    if name.strip().lower() == "phase":
                        col_phase = i
                        break
            data_rows = rows
        else:
            col_phase = 1
            data_rows = iter([header, *rows])
        for row in data_rows:
            if not row or not row[0].strip():
                continue
            try:
                frame = int(row[0].strip())
            except ValueError:
                continue
            if len(row) <= col_phase:
                continue
            mapping[frame] = _parse_phase_value(row[col_phase])
    if not mapping:
        raise ValueError(f"No phase annotations parsed from {csv_path}")
    return mapping


def _frame_tensor(bgr_small: np.ndarray, normalize) -> torch.Tensor:
    """(256,448,3) BGR -> normalized (3,256,448) float tensor."""
    rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return normalize(tensor)


@torch.no_grad()
def infer_per_frame(
    video_path: Path,
    annotation: dict[int, int],
    model,
    normalize,
    device: torch.device,
    head: str,
    sample_every: int,
    batch_size: int,
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, int]]:
    """Score single frames every ``sample_every`` frames; return per-frame
    ``(preds, probs, gts)`` dicts keyed by frame index."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchors = sorted(i for i in range(0, n_frames, sample_every) if i in annotation)
    if annotation:
        last = max(annotation)
        if last not in anchors:
            anchors.append(last)
            anchors.sort()
    anchor_set = set(anchors)

    preds: dict[int, int] = {}
    probs: dict[int, np.ndarray] = {}
    gts: dict[int, int] = {}
    pending: list[tuple[int, int]] = []  # (frame, gt) awaiting a forward pass
    pending_tensors: list[torch.Tensor] = []

    def flush() -> None:
        if not pending_tensors:
            return
        imgs = torch.stack(pending_tensors).to(device)
        logits = model(imgs)[head]
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        for (frame, gt), prob in zip(pending, p):
            preds[frame] = int(prob.argmax(-1))
            probs[frame] = prob
            gts[frame] = gt
        pending.clear()
        pending_tensors.clear()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in anchor_set:
            pending.append((idx, annotation[idx]))
            pending_tensors.append(_frame_tensor(cv2.resize(frame, (448, 256)), normalize))
            if len(pending_tensors) >= batch_size:
                flush()
        idx += 1
    flush()
    cap.release()
    return preds, probs, gts


@torch.no_grad()
def infer_temporal(
    video_path: Path,
    annotation: dict[int, int],
    model,
    normalize,
    device: torch.device,
    head: str,
    seq_len: int,
    sample_every: int,
) -> tuple[dict[int, int], dict[int, np.ndarray], dict[int, int]]:
    """Score causal windows of ``seq_len`` frames, keeping the last-frame
    prediction (mirrors ``visualise_phase.predict_video``). Anchors are spaced
    ``sample_every`` apart and require ``seq_len - 1`` frames of context."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    anchors = sorted(
        i for i in range(seq_len - 1, n_frames, sample_every) if i in annotation
    )
    if annotation:
        last = max(annotation)
        if last not in anchors and last >= seq_len - 1:
            anchors.append(last)
            anchors.sort()
    anchor_iter = iter(anchors)
    next_anchor = next(anchor_iter, None)

    preds: dict[int, int] = {}
    probs: dict[int, np.ndarray] = {}
    gts: dict[int, int] = {}
    buf: deque[np.ndarray] = deque()  # last seq_len raw (256,448,3) frames

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(cv2.resize(frame, (448, 256)))
        if len(buf) > seq_len:
            buf.popleft()
        if idx == next_anchor:
            window = torch.stack([_frame_tensor(b, normalize) for b in buf])
            window = window.unsqueeze(0).to(device)  # (1, L, C, H, W)
            logits = model(window)[head]  # (1, L, K)
            p = torch.softmax(logits[0, -1], dim=-1).cpu().numpy()
            preds[idx] = int(p.argmax(-1))
            probs[idx] = p
            gts[idx] = annotation[idx]
            next_anchor = next(anchor_iter, None)
        idx += 1
    cap.release()
    return preds, probs, gts


def _one_hot(values: list[int], n_classes: int) -> np.ndarray:
    out = np.zeros((len(values), n_classes), dtype=np.float32)
    out[np.arange(len(values)), values] = 1.0
    return out


def video_accumulator(
    preds: dict[int, int],
    probs: dict[int, np.ndarray],
    gts: dict[int, int],
    head: str,
    metric_names: list[str],
    n_classes: int,
) -> MetricAccumulator:
    acc = MetricAccumulator([head], metric_names, {head: n_classes})
    order = sorted(preds)
    if not order:
        return acc
    logits = np.stack([probs[i] for i in order])
    targets = _one_hot([gts[i] for i in order], n_classes)
    acc.add_batch({head: logits}, {head: targets})
    return acc


def load_checkpoint_state(path: Path, model) -> None:
    """Load weights into ``model`` from a checkpoint of any common format."""
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    head = args.head
    if head not in cfg.training.heads:
        raise SystemExit(f"'{head}' not in training heads {cfg.training.heads}")
    metric_names = [m.strip() for m in args.metrics.split(",") if m.strip()]
    n_classes = int(cfg.model.num_classes.get(head, DEFAULT_NUM_CLASSES[head]))
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be >= 1")
    seq_len = cfg.dataset.sequence_length

    device_name = args.device or cfg.runtime.device
    device = resolve_device(device_name, cfg.runtime.cuda_device)

    print(f"Config: {args.config}")
    print(f"Model: {cfg.model.name} | params: {cfg.model.params}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"HeiChole root: {args.root} (split={args.split}, video={args.video or 'all'})")
    print(f"Head: {head} ({n_classes} classes) | sample_every: {args.sample_every} | device: {device}")

    model = build_model(cfg.model, heads=cfg.training.heads)
    load_checkpoint_state(Path(args.checkpoint), model)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint with {sum(p.numel() for p in model.parameters()):,} params")

    temporal = bool(getattr(model, "is_mstcn", False))
    if temporal and seq_len < 2:
        raise SystemExit("MS-TCN models require dataset.sequence_length >= 2")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    videos = discover_videos(Path(args.root), args.split)
    if args.video:
        videos = [v for v in videos if v[0] == args.video]
    if not videos:
        raise SystemExit(f"Video '{args.video}' not found in split '{args.split}'")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_video: dict[str, dict[str, float]] = {}
    accumulators: list[MetricAccumulator] = []
    started = time.time()

    for i, (vid, video_path, ann_path) in enumerate(videos, start=1):
        t0 = time.time()
        annotation = load_phase_annotations(ann_path)
        print(f"[{i}/{len(videos)}] {vid}: {len(annotation):,} annotated frames -> ", end="", flush=True)
        if temporal:
            preds, probs, gts = infer_temporal(
                video_path, annotation, model, normalize, device, head, seq_len,
                args.sample_every,
            )
        else:
            preds, probs, gts = infer_per_frame(
                video_path, annotation, model, normalize, device, head,
                args.sample_every, args.batch_size,
            )
        acc = video_accumulator(preds, probs, gts, head, metric_names, n_classes)
        accumulators.append(acc)
        metrics = acc.compute()
        per_video[vid] = metrics
        n_scored = len(preds)
        print(
            f"{n_scored:,} scored frames in {time.time() - t0:.1f}s | "
            f"acc={metrics.get(f'{head}_mean_accuracy', float('nan')):.3f} "
            f"f1={metrics.get(f'{head}_f1_macro', float('nan')):.3f}"
        )

        csv_path = out_dir / f"{vid}_predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "pred_class", "pred_phase", "gt_class", "gt_phase", "confidence"])
            for frame in sorted(preds):
                writer.writerow(
                    [
                        frame,
                        preds[frame],
                        HEICHOLES_PHASES[preds[frame]],
                        gts[frame],
                        HEICHOLES_PHASES[gts[frame]],
                        round(float(probs[frame][preds[frame]]), 4),
                    ]
                )
        print(f"  Saved: {csv_path}")

    merged = MetricAccumulator.merged(accumulators).compute()
    print(f"\nMerged over {len(accumulators)} videos (edit_score = mean of per-video scores):")
    for key, value in sorted(merged.items()):
        print(f"  {key}: {value:.6f}")

    summary = {
        "args": {
            "root": args.root,
            "split": args.split,
            "video": args.video,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "model_name": cfg.model.name,
            "head": head,
            "sample_every": args.sample_every,
            "sequence_length": seq_len,
            "metrics": metric_names,
            "device": str(device),
        },
        "per_video": per_video,
        "merged": merged,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nTotal time: {time.time() - started:.1f}s")
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()