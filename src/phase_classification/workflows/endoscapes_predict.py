"""Run phase-model inference over an EndoScapes video split.

EndoScapes does not provide CholecT50-compatible phase labels, so this script
produces predictions rather than phase metrics. It recursively finds videos
under ``<root>/<split>/`` and writes one CSV per video plus a summary JSON.

Example::

    uv run phase-cls-endoscapes-predict \
        --root /mnt/nvme1/user_files/skirillova/skirillova/datasets/endoscapes \
        --split test --config configs/ms-tcn.yaml \
        --checkpoint /path/to/model.pth --sample-every 25
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from phase_classification.engine import resolve_device
from phase_classification.heads import DEFAULT_NUM_CLASSES
from phase_classification.loader import load_config
from phase_classification.models import build_model
from phase_classification.workflows import PROJECT_ROOT

VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="EndoScapes dataset root")
    parser.add_argument(
        "--split",
        default="test",
        help="Directory under root to scan for videos (default: test; e.g. val or train)",
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "ms-tcn.yaml"),
        help="Model/config YAML matching the checkpoint architecture",
    )
    parser.add_argument("--checkpoint", required=True, help="Model weights (.pt/.pth)")
    parser.add_argument("--head", default="phase", help="Output head (default: phase)")
    parser.add_argument(
        "--sample-every",
        type=int,
        default=25,
        help="Predict every Nth video frame; use 1 for every frame",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optionally evaluate one video by filename stem",
    )
    parser.add_argument("--device", default=None, help="Override configured device")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "endoscapes-predictions"),
        help="Output directory for prediction CSVs and summary.json",
    )
    return parser.parse_args()


def find_videos(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        raise SystemExit(f"Split directory does not exist: {split_dir}")
    videos = sorted(
        path
        for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    )
    if not videos:
        raise SystemExit(f"No supported video files found under {split_dir}")
    return videos


def load_checkpoint(path: Path, model: torch.nn.Module) -> None:
    """Load a raw state dict or one wrapped under a common checkpoint key."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        state = None
        for key in ("state_dict", "model", "model_state_dict", "weights"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
        if state is None and all(isinstance(value, torch.Tensor) for value in obj.values()):
            state = obj
        if state is None:
            raise ValueError(f"Could not find a model state_dict in {path}")
    else:
        raise TypeError(f"Unsupported checkpoint type in {path}: {type(obj).__name__}")

    # Common when checkpoints were saved from DistributedDataParallel.
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {path}")


def _frame_tensor(frame_bgr: np.ndarray, normalize) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (448, 256))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return normalize(tensor)


def _class_name(head: str, class_id: int, n_classes: int) -> str:
    if head == "phase" and n_classes == len(PHASE_NAMES):
        return PHASE_NAMES[class_id]
    return f"{head}_{class_id}"


@torch.inference_mode()
def predict_video(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    normalize,
    head: str,
    n_classes: int,
    sequence_length: int,
    sample_every: int,
) -> tuple[list[dict], int, float]:
    """Predict sampled frames, using causal windows for MS-TCN models."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 0.0

    temporal = bool(getattr(model, "is_mstcn", False))
    seq_len = sequence_length if temporal else 1
    frame_buffer: deque[torch.Tensor] = deque(maxlen=seq_len)
    rows: list[dict] = []
    frame_index = 0
    first_anchor = seq_len - 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_buffer.append(_frame_tensor(frame, normalize))

            if frame_index >= first_anchor and (frame_index - first_anchor) % sample_every == 0:
                if temporal:
                    images = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
                else:
                    images = frame_buffer[-1].unsqueeze(0).to(device)
                logits = model(images)[head]
                frame_logits = logits[0, -1] if temporal else logits[0]
                probabilities = torch.softmax(frame_logits, dim=-1).cpu().numpy()
                predicted = int(probabilities.argmax())
                row = {
                    "frame": frame_index,
                    "time_seconds": round(frame_index / fps, 4) if fps else "",
                    "pred_class": predicted,
                    "pred_label": _class_name(head, predicted, n_classes),
                    "confidence": round(float(probabilities[predicted]), 6),
                }
                row.update(
                    {
                        f"{head}_prob_{class_id}": round(float(probability), 6)
                        for class_id, probability in enumerate(probabilities)
                    }
                )
                rows.append(row)
            frame_index += 1
    finally:
        capture.release()
    return rows, frame_index, fps


def main() -> None:
    args = parse_args()
    if args.sample_every < 1:
        raise SystemExit("--sample-every must be >= 1")

    cfg = load_config(args.config)
    if args.head not in cfg.training.heads:
        raise SystemExit(
            f"Head {args.head!r} is not enabled in training.heads={cfg.training.heads}"
        )
    n_classes = int(cfg.model.num_classes.get(args.head, DEFAULT_NUM_CLASSES[args.head]))
    device = resolve_device(args.device or cfg.runtime.device, cfg.runtime.cuda_device)

    # Pretrained initialization is unnecessary for evaluation: the checkpoint
    # supplies the complete weights and this avoids downloading backbone weights.
    model_cfg = cfg.model.model_copy(
        update={"params": {**cfg.model.params, "pretrained": False}}
    )
    model = build_model(model_cfg, heads=list(cfg.training.heads))
    load_checkpoint(Path(args.checkpoint), model)
    model.to(device).eval()

    sequence_length = int(cfg.dataset.sequence_length)
    if getattr(model, "is_mstcn", False) and sequence_length < 2:
        raise SystemExit("MS-TCN inference requires dataset.sequence_length >= 2")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    split_dir = Path(args.root).expanduser() / args.split
    videos = find_videos(split_dir)
    if args.video:
        videos = [video for video in videos if video.stem == args.video]
        if not videos:
            raise SystemExit(f"Video {args.video!r} not found under {split_dir}")

    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "root": str(Path(args.root).expanduser()),
        "split": args.split,
        "config": str(Path(args.config).expanduser()),
        "checkpoint": str(Path(args.checkpoint).expanduser()),
        "model_name": cfg.model.name,
        "head": args.head,
        "sequence_length": sequence_length,
        "sample_every": args.sample_every,
        "device": str(device),
        "videos": {},
        "note": "Prediction-only: EndoScapes phase ground truth was not evaluated.",
    }

    started = time.time()
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.relative_to(split_dir)}", flush=True)
        rows, total_frames, fps = predict_video(
            video_path,
            model,
            device,
            normalize,
            args.head,
            n_classes,
            sequence_length,
            args.sample_every,
        )
        relative = video_path.relative_to(split_dir).with_suffix("")
        output_stem = "__".join(relative.parts)
        csv_path = output_dir / f"{output_stem}_predictions.csv"
        columns = ["frame", "time_seconds", "pred_class", "pred_label", "confidence"]
        columns.extend(f"{args.head}_prob_{i}" for i in range(n_classes))
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        summary["videos"][str(video_path.relative_to(split_dir))] = {
            "frames": total_frames,
            "fps": fps,
            "predictions": len(rows),
            "csv": str(csv_path),
        }
        print(f"  {len(rows):,} predictions -> {csv_path}")

    summary_path = output_dir / "summary.json"
    summary["elapsed_seconds"] = round(time.time() - started, 2)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
