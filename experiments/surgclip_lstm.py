#!/usr/bin/env python3
"""Extract SurgCLIP (SurgLaVi-beta) clip embeddings, run zero-shot phase
recognition, and optionally train a per-frame BiLSTM head on CholecT50.

Usage:
    python surgclip_lstm.py --dataset-dir /data/cholect50 [options]

Pipeline (mirrors surgvlp_lstm.py, swapping the SurgVLP/PeskaVLP towers for
SurgCLIP from aperezr20/SurgLaVi, MedIA'26):
    Stage 1 (embeddings): loads SurgCLIP-beta through the official `surgclip`
    package (TimeSformer-B vision tower over 16-frame clips, dual BERT text
    tower), runs one --num-frames clip per annotated frame of the chosen
    CholecT50 split through ``encode_vision`` and caches the anchor frame's
    projected, L2-normalised CLIP-space embedding as
    {"features": [N, D], "paths": [...]}. Weights auto-download from
    HuggingFace (aleperez24/SurgCLIP) on first use and are cached in
    ~/.cache/surgclip (override the location with $SURGCLIP_CACHE).

    Stage 2 (zero-shot, --zero-shot): classifies the test frames against the
    pretrained text tower on the 7 phase prompts, i.e. no training at all.
    Exits after evaluation.

    Stage 3 (BiLSTM head, default): builds contiguous frame windows
    (--sequence-length) over the cached clip embeddings and trains the same
    bidirectional LSTM + per-frame classification head as surgvlp_lstm.py
    (class-balanced cross-entropy), reporting metrics computed with
    ``phase_classification.metrics.MetricAccumulator`` (edit_score is the mean
    over per-video test sequences).

The feature datasets, BiLSTM head, batch sampler, and training loop are reused
from ``surgvlp_lstm.py`` so results stay directly comparable; only the encoder
and the zero-shot similarity differ.

Example
    python surgclip_lstm.py --dataset-dir /data/cholect50 --extract-only
    python surgclip_lstm.py --dataset-dir /data/cholect50 --zero-shot
    python surgclip_lstm.py --dataset-dir /data/cholect50 \\
        --embeddings-pt outputs/embeddings/surgclip-b-.../surgclip-b_features.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# --- Make the project's src/ package and the sibling experiment importable ---
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

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import surgclip

# Shared pieces from the SurgVLP experiment (feature datasets, BiLSTM head,
# video-coherent batch sampler, epoch loop, label helpers).
from surgvlp_lstm import (
    PHASE_NAMES,
    CholecT50Feature,
    PeskaVLPBiLSTM,
    VideoBatchSampler,
    _load_labels,
    _phase_counts,
    _phase_onehot,
    _read_prompts,
    collate_fn,
    run_epoch,
)

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX
from phase_classification.metrics import MetricAccumulator

MODEL_NAME_MAP = {
    "surgclip-b": "SurgCLIP-B",
}

# CholecT50 phase names are in phase_classification.heads / surgvlp_lstm.PHASE_NAMES;
# these prompts are index-aligned with PHASE_NAMES and taken verbatim from the
# SurgCLIP quickstart (aperezr20/SurgLaVi README / src/surgclip/MODEL.md).
DEFAULT_PROMPTS = [
    "Prepares for surgery by inserting trocars into the patient's abdominal cavity",
    "Employs grasper and hook during calot triangle dissection, manipulating gallbladder to reveal hepatic triangle, cystic duct and cystic artery",
    "Utilizes clipper to secure cystic duct and artery, followed by precise dissection using scissors",
    "Utilizes a hook to dissect the connective tissue during the dissection phase, separating gallbladder from the liver",
    "Secures the removed gallbladder in the specimen bag during the packaging phase of the procedure",
    "Employs suction and irrigation techniques to maintain a clear surgical field during the clean and coagulation phase, simultaneously coagulating bleeding vessels",
    "Handles the specimen bag during the retraction",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, help="CholecT50 root with videos/ and labels/")
    p.add_argument("--variant", default="cholect50-crossval", help="CholecT50 split variant")
    p.add_argument("--test-fold", type=int, default=1, help="Cross-validation fold for test")
    p.add_argument(
        "--model",
        default="surgclip-b",
        choices=sorted(MODEL_NAME_MAP),
        help="SurgCLIP checkpoint (SurgCLIP-beta, auto-downloaded from HuggingFace)",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Frames per clip fed to the TimeSformer vision tower (1 = single-frame ablation)",
    )
    p.add_argument(
        "--clip-mode",
        default="centered",
        choices=["centered", "online"],
        help="Window placement around the anchor frame (centered = offline, online = causal)",
    )
    p.add_argument(
        "--extract-batch-size", type=int, default=4, help="Clips per forward pass during extraction"
    )
    p.add_argument(
        "--amp", action="store_true", help="fp16 autocast for extraction forward passes (CUDA only)"
    )
    p.add_argument(
        "--embedding-dir",
        default=str(REPO_ROOT / "outputs" / "embeddings"),
        help="Where extracted embeddings are saved",
    )
    p.add_argument(
        "--embeddings-pt", default=None, help="Reuse an existing features .pt (skips extraction)"
    )
    p.add_argument("--extract-only", action="store_true", help="Stop after embedding extraction")
    p.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run zero-shot phase classification (pretrained text tower), then exit",
    )
    p.add_argument(
        "--class-prompt",
        default=None,
        help="Path to a text file with one class prompt per line (default: embedded prompts)",
    )
    p.add_argument("--sequence-length", type=int, default=16, help="LSTM window size")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=128, help="LSTM hidden size")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional LSTM")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--heads", default="phase", help="Comma-separated heads (default: phase)")
    p.add_argument(
        "--metrics",
        default="jaccard,f1_macro,mean_accuracy,edit_score",
        help="Comma-separated metric names",
    )
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "surgclip-lstm"))
    return p.parse_args()


# --- Stage 1: SurgCLIP vision tower -----------------------------------------
def _patch_surgclip_tokenizer() -> None:
    """Make surgclip.load work with modern transformers.

    surgclip's vendored BertTokenizer sets ``self.vocab`` *after*
    ``super().__init__()``, but transformers >=4.4x resolves special tokens
    during base init via ``get_vocab()`` -> AttributeError: no attribute
    'vocab'. ``transformers.BertTokenizer`` is a drop-in replacement for
    inference (same vocab file and WordPiece), so fall back to it.
    """
    original = surgclip.load_tokenizer

    def compat(config):
        try:
            return original(config)
        except Exception as exc:  # noqa: BLE001 - any tokenizer init failure
            print(
                f"surgclip vendored tokenizer unavailable ({exc}); "
                "falling back to transformers.BertTokenizer"
            )
            from transformers import BertTokenizer

            return BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)

    surgclip.load_tokenizer = compat


_patch_surgclip_tokenizer()


def load_surgclip(
    model_name: str, num_frames: int, device: torch.device
) -> tuple[nn.Module, object, object]:
    """Load SurgCLIP via the official package (weights auto-download from HF).

    ``num_frames`` configures the TimeSformer temporal extent; use 16 (the
    checkpoint's native clip length). ``num_frames=1`` disables temporal
    modeling for a cheap single-frame ablation (temporal weights of the
    checkpoint are then ignored).
    """
    temporal = num_frames > 1
    model, preprocess, tokenizer = surgclip.load(
        model_name, device=device, num_frames=num_frames, temporal_modeling=temporal
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {model_name} (num_frames={num_frames}, temporal={temporal}, "
        f"{n_params:,} params)"
    )
    return model, preprocess, tokenizer


def _window_start(anchor: int, total: int, num_frames: int, clip_mode: str) -> int:
    """Mirror VideoPreprocessor's centered/online window selection (clamped).

    centered: start = clamp(anchor - num_frames // 2) so the full window fits.
    online:   the anchor frame is the last frame of the window.
    """
    if num_frames >= total:
        return 0
    if clip_mode == "centered":
        half = num_frames // 2
        start = max(0, anchor - half)
        end = start + num_frames
        if end > total:
            end = total
            start = max(0, end - num_frames)
        return start
    if clip_mode == "online":
        return max(0, anchor + 1 - num_frames)
    raise ValueError(f"Unknown clip mode: {clip_mode!r} (choose 'centered' or 'online')")


class ClipWindowReader:
    """Preprocesses each frame once and serves sliding clip windows.

    ``surgclip.VideoPreprocessor`` re-globs the frame directory and re-opens
    every window frame for each anchor call; this reader exploits the fact
    that the window start only moves forward over a sorted frame list and
    keeps a bounded sliding buffer of preprocessed frames instead.
    Short videos are padded by repeating the last frame, exactly like
    ``VideoPreprocessor._sample``.
    """

    def __init__(self, img_paths: list[str], preprocess, num_frames: int, clip_mode: str):
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.img_paths = img_paths
        self.preprocess = preprocess
        self.num_frames = num_frames
        self.clip_mode = clip_mode
        self.total = len(img_paths)
        self._buffer: deque[tuple[int, torch.Tensor]] = deque()  # (absolute idx, tensor)
        self._next_idx = 0  # next frame index to load from disk

    def _load_frame(self, idx: int) -> torch.Tensor:
        with Image.open(self.img_paths[idx]) as img:
            return self.preprocess(img.convert("RGB"))

    def _extend_to(self, idx: int) -> None:
        while self._next_idx <= idx:
            self._buffer.append((self._next_idx, self._load_frame(self._next_idx)))
            self._next_idx += 1

    def _trim(self, start: int) -> None:
        while self._buffer and self._buffer[0][0] < start:
            self._buffer.popleft()

    def window(self, anchor: int) -> tuple[torch.Tensor, int]:
        """Return the anchor's (num_frames, C, H, W) clip and its slot index.

        The slot is the anchor frame's position inside the (possibly padded)
        window — the per-frame pooled output of the vision tower is indexed
        with it after the forward pass.
        """
        start = _window_start(anchor, self.total, self.num_frames, self.clip_mode)
        end = min(start + self.num_frames, self.total)
        self._extend_to(end - 1)
        self._trim(start)
        frames = [t for i, t in self._buffer if start <= i < end]
        while len(frames) < self.num_frames:  # pad short videos (repeat last frame)
            frames.append(frames[-1])
        return torch.stack(frames, dim=0), anchor - start


def _encode_clip_batch(
    model: nn.Module,
    clips: torch.Tensor,
    slots: torch.Tensor,
    device: torch.device,
    autocast: bool,
) -> torch.Tensor:
    """CLIP-space embedding of each clip's anchor frame.

    ``encode_vision`` returns per-frame pooled CLS embeddings (B, T, C) when
    temporal modeling is enabled (the real TimeSformer), or a single pooled
    vector (B, C) otherwise. ``slots`` indexes the anchor frame's position in
    the (B, T, C) case. The returned embedding is
    ``F.normalize(vision_proj(pooled))`` — the quantity ``SurgCLIP.get_sim``
    scores text against.
    """
    clips = clips.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast):
        _, pooled = model.encode_vision(clips)
        proj = model.vision_proj(pooled)  # (B, T, E) or (B, E)
    proj = F.normalize(proj.float(), dim=-1)
    if proj.dim() == 3:  # (B, T, E): keep the anchor frame's slot
        idx = slots.to(proj.device)
        return proj[torch.arange(len(idx), device=proj.device), idx]
    return proj  # (B, E): already one embedding per clip


@torch.no_grad()
def extract_surgclip_embeddings(
    video_datasets: list,
    model: nn.Module,
    preprocess,
    device: torch.device,
    num_frames: int,
    clip_mode: str,
    batch_size: int,
    amp: bool,
) -> tuple[torch.Tensor, list[str]]:
    """Embed every annotated frame of every video as one clip window.

    For each anchor frame the window is selected exactly like the official
    ``VideoPreprocessor`` (centered/online, clamped at video boundaries, short
    videos padded). The vision tower yields per-frame pooled embeddings for the
    whole clip; the cached feature is the anchor frame's slot — projected and
    L2-normalised, the quantity ``SurgCLIP.get_sim`` scores text against.
    """
    model.eval()
    autocast_on = amp and device.type == "cuda"
    features: list[torch.Tensor] = []
    paths: list[str] = []
    for ds in tqdm(video_datasets, desc="extract", unit="video"):
        frames = sorted(ds.frames, key=int)
        img_paths = [
            os.path.join(ds.img_dir, f"{str(frame).zfill(6)}.png") for frame in frames
        ]
        reader = ClipWindowReader(img_paths, preprocess, num_frames, clip_mode)
        video_name = Path(ds.img_dir).name
        clips: list[torch.Tensor] = []
        slots: list[int] = []
        clip_paths: list[str] = []
        for anchor in tqdm(range(len(img_paths)), desc=video_name, leave=False, unit="frame"):
            clip, slot = reader.window(anchor)
            clips.append(clip)
            slots.append(slot)
            clip_paths.append(img_paths[anchor])
            if len(clips) == batch_size:
                emb = _encode_clip_batch(
                    model, torch.stack(clips),
                    torch.tensor(slots, dtype=torch.long), device, autocast_on,
                )
                features.append(emb.detach().cpu())
                paths.extend(clip_paths)
                clips, slots, clip_paths = [], [], []
        if clips:
            emb = _encode_clip_batch(
                model, torch.stack(clips),
                torch.tensor(slots, dtype=torch.long), device, autocast_on,
            )
            features.append(emb.detach().cpu())
            paths.extend(clip_paths)
    return torch.cat(features, dim=0), paths


# --- Stage 2: zero-shot evaluation ------------------------------------------
@torch.no_grad()
def run_zero_shot(
    features: torch.Tensor,
    paths: list[str],
    model: nn.Module,
    tokenizer,
    device: torch.device,
    test_records: list[str],
    dataset_dir: str,
    prompts: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    """Zero-shot phase classification over the CholecT50 test split.

    Text features come from the SurgCLIP text tower on the class prompts;
    image features are the cached CLIP-space embeddings. Scoring follows
    ``SurgCLIP.get_sim``: normalised projections, cosine similarity, scaled by
    the learned temperature. Metrics are computed per video with
    ``MetricAccumulator`` and merged (edit_score = mean over videos); the
    repo-style mean per-video accuracy is printed as well.
    """
    heads = ["phase"]
    num_classes = {h: DEFAULT_NUM_CLASSES[h] for h in heads}

    tokens = surgclip.tokenize(prompts, tokenizer, device=device)
    _, pooled_text = model.encode_text(tokens)
    text_feats = F.normalize(model.text_proj(pooled_text.float()), dim=-1)
    temp = float(model.temp)
    print(f"Text features: {tuple(text_feats.shape)} (learned temp={temp:.4f})")

    lookup: dict[tuple[str, str], int] = {}
    for idx, path in enumerate(paths):
        p = Path(path)
        lookup[(p.parent.name, p.stem)] = idx

    per_video_acc: list[float] = []
    accumulators = []
    for video in test_records:
        label_file = os.path.join(dataset_dir, "labels", f"{video}.json")
        label_data = _load_labels(label_file)
        frame_keys = sorted(label_data, key=int)
        probs, targets = [], []
        correct = 0
        for frame in frame_keys:
            idx = lookup[(video, str(frame).zfill(6))]
            emb = features[idx].to(device)
            p = ((emb @ text_feats.T) / temp).softmax(dim=-1)
            target = torch.tensor(_phase_onehot(label_data[frame])).bool().int()
            probs.append(p.cpu())
            targets.append(target)
            if p.argmax().item() == target.argmax().item():
                correct += 1
        probs = torch.stack(probs)
        targets = torch.stack(targets)
        per_video_acc.append(correct / len(frame_keys))

        acc = MetricAccumulator(heads, metrics, num_classes)
        acc.add_batch(
            {"phase": probs.numpy()},
            {"phase": targets.numpy()},
        )
        accumulators.append(acc)

    merged = MetricAccumulator.merged(accumulators).compute()
    mean_video_acc = float(np.mean(per_video_acc))

    print(f"\nZero-shot ({len(test_records)} test videos)")
    print(f"Mean per-video accuracy: {mean_video_acc:.4f}")
    for key, value in sorted(merged.items()):
        print(f"  {key}: {value:.4f}")
    if any(k.startswith(f"{heads[0]}_f1_class_") for k in merged):
        print("Per-class F1:")
        for i, name in enumerate(PHASE_NAMES):
            key = f"{heads[0]}_f1_class_{i}"
            if key in merged:
                print(f"  {name}: {merged[key]:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "zero_shot_metrics.json", "w") as f:
        json.dump(
            {
                "test_records": test_records,
                "class_prompts": prompts,
                "mean_video_accuracy": mean_video_acc,
                "metrics": {k: float(v) for k, v in merged.items()},
            },
            f,
            indent=2,
        )
    print("Saved zero-shot metrics:", output_dir / "zero_shot_metrics.json")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print("Device:", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not heads or not metrics:
        raise ValueError("--heads and --metrics must be non-empty")
    if any(h not in DEFAULT_NUM_CLASSES for h in heads):
        raise ValueError(f"Unknown head(s): {heads}. Available: {sorted(DEFAULT_NUM_CLASSES)}")
    num_classes = {h: DEFAULT_NUM_CLASSES[h] for h in heads}

    # --- Stage 1: clip embeddings -------------------------------------------
    split = CholecT50(
        dataset_dir=args.dataset_dir,
        dataset_variant=args.variant,
        test_fold=args.test_fold,
        augmentation_list=["original"],
        normalize=True,
        sequence_length=1,
    )

    clip_model = None
    tokenizer = None
    if args.embeddings_pt:
        data = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
        features, paths = data["features"], list(data["paths"])
        print(f"Loaded cached embeddings: {tuple(features.shape)} from {args.embeddings_pt}")
    else:
        model_name = MODEL_NAME_MAP[args.model]
        clip_model, preprocess, tokenizer = load_surgclip(model_name, args.num_frames, device)

        video_datasets = (
            list(split.train_dataset.datasets)
            + list(split.val_dataset.datasets)
            + list(split.test_dataset)
        )
        seen, unique = set(), []
        for ds in video_datasets:
            if getattr(ds, "img_dir", "") in seen:
                continue
            seen.add(ds.img_dir)
            unique.append(ds)
        print(
            f"Extracting clip embeddings over {len(unique)} videos "
            f"({model_name}, num_frames={args.num_frames}, mode={args.clip_mode}, "
            f"{args.variant}, fold {args.test_fold})"
        )
        features, paths = extract_surgclip_embeddings(
            unique,
            clip_model,
            preprocess,
            device,
            args.num_frames,
            args.clip_mode,
            args.extract_batch_size,
            args.amp,
        )
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_dir = Path(args.embedding_dir) / f"{args.model}-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{args.model}_features.pt"
        torch.save(
            {
                "features": features,
                "paths": paths,
                "model": model_name,
                "variant": args.variant,
                "test_fold": args.test_fold,
                "num_frames": args.num_frames,
                "clip_mode": args.clip_mode,
            },
            out_pt,
        )
        print(f"Saved embeddings: {out_pt} ({tuple(features.shape)})")
        if args.extract_only:
            return

    print(f"Embedding dim: {features.shape[-1]}")

    # --- Stage 2: zero-shot evaluation (optional, exits) ---------------------
    if args.zero_shot:
        if clip_model is None:
            model_name = MODEL_NAME_MAP[args.model]
            clip_model, _, tokenizer = load_surgclip(model_name, args.num_frames, device)
        prompts = _read_prompts(args.class_prompt) if args.class_prompt else list(DEFAULT_PROMPTS)
        run_zero_shot(
            features,
            paths,
            clip_model,
            tokenizer,
            device,
            split.test_records,
            args.dataset_dir,
            prompts,
            metrics,
            Path(args.output_dir),
        )
        return

    # --- Stage 3: BiLSTM over the cached clip embeddings ---------------------
    if clip_model is not None:
        del clip_model, tokenizer  # free the vision tower before training the head

    feat_ds = CholecT50Feature(
        dataset_dir=args.dataset_dir,
        features=features,
        paths=paths,
        train_records=split.train_records,
        val_records=split.val_records,
        test_records=split.test_records,
        sequence_length=args.sequence_length,
    )
    train_ds, val_ds, test_ds = feat_ds.build()
    print(f"train: {len(train_ds)} val: {len(val_ds)} test: {len(test_ds)}")

    model = PeskaVLPBiLSTM(
        feat_dim=features.shape[-1],
        lstm_hidden=args.hidden,
        num_layers=args.layers,
        bidirectional=not args.no_bidirectional,
        num_classes=num_classes[heads[0]],
        dropout=args.dropout,
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    phase_counts = _phase_counts(train_ds, num_classes[heads[0]])
    phase_freq = (phase_counts / phase_counts.sum()).clamp(min=1e-9)
    class_weights = 1.0 / phase_freq
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print("Per-class phase counts:", phase_counts.int().tolist())
    print("Used class weights:    ", [f"{w:.2f}" for w in class_weights.cpu().tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    head_idx = HEAD_TUPLE_INDEX[heads[0]]

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_sampler=VideoBatchSampler(ds, args.batch_size, shuffle=shuffle),
            collate_fn=collate_fn,
            num_workers=0,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    best_primary, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_acc = MetricAccumulator(heads, metrics, num_classes)
        val_acc = MetricAccumulator(heads, metrics, num_classes)
        train_loss, _ = run_epoch(
            train_loader, model, criterion, optimizer, device, head_idx, num_classes[heads[0]],
            train_acc, train=True, desc=f"train {epoch}/{args.epochs}",
        )
        val_loss, _ = run_epoch(
            val_loader, model, criterion, optimizer, device, head_idx, num_classes[heads[0]],
            val_acc, train=False, desc=f"val   {epoch}/{args.epochs}",
        )
        train_metrics, val_metrics = train_acc.compute(), val_acc.compute()
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print(f"  train: {train_metrics}")
        print(f"  val:   {val_metrics}")
        primary = val_metrics.get(f"{heads[0]}_{metrics[0]}", -1.0)
        if primary > best_primary:
            best_primary = primary
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    test_losses, test_accs = [], []
    for video_ds in test_ds:
        loader = DataLoader(
            video_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        acc = MetricAccumulator(heads, metrics, num_classes)
        loss, _ = run_epoch(
            loader, model, criterion, optimizer, device, head_idx, num_classes[heads[0]],
            acc, train=False, desc="test",
        )
        test_losses.append(loss)
        test_accs.append(acc)
    test_metrics = MetricAccumulator.merged(test_accs).compute()
    print(f"\nTest loss={np.mean(test_losses):.4f}")
    print(f"Test metrics: {test_metrics}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, out_dir / "best.pt")
        print("Saved best checkpoint:", out_dir / "best.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "test_loss": float(np.mean(test_losses)),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            },
            f,
            indent=2,
        )
    print("Saved metrics:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
