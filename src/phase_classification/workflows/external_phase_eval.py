"""Evaluate the trained SurgCLIP temporal phase model on external datasets.

Examples::

    phase-cls-external-eval cholec80 \
        --root /mnt/nvme1/datasets/cholec80 \
        --checkpoint outputs/surgclip-ms-tcnpp-diffusion/fold1/checkpoints/best.pt \
        --config configs/surgclip-ms-tcnpp-diffusion.yaml

M2CAI16's eight phases are mapped to the model's seven Cholec phases by merging
Trocar Placement into Preparation. AutoLaparo has a different procedure and
taxonomy; its predictions are exported but classification metrics are omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.loader import load_config
from phase_classification.metrics import MetricAccumulator
from phase_classification.models.ms_tcn import CachedFeatureMSTCNPlusPlus
from phase_classification.surgclip_diffusion import (
    PHASE_PROMPTS,
    ConditionalTemporalDiffusion,
    _patch_surgclip_tokenizer,
    _score_clip_batch,
)

CHOLECT_PHASES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingAndCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningAndCoagulation",
    "GallbladderRetraction",
]
M2CAI_PHASES = [
    "TrocarPlacement",
    "Preparation",
    *CHOLECT_PHASES[1:],
]
AUTOLAPARO_PHASES = [
    "Preparation",
    "DividingLigamentAndPeritoneum",
    "DividingUterineVesselsAndLigament",
    "TransectingTheVagina",
    "SpecimenRemoval",
    "Suturing",
    "Washing",
]
_NATIVE_PHASE_NAMES = {
    "cholec80": CHOLECT_PHASES,
    "m2cai16": M2CAI_PHASES,
    "autolaparo": AUTOLAPARO_PHASES,
}

_PHASE_ALIASES = {
    "preparation": 0,
    "trocarplacement": 0,
    "calottriangledissection": 1,
    "calotdissection": 1,
    "clippingandcutting": 2,
    "clippingcutting": 2,
    "gallbladderdissection": 3,
    "gallbladderpackaging": 4,
    "cleaningandcoagulation": 5,
    "cleaningcoagulation": 5,
    "gallbladderretraction": 6,
    "gallbladderextraction": 6,
    "dividingligamentandperitoneum": 1,
    "dividinguterinevesselsandligament": 2,
    "transectingthevagina": 3,
    "specimenremoval": 4,
    "suturing": 5,
    "washing": 6,
}
_M2CAI_PHASE_ALIASES = {
    "trocarplacement": 0,
    "preparation": 1,
    "calottriangledissection": 2,
    "calotdissection": 2,
    "clippingandcutting": 3,
    "clippingcutting": 3,
    "gallbladderdissection": 4,
    "gallbladderpackaging": 5,
    "cleaningandcoagulation": 6,
    "cleaningcoagulation": 6,
    "gallbladderretraction": 7,
    "gallbladderextraction": 7,
}


@dataclass(frozen=True)
class PhaseVideo:
    video_id: str
    video_path: Path
    annotation_path: Path


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.strip().lower())


def parse_phase_value(value: str, dataset: str) -> int:
    """Convert a phase name or dataset-specific integer into a zero-based ID."""
    return _phase_for_dataset(value, dataset)


def _video_id_number(video_id: str) -> int | None:
    match = re.search(
        r"(?:video|vid|workflow_video)[_-]?(\d+)", video_id, re.IGNORECASE
    )
    return int(match.group(1)) if match else None


def discover_cholec80(root: Path, split: str) -> list[PhaseVideo]:
    videos_dir = root / "videos"
    annotations_dir = root / "phase_annotations"
    pairs: list[PhaseVideo] = []
    for annotation in sorted(annotations_dir.glob("video*-phase.txt")):
        video_id = annotation.name.removesuffix("-phase.txt")
        number = _video_id_number(video_id)
        if number is None or (split == "train" and number > 40) or (
            split == "test" and number <= 40
        ):
            continue
        video = videos_dir / f"{video_id}.mp4"
        if video.is_file():
            pairs.append(PhaseVideo(video_id, video, annotation))
    if not pairs:
        raise FileNotFoundError(
            f"No Cholec80 video/annotation pairs under {root}; expected "
            "videos/videoXX.mp4 and phase_annotations/videoXX-phase.txt."
        )
    return pairs


def _candidate_annotation(video_path: Path, roots: list[Path]) -> Path | None:
    stems = [video_path.stem]
    number = _video_id_number(video_path.stem)
    if number is not None:
        stems.extend(
            [
                f"workflow_video_{number:02d}",
                f"video{number:02d}",
                f"video_{number:02d}",
                f"{number:02d}",
                str(number),
            ]
        )
    extensions = (".txt", ".tsv", ".csv")
    for directory in roots:
        for stem in stems:
            for extension in extensions:
                candidate = directory / f"{stem}{extension}"
                if candidate.is_file():
                    return candidate
                for suffix in ("-phase", "_phase", "-Phase", "_Annotation_Phase"):
                    candidate = directory / f"{stem}{suffix}{extension}"
                    if candidate.is_file():
                        return candidate
    return None


def discover_m2cai16(root: Path, split: str) -> list[PhaseVideo]:
    annotation_dirs = [
        path
        for parent in (root, root / "train", root / "test", root / "Train", root / "Test")
        for name in ("annotations", "phase_annotations", "Annotations", "labels")
        if (path := parent / name).is_dir()
    ]
    videos: dict[str, Path] = {}
    for directory in (
        root / "videos",
        root / "Videos",
        root / "train" / "videos",
        root / "test" / "videos",
        root / "Train" / "videos",
        root / "Test" / "videos",
        root,
    ):
        if directory.is_dir():
            for path in directory.glob("workflow_video_*.mp4"):
                videos[path.stem.lower()] = path
    pairs: list[PhaseVideo] = []
    for annotation_dir in annotation_dirs:
        if not annotation_dir.is_dir():
            continue
        for annotation in sorted(annotation_dir.glob("workflow_video_*.txt")):
            number = _video_id_number(annotation.stem)
            if number is None or (split == "train" and number > 27) or (
                split == "test" and number <= 27
            ):
                continue
            video_id = annotation.stem.lower()
            video = videos.get(video_id)
            if video is not None:
                pairs.append(PhaseVideo(video.stem, video, annotation))
    if not pairs:
        raise FileNotFoundError(
            f"No annotated M2CAI16 videos under {root}; expected videos/ and "
            "annotations/workflow_video_XX.txt (CAMMA release layout)."
        )
    return pairs


def discover_autolaparo(root: Path) -> list[PhaseVideo]:
    """Discover video + sidecar labels; use --manifest for other release layouts."""
    annotation_dirs = [
        root / name
        for name in (
            "annotations",
            "phase_annotations",
            "labels",
            "Task1/annotations",
            "Task1/Annotations",
            "task1/annotations",
        )
    ]
    video_dirs = [
        root / name
        for name in (
            "videos",
            "video",
            "video_src_path",
            "Task1/videos",
            "Task1/video_src_path",
            "task1/videos",
        )
    ]
    paths: set[Path] = set()
    for directory in video_dirs:
        if directory.is_dir():
            paths.update(directory.glob("*.mp4"))
    # Also support a directory of 1-fps images from the official preprocessing.
    for directory in (root / "frame_save_path", root / "frames", root / "Task1/frames"):
        if directory.is_dir():
            paths.update(
                path
                for path in directory.iterdir()
                if path.is_dir()
                and any(path.glob("*.jpg"))
            )
    pairs = []
    for video in sorted(paths):
        annotation = _candidate_annotation(video, annotation_dirs + [root])
        if annotation is not None:
            pairs.append(PhaseVideo(video.name if video.is_dir() else video.stem, video, annotation))
    if not pairs:
        raise FileNotFoundError(
            f"No AutoLaparo Task-1 video/annotation pairs under {root}. "
            "Expected videos plus per-video CSV/TSV/TXT phase annotations, or use --manifest."
        )
    return pairs


def discover_manifest(path: Path, dataset: str) -> list[PhaseVideo]:
    """Read a portable manifest with video_id, video_path, annotation_path columns."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "video_path", "annotation_path"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Manifest must have video_id, video_path, annotation_path columns."
            )
        result = []
        for row in reader:
            video_path = Path(row["video_path"])
            annotation_path = Path(row["annotation_path"])
            if not video_path.is_absolute():
                video_path = path.parent / video_path
            if not annotation_path.is_absolute():
                annotation_path = path.parent / annotation_path
            if not video_path.exists() or not annotation_path.is_file():
                raise FileNotFoundError(f"Manifest paths do not exist: {row}")
            result.append(PhaseVideo(row["video_id"], video_path, annotation_path))
    return result


def _time_seconds(value: str) -> float:
    value = value.strip()
    if ":" not in value:
        return float(value)
    parts = [float(part) for part in value.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def _read_rows(path: Path) -> list[list[str]]:
    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Annotation file is empty: {path}")
    sample = "\n".join(lines[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        return [next(csv.reader([line], dialect)) for line in lines]
    except csv.Error:
        return [re.split(r"\s+", line.strip()) for line in lines]


def _header_indices(row: list[str]) -> dict[str, int] | None:
    names = [_normalise(cell) for cell in row]
    recognized = {
        "frame", "frameid", "frameindex", "timestamp", "time", "phase", "label",
        "workflow", "start", "startframe", "starttime", "end", "endframe", "endtime",
    }
    if not any(name in recognized for name in names):
        return None
    return {name: index for index, name in enumerate(names)}


def _phase_for_dataset(raw: str, dataset: str) -> int:
    value = raw.strip()
    key = _normalise(value)
    if dataset == "m2cai16" and key in _M2CAI_PHASE_ALIASES:
        return _M2CAI_PHASE_ALIASES[key]
    if key in _PHASE_ALIASES:
        return _PHASE_ALIASES[key]
    if key.startswith("p") and key[1:].isdigit():
        phase = int(key[1:]) - 1
        if (dataset == "m2cai16" and 0 <= phase < 8) or (
            dataset != "m2cai16" and 0 <= phase < 7
        ):
            return phase
        raise ValueError(f"{dataset} phase value is out of range: {raw!r}")
    phase = int(value)
    if dataset == "autolaparo":
        phase -= 1
    max_phase = 7 if dataset == "m2cai16" else 6
    if not 0 <= phase <= max_phase:
        raise ValueError(f"{dataset} phase value is out of range: {raw!r}")
    return phase


def sample_annotation(
    path: Path,
    dataset: str,
    sample_times: list[float],
    annotation_fps: float,
) -> list[int | None]:
    """Read dense frame labels or start/end intervals at requested video times."""
    rows = _read_rows(path)
    header = _header_indices(rows[0])
    data = rows[1:] if header is not None else rows
    result: list[int | None] = [None] * len(sample_times)
    intervals: list[tuple[float, float, int]] = []
    tolerance = 0.5 / max(annotation_fps, 1.0) + 1e-3

    for row_index, row in enumerate(data):
        if not row:
            continue
        if header is not None:
            phase_col = next(
                (header[k] for k in ("phase", "label", "workflow") if k in header),
                None,
            )
            if phase_col is None or len(row) <= phase_col:
                continue
            raw_phase = row[phase_col]
            start_col = next(
                (header[k] for k in ("starttime", "startframe", "start") if k in header),
                None,
            )
            end_col = next(
                (header[k] for k in ("endtime", "endframe", "end") if k in header),
                None,
            )
            if start_col is not None and end_col is not None:
                if len(row) <= max(start_col, end_col):
                    continue
                start_value, end_value = row[start_col], row[end_col]
                start_name = next(name for name, index in header.items() if index == start_col)
                end_name = next(name for name, index in header.items() if index == end_col)
                frame_units = "frame" in start_name or "frame" in end_name
                start = float(start_value) / annotation_fps if frame_units else _time_seconds(start_value)
                end = float(end_value) / annotation_fps if frame_units else _time_seconds(end_value)
                intervals.append((start, end, _phase_for_dataset(raw_phase, dataset)))
                continue
            time_col = next(
                (header[k] for k in ("timestamp", "time") if k in header), None
            )
            frame_col = next(
                (header[k] for k in ("frame", "frameid", "frameindex") if k in header),
                None,
            )
            if time_col is not None and len(row) > time_col:
                annotation_time = _time_seconds(row[time_col])
            elif frame_col is not None and len(row) > frame_col:
                annotation_time = float(row[frame_col]) / annotation_fps
            else:
                annotation_time = row_index / annotation_fps
        elif len(row) >= 3:
            # Common interval format: start, end, phase.
            start, end = float(row[0]) / annotation_fps, float(row[1]) / annotation_fps
            intervals.append((start, end, _phase_for_dataset(row[2], dataset)))
            continue
        elif len(row) == 2:
            # CAMMA Cholec80/M2CAI format: frame index<TAB>phase name.
            annotation_time = float(row[0]) / annotation_fps
            raw_phase = row[1]
        else:
            annotation_time = row_index / annotation_fps
            raw_phase = row[0]
        phase = _phase_for_dataset(raw_phase, dataset)
        if sample_times:
            period = sample_times[1] - sample_times[0] if len(sample_times) > 1 else 1.0
            nearest = round((annotation_time - sample_times[0]) / period)
            if (
                0 <= nearest < len(sample_times)
                and abs(sample_times[nearest] - annotation_time) <= tolerance
            ):
                result[nearest] = phase

    if intervals:
        for index, sample_time in enumerate(sample_times):
            for start, end, phase in intervals:
                if start <= sample_time <= end:
                    result[index] = phase
                    break
    if not any(value is not None for value in result):
        raise ValueError(f"No labels from {path} align with the sampled video times.")
    return result


def _cholect50_video_ids() -> set[int]:
    selector = object.__new__(CholecT50)
    folds = selector.split_selector(case="cholect50-crossval")
    return {int(video_id) for fold in folds.values() for video_id in fold}


def _filter_cholec80_overlap(videos: list[PhaseVideo]) -> tuple[list[PhaseVideo], list[str]]:
    known = _cholect50_video_ids()
    excluded = [
        item.video_id
        for item in videos
        if (number := _video_id_number(item.video_id)) is not None and number in known
    ]
    return [item for item in videos if item.video_id not in set(excluded)], excluded


def _sample_window_start(anchor: int, total: int, length: int) -> int:
    if total <= length:
        return 0
    start = max(0, anchor - length // 2)
    if start + length > total:
        start = total - length
    return start


def _iter_sampled_frames(video_path: Path, sample_seconds: float):
    """Yield RGB frames at approximately the requested rate, resized like cache frames."""
    import cv2

    if video_path.is_dir():
        paths = sorted(
            (p for p in video_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
            key=lambda p: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", p.name)],
        )
        if not paths:
            raise ValueError(f"No image frames in {video_path}")
        for index in range(0, len(paths), max(1, round(sample_seconds))):
            with Image.open(paths[index]) as image:
                yield np.asarray(image.convert("RGB").resize((448, 256)))
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise ValueError(f"Could not read frame rate for {video_path}")
    stride = max(1, round(fps * sample_seconds))
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            frame = cv2.resize(frame, (448, 256), interpolation=cv2.INTER_AREA)
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_index += 1
    capture.release()


@torch.inference_mode()
def extract_video_embeddings(
    video_path: Path,
    model,
    preprocess,
    text_features: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    num_frames: int,
    sample_seconds: float,
    window_batch_size: int,
) -> tuple[torch.Tensor, list[float]]:
    """Extract the same centered SurgCLIP features used by the CholecT50 cache."""
    import cv2

    if video_path.is_dir():
        sample_fps = 1.0 / sample_seconds
        frame_count = sum(
            1
            for p in video_path.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        total = int(np.ceil(frame_count / max(1, round(sample_seconds))))
        sample_times = [i / sample_fps for i in range(total)]
    else:
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0 or frame_count <= 0:
            raise ValueError(f"Invalid video metadata for {video_path}")
        stride = max(1, round(fps * sample_seconds))
        sample_times = [i * stride / fps for i in range((frame_count + stride - 1) // stride)]
        sample_fps = fps / stride

    total = len(sample_times)
    if total == 0:
        raise ValueError(f"No sample frames found in {video_path}")
    groups: dict[int, list[int]] = defaultdict(list)
    for anchor in range(total):
        groups[_sample_window_start(anchor, total, num_frames)].append(anchor)

    ring: deque[np.ndarray] = deque(maxlen=num_frames)
    batch_clips: list[torch.Tensor] = []
    batch_starts: list[int] = []
    embeddings: list[torch.Tensor | None] = [None] * total

    def flush() -> None:
        if not batch_clips:
            return
        clips = torch.stack(batch_clips)
        values, _ = _score_clip_batch(
            model, clips, text_features, temperature, device, amp=False
        )
        for start, embedding in zip(batch_starts, values, strict=True):
            for anchor in groups[start]:
                embeddings[anchor] = embedding.detach().cpu()
        batch_clips.clear()
        batch_starts.clear()

    def queue_window(start: int, frames: list[np.ndarray]) -> None:
        clip = torch.stack(
            [preprocess(Image.fromarray(frame)) for frame in frames]
        )
        batch_clips.append(clip)
        batch_starts.append(start)
        if len(batch_clips) >= window_batch_size:
            flush()

    for sample_index, frame in enumerate(_iter_sampled_frames(video_path, sample_seconds)):
        ring.append(frame)
        if len(ring) == num_frames:
            start = sample_index - num_frames + 1
            if start in groups:
                queue_window(start, list(ring))

    if total < num_frames:
        frames = list(ring)
        frames.extend([frames[-1]] * (num_frames - len(frames)))
        queue_window(0, frames)
    flush()
    if any(embedding is None for embedding in embeddings):
        raise RuntimeError(f"Failed to create feature windows for {video_path}")
    return torch.stack([embedding for embedding in embeddings if embedding is not None]), sample_times


def _load_surgclip(model_name: str, num_frames: int, device: torch.device):
    import surgclip

    _patch_surgclip_tokenizer(surgclip)
    model, preprocess, tokenizer = surgclip.load(
        model_name,
        device=device,
        num_frames=num_frames,
        temporal_modeling=num_frames > 1,
    )
    model.eval()
    token_ids = surgclip.tokenize(PHASE_PROMPTS, tokenizer, device=device)
    with torch.inference_mode():
        _, pooled_text = model.encode_text(token_ids)
        text_features = F.normalize(model.text_proj(pooled_text.float()), dim=-1)
        temperature = model.temp.detach().float().clamp(min=1e-6)
    return model, preprocess, text_features, temperature


def _load_temporal_models(checkpoint_path: Path, config_path: Path, device: torch.device):
    cfg = load_config(str(config_path))
    if cfg.model.name not in {"surgclip-ms-tcnpp-diffusion", "surgclip-ms-tcn-diffusion"}:
        raise ValueError("External evaluation expects a SurgCLIP MS-TCN diffusion config.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not {"coarse_state_dict", "refiner_state_dict"} <= set(checkpoint):
        raise ValueError(f"Not a combined SurgCLIP diffusion checkpoint: {checkpoint_path}")
    params: dict[str, Any] = dict(checkpoint.get("settings", cfg.model.params))
    feature_dim = int(checkpoint["feature_dim"])
    classes = cfg.model.num_classes
    if cfg.model.name == "surgclip-ms-tcnpp-diffusion":
        coarse = CachedFeatureMSTCNPlusPlus(
            feature_dim=feature_dim,
            num_classes=classes,
            heads=["phase"],
            num_refinement_stages=int(params.get("num_refinement_stages", 3)),
            generation_layers=int(params.get("generation_layers", 11)),
            refinement_layers=int(params.get("refinement_layers", 10)),
            num_f_maps=int(params.get("num_f_maps", 64)),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.5)),
            smoothing_weight=float(params.get("smoothing_weight", 0.15)),
        )
    else:
        from phase_classification.models.ms_tcn import CachedFeatureMSTCN

        coarse = CachedFeatureMSTCN(
            feature_dim=feature_dim,
            num_classes=classes,
            heads=["phase"],
            num_stages=int(params.get("num_stages", 3)),
            num_layers=int(params.get("num_layers", 8)),
            num_f_maps=int(params.get("num_f_maps", 64)),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.0)),
        )
    coarse.load_state_dict(checkpoint["coarse_state_dict"])
    coarse.to(device).eval()
    refiner = ConditionalTemporalDiffusion(
        condition_dim=classes["phase"] + feature_dim,
        num_classes=classes["phase"],
        width=int(params.get("refiner_width", 96)),
        layers=int(params.get("refiner_layers", 6)),
        diffusion_steps=int(params.get("diffusion_timesteps", 1000)),
        correction_scale=float(params.get("correction_scale", 1.0)),
    )
    refiner.load_state_dict(checkpoint["refiner_state_dict"])
    refiner.to(device).eval()
    return coarse, refiner, params, feature_dim


def _eval_phase_id(raw_id: int, dataset: str) -> int | None:
    if dataset == "cholec80":
        return raw_id
    if dataset == "m2cai16":
        # M2CAI has a separate Trocar Placement class; CholecT50's Preparation
        # phase includes this early activity, so merge it explicitly.
        return (0, 0, 1, 2, 3, 4, 5, 6)[raw_id]
    return None


def _sequence_metrics(
    probabilities: np.ndarray, target_ids: list[int], metric_names: list[str]
) -> dict[str, float]:
    acc = MetricAccumulator(["phase"], metric_names, {"phase": 7})
    targets = np.eye(7, dtype=np.float32)[np.asarray(target_ids, dtype=np.int64)]
    acc.add_batch({"phase": probabilities}, {"phase": targets})
    return acc.compute()


def evaluate_video(
    video: PhaseVideo,
    dataset: str,
    coarse,
    refiner,
    surgclip_model,
    preprocess,
    text_features: torch.Tensor,
    temperature: torch.Tensor,
    device: torch.device,
    num_frames: int,
    sample_seconds: float,
    window_batch_size: int,
    sample_steps: int,
    annotation_fps: float | None,
) -> dict[str, Any]:
    if annotation_fps is None:
        annotation_fps = {"cholec80": 25.0, "m2cai16": 25.0, "autolaparo": 1.0}[dataset]
    embeddings, sample_times = extract_video_embeddings(
        video.video_path,
        surgclip_model,
        preprocess,
        text_features,
        temperature,
        device,
        num_frames,
        sample_seconds,
        window_batch_size,
    )
    labels = sample_annotation(video.annotation_path, dataset, sample_times, annotation_fps)
    with torch.inference_mode():
        features = embeddings.unsqueeze(0).to(device)
        coarse_logits = coarse(features)["phase"][0]
        coarse_probabilities = coarse_logits.softmax(dim=-1)
        condition = torch.cat(
            (coarse_probabilities, F.normalize(features[0], dim=-1)), dim=-1
        ).unsqueeze(0)
        refined_probabilities = refiner.sample_probabilities(
            condition,
            sample_steps=sample_steps,
            num_samples=1,
            seed=42,
        )[0]
    coarse_probs = coarse_probabilities.cpu().numpy()
    refined_probs = refined_probabilities.cpu().numpy()

    scored = [
        (index, label, _eval_phase_id(label, dataset))
        for index, label in enumerate(labels)
        if label is not None
    ]
    metrics: dict[str, Any] = {"metric_semantics": "not_comparable" if dataset == "autolaparo" else "mapped_to_cholect7"}
    if dataset != "autolaparo" and scored:
        indices = [item[0] for item in scored]
        target_ids = [int(item[2]) for item in scored]
        metrics["coarse"] = _sequence_metrics(coarse_probs[indices], target_ids, ["f1_macro", "jaccard", "mean_accuracy", "edit_score"])
        metrics["refined"] = _sequence_metrics(refined_probs[indices], target_ids, ["f1_macro", "jaccard", "mean_accuracy", "edit_score"])

    rows = []
    for index, raw_label in enumerate(labels):
        eval_label = _eval_phase_id(raw_label, dataset) if raw_label is not None else None
        coarse_id = int(coarse_probs[index].argmax())
        refined_id = int(refined_probs[index].argmax())
        rows.append(
            {
                "sample_index": index,
                "time_seconds": round(sample_times[index], 3),
                "ground_truth_id": raw_label,
                "ground_truth_phase": (
                    _NATIVE_PHASE_NAMES[dataset][raw_label]
                    if raw_label is not None
                    else None
                ),
                "evaluation_target_id": eval_label,
                "coarse_prediction_id": coarse_id,
                "coarse_prediction": CHOLECT_PHASES[coarse_id],
                "coarse_confidence": round(float(coarse_probs[index, coarse_id]), 6),
                "refined_prediction_id": refined_id,
                "refined_prediction": CHOLECT_PHASES[refined_id],
                "refined_confidence": round(float(refined_probs[index, refined_id]), 6),
            }
        )
    return {
        "video_id": video.video_id,
        "video_path": str(video.video_path),
        "annotation_path": str(video.annotation_path),
        "sample_count": len(sample_times),
        "annotated_sample_count": len(scored),
        "metrics": metrics,
        "predictions": rows,
        "_metric_targets": [int(item[2]) for item in scored if item[2] is not None],
        "_coarse_metric_probabilities": (
            coarse_probs[[item[0] for item in scored]].tolist()
            if dataset != "autolaparo" and scored
            else []
        ),
        "_refined_metric_probabilities": (
            refined_probs[[item[0] for item in scored]].tolist()
            if dataset != "autolaparo" and scored
            else []
        ),
    }


def _write_video_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("cholec80", "m2cai16", "autolaparo"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("all", "train", "test"), default="test")
    parser.add_argument("--video", default=None, help="Evaluate one video ID, e.g. video41.")
    parser.add_argument("--config", type=Path, default=Path("configs/surgclip-ms-tcnpp-diffusion.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--surgclip-model", default="SurgCLIP-B")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--sample-seconds", type=float, default=1.0)
    parser.add_argument("--annotation-fps", type=float, default=None)
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--include-cholect50-overlap", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.num_frames < 1 or args.sample_seconds <= 0 or args.window_batch_size < 1:
        raise SystemExit("num-frames/window-batch-size must be positive; sample-seconds > 0")
    if args.manifest is not None:
        videos = discover_manifest(args.manifest, args.dataset)
    elif args.dataset == "cholec80":
        videos = discover_cholec80(args.root, args.split)
    elif args.dataset == "m2cai16":
        videos = discover_m2cai16(args.root, args.split)
    else:
        videos = discover_autolaparo(args.root)

    excluded: list[str] = []
    if args.dataset == "cholec80" and not args.include_cholect50_overlap:
        videos, excluded = _filter_cholec80_overlap(videos)
        if not videos:
            raise SystemExit("All selected Cholec80 videos overlap CholecT50; use --include-cholect50-overlap to override.")
    if args.video is not None:
        videos = [video for video in videos if video.video_id.lower() == args.video.lower()]
        if not videos:
            raise SystemExit(
                f"Video {args.video!r} is not in the selected split. For a "
                "CholecT50-overlapping video, pass --include-cholect50-overlap."
            )
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    device = resolve_device(args.device)
    print(f"Dataset={args.dataset}; videos={len(videos)}; device={device}")
    if excluded:
        print(f"Excluded {len(excluded)} CholecT50-overlapping videos: {', '.join(excluded)}")

    coarse, refiner, settings, feature_dim = _load_temporal_models(
        args.checkpoint, args.config, device
    )
    if int(settings.get("diffusion_timesteps", 1000)) < args.sample_steps:
        raise SystemExit("--sample-steps exceeds the checkpoint diffusion schedule")
    surgclip_model, preprocess, text_features, temperature = _load_surgclip(
        args.surgclip_model, args.num_frames, device
    )
    output = args.output or Path("outputs/external-eval") / args.dataset / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)

    results = []
    coarse_accumulators: list[MetricAccumulator] = []
    refined_accumulators: list[MetricAccumulator] = []
    for index, video in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video.video_id} ...", flush=True)
        started = time.time()
        result = evaluate_video(
            video,
            args.dataset,
            coarse,
            refiner,
            surgclip_model,
            preprocess,
            text_features,
            temperature,
            device,
            args.num_frames,
            args.sample_seconds,
            args.window_batch_size,
            args.sample_steps,
            args.annotation_fps,
        )
        _write_video_predictions(output / f"{video.video_id}_predictions.csv", result.pop("predictions"))
        targets = result.pop("_metric_targets")
        coarse_probabilities = result.pop("_coarse_metric_probabilities")
        refined_probabilities = result.pop("_refined_metric_probabilities")
        if targets:
            target_matrix = np.eye(7, dtype=np.float32)[np.asarray(targets)]
            for probabilities, accumulators in (
                (coarse_probabilities, coarse_accumulators),
                (refined_probabilities, refined_accumulators),
            ):
                accumulator = MetricAccumulator(
                    ["phase"],
                    ["f1_macro", "jaccard", "mean_accuracy", "edit_score"],
                    {"phase": 7},
                )
                accumulator.add_batch(
                    {"phase": np.asarray(probabilities)},
                    {"phase": target_matrix},
                )
                accumulators.append(accumulator)
        results.append(result)
        if "coarse" in result["metrics"]:
            print(
                f"  coarse F1={result['metrics']['coarse']['phase_f1_macro']:.4f}; "
                f"refined F1={result['metrics']['refined']['phase_f1_macro']:.4f}; "
                f"{time.time() - started:.1f}s",
                flush=True,
            )
        else:
            print(f"  predictions exported; taxonomy is not comparable ({time.time() - started:.1f}s)", flush=True)

    summary: dict[str, Any] = {
        "dataset": args.dataset,
        "root": str(args.root),
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "surgclip_model": args.surgclip_model,
        "num_frames": args.num_frames,
        "sample_seconds": args.sample_seconds,
        "feature_dim": feature_dim,
        "excluded_cholect50_overlap": excluded,
        "phase_mapping": (
            "M2CAI TrocarPlacement and Preparation merged into CholecT50 Preparation"
            if args.dataset == "m2cai16"
            else "Cholec80 7-phase labels map directly to CholecT50"
            if args.dataset == "cholec80"
            else "different hysterectomy taxonomy; predictions only, no classification metrics"
        ),
        "videos": results,
    }
    metrics_available = [
        result for result in results if "coarse" in result.get("metrics", {})
    ]
    if metrics_available:
        summary["coarse_mean_video_macro_f1"] = float(
            np.mean([item["metrics"]["coarse"]["phase_f1_macro"] for item in metrics_available])
        )
        summary["refined_mean_video_macro_f1"] = float(
            np.mean([item["metrics"]["refined"]["phase_f1_macro"] for item in metrics_available])
        )
        summary["coarse_pooled_metrics"] = MetricAccumulator.merged(
            coarse_accumulators
        ).compute()
        summary["refined_pooled_metrics"] = MetricAccumulator.merged(
            refined_accumulators
        ).compute()
    summary_path = output / "metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved cross-dataset evaluation to {summary_path}")
    if metrics_available:
        print(
            f"Mean per-video macro-F1: coarse={summary['coarse_mean_video_macro_f1']:.4f}; "
            f"refined={summary['refined_mean_video_macro_f1']:.4f}"
        )
    elif args.dataset == "autolaparo":
        print("No accuracy/F1 reported: AutoLaparo's hysterectomy phase taxonomy differs from the trained Cholec classifier.")


if __name__ == "__main__":
    main()
