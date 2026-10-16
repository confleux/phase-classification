"""CholecT50 full-video datasets backed by frozen SurgCLIP feature caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX


@dataclass
class FeatureDataLoaders:
    """One complete video per item/batch; test loaders remain video-separated."""

    train: DataLoader
    val: DataLoader
    test: list[DataLoader]
    feature_dim: int
    videos: dict[str, list[str]]


class FeatureVideoDataset(Dataset):
    """A single full-video feature sequence and its five CholecT50 label heads."""

    def __init__(self, features: torch.Tensor, labels: tuple[torch.Tensor, ...]):
        self.features = features.float()
        self.labels = labels

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if index != 0:
            raise IndexError(index)
        return self.features, self.labels


def _phase_labels(dataset: Any, video_id: str) -> tuple[torch.Tensor, ...]:
    frame_order = sorted(dataset.frames, key=int)
    labels_by_head = [
        torch.zeros(len(frame_order), DEFAULT_NUM_CLASSES[head], dtype=torch.float32)
        for head, _ in sorted(HEAD_TUPLE_INDEX.items(), key=lambda item: item[1])
    ]
    for frame_index, frame in enumerate(frame_order):
        counts = np.zeros(DEFAULT_NUM_CLASSES["phase"], dtype=np.float32)
        for row in dataset.label_data[frame]:
            if len(row) > 14 and row[14] >= 0:
                phase = int(row[14])
                if phase >= len(counts):
                    raise ValueError(f"Invalid phase id {phase} at {video_id}/{frame}.")
                counts[phase] += 1
        if not counts.any():
            raise ValueError(f"No phase annotation at {video_id}/{frame}.")
        labels_by_head[HEAD_TUPLE_INDEX["phase"]][frame_index] = torch.from_numpy(
            (counts > 0).astype(np.float32)
        )
    return tuple(labels_by_head)


def _cache_video_features(cache: dict[str, Any]) -> dict[str, torch.Tensor]:
    if isinstance(cache.get("splits"), dict):
        by_video = {}
        for items in cache["splits"].values():
            for item in items:
                video_id = str(item["video_id"])
                if video_id in by_video:
                    raise ValueError(f"Duplicate video in feature cache: {video_id}.")
                by_video[video_id] = torch.as_tensor(item["embeddings"]).float()
        return by_video

    if "features" not in cache or "paths" not in cache:
        raise ValueError(
            "Unsupported feature cache; expected {'features', 'paths'} or per-video 'splits'."
        )
    features = torch.as_tensor(cache["features"]).float()
    paths = cache["paths"]
    if features.ndim != 2 or len(features) != len(paths):
        raise ValueError("Flat cache must contain features [N, D] and N paths.")
    by_frame = {
        (Path(str(path)).parent.name, Path(str(path)).stem): features[index]
        for index, path in enumerate(paths)
    }
    return _group_flat_features(by_frame)


def _group_flat_features(
    by_frame: dict[tuple[str, str], torch.Tensor]
) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[tuple[int, torch.Tensor]]] = {}
    for (video_id, frame), feature in by_frame.items():
        grouped.setdefault(video_id, []).append((int(frame), feature))
    return {
        video_id: torch.stack([feature for _, feature in sorted(items)])
        for video_id, items in grouped.items()
    }


def build_feature_dataloaders(
    dataset_dir: str | Path,
    cache_path: str | Path,
    variant: str = "cholect50-crossval",
    test_fold: int = 1,
    seed: int = 42,
) -> FeatureDataLoaders:
    """Align a repository SurgCLIP cache to CholecT50 and create video batches."""
    if not variant.startswith("cholect"):
        raise ValueError("Cached SurgCLIP feature probing currently supports CholecT50 variants only.")
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    cached_variant = metadata.get("dataset_variant", cache.get("variant"))
    cached_fold = metadata.get("test_fold", cache.get("test_fold"))
    if cached_variant is not None and cached_variant != variant:
        raise ValueError(f"Cache variant {cached_variant!r} does not match {variant!r}.")
    if cached_fold is not None and int(cached_fold) != test_fold:
        raise ValueError(f"Cache fold {cached_fold} does not match requested fold {test_fold}.")

    split = CholecT50(
        dataset_dir=str(dataset_dir),
        dataset_variant=variant,
        test_fold=test_fold,
        augmentation_list=["original"],
        normalize=False,
        sequence_length=1,
    )
    datasets = {
        "train": list(split.train_dataset.datasets),
        "validation": list(split.val_dataset.datasets),
        "test": list(split.test_dataset),
    }
    cached = _cache_video_features(cache)
    saved_targets = {
        str(item["video_id"]): torch.as_tensor(item["targets"], dtype=torch.long)
        for items in cache.get("splits", {}).values()
        for item in items
        if "targets" in item
    }

    built: dict[str, list[FeatureVideoDataset]] = {}
    video_ids: dict[str, list[str]] = {}
    feature_dim = None
    for name, video_datasets in datasets.items():
        built[name], video_ids[name] = [], []
        for source in video_datasets:
            video_id = Path(source.img_dir).name
            if video_id not in cached:
                raise ValueError(f"Feature cache has no embeddings for {video_id}.")
            labels = _phase_labels(source, video_id)
            features = cached[video_id]
            if features.ndim != 2:
                raise ValueError(f"{video_id}: expected features shaped (T, D).")
            phase_targets = labels[HEAD_TUPLE_INDEX["phase"]].argmax(dim=-1)
            if len(features) != len(phase_targets):
                raise ValueError(
                    f"{video_id}: cache has {len(features)} features for "
                    f"{len(phase_targets)} annotated frames."
                )
            if video_id in saved_targets and not torch.equal(saved_targets[video_id], phase_targets):
                raise ValueError(f"Cached targets do not align with CholecT50 for {video_id}.")
            feature_dim = int(features.shape[-1]) if feature_dim is None else feature_dim
            if features.shape[-1] != feature_dim:
                raise ValueError(f"Inconsistent feature dimension in video {video_id}.")
            built[name].append(FeatureVideoDataset(features, labels))
            video_ids[name].append(video_id)

    if not built["train"] or not built["validation"] or not built["test"]:
        raise ValueError("Feature cache produced an empty train, validation, or test split.")

    train = DataLoader(
        ConcatDataset(built["train"]), batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(seed)
    )
    val = DataLoader(ConcatDataset(built["validation"]), batch_size=1, shuffle=False)
    test = [DataLoader(video, batch_size=1, shuffle=False) for video in built["test"]]
    return FeatureDataLoaders(train, val, test, int(feature_dim), video_ids)
