"""HeiChole phase-recognition dataset loader.

Expected local layout (a convenience convention; the challenge page does not
specify archive filenames)::

    <root>/
      videos/<case-id>.mp4
      labels/<case-id>_Phase.csv

Videos may use any OpenCV-supported extension. Labels are CSV files with the
official ``Frame,Phase`` columns and zero-based frame/phase IDs. A label named
``<case-id>.csv`` is accepted as well. The loader uses the public training
cases to make deterministic five-fold splits; ``test_fold`` selects the
held-out fold and the next fold is used for validation.

HeiChole phase labels map directly to seven single-class phase outputs. The
five-head return shape mirrors the CholecT50 loader for compatibility; the
non-phase heads are zero-filled and should not be enabled for HeiChole runs.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg"}
_PHASE_NAMES = {"phase", "phaseid", "surgicalphase"}
_FRAME_NAMES = {"frame", "frameid", "frameindex"}


def _normalized_stem(path: Path) -> str:
    stem = path.stem.casefold()
    for suffix in ("_phase", "-phase", "_phases", "-phases"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _read_phase_csv(path: Path) -> dict[int, int]:
    """Read the challenge ``Frame,Phase`` annotation CSV."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"HeiChole annotation CSV is empty: {path}")

        columns = {name.strip().casefold().replace("_", ""): name for name in reader.fieldnames}
        frame_column = next(
            (columns[name] for name in _FRAME_NAMES if name in columns), None
        )
        phase_column = next(
            (columns[name] for name in _PHASE_NAMES if name in columns), None
        )
        if frame_column is None or phase_column is None:
            raise ValueError(
                f"Expected Frame and Phase columns in HeiChole CSV {path}; "
                f"found {reader.fieldnames}"
            )

        annotations: dict[int, int] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(row[frame_column])
                phase = int(row[phase_column])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frame/phase value in {path} at line {line_number}"
                ) from exc
            if frame < 0:
                raise ValueError(f"Frame indices must be non-negative in {path}")
            if not 0 <= phase < 7:
                raise ValueError(f"HeiChole phase IDs must be in [0, 6] in {path}")
            if frame in annotations:
                raise ValueError(f"Duplicate frame {frame} in HeiChole CSV {path}")
            annotations[frame] = phase

    if not annotations:
        raise ValueError(f"HeiChole annotation CSV contains no labeled frames: {path}")
    return annotations


def _load_video_frame(video_path: Path, frame_index: int) -> Image.Image:
    """Decode one zero-based video frame to an RGB PIL image."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Reading HeiChole videos requires opencv-python-headless. "
            "Install the project with its dev dependencies or add that package."
        ) from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open HeiChole video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            raise IndexError(
                f"Frame {frame_index} is not available in HeiChole video {video_path}"
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)
    finally:
        capture.release()


class HeiCholeVideo(Dataset):
    """Frame samples (or contiguous temporal windows) from one operation."""

    def __init__(
        self,
        video_path: Path,
        annotations: dict[int, int],
        transform,
        sequence_length: int = 1,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.video_path = video_path
        self.annotations = annotations
        self.frames = sorted(annotations)
        self.transform = transform
        self.sequence_length = sequence_length
        if sequence_length == 1:
            self.starts = list(range(len(self.frames)))
        else:
            self.starts = [
                start
                for start in range(len(self.frames) - sequence_length + 1)
                if all(
                    self.frames[start + offset] == self.frames[start] + offset
                    for offset in range(sequence_length)
                )
            ]

    def __len__(self) -> int:
        return len(self.starts)

    @staticmethod
    def _labels(phases: list[int]) -> tuple[torch.Tensor, ...]:
        # Preserve the canonical CholecT50 tuple order: triplet, tool, verb,
        # target, phase. HeiChole currently supervises only the phase head.
        labels = [
            torch.zeros((len(phases), 100), dtype=torch.float32),
            torch.zeros((len(phases), 6), dtype=torch.float32),
            torch.zeros((len(phases), 10), dtype=torch.float32),
            torch.zeros((len(phases), 15), dtype=torch.float32),
            torch.zeros((len(phases), 7), dtype=torch.float32),
        ]
        labels[4][torch.arange(len(phases)), torch.tensor(phases)] = 1.0
        if len(phases) == 1:
            return tuple(label[0] for label in labels)
        return tuple(labels)

    def __getitem__(self, index: int):
        start = self.starts[index]
        frame_ids = self.frames[start : start + self.sequence_length]
        images = [
            self.transform(_load_video_frame(self.video_path, frame_id))
            for frame_id in frame_ids
        ]
        phases = [self.annotations[frame_id] for frame_id in frame_ids]
        labels = self._labels(phases)
        if self.sequence_length == 1:
            return images[0], labels
        return torch.stack(images), labels


class HeiChole:
    """Build train/validation/per-video-test datasets from a HeiChole root."""

    def __init__(
        self,
        dataset_dir: str | Path,
        test_fold: int = 1,
        augmentation_list: list[str] | None = None,
        normalize: bool = True,
        sequence_length: int = 1,
    ) -> None:
        if not 1 <= test_fold <= 5:
            raise ValueError("test_fold must be in [1, 5]")
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.dataset_dir = Path(dataset_dir)
        self.sequence_length = sequence_length
        train_transform, test_transform = self._transforms(
            augmentation_list or ["original"], normalize
        )

        videos = self._find_videos()
        labels = self._find_labels()
        cases = [(video, labels.get(video.stem.casefold())) for video in videos]
        missing_labels = [video.name for video, label in cases if label is None]
        if missing_labels:
            raise FileNotFoundError(
                "Missing matching HeiChole phase CSV for video(s): "
                + ", ".join(missing_labels)
            )
        cases = [(video, label) for video, label in cases if label is not None]
        if len(cases) < 5:
            raise ValueError(
                f"HeiChole five-fold splitting requires at least 5 labeled videos; "
                f"found {len(cases)}"
            )

        random.Random(42).shuffle(cases)
        folds = [cases[index::5] for index in range(5)]
        test_index = test_fold - 1
        val_index = test_index % 5
        val_index = (val_index + 1) % 5
        self.test_records = [video.stem for video, _ in folds[test_index]]
        self.val_records = [video.stem for video, _ in folds[val_index]]
        self.train_records = [
            video.stem
            for fold_index, fold in enumerate(folds)
            if fold_index not in {test_index, val_index}
            for video, _ in fold
        ]

        case_datasets: dict[str, HeiCholeVideo] = {}
        for video, label_path in cases:
            annotations = _read_phase_csv(label_path)
            case_datasets[video.stem] = HeiCholeVideo(
                video_path=video,
                annotations=annotations,
                transform=train_transform,
                sequence_length=sequence_length,
            )

        self.train_dataset = self._combine(
            [case_datasets[name] for name in self.train_records], "train"
        )
        self.val_dataset = self._combine(
            [case_datasets[name] for name in self.val_records], "validation"
        )
        self.test_dataset = [
            self._with_transform(case_datasets[name], test_transform)
            for name in self.test_records
        ]

    def _find_videos(self) -> list[Path]:
        videos_dir = self.dataset_dir / "videos"
        if not videos_dir.is_dir():
            raise FileNotFoundError(f"Missing HeiChole videos directory: {videos_dir}")
        return sorted(
            (
                path
                for path in videos_dir.iterdir()
                if path.is_file() and path.suffix.casefold() in _VIDEO_EXTENSIONS
            ),
            key=lambda path: path.stem.casefold(),
        )

    def _find_labels(self) -> dict[str, Path]:
        labels_dir = self.dataset_dir / "labels"
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing HeiChole labels directory: {labels_dir}")
        labels: dict[str, Path] = {}
        for path in sorted(labels_dir.glob("*.csv")):
            case_id = _normalized_stem(path)
            if case_id in labels:
                raise ValueError(f"Multiple HeiChole phase CSVs found for {case_id}")
            labels[case_id] = path
        return labels

    @staticmethod
    def _transforms(augmentation_list: list[str], normalize: bool):
        augmentations = {
            "original": lambda: transforms.Lambda(lambda image: image),
            "vflip": lambda: transforms.RandomVerticalFlip(0.4),
            "hflip": lambda: transforms.RandomHorizontalFlip(0.4),
            "contrast": lambda: transforms.ColorJitter(
                brightness=0.1, contrast=0.2, saturation=0, hue=0
            ),
            "rot90": lambda: transforms.RandomRotation(90, expand=True),
            "brightness": lambda: transforms.RandomAdjustSharpness(1.6, p=0.5),
            "random_contrast": lambda: transforms.RandomAutocontrast(p=0.5),
        }
        unknown = set(augmentation_list) - augmentations.keys()
        if unknown:
            raise ValueError(f"Unknown HeiChole augmentation(s): {sorted(unknown)}")
        train_ops = [transforms.Resize((256, 448))]
        train_ops.extend(augmentations[name]() for name in augmentation_list)
        train_ops.extend([transforms.Resize((256, 448)), transforms.ToTensor()])
        test_ops = [transforms.Resize((256, 448)), transforms.ToTensor()]
        if normalize:
            normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            train_ops.append(normalize_transform)
            test_ops.append(normalize_transform)
        return transforms.Compose(train_ops), transforms.Compose(test_ops)

    def _combine(self, datasets: list[Dataset], split_name: str) -> Dataset:
        nonempty = [dataset for dataset in datasets if len(dataset) > 0]
        if not nonempty:
            raise ValueError(
                f"HeiChole {split_name} split is empty "
                f"(check annotations and sequence_length)."
            )
        return ConcatDataset(nonempty)

    @staticmethod
    def _with_transform(dataset: HeiCholeVideo, transform) -> HeiCholeVideo:
        dataset.transform = transform
        return dataset

    def build(self):
        """Return ``(train_dataset, val_dataset, per_video_test_datasets)``."""
        return self.train_dataset, self.val_dataset, self.test_dataset
