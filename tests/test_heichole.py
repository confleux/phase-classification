from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image
from torchvision import transforms

from phase_classification.config.dataset import DatasetConfig
from phase_classification.data import heichole
from phase_classification.data.builders import build_dataset
from phase_classification.data.heichole import (
    HeiChole,
    HeiCholeVideo,
    _read_phase_csv,
)


def _write_annotations(path: Path, rows: list[tuple[int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Phase"])
        writer.writerows(rows)


def test_read_phase_csv_parses_official_frame_phase_columns(tmp_path):
    csv_path = tmp_path / "case01_Phase.csv"
    _write_annotations(csv_path, [(0, 0), (4, 6)])

    assert _read_phase_csv(csv_path) == {0: 0, 4: 6}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([(0, 7)], r"phase IDs must be in \[0, 6\]"),
        ([(0, 1), (0, 2)], "Duplicate frame 0"),
    ],
)
def test_read_phase_csv_rejects_invalid_labels(tmp_path, rows, message):
    csv_path = tmp_path / "case01.csv"
    _write_annotations(csv_path, rows)

    with pytest.raises(ValueError, match=message):
        _read_phase_csv(csv_path)


def test_video_dataset_emits_phase_head_and_contiguous_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(
        heichole,
        "_load_video_frame",
        lambda _path, frame: Image.new("RGB", (4, 4), color=(frame, 0, 0)),
    )
    dataset = HeiCholeVideo(
        video_path=tmp_path / "case.mp4",
        annotations={0: 0, 1: 2, 3: 6},
        transform=transforms.ToTensor(),
        sequence_length=2,
    )

    images, labels = dataset[0]
    assert len(dataset) == 1  # The missing frame 2 prevents a second window.
    assert images.shape == (2, 3, 4, 4)
    assert [label.shape for label in labels] == [
        (2, 100),
        (2, 6),
        (2, 10),
        (2, 15),
        (2, 7),
    ]
    assert labels[4].argmax(dim=1).tolist() == [0, 2]
    assert labels[0].count_nonzero() == 0


def test_heichole_five_fold_builds_per_video_test_datasets(monkeypatch, tmp_path):
    videos_dir = tmp_path / "videos"
    labels_dir = tmp_path / "labels"
    videos_dir.mkdir()
    labels_dir.mkdir()
    for case_index in range(5):
        (videos_dir / f"case{case_index}.mp4").touch()
        _write_annotations(
            labels_dir / f"case{case_index}_Phase.csv", [(0, case_index), (1, case_index)]
        )

    monkeypatch.setattr(
        heichole,
        "_load_video_frame",
        lambda _path, _frame: Image.new("RGB", (4, 4), color="black"),
    )
    dataset = build_dataset(
        DatasetConfig(
            dataset_variant="heichole",
            root_dir=tmp_path,
            test_fold=1,
            augmentations=["original"],
            normalize=False,
        )
    )

    assert isinstance(dataset, HeiChole)
    assert len(dataset.train_records) == 3
    assert len(dataset.val_records) == 1
    assert len(dataset.test_records) == 1
    assert len(dataset.test_dataset) == 1
    assert len(dataset.train_dataset) == 6
    image, labels = dataset.test_dataset[0][0]
    assert image.shape == (3, 256, 448)
    assert labels[4].shape == (7,)
    assert labels[4].sum().item() == 1
    assert dataset.build() == (
        dataset.train_dataset,
        dataset.val_dataset,
        dataset.test_dataset,
    )


def test_heichole_requires_matching_phase_csv(tmp_path):
    (tmp_path / "videos").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "videos" / "case.mp4").touch()

    with pytest.raises(FileNotFoundError, match="matching HeiChole phase CSV"):
        HeiChole(tmp_path, augmentation_list=["original"])


def test_heichole_uses_five_fold_index():
    with pytest.raises(ValueError, match="test_fold must be in \\[1, 5\\]"):
        HeiChole("/tmp/not-used", test_fold=6)
