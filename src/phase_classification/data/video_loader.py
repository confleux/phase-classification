"""Per-video dataset variant of the vendored CholecT50 loader.

Mirrors :mod:`phase_classification.data.cholec_t50` (same splits, transforms
and label encoding) but each ``__getitem__`` returns *all* frames and labels
of a single video instead of a single frame/window. Intended to be wrapped in
a ``DataLoader`` with ``batch_size=1`` so that one iteration yields one video:

    images:  FloatTensor of shape [num_frames, C, H, W]
    labels:  tuple of IntTensors, each of shape [num_frames, num_classes]
             in the canonical order
             (triplet, tool, verb, target, phase)

This file is separate from the vendored CAMMA reference loader on purpose:
the reference must stay unmodified.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from phase_classification.data.cholec_t50 import CholecT50


class T50Video(Dataset):
    """A single CholecT50 video; ``__getitem__`` returns the whole video."""

    def __init__(self, img_dir, label_file, transform=None, target_transform=None):
        with open(label_file, "rb") as f:
            label_data = json.load(f)
        self.label_data = label_data["annotations"]
        self.frames = sorted(self.label_data.keys(), key=lambda k: int(k))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        # One item == one whole video. Returning the frame count here would
        # make a DataLoader iterate the same video once per frame.
        return 1

    def get_binary_labels(self, labels):
        # Same encoding as the reference loader.
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])
        phase_label = np.zeros([7])
        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0:
                triplet_label[triplet[0]] += 1
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] += 1
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] += 1
            target = label[8:14]
            if target[0] != -1.0:
                target_label[target[0]] += 1
            phase = label[14:15]
            if phase[0] != -1.0:
                phase_label[phase[0]] += 1
        return (triplet_label, tool_label, verb_label, target_label, phase_label)

    def __getitem__(self, index):
        """Return ``(images, labels)`` for the whole video.

        Only ``index=0`` is meaningful (one item == one video); any index is
        accepted so it works transparently inside a ``DataLoader``.
        """
        frame_tensors: list[torch.Tensor] = []
        stacked_components: list[list[torch.Tensor]] = [[] for _ in range(5)]
        for frame_key in self.frames:
            basename = f"{str(frame_key).zfill(6)}.png"
            img_path = os.path.join(self.img_dir, basename)
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            frame_tensors.append(image)
            label_tuple = self.get_binary_labels(self.label_data[frame_key])
            if self.target_transform:
                label_tuple = self.target_transform(label_tuple)
            for k, tensor in enumerate(label_tuple):
                stacked_components[k].append(tensor)
        images = torch.stack(frame_tensors, dim=0)
        labels = tuple(torch.stack(parts, dim=0) for parts in stacked_components)
        return images, labels


class CholecT50Video(CholecT50):
    """CholecT50 with per-video datasets instead of per-frame ones.

    Inherits all split/augmentation logic from the reference loader and only
    overrides how the underlying ``Dataset`` objects are built. Each element
    of ``train_dataset`` / ``val_dataset`` / ``test_dataset`` is a
    :class:`T50Video` covering exactly one video.
    """

    def _make_video_dataset(self, video, transform):
        img_dir = os.path.join(self.dataset_dir, "videos", video)
        label_file = os.path.join(self.dataset_dir, "labels", f"{video}.json")
        return T50Video(
            img_dir=img_dir,
            label_file=label_file,
            transform=transform,
            target_transform=self.target_transform,
        )

    def build_train_dataset(self, transform):
        iterable_dataset = []
        for video in self.train_records:
            dataset = self._make_video_dataset(video, transform)
            if len(dataset.frames) > 0:
                iterable_dataset.append(dataset)
        if len(iterable_dataset) == 0:
            raise ValueError("Train dataset is empty (check videos directory).")
        self.train_dataset = iterable_dataset

    def build_val_dataset(self, transform):
        iterable_dataset = []
        for video in self.val_records:
            dataset = self._make_video_dataset(video, transform)
            if len(dataset.frames) > 0:
                iterable_dataset.append(dataset)
        if len(iterable_dataset) == 0:
            raise ValueError("Val dataset is empty (check videos directory).")
        self.val_dataset = iterable_dataset

    def build_test_dataset(self, transform):
        iterable_dataset = []
        for video in self.test_records:
            dataset = self._make_video_dataset(video, transform)
            if len(dataset.frames) > 0:
                iterable_dataset.append(dataset)
        self.test_dataset = iterable_dataset


def build_video_dataloaders(
    cholec: CholecT50Video,
    batch_size: int = 1,
    num_workers: int = 0,
):
    """Wrap each per-video split in a ``DataLoader`` yielding one video per step.

    Requires ``batch_size == 1`` since videos have different lengths and
    cannot be collated together.
    """
    from torch.utils.data import DataLoader

    if batch_size != 1:
        raise ValueError("Per-video loading requires batch_size=1.")
    common = {"batch_size": 1, "num_workers": num_workers}
    train_loader = DataLoader(cholec.train_dataset, shuffle=True, **common)
    val_loader = DataLoader(cholec.val_dataset, shuffle=False, **common)
    test_loaders = [
        DataLoader(videos, shuffle=False, **common) for videos in cholec.test_dataset
    ]
    return train_loader, val_loader, test_loaders
