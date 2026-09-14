"""Shared fixtures and tiny stubs for the test suite."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from phase_classification.heads import HEAD_TUPLE_INDEX

NUM_CLASSES = {
    "phase": 7,
    "tool": 6,
    "verb": 10,
    "target": 15,
    "triplet": 100,
}


class StubClassifier(nn.Module):
    """Mean-pool pixels -> shared projection -> one linear head per label head."""

    def __init__(self, heads: list[str], num_classes: dict[str, int], in_dim: int = 3):
        super().__init__()
        self.heads = list(heads)
        self.proj = nn.Linear(in_dim, 8)
        self.heads_block = nn.ModuleDict(
            {h: nn.Linear(8, num_classes[h]) for h in self.heads}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.proj(x.float().mean(dim=(2, 3)))
        return {h: head(feats) for h, head in self.heads_block.items()}


class TinyPhaseDataset(Dataset):
    """Four 3x8x8 frames with per-head labels in the vendored tuple order."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int):
        labels: list[torch.Tensor] = [torch.zeros(1) for _ in range(5)]
        phase = torch.zeros(NUM_CLASSES["phase"])
        phase[index % NUM_CLASSES["phase"]] = 1.0
        labels[HEAD_TUPLE_INDEX["phase"]] = phase
        return torch.rand(3, 8, 8), tuple(labels)


def make_label_row(
    triplet: float = -1.0,
    tool: float = -1.0,
    verb: float = -1.0,
    target: float = -1.0,
    phase: float = -1.0,
) -> list[float]:
    """A CholecT50 annotation row: [triplet, tool*6, verb, target*6, phase]."""
    return [
        triplet,
        tool, -1.0, -1.0, -1.0, -1.0, -1.0,
        verb,
        target, -1.0, -1.0, -1.0, -1.0, -1.0,
        phase,
    ]


def make_labels_batch(
    heads: list[str], batch_size: int = 2, num_classes: dict[str, int] | None = None
) -> tuple[torch.Tensor, ...]:
    """Random multi-hot label tuple in the vendored loader's head order."""
    nc = num_classes or NUM_CLASSES
    slots: list[torch.Tensor | None] = [None] * 5
    for head in heads:
        k = nc[head]
        t = torch.zeros(batch_size, k)
        t[torch.arange(batch_size), torch.randint(0, k, (batch_size,))] = 1.0
        slots[HEAD_TUPLE_INDEX[head]] = t
    return tuple(s if s is not None else torch.zeros(batch_size, 1) for s in slots)


@pytest.fixture
def num_classes() -> dict[str, int]:
    return dict(NUM_CLASSES)
