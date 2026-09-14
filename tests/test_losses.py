"""Tests for MultiHeadLoss, MSTCNLoss and compute_class_weights."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from conftest import make_label_row, make_labels_batch

from phase_classification.heads import head_index
from phase_classification.losses import (
    MSTCNLoss,
    MultiHeadLoss,
    compute_class_weights,
)


def test_head_index():
    assert head_index("triplet") == 0
    assert head_index("tool") == 1
    assert head_index("verb") == 2
    assert head_index("target") == 3
    assert head_index("phase") == 4


class TestMultiHeadLoss:
    def _loss(self, heads, loss="cross_entropy", class_weights=None):
        return MultiHeadLoss(heads, loss, class_weights=class_weights)

    def test_single_label_ce(self):
        loss_fn = self._loss(["phase"])
        logits = {"phase": torch.randn(3, 7)}
        labels = make_labels_batch(["phase"], batch_size=3)
        out = loss_fn(logits, labels)
        expected = F.cross_entropy(logits["phase"], labels[4].argmax(dim=1))
        assert torch.allclose(out["phase"], expected)
        assert out["total"] == out["phase"]

    def test_multi_label_bce(self):
        loss_fn = self._loss(["tool"])
        logits = {"tool": torch.randn(3, 6)}
        labels = make_labels_batch(["tool"], batch_size=3)
        out = loss_fn(logits, labels)
        expected = F.binary_cross_entropy_with_logits(logits["tool"], labels[1])
        assert torch.allclose(out["tool"], expected)

    def test_focal_differs_from_ce(self):
        loss_fn = self._loss(["phase"], loss="focal")
        logits = torch.randn(4, 7)
        labels = make_labels_batch(["phase"], batch_size=4)
        out = loss_fn({"phase": logits}, labels)
        ce = F.cross_entropy(logits, labels[4].argmax(dim=1))
        assert out["phase"] >= 0
        assert abs(out["phase"].item() - ce.item()) > 1e-3

    def test_weighted_ce(self):
        w = torch.tensor([0.1] * 6 + [2.0])
        loss_fn = self._loss(["phase"], class_weights={"phase": w})
        logits = torch.randn(4, 7)
        labels = make_labels_batch(["phase"], batch_size=4)
        out = loss_fn({"phase": logits}, labels)
        expected = F.cross_entropy(logits, labels[4].argmax(dim=1), weight=w)
        assert torch.allclose(out["phase"], expected)

    def test_sequence_label_takes_last_frame(self):
        loss_fn = self._loss(["phase"])
        logits = {"phase": torch.randn(2, 7)}
        labels = list(make_labels_batch(["phase"], batch_size=2))
        seq = torch.zeros(2, 5, 7)
        seq[:, -1] = labels[4]
        labels[4] = seq
        out = loss_fn(logits, tuple(labels))
        expected = F.cross_entropy(logits["phase"], seq[:, -1].argmax(dim=1))
        assert torch.allclose(out["phase"], expected)

    def test_skips_heads_without_logits(self):
        loss_fn = self._loss(["phase", "tool"])
        logits = {"phase": torch.randn(2, 7)}
        labels = make_labels_batch(["phase"], batch_size=2)
        out = loss_fn(logits, labels)
        assert set(out) == {"phase", "total"}

    def test_empty_heads_total_zero(self):
        loss_fn = self._loss(["phase"])
        logits = {"tool": torch.randn(2, 6)}
        labels = make_labels_batch(["tool"], batch_size=2)
        out = loss_fn(logits, labels)
        assert out["total"].item() == 0.0

    def test_invalid_loss_raises(self):
        with pytest.raises(ValueError, match="Unsupported loss"):
            MultiHeadLoss(["phase"], "bogus")


class StubMSTCN:
    is_mstcn = True

    def __init__(self, stage_logits):
        self.stage_logits = stage_logits


class TestMSTCNLoss:
    def _sequence_labels(self, b=2, length=6, k=7):
        target = torch.zeros(b, length, k)
        idx = torch.randint(0, k, (b, length))
        target[
            torch.arange(b).unsqueeze(1), torch.arange(length).unsqueeze(0), idx
        ] = 1.0
        labels = [torch.zeros(b, 1) for _ in range(5)]
        labels[4] = target
        return tuple(labels)

    def test_per_stage_supervised(self):
        stages = [torch.randn(2, 6, 7) for _ in range(3)]
        loss_fn = MSTCNLoss(StubMSTCN({"phase": stages}), ["phase"])
        out = loss_fn({"phase": torch.randn(2, 6, 7)}, self._sequence_labels())
        assert out["total"].item() > 0
        assert torch.isfinite(out["total"])

    def test_single_frame_batch(self):
        stages = [torch.randn(2, 1, 7)]
        loss_fn = MSTCNLoss(StubMSTCN({"phase": stages}), ["phase"])
        labels = [torch.zeros(2, 1) for _ in range(5)]
        labels[4] = torch.zeros(2, 7)
        labels[4][0, 3] = 1.0
        out = loss_fn({"phase": torch.randn(2, 7)}, tuple(labels))
        assert torch.isfinite(out["total"])

    def test_smoothing_zero_for_length_one(self):
        stages = [torch.randn(2, 1, 7)]
        loss_fn = MSTCNLoss(StubMSTCN({"phase": stages}), ["phase"])
        target = torch.zeros(2, 1, 7)
        target[:, 0, 2] = 1.0
        labels = [torch.zeros(2, 1) for _ in range(5)]
        labels[4] = target
        out = loss_fn({"phase": torch.randn(2, 7)}, tuple(labels))
        expected = F.cross_entropy(
            stages[0].reshape(-1, 7), torch.full((2,), 2, dtype=torch.long)
        )
        assert torch.allclose(out["total"], expected)

    def test_missing_head_zero(self):
        loss_fn = MSTCNLoss(StubMSTCN({}), ["phase"])
        out = loss_fn({"phase": torch.randn(2, 6, 7)}, self._sequence_labels())
        assert out["total"].item() == 0.0

    def test_multi_label_head_bce_shape(self):
        # (B, L=4, K=6) tool stages; BCE must be computed on (B, L, K), and the
        # transposed stage must not be silently broadcast against the target.
        stages = [torch.randn(2, 4, 6)]
        loss_fn = MSTCNLoss(StubMSTCN({"tool": stages}), ["tool"])
        target = torch.zeros(2, 4, 6)
        target[:, :, 1] = 1.0
        labels = [torch.zeros(2, 1) for _ in range(5)]
        labels[1] = target
        out = loss_fn({"tool": torch.randn(2, 4, 6)}, tuple(labels))
        expected_bce = F.binary_cross_entropy_with_logits(stages[0], target)
        assert torch.isfinite(out["total"])
        # total = BCE + smoothing (smoothing >= 0)
        assert out["total"].item() >= expected_bce.item() - 1e-6


class TestComputeClassWeights:
    def test_fast_path_from_label_jsons(self, tmp_path):
        from phase_classification.data.cholec_t50 import T50

        label_file = tmp_path / "VID01.json"
        label_file.write_text(
            json.dumps(
                {
                    "annotations": {
                        "0": [make_label_row(phase=0, tool=1)],
                        "1": [make_label_row(phase=1, tool=1)],
                        "2": [make_label_row(phase=1, tool=2)],
                    }
                }
            )
        )
        ds = T50(img_dir=str(tmp_path), label_file=str(label_file))
        weights = compute_class_weights(
            SimpleNamespace(datasets=[ds]), ["phase", "tool"], {"phase": 7, "tool": 6}
        )
        counts_phase = torch.tensor([1.0, 2.0, 0, 0, 0, 0, 0])
        counts_phase[counts_phase == 0] = 2.0
        assert torch.allclose(
            weights["phase"], (counts_phase.sum() / (7 * counts_phase)).float()
        )
        counts_tool = torch.tensor([0.0, 2.0, 1.0, 0, 0, 0])
        counts_tool[counts_tool == 0] = 2.0
        assert torch.allclose(
            weights["tool"], (counts_tool.sum() / (6 * counts_tool)).float()
        )
