"""Tests for MetricAccumulator and the metric functions."""

from __future__ import annotations

import numpy as np
import pytest

from phase_classification.metrics import (
    MetricAccumulator,
    _confusion_matrix,
    _edit_score,
    _f1_macro,
    _jaccard_mean,
    _mean_accuracy,
    _per_class_accuracy,
)

TARGET = np.array([0, 0, 1, 1])
PRED = np.array([0, 1, 1, 1])
CM = np.array([[1, 1], [0, 2]])


class TestMetricFunctions:
    def test_confusion_matrix(self):
        assert np.array_equal(_confusion_matrix(TARGET, PRED, 2), CM)

    def test_per_class_accuracy(self):
        assert np.allclose(_per_class_accuracy(CM), [0.5, 1.0])

    def test_mean_accuracy(self):
        assert _mean_accuracy(CM) == pytest.approx(0.75)

    def test_jaccard_mean(self):
        assert _jaccard_mean(CM) == pytest.approx((0.5 + 2 / 3) / 2)

    def test_f1_macro(self):
        assert _f1_macro(CM) == pytest.approx((2 / 3 + 0.8) / 2)

    def test_edit_score_identical(self):
        assert _edit_score([1, 2, 3], [1, 2, 3]) == 1.0

    def test_edit_score_one_substitution(self):
        assert _edit_score([1, 2], [1, 3]) == pytest.approx(0.5)

    def test_edit_score_empty(self):
        assert _edit_score([], []) == 0.0

    def test_edit_score_insertion(self):
        assert _edit_score([1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.75)

    @pytest.mark.parametrize("seed", range(5))
    def test_edit_score_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        a = rng.integers(0, 4, size=int(rng.integers(5, 40)))
        b = rng.integers(0, 4, size=int(rng.integers(5, 40)))
        denom = max(len(a), len(b))
        expected = 1 - _reference_levenshtein(a, b) / denom
        assert _edit_score(a, b) == pytest.approx(expected)


def _reference_levenshtein(a, b) -> int:
    """Slow Wagner-Fischer reference implementation for cross-checking."""
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int64)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[n, m])


def one_hot(values: list[int], k: int) -> np.ndarray:
    out = np.zeros((len(values), k))
    out[np.arange(len(values)), values] = 1.0
    return out


class TestMetricAccumulator:
    def _accumulator(self, metric_names: list[str]) -> MetricAccumulator:
        return MetricAccumulator(["phase"], metric_names, {"phase": 2})

    def test_scalar_metrics(self):
        acc = self._accumulator(
            ["jaccard", "f1_macro", "mean_accuracy", "f1_per_class", "accuracy_per_class"]
        )
        acc.add_batch(
            {"phase": np.array([[5.0, 0.0], [0.0, 5.0], [0.0, 5.0], [0.0, 5.0]])},
            {"phase": one_hot([0, 0, 1, 1], 2)},
        )
        results = acc.compute()
        assert results["phase_jaccard"] == pytest.approx((0.5 + 2 / 3) / 2)
        assert results["phase_f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2)
        assert results["phase_mean_accuracy"] == pytest.approx(0.75)
        assert results["phase_f1_class_0"] == pytest.approx(2 / 3)
        assert results["phase_accuracy_class_1"] == 1.0

    def test_sequence_logits_reshaped(self):
        acc = self._accumulator(["jaccard"])
        logits = np.array([[[5.0, 0.0], [0.0, 5.0]]])
        targets = one_hot([0, 1], 2).reshape(1, 2, 2)
        acc.add_batch({"phase": logits}, {"phase": targets})
        assert acc.compute()["phase_jaccard"] == 1.0

    def test_missing_head_skipped(self):
        acc = self._accumulator(["jaccard"])
        acc.add_batch({}, {})
        assert acc.compute() == {}

    def test_edit_score_accumulated(self):
        acc = self._accumulator(["edit_score"])
        logits = np.array([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
        acc.add_batch({"phase": logits}, {"phase": one_hot([0, 1, 0], 2)})
        assert acc.compute()["phase_edit_score"] == 1.0

    def test_unknown_metric_raises(self):
        acc = self._accumulator(["mystery"])
        acc.add_batch(
            {"phase": np.array([[5.0, 0.0]])}, {"phase": one_hot([0], 2)}
        )
        with pytest.raises(ValueError, match="Unknown metric"):
            acc.compute()


class TestMergedAccumulator:
    def test_merge_pools_confusion_metrics(self):
        def make(pred_batch, target_batch):
            acc = MetricAccumulator(["phase"], ["jaccard"], {"phase": 2})
            acc.add_batch({"phase": pred_batch}, {"phase": target_batch})
            return acc

        perfect = make(
            np.array([[5.0, 0.0], [0.0, 5.0]]), one_hot([0, 1], 2)
        )
        all_wrong = make(np.array([[5.0, 0.0], [5.0, 0.0]]), one_hot([1, 1], 2))
        merged = MetricAccumulator.merged([perfect, all_wrong])
        # pooled cm = [[1, 0], [2, 1]] -> jaccard (1/3 + 1/3) / 2
        assert merged.compute()["phase_jaccard"] == pytest.approx(1 / 3)

    def test_merge_averages_edit_scores(self):
        def make(logits, targets):
            acc = MetricAccumulator(["phase"], ["edit_score"], {"phase": 2})
            acc.add_batch({"phase": logits}, {"phase": targets})
            return acc

        perfect = make(
            np.array([[5.0, 0.0], [0.0, 5.0]]), one_hot([0, 1], 2)
        )
        all_wrong = make(np.array([[5.0, 0.0], [5.0, 0.0]]), one_hot([1, 1], 2))
        merged = MetricAccumulator.merged([perfect, all_wrong])
        assert merged.compute()["phase_edit_score"] == pytest.approx(0.5)

    def test_merge_skips_empty_sequences(self):
        filled = MetricAccumulator(["phase"], ["edit_score"], {"phase": 2})
        filled.add_batch({"phase": np.array([[5.0, 0.0]])}, {"phase": one_hot([0], 2)})
        empty = MetricAccumulator(["phase"], ["edit_score"], {"phase": 2})
        merged = MetricAccumulator.merged([filled, empty])
        assert merged.compute()["phase_edit_score"] == 1.0
