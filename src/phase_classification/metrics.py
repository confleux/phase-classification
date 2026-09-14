"""Metric computation for (optionally multi-head) CholecT50 classification.

Predictions and targets are collected per head as integer class indices over a
dataset and reduced at the end. Metric names follow ``ValidationConfig.metrics``:

- ``jaccard`` -> per-head mean per-class IoU (``<head>_jaccard``)
- ``f1_macro`` -> per-head macro F1 (``<head>_f1_macro``)
- ``f1_per_class`` -> per-head per-class F1 (``<head>_f1_class_<i>``)
- ``mean_accuracy`` -> per-head mean per-class accuracy (``<head>_mean_accuracy``)
- ``accuracy_per_class`` -> per-head per-class accuracy (``<head>_accuracy_class_<i>``)
- ``edit_score`` -> per-head Levenshtein edit score (``<head>_edit_score``)

One accumulator instance represents one sequence (typically one video). Use
:meth:`MetricAccumulator.merged` to combine per-video accumulators: the
confusion-matrix metrics are pooled over all frames, while ``edit_score``
becomes the mean of the per-video scores (the standard Edit Score semantics).
"""

from collections import defaultdict
from collections.abc import Callable

import numpy as np
from rapidfuzz.distance import Levenshtein


class MetricAccumulator:
    """Collect per-head prediction/target pairs and reduce to scalar metrics."""

    def __init__(self, heads: list[str], metric_names: list[str], num_classes: dict[str, int]):
        self.heads = list(heads)
        self.metric_names = list(metric_names)
        self.num_classes = num_classes
        self._pred: dict[str, list[np.ndarray]] = defaultdict(list)
        self._target: dict[str, list[np.ndarray]] = defaultdict(list)
        # Only set by :meth:`merged` for the averaged per-sequence edit score.
        self._sequence_edit_scores: dict[str, list[float]] | None = None

    def _append(self, name: str, pred: np.ndarray, target: np.ndarray) -> None:
        self._pred[name].append(np.asarray(pred).reshape(-1))
        self._target[name].append(np.asarray(target).reshape(-1))

    def add_batch(
        self, logits: dict[str, np.ndarray], targets: dict[str, np.ndarray]
    ) -> None:
        for name in self.heads:
            if name not in logits or name not in targets:
                continue
            lo = np.asarray(logits[name])
            tg = np.asarray(targets[name])
            if lo.ndim == 3:  # (B, L, K) per-frame sequence -> (B*L, K)
                lo = lo.reshape(-1, lo.shape[-1])
                tg = tg.reshape(-1, tg.shape[-1])
            pred = np.argmax(lo, axis=1)
            tgt = np.argmax(tg, axis=1)
            self._append(name, pred, tgt)

    def _concat(self, store: dict[str, list[np.ndarray]], name: str) -> np.ndarray:
        chunks = store.get(name) or []
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.int64)

    def _sequence_edit_score(self, name: str) -> float | None:
        """Edit score over this accumulator's own (single-sequence) data."""
        if not self._pred.get(name):
            return None
        return _edit_score(self._concat(self._pred, name), self._concat(self._target, name))

    def _edit_score_values(self, name: str) -> dict[str, float]:
        if self._sequence_edit_scores is not None:  # merged: average per sequence
            scores = self._sequence_edit_scores.get(name, [])
        else:  # single sequence
            score = self._sequence_edit_score(name)
            scores = [score] if score is not None else []
        return {"edit_score": float(np.mean(scores))} if scores else {}

    @classmethod
    def merged(cls, accumulators: list[MetricAccumulator]) -> MetricAccumulator:
        """Combine per-sequence accumulators (e.g. one per test video).

        Confusion-matrix metrics are pooled over all frames; ``edit_score``
        becomes the mean of the per-sequence scores (Edit Score semantics).
        """
        first = accumulators[0]
        out = cls(first.heads, first.metric_names, first.num_classes)
        for name in first.heads:
            out._pred[name] = [
                chunk for acc in accumulators for chunk in acc._pred.get(name, [])
            ]
            out._target[name] = [
                chunk for acc in accumulators for chunk in acc._target.get(name, [])
            ]
        out._sequence_edit_scores = {}
        for name in first.heads:
            scores = [
                s
                for acc in accumulators
                if (s := acc._sequence_edit_score(name)) is not None
            ]
            if scores:
                out._sequence_edit_scores[name] = scores
        return out

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for name in self.heads:
            if not self._pred.get(name):
                continue
            pred = self._concat(self._pred, name)
            target = self._concat(self._target, name)
            if pred.size == 0:
                continue
            n = int(self.num_classes.get(name, int(pred.max()) + 1))
            cm = _confusion_matrix(target, pred, n)
            for metric in self.metric_names:
                if metric == "edit_score":
                    values = self._edit_score_values(name)
                else:
                    values = _cm_metric_values(metric, cm)
                for key, value in values.items():
                    results[f"{name}_{key}"] = float(value)
        return results


def _confusion_matrix(target, pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    return cm


# Confusion-matrix metrics, keyed by their ValidationConfig.metrics name.
_CM_METRICS: dict[str, Callable[[np.ndarray], dict[str, float]]] = {
    "jaccard": lambda cm: {"jaccard": _jaccard_mean(cm)},
    "f1_macro": lambda cm: {"f1_macro": _f1_macro(cm)},
    "mean_accuracy": lambda cm: {"mean_accuracy": _mean_accuracy(cm)},
    "f1_per_class": lambda cm: {
        f"f1_class_{i}": float(v) for i, v in enumerate(_per_class_f1(cm))
    },
    "accuracy_per_class": lambda cm: {
        f"accuracy_class_{i}": float(v) for i, v in enumerate(_per_class_accuracy(cm))
    },
}


def _cm_metric_values(metric: str, cm: np.ndarray) -> dict[str, float]:
    fn = _CM_METRICS.get(metric)
    if fn is None:
        raise ValueError(f"Unknown metric '{metric}'.")
    return fn(cm)


def _per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    return np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
    )


def _mean_accuracy(cm: np.ndarray) -> float:
    return float(_per_class_accuracy(cm).mean())


def _per_class_jaccard(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    union = tp + fp + fn
    return np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)


def _jaccard_mean(cm: np.ndarray) -> float:
    return float(_per_class_jaccard(cm).mean())


def _per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )


def _f1_macro(cm: np.ndarray) -> float:
    return float(_per_class_f1(cm).mean())


def _edit_score(pred, target) -> float:
    """Edit Score = 1 - normalised Levenshtein distance over one sequence.

    The predicted and target class sequences are compared with the Levenshtein
    distance (insertions + deletions + substitutions). The score lies in [0, 1]
    and is higher when the predicted sequence matches the ground-truth sequence
    more closely.
    """
    denom = max(len(pred), len(target))
    if denom == 0:
        return 0.0
    lev = Levenshtein.distance(
        np.asarray(pred, dtype=np.int64).tolist(),
        np.asarray(target, dtype=np.int64).tolist(),
    )
    return float(1.0 - lev / denom)