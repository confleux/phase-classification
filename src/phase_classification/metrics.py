"""Metric computation for (optionally multi-head) CholecT50 classification.

Predictions and targets are collected per head as integer class indices over a
dataset and reduced at the end. Metric names follow ``ValidationConfig.metrics``:

- ``jaccard`` -> per-head mean per-class IoU (``<head>_jaccard``)
- ``f1_macro`` -> per-head macro F1 (``<head>_f1_macro``)
- ``f1_per_class`` -> per-head per-class F1 (``<head>_f1_class_<i>``)
- ``mean_accuracy`` -> per-head mean per-class accuracy (``<head>_mean_accuracy``)
- ``accuracy_per_class`` -> per-head per-class accuracy (``<head>_accuracy_class_<i>``)
- ``edit_score`` -> per-head sequence-level Levenshtein edit score (``<head>_edit_score``)
"""

from collections import defaultdict

import numpy as np


class MetricAccumulator:
    """Collect per-head prediction/target pairs and reduce to scalar metrics."""

    def __init__(self, heads: list[str], metric_names: list[str], num_classes: dict[str, int]):
        self.heads = list(heads)
        self.metric_names = list(metric_names)
        self.num_classes = num_classes
        self._pred: dict[str, list[np.ndarray]] = defaultdict(list)
        self._target: dict[str, list[np.ndarray]] = defaultdict(list)

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

    def compute(self) -> dict[str, float]:
        results: dict[str, float] = {}
        for name in self.heads:
            if name not in self._pred:
                continue
            pred = np.concatenate(self._pred[name]) if self._pred[name] else np.array([])
            target = (
                np.concatenate(self._target[name]) if self._target[name] else np.array([])
            )
            if pred.size == 0:
                continue
            n = int(self.num_classes.get(name, int(pred.max()) + 1))
            cm = _confusion_matrix(target, pred, n)
            for metric in self.metric_names:
                values = self._metric_values(metric, cm, pred, target, n)
                for key, value in values.items():
                    results[f"{name}_{key}"] = float(value)
        return results

    def _metric_values(
        self, metric: str, cm: np.ndarray, pred, target, n: int
    ) -> dict[str, float]:
        if metric == "edit_score":
            return {"edit_score": _edit_score(pred, target)}
        if metric == "jaccard":
            return {"jaccard": _jaccard_mean(cm)}
        if metric == "f1_macro":
            return {"f1_macro": _f1_macro(cm)}
        if metric == "mean_accuracy":
            return {"mean_accuracy": _mean_accuracy(cm)}
        if metric == "f1_per_class":
            f1s = _per_class_f1(cm)
            return {f"f1_class_{i}": float(v) for i, v in enumerate(f1s)}
        if metric == "accuracy_per_class":
            accs = _per_class_accuracy(cm)
            return {f"accuracy_class_{i}": float(v) for i, v in enumerate(accs)}
        raise ValueError(f"Unknown metric '{metric}'.")


def _confusion_matrix(target, pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    return cm


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


def _edit_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Sequence-level Edit Score = 1 - normalised Levenshtein distance.

    The predicted and target class sequences are compared with the Levenshtein
    distance (insertions + deletions + substitutions). The score lies in
    [0, 1] and is higher when the predicted sequence matches the ground-truth
    sequence more closely.
    """
    denom = max(len(pred), len(target))
    if denom == 0:
        return 0.0
    lev = _levenshtein(np.asarray(pred, dtype=np.int64), np.asarray(target, dtype=np.int64))
    return float(1.0 - lev / denom)


def _levenshtein(a: np.ndarray, b: np.ndarray) -> int:
    """Wagner-Fischer Levenshtein distance between two integer sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        ai = a[i - 1]
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]