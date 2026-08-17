"""Metric computation for (optionally multi-head) CholecT50 classification.

Predictions and targets are collected per head as integer class indices over a
dataset and reduced at the end. Metric names follow ``ValidationConfig.metrics``:
``accuracy``, ``accuracy_per_phase`` (per-class balanced accuracy), ``f1_macro``,
``f1_micro`` and ``jaccard`` (per-class intersection-over-union).
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
            pred = np.argmax(logits[name], axis=1)
            tgt = np.argmax(targets[name], axis=1)
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
                value = _reduce_metric(metric, cm)
                results[f"{name}_{metric}"] = float(value)
        return results


def _confusion_matrix(target, pred, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (target, pred), 1)
    return cm


def _reduce_metric(metric: str, cm: np.ndarray) -> float:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = cm.sum()
    if metric == "accuracy":
        return float(tp.sum() / denom) if denom else 0.0
    if metric == "accuracy_per_phase":
        per_class = np.divide(
            tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
        )
        return float(per_class.mean())
    if metric == "f1_macro":
        return _f1(tp, fp, fn, average="macro")
    if metric == "f1_micro":
        return _f1(tp, fp, fn, average="micro")
    if metric == "jaccard":
        union = tp + fn + fp
        iou = np.divide(tp, union, out=np.zeros_like(tp, dtype=float), where=union > 0)
        return float(iou.mean())
    raise ValueError(f"Unknown metric '{metric}'.")


def _f1(tp, fp, fn, average: str) -> float:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=float),
        where=(precision + recall) > 0,
    )
    if average == "macro":
        return float(f1.mean())
    tp_sum = tp.sum()
    denom = tp_sum + 0.5 * (fp.sum() + fn.sum())
    return float(tp_sum / denom) if denom else 0.0
