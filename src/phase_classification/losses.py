"""Loss construction from config.

CholecT50 labels are binary (multi-hot) vectors per head. For single-label
heads (e.g. ``phase``, ``verb``) we take the argmax as the class index and use
``CrossEntropyLoss``; for multi-label heads (e.g. ``tool``, ``triplet``) we use
``BCEWithLogitsLoss`` on the binary vectors. A ``MultiHeadLoss`` sums the losses
over the heads selected in the training config.
"""

import torch
from torch import nn

from phase_classification.heads import HEAD_TUPLE_INDEX, SINGLE_LABEL_HEADS


def compute_class_weights(
    dataset,
    heads: list[str],
    num_classes: dict[str, int],
    max_count: float = 1e6,
) -> dict[str, torch.Tensor]:
    """Compute inverse-frequency class weights from a dataset's label files.

    Reads only the per-video JSON annotations (no image decoding), accumulating
    per-head class histograms, and returns ``head -> weight tensor (K,)``
    normalised to mean 1. Classes absent from the split get the maximum weight.

    ``dataset`` must be a ``ConcatDataset``-like object whose ``.datasets`` are
    per-video CholecT50 datasets exposing ``.label_data`` and
    ``.get_binary_labels`` (i.e. the train split of the vendored loader).
    """
    counts = {h: torch.zeros(num_classes[h], dtype=torch.float64) for h in heads}

    video_datasets = getattr(dataset, "datasets", None)
    if video_datasets is None or not all(
        hasattr(ds, "label_data") for ds in video_datasets
    ):
        raise TypeError(
            "compute_class_weights expects a split of per-video CholecT50 "
            f"datasets with .label_data; got {type(dataset).__name__}."
        )
    for ds in video_datasets:
        for frame_labels in ds.label_data.values():
            encoded = ds.get_binary_labels(frame_labels)
            for h in heads:
                t = encoded[HEAD_TUPLE_INDEX[h]]
                counts[h] += torch.as_tensor(t, dtype=torch.float64)

    weights: dict[str, torch.Tensor] = {}
    for h in heads:
        c = counts[h]
        c[c == 0] = c.max().clamp(min=1)  # unseen classes get the mildest weight
        w = (c.sum() / (len(c) * c)).float()  # inverse frequency, mean-1 normalised
        weights[h] = w.clamp(max=max_count / max(c.max().item(), 1.0))
    return weights


def _head_target(label: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reduce a per-sample label tensor to shape ``(B, K)`` on ``device``.

    Sequence datasets return labels of shape ``(B, L, K)``; we predict the
    current frame, so we take the last frame of the window.
    """
    t = label
    if t.ndim == 3:
        t = t[:, -1, :]
    return t.to(device).float()


def _build_losses(
    heads: list[str], class_weights: dict[str, torch.Tensor] | None
) -> tuple[
    nn.CrossEntropyLoss,
    nn.BCEWithLogitsLoss,
    dict[str, nn.CrossEntropyLoss],
    dict[str, nn.BCEWithLogitsLoss],
]:
    """Build the shared unweighted + per-head weighted CE/BCE losses."""
    w = class_weights or {}
    return (
        nn.CrossEntropyLoss(),
        nn.BCEWithLogitsLoss(),
        {h: nn.CrossEntropyLoss(weight=w[h]) for h in heads if h in w},
        {h: nn.BCEWithLogitsLoss(pos_weight=w[h]) for h in heads if h in w},
    )


class MultiHeadLoss(nn.Module):
    """Aggregate per-head classification losses for the configured heads.

    ``class_weights`` (optional, from :func:`compute_class_weights`) applies
    inverse-frequency weighting: per-class weights in CE, ``pos_weight`` in BCE.
    """

    def __init__(
        self,
        heads: list[str],
        loss: str = "cross_entropy",
        class_weights: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.heads = list(heads)
        self.loss = loss
        self._ce, self._bce, self._weighted_ce, self._weighted_bce = _build_losses(
            self.heads, class_weights
        )
        if loss not in ("cross_entropy", "focal"):
            raise ValueError(
                f"Unsupported loss '{loss}'. Choose from 'cross_entropy' or 'focal'."
            )

    def _single(
        self, name: str, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if self.loss == "focal":
            return self._focal(logits, targets)
        ce = self._weighted_ce.get(name, self._ce)
        return ce(logits, targets.argmax(dim=1))

    def _multi(
        self, name: str, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce = self._weighted_bce.get(name, self._bce)
        return bce(logits, targets)

    def _focal(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_idx = targets.argmax(dim=1)
        pt = probs.gather(1, targets_idx.unsqueeze(1)).squeeze(1)
        return (-((1.0 - pt) ** 2.0) * torch.log(pt + 1e-12)).mean()

    def forward(
        self, logits: dict[str, torch.Tensor], labels: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        device = next(iter(logits.values())).device
        per_head: dict[str, torch.Tensor] = {}

        def empty() -> torch.Tensor:
            return torch.zeros((), device=device)

        for name in self.heads:
            if name not in logits:
                continue
            lo = logits[name]
            idx = HEAD_TUPLE_INDEX[name]
            target = _head_target(labels[idx], lo.device)
            if name in SINGLE_LABEL_HEADS:
                per_head[name] = self._single(name, lo, target)
            else:
                per_head[name] = self._multi(name, lo, target)

        total = sum(per_head.values(), empty()) if per_head else empty()
        per_head["total"] = total
        return per_head


class MSTCNLoss(nn.Module):
    """MS-TCN loss: per-frame CE/BCE summed over every stage + smoothing loss.

    The model passes its per-head, per-stage (final) logits via ``stage_logits``
    (a dict of ``head -> list[Tensor]`` of shape ``(B, L, K)``, set during
    ``forward``). Each stage's predictions are supervised with per-frame CE
    (single-label heads) or BCE (multi-label heads), and a truncated-MSE
    smoothing loss penalises over-segmentation (each stage).
    """

    def __init__(
        self,
        model,
        heads: list[str],
        tau: float = 4.0,
        class_weights: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.model = model
        self.heads = list(heads)
        self.tau = tau
        self.smoothing_on_log_probs = bool(
            getattr(model, "smoothing_on_log_probs", False)
        )
        self.smoothing_weight = float(getattr(model, "smoothing_weight", 1.0))
        self._ce, self._bce, self._weighted_ce, self._weighted_bce = _build_losses(
            self.heads, class_weights
        )

    def _smoothing(self, logits: torch.Tensor) -> torch.Tensor:
        values = (
            torch.log_softmax(logits, dim=1)
            if self.smoothing_on_log_probs
            else torch.softmax(logits, dim=1)
        )
        if values.shape[-1] <= 1:
            return torch.zeros((), device=values.device)
        diff = (values[:, :, 1:] - values[:, :, :-1]).abs()
        if self.smoothing_on_log_probs:
            diff = diff.clamp(max=self.tau).square()
        else:
            diff = diff.square().clamp(max=self.tau)
        return self.smoothing_weight * diff.mean()

    def forward(
        self, logits: dict[str, torch.Tensor], labels: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        device = next(iter(logits.values())).device
        per_head: dict[str, torch.Tensor] = {}
        for name in self.heads:
            stages = self.model.stage_logits.get(name)
            if not stages:
                continue
            target = labels[HEAD_TUPLE_INDEX[name]].to(device).float()
            if target.ndim == 2:  # (B, K) single-frame batch -> (B, 1, K)
                target = target.unsqueeze(1)
            head_loss = torch.zeros((), device=device)
            for stage_out in stages:  # (B, L, K)
                stage_out = stage_out.transpose(1, 2)  # (B, K, L)
                if name in SINGLE_LABEL_HEADS:
                    ce = self._weighted_ce.get(name, self._ce)
                    head_loss = head_loss + ce(
                        stage_out.transpose(1, 2).reshape(-1, stage_out.shape[1]),
                        target.reshape(-1, target.shape[-1]).argmax(dim=1),
                    )
                else:
                    bce = self._weighted_bce.get(name, self._bce)
                    head_loss = head_loss + bce(stage_out.transpose(1, 2), target)
                head_loss = head_loss + self._smoothing(stage_out)
            per_head[name] = head_loss
        total = sum(per_head.values()) if per_head else torch.zeros((), device=device)
        per_head["total"] = total
        return per_head
