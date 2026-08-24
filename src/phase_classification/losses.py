"""Loss construction from config.

CholecT50 labels are binary (multi-hot) vectors per head. For single-label
heads (e.g. ``phase``, ``verb``) we take the argmax as the class index and use
``CrossEntropyLoss``; for multi-label heads (e.g. ``tool``, ``triplet``) we use
``BCEWithLogitsLoss`` on the binary vectors. A ``MultiHeadLoss`` sums the losses
over the heads selected in the training config.
"""

import torch
from torch import nn

HEAD_TUPLE_INDEX = {
    "triplet": 0,
    "tool": 1,
    "verb": 2,
    "target": 3,
    "phase": 4,
}

# Heads where a frame normally has exactly one active class.
SINGLE_LABEL_HEADS = ("phase", "verb")


def head_index(name: str) -> int:
    """Return the position of ``name`` in the loader's label tuple."""
    return HEAD_TUPLE_INDEX[name]


def compute_class_weights(
    loader,
    heads: list[str],
    num_classes: dict[str, int],
    max_count: float = 1e6,
) -> dict[str, torch.Tensor]:
    """Compute inverse-frequency class weights from a dataloader's labels.

    Iterates once over ``loader`` (no model forward), accumulating per-head
    class histograms, and returns ``head -> weight tensor of shape (K,)``
    normalised to mean 1. Classes absent from the split get the maximum weight.
    """
    counts = {h: torch.zeros(num_classes[h], dtype=torch.float64) for h in heads}
    for _, labels in loader:
        for h in heads:
            t = labels[HEAD_TUPLE_INDEX[h]]
            if t.ndim == 3:  # (B, L, K) sequence -> flatten frames
                t = t.reshape(-1, t.shape[-1])
            idx = t.argmax(dim=1)
            counts[h] += torch.bincount(idx, minlength=num_classes[h]).double()
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
        w = class_weights or {}
        self._ce = nn.CrossEntropyLoss()
        self._bce = nn.BCEWithLogitsLoss()
        self._weighted_ce: dict[str, nn.CrossEntropyLoss] = {
            h: nn.CrossEntropyLoss(weight=w[h]) for h in self.heads if h in w
        }
        self._weighted_bce: dict[str, nn.BCEWithLogitsLoss] = {
            h: nn.BCEWithLogitsLoss(pos_weight=w[h]) for h in self.heads if h in w
        }
        if loss not in ("cross_entropy", "focal"):
            raise ValueError(
                f"Unsupported loss '{loss}'. Choose from 'cross_entropy' or 'focal'."
            )

    def _single(self, name: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss == "focal":
            return self._focal(logits, targets)
        ce = self._weighted_ce.get(name, self._ce)
        return ce(logits, targets.argmax(dim=1))

    def _multi(self, name: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        w = class_weights or {}
        self._ce = nn.CrossEntropyLoss()
        self._bce = nn.BCEWithLogitsLoss()
        self._weighted_ce: dict[str, nn.CrossEntropyLoss] = {
            h: nn.CrossEntropyLoss(weight=w[h]) for h in self.heads if h in w
        }
        self._weighted_bce: dict[str, nn.BCEWithLogitsLoss] = {
            h: nn.BCEWithLogitsLoss(pos_weight=w[h]) for h in self.heads if h in w
        }

    def _sequence_target(self, label: torch.Tensor, device: torch.device) -> torch.Tensor:
        return label.to(device).float()

    def _smoothing(self, logits: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)  # (B, K, L)
        if prob.shape[-1] <= 1:
            return torch.zeros((), device=prob.device)
        diff = (prob[:, :, 1:] - prob[:, :, :-1]) ** 2
        return torch.clamp(diff, max=self.tau).mean()

    def forward(
        self, logits: dict[str, torch.Tensor], labels: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        device = next(iter(logits.values())).device
        per_head: dict[str, torch.Tensor] = {}
        for name in self.heads:
            stages = self.model.stage_logits.get(name)
            if not stages:
                continue
            target = self._sequence_target(labels[HEAD_TUPLE_INDEX[name]], device)
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
                    head_loss = head_loss + bce(stage_out, target)
                head_loss = head_loss + self._smoothing(stage_out)
            per_head[name] = head_loss
        total = sum(per_head.values()) if per_head else torch.zeros((), device=device)
        per_head["total"] = total
        return per_head
