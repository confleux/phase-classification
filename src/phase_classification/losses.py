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
    """Aggregate per-head classification losses for the configured heads."""

    def __init__(self, heads: list[str], loss: str = "cross_entropy"):
        super().__init__()
        self.heads = list(heads)
        self.loss = loss
        self._ce = nn.CrossEntropyLoss()
        self._bce = nn.BCEWithLogitsLoss()
        if loss not in ("cross_entropy", "focal"):
            raise ValueError(
                f"Unsupported loss '{loss}'. Choose from 'cross_entropy' or 'focal'."
            )

    def _single(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss == "focal":
            return self._focal(logits, targets)
        return self._ce(logits, targets.argmax(dim=1))

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
                per_head[name] = self._single(lo, target)
            else:
                per_head[name] = self._bce(lo, target)

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

    def __init__(self, model, heads: list[str], tau: float = 4.0):
        super().__init__()
        self.model = model
        self.heads = list(heads)
        self.tau = tau
        self._ce = nn.CrossEntropyLoss()
        self._bce = nn.BCEWithLogitsLoss()

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
            head_loss = torch.zeros((), device=device)
            for stage_out in stages:  # (B, L, K)
                stage_out = stage_out.transpose(1, 2)  # (B, K, L)
                if name in SINGLE_LABEL_HEADS:
                    head_loss = head_loss + self._ce(stage_out, target.argmax(dim=2))
                else:
                    head_loss = head_loss + self._bce(stage_out, target)
                head_loss = head_loss + self._smoothing(stage_out)
            per_head[name] = head_loss
        total = sum(per_head.values()) if per_head else torch.zeros((), device=device)
        per_head["total"] = total
        return per_head
