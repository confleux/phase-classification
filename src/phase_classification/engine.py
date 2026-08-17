"""Training engine: optimizer/scheduler construction and epoch loops.

This module is pure torch (no MLflow). It trains/validates/tests a model over
torch DataLoaders and returns scalar metrics/losses for the caller to log.
The test split of CholecT50 is a list of per-video datasets, so run
:meth:`Trainer.evaluate` once per video loader and merge accumulators.
"""

from __future__ import annotations

import torch
from torch import nn

from phase_classification.config.training import TrainingConfig
from phase_classification.losses import MultiHeadLoss, head_index
from phase_classification.metrics import MetricAccumulator


def resolve_device(name: str, cuda_device: int | None = None) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda is not available.")
        return torch.device(f"cuda:{cuda_device}" if cuda_device is not None else "cuda")
    if name != "auto":
        raise ValueError(f"Unknown device '{name}'.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"Unknown optimizer '{cfg.optimizer}'.")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainingConfig,
    steps_per_epoch: int,
    higher_is_better: bool = False,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.step_size, gamma=cfg.gamma
        )
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs * steps_per_epoch
        )
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if higher_is_better else "min",
            factor=cfg.gamma,
            patience=cfg.step_size,
        )
    raise ValueError(f"Unknown scheduler '{cfg.scheduler}'.")


class Trainer:
    """Owns the model, device, loss, optimizer and scheduler; runs epochs."""

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig,
        device: torch.device,
        metric_names: list[str],
        num_classes: dict[str, int],
        mixed_precision: str = "fp32",
        higher_is_better: bool = False,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.metric_names = metric_names
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        self.loss_fn = MultiHeadLoss(cfg.heads, cfg.loss)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(
            self.optimizer, cfg, steps_per_epoch=1, higher_is_better=higher_is_better
        )
        self.scaler = (
            torch.amp.GradScaler("cuda") if mixed_precision == "fp16" else None
        )

    def _to_device(
        self, images: torch.Tensor, labels: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        images = images.to(self.device, non_blocking=True)
        labels = tuple(t.to(self.device, non_blocking=True) for t in labels)
        return images, labels

    def _forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.mixed_precision == "fp16":
            with torch.amp.autocast("cuda"):
                return self.model(images)
        return self.model(images)

    def training_step(
        self,
        images,
        labels,
        accumulator: MetricAccumulator | None = None,
    ) -> float:
        images, labels = self._to_device(images, labels)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self._forward(images)
        if accumulator is not None:
            logits_np = {
                name: logits[name].detach().cpu().numpy() for name in logits
            }
            targets = {
                name: labels[head_index(name)].cpu().numpy()
                for name in self.cfg.heads
            }
            accumulator.add_batch(logits_np, targets)
        losses = self.loss_fn(logits, labels)
        loss = losses["total"]
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.cfg.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()
        return float(loss.detach().item())

    def train_epoch(
        self,
        loader,
        accumulator: MetricAccumulator | None = None,
    ) -> tuple[float, dict[str, float]]:
        self.model.train()
        total, count = 0.0, 0
        for images, labels in loader:
            total += self.training_step(images, labels, accumulator=accumulator)
            count += 1
        loss = total / max(count, 1)
        metrics = accumulator.compute() if accumulator is not None else {}
        return loss, metrics

    @torch.no_grad()
    def evaluate_accumulate(self, loader, accumulator: MetricAccumulator) -> None:
        self.model.eval()
        for images, labels in loader:
            images, labels = self._to_device(images, labels)
            logits = self.model(images)
            logits_np = {name: logits[name].cpu().numpy() for name in logits}
            targets = {
                name: labels[head_index(name)].cpu().numpy() for name in self.cfg.heads
            }
            accumulator.add_batch(logits_np, targets)

    def _accumulator(self) -> MetricAccumulator:
        return MetricAccumulator(self.cfg.heads, self.metric_names, self.num_classes)

    @torch.no_grad()
    def evaluate(self, loader) -> dict[str, float]:
        acc = self._accumulator()
        self.evaluate_accumulate(loader, acc)
        return acc.compute()

    def step_scheduler(self, metric: float | None = None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
