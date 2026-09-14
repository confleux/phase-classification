"""Tests for the training engine (device, optimizers, schedulers, Trainer)."""

from __future__ import annotations

import pytest
import torch
from conftest import StubClassifier, TinyPhaseDataset, make_labels_batch
from torch.utils.data import DataLoader

from phase_classification.config.training import TrainingConfig
from phase_classification.engine import (
    Trainer,
    build_optimizer,
    build_scheduler,
    resolve_device,
)


class TestResolveDevice:
    def test_cpu(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown device"):
            resolve_device("tpu")


class TestOptimizer:
    def _model(self):
        return StubClassifier(["phase"], {"phase": 7})

    def test_sgd(self):
        cfg = TrainingConfig(optimizer="sgd", learning_rate=0.1, momentum=0.9)
        assert isinstance(build_optimizer(self._model(), cfg), torch.optim.SGD)

    def test_adam(self):
        cfg = TrainingConfig(optimizer="adam")
        assert isinstance(build_optimizer(self._model(), cfg), torch.optim.Adam)

    def test_adamw(self):
        cfg = TrainingConfig(optimizer="adamw")
        assert isinstance(build_optimizer(self._model(), cfg), torch.optim.AdamW)

    def test_unknown_raises(self):
        cfg = TrainingConfig.model_construct(optimizer="bogus")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(self._model(), cfg)


class TestScheduler:
    def _opt(self):
        return build_optimizer(StubClassifier(["phase"], {"phase": 7}), TrainingConfig())

    def test_none(self):
        cfg = TrainingConfig(scheduler="none")
        assert build_scheduler(self._opt(), cfg) is None

    def test_step(self):
        cfg = TrainingConfig(scheduler="step")
        s = build_scheduler(self._opt(), cfg)
        assert isinstance(s, torch.optim.lr_scheduler.StepLR)

    def test_cosine(self):
        cfg = TrainingConfig(scheduler="cosine", epochs=3)
        s = build_scheduler(self._opt(), cfg)
        assert isinstance(s, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert s.T_max == 3

    def test_plateau(self):
        cfg = TrainingConfig(scheduler="plateau")
        s = build_scheduler(self._opt(), cfg, higher_is_better=True)
        assert isinstance(s, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert s.mode == "max"


class TestTrainer:
    def _trainer(self, **overrides) -> Trainer:
        cfg = TrainingConfig(
            heads=["phase"], optimizer="sgd", learning_rate=0.05, **overrides
        )
        model = StubClassifier(["phase"], {"phase": 7})
        return Trainer(
            model,
            cfg,
            torch.device("cpu"),
            metric_names=["jaccard", "f1_macro"],
            num_classes={"phase": 7},
        )

    def test_training_step_updates_params(self):
        trainer = self._trainer()
        images = torch.randn(4, 3, 8, 8)
        labels = make_labels_batch(["phase"], batch_size=4)
        before = trainer.model.proj.weight.detach().clone()
        loss = trainer.training_step(images, labels)
        assert loss > 0
        assert not torch.equal(before, trainer.model.proj.weight)

    def test_train_epoch_returns_loss_and_metrics(self):
        trainer = self._trainer()
        loader = DataLoader(TinyPhaseDataset(), batch_size=2)
        acc = trainer.new_accumulator()
        loss, metrics = trainer.train_epoch(loader, accumulator=acc)
        assert loss > 0
        assert "phase_jaccard" in metrics

    def test_evaluate(self):
        trainer = self._trainer()
        metrics = trainer.evaluate(DataLoader(TinyPhaseDataset(), batch_size=2))
        assert 0 <= metrics["phase_jaccard"] <= 1
        assert 0 <= metrics["phase_f1_macro"] <= 1

    def test_scheduler_step(self):
        trainer = self._trainer(scheduler="step", step_size=1)
        lr0 = trainer.optimizer.param_groups[0]["lr"]
        trainer.step_scheduler()
        assert trainer.optimizer.param_groups[0]["lr"] != lr0

    def test_grad_clip(self):
        trainer = self._trainer(grad_clip_norm=1.0)
        loss = trainer.training_step(
            torch.randn(4, 3, 8, 8), make_labels_batch(["phase"], batch_size=4)
        )
        assert loss > 0
