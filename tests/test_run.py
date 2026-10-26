"""Tests for run.py pure helpers and orchestration pieces."""

from __future__ import annotations

import re
from types import SimpleNamespace

import torch
from conftest import StubClassifier
from torch.utils.data import DataLoader, TensorDataset

from phase_classification.config import ExperimentConfig
from phase_classification.data.dataloaders import DataLoaders, build_test_loaders
from phase_classification.metrics import LossAccumulator
from phase_classification.run import (
    _count_parameters,
    _higher_is_better,
    _is_better,
    _loss_key,
    _losses,
    _train_epochs,
)
from phase_classification.tracking import _git_commit_hash


class TestIsBetter:
    def test_higher(self):
        assert _is_better(0.6, 0.5, higher_better=True)
        assert not _is_better(0.5, 0.5, higher_better=True)
        assert _is_better(0.4, 0.5, higher_better=False)


class TestHigherIsBetter:
    def test_all_higher(self):
        assert _higher_is_better(["jaccard", "f1_macro"])

    def test_empty(self):
        assert not _higher_is_better([])

    def test_unknown_metric(self):
        assert not _higher_is_better(["jaccard", "mystery"])


def test_git_commit_hash_shape():
    commit = _git_commit_hash()
    assert commit is None or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_build_test_loaders_wraps_each_video_dataset():
    ds1, ds2 = TensorDataset(torch.zeros(2)), TensorDataset(torch.zeros(3))
    dl = DataLoaders(
        train=DataLoader(ds1, batch_size=2),
        val=DataLoader(ds1, batch_size=2),
        test=[ds1, ds2],
    )
    cfg = ExperimentConfig.model_validate(
        {
            "dataset": {"root_dir": "/tmp"},
            "dataloader": {"batch_size": 2, "num_workers": 0},
        }
    )
    loaders = build_test_loaders(dl, cfg)
    assert len(loaders) == 2
    assert all(loader.batch_size == 2 for loader in loaders)


def test_loss_key_mapping():
    assert _loss_key("total") == "loss"
    assert _loss_key("phase") == "phase_loss"


def test_losses_prefixing():
    acc = LossAccumulator(["phase"])
    acc.add({"phase": torch.tensor(2.0), "total": torch.tensor(3.0)})
    assert _losses(acc, "train") == {"train_loss": 3.0, "train_phase_loss": 2.0}
    assert _losses(acc, "") == {"loss": 3.0, "phase_loss": 2.0}


def test_train_epochs_stops_after_validation_patience(tmp_path, monkeypatch):
    from phase_classification import run as run_module

    cfg = ExperimentConfig.model_validate(
        {
            "dataset": {"root_dir": str(tmp_path)},
            "training": {
                "heads": ["phase"],
                "epochs": 10,
                "early_stopping_patience": 2,
            },
        }
    )
    values = iter([0.5, 0.49, 0.48, 0.7])
    monkeypatch.setattr(
        run_module,
        "_epoch_metrics",
        lambda *args, **kwargs: {"val_phase_f1_macro": next(values)},
    )
    logged = []
    monkeypatch.setattr(
        run_module.mlflow,
        "log_metrics",
        lambda metrics, step=None: logged.append((metrics, step)),
    )

    class LossAccumulatorStub:
        def compute(self):
            return {"total": 1.0}

    class TrainerStub:
        epochs = 0

        def new_accumulator(self):
            return object()

        def new_loss_accumulator(self):
            return LossAccumulatorStub()

        def train_epoch(self, *args, **kwargs):
            self.epochs += 1
            return 1.0, {}

        def step_scheduler(self, metric=None):
            pass

    trainer = TrainerStub()
    model = torch.nn.Linear(1, 1)
    best_state = _train_epochs(
        trainer,
        model,
        SimpleNamespace(train=object(), val=object()),
        cfg,
        higher_better=True,
        metric_names=["f1_macro"],
    )

    assert trainer.epochs == 3
    assert best_state is not None
    assert any(
        metrics == {"best_val_phase_f1_macro": 0.5, "best_epoch": 1.0} and step == 1
        for metrics, step in logged
    )


def test_count_parameters_counts_all_params():
    model = StubClassifier(["phase"], {"phase": 7})
    counts = _count_parameters(model)
    assert counts["total"] == sum(p.numel() for p in model.parameters())
    assert counts["trainable"] == counts["total"]
    assert counts["total"] > 0


def test_count_parameters_tracks_frozen_backbone():
    model = StubClassifier(["phase"], {"phase": 7})
    for param in model.proj.parameters():
        param.requires_grad_(False)
    counts = _count_parameters(model)
    assert counts["total"] > counts["trainable"]
    assert counts["trainable"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
