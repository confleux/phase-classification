"""Smoke tests for Meta-SurDiff models, bilevel updates, and run lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset, DataLoader

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.data.surgclip_features import FeatureVideoDataset
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX
from phase_classification.meta_surdiff import _meta_step, run_meta_surdiff
from phase_classification.models.meta_surdiff import (
    ConditionalLabelDiffusion,
    MetaSurDiffSystem,
    MetaWeightNet,
)
from phase_classification.models.ms_tcn import CachedFeatureMSTCN


def _video(features: torch.Tensor, phase_ids: list[int]) -> FeatureVideoDataset:
    length = len(phase_ids)
    labels = [
        torch.zeros(length, classes)
        for classes in DEFAULT_NUM_CLASSES.values()
    ]
    labels[HEAD_TUPLE_INDEX["phase"]] = torch.nn.functional.one_hot(
        torch.tensor(phase_ids), DEFAULT_NUM_CLASSES["phase"]
    ).float()
    return FeatureVideoDataset(features, tuple(labels))


def test_meta_weighted_bilevel_step_updates_both_parameter_sets():
    coarse = CachedFeatureMSTCN(
        4, {"phase": 7}, ["phase"], num_stages=2, num_layers=1, num_f_maps=8
    )
    system = MetaSurDiffSystem(
        coarse, ConditionalLabelDiffusion(7, timesteps=4, hidden_dim=8)
    )
    weight_net = MetaWeightNet(hidden_dim=4)
    optimizer = torch.optim.AdamW(system.parameters(), lr=1e-4)
    weight_optimizer = torch.optim.AdamW(weight_net.parameters(), lr=1e-3)

    train_features = torch.randn(1, 8, 4)
    train_targets = torch.randint(0, 7, (1, 8))
    train_mask = torch.ones(1, 8, dtype=torch.bool)
    meta_features = train_features.expand(7, -1, -1).clone()
    meta_targets = torch.arange(7).view(7, 1).expand(-1, 8)
    meta_mask = torch.zeros_like(meta_targets, dtype=torch.bool)
    meta_mask[:, 4] = True
    before = [parameter.detach().clone() for parameter in weight_net.parameters()]

    loss, weight_std = _meta_step(
        system,
        weight_net,
        optimizer,
        weight_optimizer,
        (train_features, train_targets, train_mask),
        (meta_features, meta_targets, meta_mask),
        inner_learning_rate=1e-4,
    )

    assert loss > 0
    assert weight_std >= 0
    assert any(not torch.equal(old, new) for old, new in zip(before, weight_net.parameters()))


def test_meta_surdiff_two_stage_lifecycle_smoke(tmp_path, monkeypatch):
    phase_ids = list(range(7)) + [0]
    video_train = _video(torch.randn(len(phase_ids), 4), phase_ids)
    video_val = _video(torch.randn(len(phase_ids), 4), phase_ids)
    video_test = _video(torch.randn(len(phase_ids), 4), phase_ids)
    data = SimpleNamespace(
        feature_dim=4,
        train=DataLoader(ConcatDataset([video_train]), batch_size=1, shuffle=True),
        val=DataLoader(ConcatDataset([video_val]), batch_size=1),
        test=[video_test],
    )
    config = ExperimentConfig.model_validate(
        {
            "project": {"output_dir": str(tmp_path), "save_checkpoints": False},
            "dataset": {
                "root_dir": str(tmp_path),
                "feature_cache": str(tmp_path / "features.pt"),
            },
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "model": {
                "name": "surgclip-meta-surdiff",
                "num_classes": {"phase": 7},
                "params": {
                    "feature_dim": 4,
                    "num_stages": 2,
                    "num_layers": 1,
                    "num_f_maps": 8,
                    "diffusion_timesteps": 4,
                    "diffusion_hidden_dim": 8,
                    "meta_hidden_dim": 4,
                    "meta_epochs": 1,
                    "meta_steps_per_epoch": 1,
                    "crop_length": 8,
                    "sampling_steps": 2,
                },
            },
            "training": {
                "heads": ["phase"],
                "epochs": 1,
                "scheduler": "none",
            },
            "validation": {"metrics": ["f1_macro", "edit_score"]},
            "runtime": {"device": "cpu", "mixed_precision": "fp32", "seed": 7},
        }
    )
    logged_metrics = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda values, step=None: logged_metrics.append(values))
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    previous_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")

    try:
        result = run_meta_surdiff(config, data, torch.device("cpu"))
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    assert "test_coarse_phase_f1_macro" in result
    assert "test_refined_phase_f1_macro" in result
    assert logged_metrics
    assert not (tmp_path / "checkpoints" / "best.pt").exists()


def test_run_lifecycle_dispatches_meta_surdiff(tmp_path, monkeypatch):
    from phase_classification import meta_surdiff as training_module
    from phase_classification import run as run_module

    config = ExperimentConfig.model_validate(
        {
            "dataset": {
                "root_dir": str(tmp_path),
                "feature_cache": str(tmp_path / "features.pt"),
            },
            "model": {"name": "surgclip-meta-surdiff", "params": {"feature_dim": 4}},
            "training": {"heads": ["phase"]},
        }
    )
    dataloaders = object()
    monkeypatch.setattr(run_module, "build_dataloaders", lambda cfg: dataloaders)
    monkeypatch.setattr(run_module, "resolve_device", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(run_module, "_validate_feature_probe", lambda cfg, dl: None)
    monkeypatch.setattr(
        training_module,
        "run_meta_surdiff",
        lambda cfg, dl, device: {"test_refined_phase_f1_macro": 0.8},
    )
    ended = []
    monkeypatch.setattr(run_module, "_end_mlflow_run", lambda status: ended.append(status))

    result = run_module._run_lifecycle(config)

    assert result == {"test_refined_phase_f1_macro": 0.8}
    assert ended == ["FINISHED"]
