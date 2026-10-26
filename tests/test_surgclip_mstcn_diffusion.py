"""End-to-end smoke test for cached MS-TCN plus residual diffusion."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import ConcatDataset, DataLoader

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.data.surgclip_features import FeatureVideoDataset
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX
from phase_classification.losses import MSTCNLoss
from phase_classification.models.ms_tcn import CachedFeatureMSTCNPlusPlus
from phase_classification.surgclip_mstcn_diffusion import run_surgclip_mstcn_diffusion


def _video() -> FeatureVideoDataset:
    phase_ids = [0, 1, 2, 3, 4, 5, 6, 0]
    labels = [
        torch.zeros(len(phase_ids), classes)
        for classes in DEFAULT_NUM_CLASSES.values()
    ]
    labels[HEAD_TUPLE_INDEX["phase"]] = torch.nn.functional.one_hot(
        torch.tensor(phase_ids), DEFAULT_NUM_CLASSES["phase"]
    ).float()
    return FeatureVideoDataset(torch.randn(len(phase_ids), 4), tuple(labels))


def test_cached_mstcnpp_prediction_and_refinement_stages():
    model = CachedFeatureMSTCNPlusPlus(
        feature_dim=8,
        num_classes={"phase": 7},
        heads=["phase"],
        num_refinement_stages=2,
        generation_layers=3,
        refinement_layers=2,
        num_f_maps=8,
        dropout=0.0,
    )
    features = torch.randn(1, 12, 8)
    targets = torch.nn.functional.one_hot(torch.randint(0, 7, (1, 12)), 7).float()
    labels = (
        torch.zeros(1, 12, 100),
        torch.zeros(1, 12, 6),
        torch.zeros(1, 12, 10),
        torch.zeros(1, 12, 15),
        targets,
    )

    logits = model(features)
    loss = MSTCNLoss(model, ["phase"])(logits, labels)["total"]
    loss.backward()
    assert logits["phase"].shape == (1, 12, 7)
    assert len(model.stage_logits["phase"]) == 3
    assert model.smoothing_on_log_probs
    assert model.smoothing_weight == 0.15
    assert torch.isfinite(loss)


@pytest.mark.parametrize(
    ("model_name", "architecture_params"),
    [
        ("surgclip-ms-tcn-diffusion", {"num_stages": 2, "num_layers": 1}),
        (
            "surgclip-ms-tcnpp-diffusion",
            {
                "num_refinement_stages": 1,
                "generation_layers": 2,
                "refinement_layers": 1,
                "dropout": 0.0,
            },
        ),
    ],
)
def test_combined_feature_tcn_diffusion_training_smoke(
    tmp_path, monkeypatch, model_name, architecture_params
):
    train, val, test = _video(), _video(), _video()
    data = SimpleNamespace(
        train=DataLoader(ConcatDataset([train]), batch_size=1),
        val=DataLoader(ConcatDataset([val]), batch_size=1),
        test=[test],
    )
    config = ExperimentConfig.model_validate(
        {
            "project": {
                "output_dir": str(tmp_path / model_name),
                "save_checkpoints": False,
            },
            "dataset": {
                "root_dir": str(tmp_path),
                "feature_cache": str(tmp_path / "features.pt"),
            },
            "dataloader": {"batch_size": 1, "num_workers": 0},
            "model": {
                "name": model_name,
                "num_classes": {"phase": 7},
                "params": {
                    "feature_dim": 4,
                    "num_f_maps": 8,
                    "refiner_epochs": 1,
                    "refiner_steps_per_epoch": 1,
                    "refiner_batch_size": 1,
                    "refiner_crop_length": 8,
                    "refiner_width": 8,
                    "refiner_layers": 1,
                    "diffusion_timesteps": 4,
                    "sampling_steps": 2,
                    **architecture_params,
                },
            },
            "training": {
                "heads": ["phase"],
                "epochs": 1,
                "scheduler": "none",
            },
            "validation": {"metrics": ["f1_macro", "edit_score"]},
            "runtime": {"device": "cpu", "mixed_precision": "fp32", "seed": 42},
        }
    )
    logged = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda values, step=None: logged.append(values))

    result = run_surgclip_mstcn_diffusion(config, data, torch.device("cpu"))

    assert "test_coarse_phase_f1_macro" in result
    assert "test_refined_phase_f1_macro" in result
    assert any("refiner_train_noise_loss" in metrics for metrics in logged)


def test_run_lifecycle_dispatches_surgclip_mstcn_diffusion(tmp_path, monkeypatch):
    from phase_classification import run as run_module
    from phase_classification import surgclip_mstcn_diffusion as training_module

    config = ExperimentConfig.model_validate(
        {
            "dataset": {
                "root_dir": str(tmp_path),
                "feature_cache": str(tmp_path / "features.pt"),
            },
            "model": {
                "name": "surgclip-ms-tcn-diffusion",
                "params": {"feature_dim": 4},
            },
            "training": {"heads": ["phase"]},
        }
    )
    dataloaders = object()
    monkeypatch.setattr(run_module, "build_dataloaders", lambda cfg: dataloaders)
    monkeypatch.setattr(run_module, "resolve_device", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(run_module, "_validate_feature_probe", lambda cfg, dl: None)
    monkeypatch.setattr(
        training_module,
        "run_surgclip_mstcn_diffusion",
        lambda cfg, dl, device: {"test_refined_phase_f1_macro": 0.8},
    )
    ended = []
    monkeypatch.setattr(run_module, "end_run", lambda status: ended.append(status))

    result = run_module._run_lifecycle(config)

    assert result == {"test_refined_phase_f1_macro": 0.8}
    assert ended == ["FINISHED"]
