"""Schema tests for the pydantic config models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phase_classification.config import ExperimentConfig
from phase_classification.config.dataset import DatasetConfig
from phase_classification.config.model import ModelConfig
from phase_classification.config.runtime import RuntimeConfig


def make_cfg(data: dict | None = None) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {"dataset": {"root_dir": "/tmp/data"}, **(data or {})}
    )


class TestDefaults:
    def test_minimal_config_fills_defaults(self):
        cfg = make_cfg()
        assert cfg.project.name == "phase-classification"
        assert cfg.dataset.dataset_variant == "cholect50-crossval"
        assert cfg.dataloader.batch_size == 32
        assert cfg.model.name == "resnet18"
        assert cfg.model.num_classes["phase"] == 7
        assert cfg.training.epochs == 50
        assert cfg.training.heads == ["phase"]
        assert cfg.validation.metrics == [
            "jaccard",
            "f1_macro",
            "f1_per_class",
            "mean_accuracy",
            "accuracy_per_class",
            "edit_score",
        ]
        assert cfg.runtime.device == "auto"
        assert cfg.runtime.seed == 42

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            make_cfg({"dataset": {"root_dir": "/tmp", "bogus": 1}})


class TestValidators:
    def test_cuda_device_requires_cuda_setting(self):
        with pytest.raises(ValidationError):
            RuntimeConfig(device="cpu", cuda_device=0)

    def test_num_classes_rejects_unknown_head(self):
        with pytest.raises(ValidationError):
            ModelConfig(num_classes={"bogus": 5})

    def test_num_classes_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            ModelConfig(num_classes={"phase": 0})

    def test_image_size_field_removed(self):
        with pytest.raises(ValidationError):
            DatasetConfig(root_dir="/tmp", image_size=[256, 448])

    def test_test_fold_bounds(self):
        with pytest.raises(ValidationError):
            DatasetConfig(root_dir="/tmp", test_fold=6)


class TestMlflowParams:
    def test_params_flattened(self):
        cfg = make_cfg({"mlflow": {"experiment_name": "exp"}})
        params = cfg.to_mlflow_params()
        assert params["mlflow.experiment_name"] == "exp"
        assert params["training.epochs"] == "50"

    def test_secrets_excluded(self):
        cfg = make_cfg(
            {
                "mlflow": {
                    "aws_access_key_id": "minio",
                    "aws_secret_access_key": "miniopass",
                }
            }
        )
        params = cfg.to_mlflow_params()
        assert "mlflow.aws_access_key_id" not in params
        assert "mlflow.aws_secret_access_key" not in params

    def test_public_config_redacts_secrets(self):
        cfg = make_cfg(
            {
                "mlflow": {
                    "aws_access_key_id": "local-user",
                    "aws_secret_access_key": "do-not-display",
                }
            }
        )
        public_config = cfg.to_public_dict()
        assert public_config["mlflow"]["aws_access_key_id"] == "[REDACTED]"
        assert public_config["mlflow"]["aws_secret_access_key"] == "[REDACTED]"

    def test_lists_comma_joined(self):
        cfg = make_cfg({"training": {"heads": ["phase", "tool"]}})
        assert cfg.to_mlflow_params()["training.heads"] == "phase,tool"
