"""Tests for MLflow and S3 client setup helpers."""

from __future__ import annotations

import os

import pytest

from phase_classification import tracking as tracking_module
from phase_classification.config import ExperimentConfig
from phase_classification.config.mlflow import MLflowConfig
from phase_classification.tracking import configure_s3_client


def test_configure_s3_client_uses_config_values(monkeypatch):
    for name in (
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = MLflowConfig(
        s3_endpoint_url="http://minio:9000",
        aws_access_key_id="minio",
        aws_secret_access_key="miniopass",
        aws_region="us-east-1",
    )

    configure_s3_client(cfg)

    assert os.environ["MLFLOW_S3_ENDPOINT_URL"] == "http://minio:9000"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "minio"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "miniopass"
    assert os.environ["AWS_DEFAULT_REGION"] == "us-east-1"


def test_configure_s3_client_preserves_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "from-env")
    cfg = MLflowConfig(aws_access_key_id="from-config")

    configure_s3_client(cfg)

    assert os.environ["AWS_ACCESS_KEY_ID"] == "from-env"


def test_start_run_finalizes_when_metadata_logging_fails(monkeypatch, tmp_path):
    cfg = ExperimentConfig.model_validate({"dataset": {"root_dir": str(tmp_path)}})
    ended = []
    monkeypatch.setattr(tracking_module.mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(
        tracking_module.mlflow, "start_run", lambda **kwargs: object()
    )
    monkeypatch.setattr(tracking_module.mlflow, "active_run", lambda: object())
    monkeypatch.setattr(
        tracking_module.mlflow, "end_run", lambda status: ended.append(status)
    )
    monkeypatch.setattr(tracking_module, "_git_commit_hash", lambda: None)
    monkeypatch.setattr(
        tracking_module.mlflow,
        "log_params",
        lambda params: (_ for _ in ()).throw(RuntimeError("tracking unavailable")),
    )

    with pytest.raises(RuntimeError, match="tracking unavailable"):
        tracking_module.start_run(cfg)

    assert ended == ["FAILED"]
