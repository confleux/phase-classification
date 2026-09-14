"""Tests for run.py pure helpers and orchestration pieces."""

from __future__ import annotations

import os
import re

import torch
from torch.utils.data import DataLoader, TensorDataset

from phase_classification.config import ExperimentConfig
from phase_classification.config.mlflow import MLflowConfig
from phase_classification.data.dataloaders import DataLoaders, build_test_loaders
from phase_classification.run import (
    _git_commit_hash,
    _higher_is_better,
    _is_better,
    _set_s3_env,
)


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


def test_set_s3_env(monkeypatch):
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
    _set_s3_env(cfg)
    assert os.environ["MLFLOW_S3_ENDPOINT_URL"] == "http://minio:9000"
    assert os.environ["AWS_ACCESS_KEY_ID"] == "minio"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "miniopass"


def test_set_s3_env_env_wins(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "from-env")
    cfg = MLflowConfig(aws_access_key_id="from-config")
    _set_s3_env(cfg)
    assert os.environ["AWS_ACCESS_KEY_ID"] == "from-env"


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
