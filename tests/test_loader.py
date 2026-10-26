"""Tests for YAML config loading and on-disk path checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase_classification.loader import ConfigLoadError, check_paths, load_config


def write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(content)
    return path


class TestLoadConfig:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigLoadError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_invalid_yaml(self, tmp_path):
        path = write_yaml(tmp_path, "dataset: [unclosed")
        with pytest.raises(ConfigLoadError, match="Invalid YAML"):
            load_config(path)

    def test_schema_violation(self, tmp_path):
        path = write_yaml(tmp_path, "dataset:\n  root_dir: /tmp\n  bogus: 1\n")
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_config(path)

    def test_valid_minimal(self, tmp_path):
        path = write_yaml(tmp_path, "dataset:\n  root_dir: /tmp\n")
        cfg = load_config(path)
        assert cfg.dataset.root_dir == Path("/tmp")

    def test_environment_variables_override_yaml_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-user")
        path = write_yaml(
            tmp_path,
            "dataset:\n"
            "  root_dir: /tmp\n"
            "mlflow:\n"
            "  s3_endpoint_url: ${MLFLOW_S3_ENDPOINT_URL:-http://localhost:9000}\n"
            "  aws_access_key_id: ${AWS_ACCESS_KEY_ID:-minio}\n"
            "  aws_secret_access_key: ${AWS_SECRET_ACCESS_KEY:-miniopass}\n",
        )
        cfg = load_config(path)
        assert cfg.mlflow.s3_endpoint_url == "http://localhost:9000"
        assert cfg.mlflow.aws_access_key_id == "test-user"
        assert cfg.mlflow.aws_secret_access_key == "miniopass"

    def test_missing_required_environment_variable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("REQUIRED_CONFIG_VALUE", raising=False)
        path = write_yaml(
            tmp_path,
            "dataset:\n  root_dir: /tmp\n"
            "mlflow:\n  tracking_uri: ${REQUIRED_CONFIG_VALUE}\n",
        )
        with pytest.raises(ConfigLoadError, match="REQUIRED_CONFIG_VALUE"):
            load_config(path)


class TestCheckPaths:
    def test_missing_root(self, tmp_path):
        cfg = load_config(
            write_yaml(tmp_path, "dataset:\n  root_dir: /nonexistent-xyz\n")
        )
        assert check_paths(cfg) == ["dataset root not found: /nonexistent-xyz"]

    def test_missing_videos_labels(self, tmp_path):
        root = tmp_path / "data"
        root.mkdir()
        cfg = load_config(write_yaml(tmp_path, f"dataset:\n  root_dir: {root}\n"))
        missing = check_paths(cfg)
        assert len(missing) == 2

    def test_ok(self, tmp_path):
        root = tmp_path / "data"
        (root / "videos").mkdir(parents=True)
        (root / "labels").mkdir()
        cfg = load_config(write_yaml(tmp_path, f"dataset:\n  root_dir: {root}\n"))
        assert check_paths(cfg) == []
