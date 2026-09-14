"""Tests for the phase-cls CLI (validate-config / inspect)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase_classification.cli import main


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text("dataset:\n  root_dir: /tmp/data\n")
    return path


def test_validate_config_ok(config_file, capsys):
    main(["validate-config", str(config_file)])
    assert "is valid" in capsys.readouterr().out


def test_validate_config_error_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.yaml"
    path.write_text("bogus: true\n")
    with pytest.raises(SystemExit) as exc:
        main(["validate-config", str(path)])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_validate_config_check_paths_missing(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(f"dataset:\n  root_dir: {tmp_path / 'nope'}\n")
    with pytest.raises(SystemExit) as exc:
        main(["validate-config", str(path), "--check-paths"])
    assert exc.value.code == 3


def test_inspect_json(config_file, capsys):
    main(["inspect", str(config_file), "--format", "json"])
    data = json.loads(capsys.readouterr().out)
    assert data["dataset"]["root_dir"] == "/tmp/data"


def test_inspect_mlflow(config_file, capsys):
    main(["inspect", str(config_file), "--format", "mlflow"])
    params = json.loads(capsys.readouterr().out)
    assert "training.epochs" in params
