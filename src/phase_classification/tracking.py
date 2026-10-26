"""MLflow run setup and client-side artifact-store configuration."""

from __future__ import annotations

import os
import subprocess
import sys

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.config.mlflow import MLflowConfig


def _git_commit_hash() -> str | None:
    """Return the current git HEAD commit hash, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def configure_s3_client(mlflow_cfg: MLflowConfig) -> None:
    """Expose configured S3/MinIO values to boto3 without overriding the env."""
    for env_name, value in (
        ("MLFLOW_S3_ENDPOINT_URL", mlflow_cfg.s3_endpoint_url),
        ("AWS_ACCESS_KEY_ID", mlflow_cfg.aws_access_key_id),
        ("AWS_SECRET_ACCESS_KEY", mlflow_cfg.aws_secret_access_key),
        ("AWS_DEFAULT_REGION", mlflow_cfg.aws_region),
    ):
        if value:
            os.environ.setdefault(env_name, value)


def start_run(cfg: ExperimentConfig) -> None:
    """Configure tracking, start the run, and record its static metadata."""
    configure_s3_client(cfg.mlflow)
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.start_run(run_name=cfg.mlflow.run_name, nested=False)
    try:
        for key, value in cfg.mlflow.tags.items():
            mlflow.set_tag(key, value)
        commit_hash = _git_commit_hash()
        if commit_hash is not None:
            mlflow.set_tag("gitcommithash", commit_hash)
        mlflow.log_params(cfg.to_mlflow_params())
    except Exception:
        end_run("FAILED")
        raise


def end_run(status: str) -> None:
    """End the active MLflow run with the requested status, if one exists."""
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except mlflow.exceptions.MlflowException as exc:
        print(f"warning: failed to finalize MLflow run ({exc})", file=sys.stderr)
