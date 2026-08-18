"""End-to-end run orchestration: train, validate, test, and log to MLflow.

A single :func:`run_training` call performs the complete lifecycle for one run:

1. build dataloaders, model, and trainer,
2. train for ``training.epochs`` with a validation pass per epoch,
3. evaluate once on the held-out test split,
4. log config params + metrics to MLflow and save the model artifact.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.data.dataloaders import (
    DatasetUnavailableError,
    build_dataloaders,
)
from phase_classification.engine import Trainer, resolve_device
from phase_classification.models import build_model


class RunError(Exception):
    """Raised when a training run cannot start or completes abnormally."""


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


def _start_mlflow(cfg: ExperimentConfig) -> None:
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.start_run(run_name=cfg.mlflow.run_name, nested=False)
    for key, value in cfg.mlflow.tags.items():
        mlflow.set_tag(key, value)
    commit_hash = _git_commit_hash()
    if commit_hash is not None:
        mlflow.set_tag("gitcommithash", commit_hash)
    mlflow.log_params(cfg.to_mlflow_params())


def _test_loaders(dl, dataloader_cfg) -> list[DataLoader]:
    """Per-video dataloaders for the CholecT50 test split (a list of datasets)."""
    datasets = dl.test.dataset
    loaders = []
    for ds in datasets:
        loaders.append(
            DataLoader(
                ds,
                batch_size=dataloader_cfg.batch_size,
                shuffle=False,
                num_workers=dataloader_cfg.num_workers,
                pin_memory=dataloader_cfg.pin_memory,
            )
        )
    return loaders


def _epoch_metrics(trainer: Trainer, loader, prefix: str) -> dict[str, float]:
    metrics = trainer.evaluate(loader)
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def _end_mlflow_run(status: str) -> None:
    """End the active MLflow run with the given status (if any run is active)."""
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except mlflow.exceptions.MlflowException as exc:
        print(f"warning: failed to finalize MLflow run ({exc})", file=sys.stderr)


def run_training(cfg: ExperimentConfig) -> dict[str, float]:
    """Execute the full training/validation/test lifecycle, logging to MLflow."""
    if cfg.validation.batch_size:
        raise RunError("validation.batch_size is not wired to the training loop yet.")
    loader_cfg = cfg.dataloader
    _start_mlflow(cfg)
    try:
        return _run_lifecycle(cfg, loader_cfg)
    except Exception:
        _end_mlflow_run("FAILED")
        raise


def _run_lifecycle(cfg: ExperimentConfig, loader_cfg) -> dict[str, float]:
    try:
        dl = build_dataloaders(cfg)
    except DatasetUnavailableError as exc:
        raise RunError(f"Cannot build dataloaders: {exc}")

    device = resolve_device(cfg.runtime.device, cfg.runtime.cuda_device)
    model = build_model(cfg.model, heads=cfg.training.heads)
    metric_names = list(cfg.validation.metrics)
    higher_better = _higher_is_better(metric_names)
    trainer = Trainer(
        model,
        cfg.training,
        device,
        metric_names=metric_names,
        num_classes=cfg.model.num_classes,
        mixed_precision=cfg.runtime.mixed_precision,
        higher_is_better=higher_better,
    )

    best_metric = float("-inf") if higher_better else float("inf")
    best_state = None
    primary_metric = f"val_{cfg.training.heads[0]}_{metric_names[0]}" if metric_names else None

    try:
        for epoch in range(1, cfg.training.epochs + 1):
            train_accumulator = trainer._accumulator()
            train_loss, train_metrics = trainer.train_epoch(
                dl.train, accumulator=train_accumulator
            )
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            val_metrics = _epoch_metrics(trainer, dl.val, prefix="val")
            mlflow.log_metrics(
                {"train_loss": train_loss, **train_metrics, **val_metrics},
                step=epoch,
            )
            if primary_metric is not None and val_metrics.get(primary_metric) is not None:
                m = val_metrics[primary_metric]
                if _is_better(m, best_metric, higher_better):
                    best_metric = m
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
            metric = None
            if cfg.training.scheduler == "plateau" and primary_metric is not None:
                metric = val_metrics.get(primary_metric)
            trainer.step_scheduler(metric)
            print(
                f"epoch {epoch}/{cfg.training.epochs} "
                f"train_loss={train_loss:.4f} "
                + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )
    except KeyboardInterrupt:
        print("Training interrupted; finishing with the best checkpoint.")

    test_metrics: dict[str, float] = {}
    accumulator = trainer._accumulator()
    for test_loader in _test_loaders(dl, loader_cfg):
        trainer.evaluate_accumulate(test_loader, accumulator)
    for key, value in accumulator.compute().items():
        test_metrics[f"test_{key}"] = value
    mlflow.log_metrics(test_metrics)

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
    if cfg.project.save_checkpoints:
        out_dir = Path(cfg.project.output_dir) / "checkpoints"
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, out_dir / "best.pt")
        print(f"Saved best checkpoint to {out_dir / 'best.pt'}")

    _save_model_to_mlflow(cfg, model, best_state)

    _end_mlflow_run("FINISHED")
    return test_metrics


def _save_model_to_mlflow(cfg: ExperimentConfig, model, state_dict) -> None:
    model.load_state_dict(state_dict)
    model.eval()
    if cfg.project.save_checkpoints:
        mlflow.pytorch.log_model(
            model,
            name="model",
            serialization_format="pickle",
        )
    if cfg.mlflow.log_artifacts_to is not None:
        artifact_dir = Path(cfg.mlflow.log_artifacts_to)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / "model.pt"
        torch.save(state_dict, artifact_path)
        mlflow.log_artifact(str(artifact_path))


def _is_better(value: float, best: float, higher_better: bool) -> bool:
    return value > best + 1e-9 if higher_better else value < best - 1e-9


_HIGHER_BETTER = {
    "jaccard",
    "f1_macro",
    "f1_per_class",
    "mean_accuracy",
    "accuracy_per_class",
    "edit_score",
}


def _higher_is_better(metric_names: list[str]) -> bool:
    return bool(metric_names) and all(m in _HIGHER_BETTER for m in metric_names)
