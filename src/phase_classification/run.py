"""End-to-end run orchestration: train, validate, test, and log to MLflow.

A single :func:`run_training` call performs the complete lifecycle for one run:

1. build dataloaders, model, and trainer,
2. train for ``training.epochs`` with a validation pass per epoch,
3. evaluate once on the held-out test split,
4. log config params + metrics to MLflow and save the model artifact.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.data.dataloaders import (
    DatasetUnavailableError,
    build_dataloaders,
    build_test_loaders,
)
from phase_classification.engine import Trainer, resolve_device
from phase_classification.metrics import LossAccumulator, MetricAccumulator
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
    _set_s3_env(cfg.mlflow)
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


def _set_s3_env(mlflow_cfg) -> None:
    """Expose S3/MinIO client credentials to boto3 (env vars win over config).

    Without these the training process cannot upload the model artifact to the
    S3 artifact store and fails with ``NoCredentialsError``.
    """
    for env_name, value in (
        ("MLFLOW_S3_ENDPOINT_URL", mlflow_cfg.s3_endpoint_url),
        ("AWS_ACCESS_KEY_ID", mlflow_cfg.aws_access_key_id),
        ("AWS_SECRET_ACCESS_KEY", mlflow_cfg.aws_secret_access_key),
        ("AWS_DEFAULT_REGION", mlflow_cfg.aws_region),
    ):
        if value:
            os.environ.setdefault(env_name, value)


def _loss_key(name: str) -> str:
    """MLflow metric name for a loss component: 'total' -> 'loss', else '<head>_loss'."""
    return "loss" if name == "total" else f"{name}_loss"


def _losses(accumulator: LossAccumulator, prefix: str) -> dict[str, float]:
    return {
        _loss_key(k) if not prefix else f"{prefix}_{_loss_key(k)}": v
        for k, v in accumulator.compute().items()
    }


def _epoch_metrics(trainer: Trainer, loader, prefix: str) -> dict[str, float]:
    accumulator = trainer.new_accumulator()
    loss_accumulator = trainer.new_loss_accumulator()
    trainer.evaluate_accumulate(
        loader, accumulator, loss_accumulator=loss_accumulator
    )
    metrics = {f"{prefix}_{k}": v for k, v in accumulator.compute().items()}
    return {**_losses(loss_accumulator, prefix), **metrics}


def _end_mlflow_run(status: str) -> None:
    """End the active MLflow run with the given status (if any run is active)."""
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
    except mlflow.exceptions.MlflowException as exc:
        print(f"warning: failed to finalize MLflow run ({exc})", file=sys.stderr)


def run_training(cfg: ExperimentConfig) -> dict[str, float]:
    """Execute the full training/validation/test lifecycle, logging to MLflow."""
    _start_mlflow(cfg)
    try:
        return _run_lifecycle(cfg)
    except Exception:
        _end_mlflow_run("FAILED")
        raise


def _clone_state_dict(model) -> dict:
    """Detached CPU copy of the model's state dict (for checkpointing)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _class_weights(
    cfg: ExperimentConfig, dl, device: torch.device
) -> dict[str, torch.Tensor] | None:
    """Inverse-frequency class weights when ``class_weighting == 'inverse'``."""
    if cfg.training.class_weighting != "inverse":
        return None
    from phase_classification.losses import compute_class_weights

    print("Computing inverse-frequency class weights from the train split...")
    # Use the underlying dataset (label JSONs only) instead of the loader
    # to avoid decoding every image just for a histogram.
    class_weights = compute_class_weights(
        dl.train.dataset, cfg.training.heads, cfg.model.num_classes
    )
    class_weights = {h: w.to(device) for h, w in class_weights.items()}
    for head, w in class_weights.items():
        print(f"  {head} weights: {[round(x, 3) for x in w.tolist()]}")
    return class_weights


def _train_epochs(
    trainer: Trainer,
    model,
    dl,
    cfg: ExperimentConfig,
    higher_better: bool,
    metric_names: list[str],
):
    """Train and validate each epoch; return the best-val state dict (or None).

    ``None`` means no validation metric was ever recorded (e.g. zero epochs
    or an empty metric list); the caller then falls back to the final model.
    """
    best_metric = float("-inf") if higher_better else float("inf")
    best_state = None
    primary_metric = (
        f"val_{cfg.training.heads[0]}_{metric_names[0]}" if metric_names else None
    )

    try:
        for epoch in range(1, cfg.training.epochs + 1):
            train_accumulator = trainer.new_accumulator()
            train_loss_accumulator = trainer.new_loss_accumulator()
            train_loss, train_metrics = trainer.train_epoch(
                dl.train,
                accumulator=train_accumulator,
                loss_accumulator=train_loss_accumulator,
            )
            train_losses = _losses(train_loss_accumulator, prefix="train")
            train_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
            val_metrics = _epoch_metrics(trainer, dl.val, prefix="val")
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    **train_losses,
                    **train_metrics,
                    **val_metrics,
                },
                step=epoch,
            )
            value = val_metrics.get(primary_metric) if primary_metric else None
            if value is not None and _is_better(value, best_metric, higher_better):
                best_metric = value
                best_state = _clone_state_dict(model)
            scheduler_metric = (
                value if cfg.training.scheduler == "plateau" and value is not None else None
            )
            trainer.step_scheduler(scheduler_metric)
            print(
                f"epoch {epoch}/{cfg.training.epochs} "
                f"train_loss={train_loss:.4f} "
                + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            )
    except KeyboardInterrupt:
        print("Training interrupted; finishing with the best checkpoint.")
    return best_state


def _evaluate_test(trainer: Trainer, dl, cfg: ExperimentConfig) -> dict[str, float]:
    """Evaluate each held-out test video; pool CM metrics, average edit score."""
    accumulators = []
    loss_accumulators = []
    for test_loader in build_test_loaders(dl, cfg):
        accumulator = trainer.new_accumulator()
        loss_accumulator = trainer.new_loss_accumulator()
        trainer.evaluate_accumulate(
            test_loader, accumulator, loss_accumulator=loss_accumulator
        )
        accumulators.append(accumulator)
        loss_accumulators.append(loss_accumulator)
    if not accumulators:
        return {}
    losses = _losses(LossAccumulator.merged(loss_accumulators), prefix="")
    metrics = MetricAccumulator.merged(accumulators).compute()
    return {**losses, **metrics}


def _run_lifecycle(cfg: ExperimentConfig) -> dict[str, float]:
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
        class_weights=_class_weights(cfg, dl, device),
    )

    best_state = _train_epochs(trainer, model, dl, cfg, higher_better, metric_names)

    test_metrics = {
        f"test_{k}": v for k, v in _evaluate_test(trainer, dl, cfg).items()
    }
    mlflow.log_metrics(test_metrics)

    if best_state is None:
        best_state = _clone_state_dict(model)
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
        name = f"{cfg.project.name}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
        mlflow.pytorch.log_model(
            model,
            name=name,
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


_HIGHER_IS_BETTER_METRICS = {
    "jaccard",
    "f1_macro",
    "f1_per_class",
    "mean_accuracy",
    "accuracy_per_class",
    "edit_score",
}


def _higher_is_better(metric_names: list[str]) -> bool:
    return bool(metric_names) and all(
        m in _HIGHER_IS_BETTER_METRICS for m in metric_names
    )
