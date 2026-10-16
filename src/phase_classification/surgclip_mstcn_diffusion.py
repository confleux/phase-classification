"""Train a cached-feature MS-TCN and refine its scores with SurgCLIP diffusion."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.engine import Trainer
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX
from phase_classification.models.ms_tcn import CachedFeatureMSTCN
from phase_classification.run import (
    _class_weights,
    _clone_state_dict,
    _higher_is_better,
    _train_epochs,
)
from phase_classification.surgclip_diffusion import VideoScores, _evaluate, _fit_refiner


def _to_video_scores(model: CachedFeatureMSTCN, videos: list, device: torch.device):
    model.eval()
    output = []
    with torch.inference_mode():
        for video_index, video in enumerate(videos):
            features = video.features.unsqueeze(0).to(device)
            scores = model(features)["phase"][0].cpu()
            targets = video.labels[HEAD_TUPLE_INDEX["phase"]].argmax(dim=-1).long()
            output.append(
                VideoScores(
                    getattr(video, "video_id", f"video_{video_index + 1:03d}"),
                    video.features.float().cpu(),
                    scores,
                    targets,
                )
            )
    return output


def _log_refiner_epoch(
    pretrain_epochs: int,
    epoch: int,
    noise_loss: float,
    validation: dict[str, float],
    video_average: dict[str, float],
) -> None:
    mlflow.log_metrics(
        {
            "refiner_train_noise_loss": noise_loss,
            **{f"refiner_val_{key}": value for key, value in validation.items()},
            **{f"refiner_val_{key}": value for key, value in video_average.items()},
        },
        step=pretrain_epochs + epoch,
    )


def run_surgclip_mstcn_diffusion(
    cfg: ExperimentConfig, data, device: torch.device
) -> dict[str, float]:
    """Run coarse MS-TCN pretraining followed by residual diffusion refinement."""
    torch.manual_seed(cfg.runtime.seed)
    np.random.seed(cfg.runtime.seed)
    params = cfg.model.params
    classes = cfg.model.num_classes["phase"]
    if classes != DEFAULT_NUM_CLASSES["phase"] or cfg.training.heads != ["phase"]:
        raise ValueError("MS-TCN diffusion refinement supports the seven-class phase head only.")
    feature_dim = int(data.train.dataset.datasets[0].features.shape[-1])
    if feature_dim != int(params.get("feature_dim", 256)):
        raise ValueError("model.params.feature_dim must match the cached SurgCLIP features.")

    coarse = CachedFeatureMSTCN(
        feature_dim=feature_dim,
        num_classes=cfg.model.num_classes,
        heads=["phase"],
        num_stages=int(params.get("num_stages", 3)),
        num_layers=int(params.get("num_layers", 8)),
        num_f_maps=int(params.get("num_f_maps", 64)),
        kernel_size=int(params.get("kernel_size", 3)),
    ).to(device)
    trainer = Trainer(
        coarse,
        cfg.training,
        device,
        metric_names=list(cfg.validation.metrics),
        num_classes=cfg.model.num_classes,
        mixed_precision=cfg.runtime.mixed_precision,
        higher_is_better=_higher_is_better(list(cfg.validation.metrics)),
        class_weights=_class_weights(cfg, data, device),
    )
    best_coarse = _train_epochs(
        trainer,
        coarse,
        data,
        cfg,
        _higher_is_better(list(cfg.validation.metrics)),
        list(cfg.validation.metrics),
    )
    if best_coarse is not None:
        coarse.load_state_dict(best_coarse)
    else:
        best_coarse = _clone_state_dict(coarse)

    train_scores = _to_video_scores(coarse, data.train.dataset.datasets, device)
    val_scores = _to_video_scores(coarse, data.val.dataset.datasets, device)
    test_scores = _to_video_scores(coarse, data.test, device)
    args = Namespace(
        epochs=int(params.get("refiner_epochs", 20)),
        steps_per_epoch=int(params.get("refiner_steps_per_epoch", 100)),
        batch_size=int(params.get("refiner_batch_size", 8)),
        crop_length=int(params.get("refiner_crop_length", 256)),
        width=int(params.get("refiner_width", 96)),
        layers=int(params.get("refiner_layers", 6)),
        diffusion_steps=int(params.get("diffusion_timesteps", 1000)),
        correction_scale=float(params.get("correction_scale", 1.0)),
        learning_rate=float(params.get("refiner_learning_rate", 2e-4)),
        weight_decay=float(params.get("refiner_weight_decay", 1e-4)),
        sample_steps=int(params.get("sampling_steps", 25)),
        test_samples=int(params.get("test_samples", 1)),
        seed=cfg.runtime.seed,
        auxiliary_ce_weight=float(params.get("auxiliary_ce_weight", 1.0)),
    )
    refiner, best_validation = _fit_refiner(
        train_scores,
        val_scores,
        args,
        device,
        on_epoch=lambda epoch, loss, validation, video_average: _log_refiner_epoch(
            cfg.training.epochs, epoch, loss, validation, video_average
        ),
    )
    coarse_metrics, coarse_video_average, _ = _evaluate(
        test_scores, None, device, args.sample_steps, num_samples=1, seed=args.seed
    )
    refined_metrics, refined_video_average, _ = _evaluate(
        test_scores,
        refiner,
        device,
        args.sample_steps,
        num_samples=args.test_samples,
        seed=args.seed,
    )
    test_metrics = {
        **{f"test_coarse_{key}": value for key, value in coarse_metrics.items()},
        **{f"test_coarse_{key}": value for key, value in coarse_video_average.items()},
        **{f"test_refined_{key}": value for key, value in refined_metrics.items()},
        **{f"test_refined_{key}": value for key, value in refined_video_average.items()},
    }
    mlflow.log_metrics(test_metrics)

    checkpoint = {
        "coarse_state_dict": best_coarse,
        "refiner_state_dict": _clone_state_dict(refiner),
        "feature_dim": feature_dim,
        "settings": dict(params),
        "best_refiner_validation_metrics": best_validation,
    }
    if cfg.project.save_checkpoints:
        checkpoint_dir = Path(cfg.project.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, checkpoint_path)
        mlflow.pytorch.log_model(
            coarse,
            name=f"{cfg.project.name}_coarse_mstcn",
            serialization_format="pickle",
        )
        mlflow.pytorch.log_model(
            refiner,
            name=f"{cfg.project.name}_diffusion_refiner",
            serialization_format="pickle",
        )
        mlflow.log_artifact(str(checkpoint_path))
    if cfg.mlflow.log_artifacts_to is not None:
        artifact_dir = Path(cfg.mlflow.log_artifacts_to)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifact_dir / "surgclip_mstcn_diffusion.pt"
        torch.save(checkpoint, checkpoint_path)
        mlflow.log_artifact(str(checkpoint_path))
    return test_metrics
