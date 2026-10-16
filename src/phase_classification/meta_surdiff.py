"""Two-stage SurgCLIP + MS-TCN + Meta-SurDiff training lifecycle."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

import mlflow
from phase_classification.config import ExperimentConfig
from phase_classification.engine import Trainer
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX
from phase_classification.metrics import MetricAccumulator
from phase_classification.models.meta_surdiff import (
    ConditionalLabelDiffusion,
    MetaSurDiffSystem,
    MetaWeightNet,
)
from phase_classification.models.ms_tcn import CachedFeatureMSTCN


def _phase_targets(video) -> torch.Tensor:
    return video.labels[HEAD_TUPLE_INDEX["phase"]].argmax(dim=-1)


def _validation_metrics(
    model: nn.Module,
    videos: list,
    device: torch.device,
    metric_names: list[str],
    refined: bool = False,
    sampling_steps: int | None = None,
) -> dict[str, float]:
    accumulators = []
    model.eval()
    with torch.inference_mode():
        for video in videos:
            features = video.features.unsqueeze(0).to(device)
            if refined:
                output = model(features, sampling_steps=sampling_steps)["refined_probs"]
            elif isinstance(model, MetaSurDiffSystem):
                output = model(features)["coarse_logits"]
            else:
                output = model(features)["phase"]
            targets = _phase_targets(video).numpy()
            accumulator = MetricAccumulator(
                ["phase"], metric_names, {"phase": DEFAULT_NUM_CLASSES["phase"]}
            )
            accumulator.add_batch(
                {"phase": output[0].detach().cpu().numpy()},
                {
                    "phase": np.eye(DEFAULT_NUM_CLASSES["phase"], dtype=np.int32)[targets]
                },
            )
            accumulators.append(accumulator)
    return MetricAccumulator.merged(accumulators).compute()


def _phase_class_weights(videos: list, device: torch.device) -> torch.Tensor:
    counts = sum(
        (video.labels[HEAD_TUPLE_INDEX["phase"]].sum(dim=0) for video in videos),
        start=torch.zeros(DEFAULT_NUM_CLASSES["phase"]),
    )
    counts[counts == 0] = counts.max().clamp(min=1)
    return (counts.sum() / (len(counts) * counts)).to(device)


def _save_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _crop(video, crop_length: int, start: int, anchor: int | None = None):
    features = video.features
    targets = _phase_targets(video)
    length = len(targets)
    start = max(0, min(start, max(length - crop_length, 0)))
    stop = min(start + crop_length, length)
    x, y = features[start:stop], targets[start:stop]
    valid = torch.zeros(crop_length, dtype=torch.bool)
    valid[: len(y)] = True
    if len(y) < crop_length:
        x = torch.cat((x, x[-1:].expand(crop_length - len(y), -1)))
        y = torch.cat((y, y[-1:].expand(crop_length - len(y))))
    if anchor is not None:
        valid.zero_()
        valid[anchor - start] = True
    return x, y, valid


def _sample_training_crop(videos: list, crop_length: int, device: torch.device):
    video = videos[int(np.random.randint(len(videos)))]
    length = len(video.features)
    start = int(np.random.randint(max(length - crop_length + 1, 1)))
    x, y, mask = _crop(video, crop_length, start)
    return x.unsqueeze(0).to(device), y.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)


def _meta_anchors(videos: list) -> dict[int, list[tuple[int, int]]]:
    anchors = {phase: [] for phase in range(DEFAULT_NUM_CLASSES["phase"])}
    for video_index, video in enumerate(videos):
        for frame_index, phase in enumerate(_phase_targets(video).tolist()):
            anchors[phase].append((video_index, frame_index))
    absent = [phase for phase, locations in anchors.items() if not locations]
    if absent:
        raise ValueError(f"Cannot build balanced meta set; absent train phases: {absent}.")
    return anchors


def _sample_balanced_meta_batch(
    videos: list,
    anchors: dict[int, list[tuple[int, int]]],
    crop_length: int,
    device: torch.device,
):
    crops = []
    for phase in range(DEFAULT_NUM_CLASSES["phase"]):
        video_index, frame_index = anchors[phase][
            int(np.random.randint(len(anchors[phase])))
        ]
        video = videos[video_index]
        start = frame_index - crop_length // 2
        x, y, mask = _crop(video, crop_length, start, anchor=frame_index)
        crops.append((x, y, mask))
    return tuple(
        torch.stack([crop[index] for crop in crops]).to(device) for index in range(3)
    )


def _disable_batch_norm_updates(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def _meta_step(
    system: MetaSurDiffSystem,
    weight_net: MetaWeightNet,
    optimizer: torch.optim.Optimizer,
    weight_optimizer: torch.optim.Optimizer,
    train_batch,
    meta_batch,
    inner_learning_rate: float,
) -> tuple[float, float]:
    train_x, train_y, train_mask = train_batch
    meta_x, meta_y, meta_mask = meta_batch
    classes = DEFAULT_NUM_CLASSES["phase"]
    train_t = torch.randint(system.diffusion.timesteps, (len(train_x),), device=train_x.device)
    train_noise = torch.randn(*train_y.shape, classes, device=train_x.device)
    parameters = OrderedDict(system.named_parameters())
    train_losses = functional_call(
        system, parameters, (train_x, train_y, train_t, train_noise)
    )
    selected_train_losses = train_losses[train_mask]
    train_weights = weight_net(selected_train_losses.detach())
    virtual_loss = (train_weights * selected_train_losses).mean()
    gradients = torch.autograd.grad(
        virtual_loss, tuple(parameters.values()), create_graph=True, allow_unused=True
    )
    fast_parameters = OrderedDict(
        (
            name,
            parameter
            if gradient is None
            else parameter - inner_learning_rate * gradient,
        )
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True)
    )

    meta_t = torch.randint(system.diffusion.timesteps, (len(meta_x),), device=meta_x.device)
    meta_noise = torch.randn(*meta_y.shape, classes, device=meta_x.device)
    meta_losses = functional_call(
        system, fast_parameters, (meta_x, meta_y, meta_t, meta_noise)
    )
    meta_loss = meta_losses[meta_mask].mean()
    weight_optimizer.zero_grad(set_to_none=True)
    weight_gradients = torch.autograd.grad(
        meta_loss, tuple(weight_net.parameters()), allow_unused=True
    )
    for parameter, gradient in zip(
        weight_net.parameters(), weight_gradients, strict=True
    ):
        parameter.grad = gradient
    weight_optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    actual_losses = system(train_x, train_y, train_t, train_noise)
    actual_selected = actual_losses[train_mask]
    with torch.no_grad():
        actual_weights = weight_net(actual_selected.detach())
    actual_loss = (actual_weights * actual_selected).mean()
    actual_loss.backward()
    nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
    optimizer.step()
    return float(actual_loss.detach()), float(actual_weights.detach().std(unbiased=False))


def _evaluate_system(
    system: MetaSurDiffSystem,
    videos: list,
    device: torch.device,
    metrics: list[str],
    sampling_steps: int,
) -> tuple[dict[str, float], dict[str, float]]:
    coarse_accumulators = []
    refined_accumulators = []
    model = system
    model.eval()
    with torch.inference_mode():
        for video in videos:
            features = video.features.unsqueeze(0).to(device)
            output = model(features, sampling_steps=sampling_steps)
            targets = _phase_targets(video).numpy()
            target_onehot = np.eye(DEFAULT_NUM_CLASSES["phase"], dtype=np.int32)[targets]
            coarse_acc = MetricAccumulator(
                ["phase"], metrics, {"phase": DEFAULT_NUM_CLASSES["phase"]}
            )
            refined_acc = MetricAccumulator(
                ["phase"], metrics, {"phase": DEFAULT_NUM_CLASSES["phase"]}
            )
            coarse_acc.add_batch(
                {"phase": output["coarse_logits"][0].cpu().numpy()},
                {"phase": target_onehot},
            )
            refined_acc.add_batch(
                {"phase": output["refined_probs"][0].cpu().numpy()},
                {"phase": target_onehot},
            )
            coarse_accumulators.append(coarse_acc)
            refined_accumulators.append(refined_acc)
    return (
        MetricAccumulator.merged(coarse_accumulators).compute(),
        MetricAccumulator.merged(refined_accumulators).compute(),
    )


def run_meta_surdiff(
    cfg: ExperimentConfig, data, device: torch.device
) -> dict[str, float]:
    """Pretrain MS-TCN, then meta-train conditional classification diffusion."""
    torch.manual_seed(cfg.runtime.seed)
    np.random.seed(cfg.runtime.seed)
    params = cfg.model.params
    classes = cfg.model.num_classes["phase"]
    if classes != 7 or cfg.training.heads != ["phase"]:
        raise ValueError("Meta-SurDiff currently supports the seven-class phase head only.")
    if cfg.runtime.mixed_precision != "fp32":
        raise ValueError("Meta-SurDiff bilevel training currently requires fp32.")
    feature_dim = int(data.feature_dim)
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
    metric_names = list(cfg.validation.metrics)
    trainer = Trainer(
        coarse,
        cfg.training,
        device,
        metric_names,
        cfg.model.num_classes,
        mixed_precision="fp32",
        higher_is_better=True,
        class_weights=(
            {"phase": _phase_class_weights(data.train.dataset.datasets, device)}
            if cfg.training.class_weighting == "inverse"
            else None
        ),
    )
    best_score = -float("inf")
    best_coarse = None
    primary = f"phase_{metric_names[0]}" if metric_names else "phase_f1_macro"
    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, _ = trainer.train_epoch(data.train)
        validation = _validation_metrics(
            coarse, data.val.dataset.datasets, device, metric_names
        )
        mlflow.log_metrics(
            {
                "coarse_train_loss": train_loss,
                **{f"coarse_val_{key}": value for key, value in validation.items()},
            },
            step=epoch,
        )
        score = validation.get(primary, -float("inf"))
        if score > best_score:
            best_score, best_coarse = score, _save_state(coarse)
        trainer.step_scheduler(score)
        print(f"coarse {epoch}/{cfg.training.epochs}: loss={train_loss:.4f} val={score:.4f}")
    if best_coarse is None:
        raise RuntimeError("Coarse MS-TCN training produced no valid checkpoint.")
    coarse.load_state_dict(best_coarse)
    if cfg.project.save_checkpoints:
        output_dir = Path(cfg.project.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_coarse, output_dir / "coarse_pretrained.pt")

    diffusion = ConditionalLabelDiffusion(
        num_classes=classes,
        timesteps=int(params.get("diffusion_timesteps", 100)),
        hidden_dim=int(params.get("diffusion_hidden_dim", 512)),
        beta_start=float(params.get("beta_start", 1e-4)),
        beta_end=float(params.get("beta_end", 0.02)),
    )
    system = MetaSurDiffSystem(coarse, diffusion).to(device)
    weight_net = MetaWeightNet(int(params.get("meta_hidden_dim", 100))).to(device)
    theta_lr = float(params.get("theta_learning_rate", 1e-5))
    meta_lr = float(params.get("meta_learning_rate", 1e-3))
    theta_optimizer = torch.optim.AdamW(system.parameters(), lr=theta_lr, weight_decay=0.0)
    meta_optimizer = torch.optim.AdamW(weight_net.parameters(), lr=meta_lr, weight_decay=0.0)
    crop_length = int(params.get("crop_length", 256))
    meta_epochs = int(params.get("meta_epochs", 20))
    steps_per_epoch = int(params.get("meta_steps_per_epoch", 100))
    sampling_steps = int(params.get("sampling_steps", diffusion.timesteps))
    inner_lr = float(params.get("inner_learning_rate", theta_lr))
    train_videos = data.train.dataset.datasets
    balanced_anchors = _meta_anchors(train_videos)
    best_refined = -float("inf")
    best_system = None
    best_weight_net = None
    primary = f"phase_{metric_names[0]}" if metric_names else "phase_f1_macro"

    for epoch in range(1, meta_epochs + 1):
        system.train()
        _disable_batch_norm_updates(system)
        losses, weight_spreads = [], []
        for _ in range(steps_per_epoch):
            train_batch = _sample_training_crop(train_videos, crop_length, device)
            meta_batch = _sample_balanced_meta_batch(
                train_videos, balanced_anchors, crop_length, device
            )
            loss, weight = _meta_step(
                system,
                weight_net,
                theta_optimizer,
                meta_optimizer,
                train_batch,
                meta_batch,
                inner_lr,
            )
            losses.append(loss)
            weight_spreads.append(weight)

        coarse_metrics, refined_metrics = _evaluate_system(
            system, data.val.dataset.datasets, device, metric_names, sampling_steps
        )
        validation_values = {
            **{f"meta_val_coarse_{key}": value for key, value in coarse_metrics.items()},
            **{f"meta_val_refined_{key}": value for key, value in refined_metrics.items()},
        }
        step = cfg.training.epochs + epoch
        mlflow.log_metrics(
            {
                "meta_train_loss": float(np.mean(losses)),
                "meta_weight_std": float(np.mean(weight_spreads)),
                **validation_values,
            },
            step=step,
        )
        score = refined_metrics.get(primary, -float("inf"))
        print(
            f"Meta-SurDiff {epoch}/{meta_epochs}: loss={np.mean(losses):.4f} "
            f"val_refined_{primary}={score:.4f}"
        )
        if score > best_refined:
            best_refined = score
            best_system = _save_state(system)
            best_weight_net = _save_state(weight_net)

    if best_system is None or best_weight_net is None:
        raise RuntimeError("Meta-SurDiff training produced no valid checkpoint.")
    system.load_state_dict(best_system)
    weight_net.load_state_dict(best_weight_net)
    coarse_test, refined_test = _evaluate_system(
        system, data.test, device, metric_names, sampling_steps
    )
    test_metrics = {
        **{f"test_coarse_{key}": value for key, value in coarse_test.items()},
        **{f"test_refined_{key}": value for key, value in refined_test.items()},
    }
    mlflow.log_metrics(test_metrics)

    output_dir = Path(cfg.project.output_dir)
    checkpoint = {
        "system_state_dict": best_system,
        "meta_weight_state_dict": best_weight_net,
        "settings": dict(params),
        "best_validation_refined_metric": best_refined,
    }
    if cfg.project.save_checkpoints:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "best.pt"
        torch.save(checkpoint, checkpoint_path)
    if cfg.project.save_checkpoints:
        system.eval()
        mlflow.pytorch.log_model(
            system,
            name=f"{cfg.project.name}_best",
            serialization_format="pickle",
        )
        mlflow.log_artifact(str(checkpoint_path))
    if cfg.mlflow.log_artifacts_to is not None:
        artifact_path = Path(cfg.mlflow.log_artifacts_to)
        artifact_path.mkdir(parents=True, exist_ok=True)
        checkpoint_artifact = artifact_path / "best.pt"
        torch.save(checkpoint, checkpoint_artifact)
        mlflow.log_artifact(str(checkpoint_artifact))
        destination = artifact_path / "meta_weight_net.pt"
        torch.save(best_weight_net, destination)
        mlflow.log_artifact(str(destination))
    return test_metrics
