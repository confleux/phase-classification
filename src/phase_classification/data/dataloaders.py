"""DataLoader construction wired to the training loop.

Builds ``torch.utils.data.DataLoader`` objects over the datasets produced by
:func:`phase_classification.data.builders.build_dataset`, applying the
``DataLoaderConfig`` knobs and seeding from ``RuntimeConfig.seed``.
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from phase_classification.config import ExperimentConfig
from phase_classification.data.builders import build_dataset
from phase_classification.loader import check_paths


class DatasetUnavailableError(Exception):
    """Raised when the dataset root is required but is missing on disk."""


def _seed_worker(worker_id: int, seed: int) -> None:
    """Per-worker RNG seeding (module-level so it can be pickled by workers)."""
    torch.manual_seed(seed + worker_id)


@dataclass
class DataLoaders:
    """Named train/val/test dataloaders for the training loop.

    In ``dataset.loader == "video"`` mode ``test`` is a *list* of per-video
    dataloaders (one whole video per step) instead of a single loader.
    """

    train: DataLoader
    val: DataLoader
    test: DataLoader | list[DataLoader]


def _make_loader(dataset: Dataset, shuffle: bool, cfg, worker_init_fn) -> DataLoader:
    if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
        kwargs = {"prefetch_factor": cfg.prefetch_factor}
    else:
        kwargs = {}
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
        **kwargs,
    )


def _make_video_loader(dataset: Dataset, shuffle: bool, cfg, worker_init_fn) -> DataLoader:
    """DataLoader yielding one whole video per step (batch_size forced to 1)."""
    if cfg.num_workers > 0 and cfg.prefetch_factor is not None:
        kwargs = {"prefetch_factor": cfg.prefetch_factor}
    else:
        kwargs = {}
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=cfg.num_workers,
        shuffle=shuffle,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        worker_init_fn=worker_init_fn,
        **kwargs,
    )


def build_dataloaders(cfg: ExperimentConfig) -> DataLoaders:
    """Build train/val/test dataloaders from an experiment config.

    Requires the dataset root to be present on disk (raises
    :class:`DatasetUnavailableError` otherwise).

    ``dataset.loader`` selects the style:
      - "frame" (default): per-frame/window batches as configured by
        ``dataloader.*``.
      - "video": one video per iteration; ``batch_size`` is forced to 1 and
        ``drop_last`` ignored. The test loader iterates over per-video test
        datasets.
    """
    missing = check_paths(cfg)
    if missing:
        raise DatasetUnavailableError("; ".join(missing))

    dataset = build_dataset(cfg.dataset)
    train_ds, val_ds, test_ds = dataset.build()
    seed = cfg.runtime.seed

    torch.manual_seed(seed)
    worker_init_fn = partial(_seed_worker, seed=seed)
    dl_cfg = cfg.dataloader
    if cfg.dataset.loader == "video":
        # train/val are lists of per-video datasets -> concatenate into one
        # dataset; each item is a whole video (batch_size forced to 1).
        return DataLoaders(
            train=_make_video_loader(
                ConcatDataset(train_ds),
                shuffle=dl_cfg.shuffle,
                cfg=dl_cfg,
                worker_init_fn=worker_init_fn,
            ),
            val=_make_video_loader(
                ConcatDataset(val_ds),
                shuffle=False,
                cfg=dl_cfg,
                worker_init_fn=worker_init_fn,
            ),
            test=[
                _make_video_loader(
                    videos, shuffle=False, cfg=dl_cfg, worker_init_fn=worker_init_fn
                )
                for videos in test_ds
            ],
        )
    return DataLoaders(
        train=_make_loader(
            train_ds, shuffle=dl_cfg.shuffle, cfg=dl_cfg, worker_init_fn=worker_init_fn
        ),
        val=_make_loader(
            val_ds, shuffle=False, cfg=dl_cfg, worker_init_fn=worker_init_fn
        ),
        test=_make_loader(
            test_ds, shuffle=False, cfg=dl_cfg, worker_init_fn=worker_init_fn
        ),
    )
