"""DataLoader construction wired to the training loop.

Builds ``torch.utils.data.DataLoader`` objects over the datasets produced by
:func:`phase_classification.data.builders.build_dataset`, applying the
``DataLoaderConfig`` knobs and seeding from ``RuntimeConfig.seed``.
"""

from dataclasses import dataclass
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset

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
    """Train/val dataloaders plus the per-video datasets of the test split.

    The CholecT50 test split is a list of per-video datasets (each video is a
    sequence for edit-score evaluation); use :func:`build_test_loaders` to
    wrap them.
    """

    train: DataLoader
    val: DataLoader
    test: list[Dataset]


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


def build_test_loaders(dl: DataLoaders, cfg: ExperimentConfig) -> list[DataLoader]:
    """Per-video test dataloaders honouring the configured DataLoader knobs."""
    worker_init_fn = partial(_seed_worker, seed=cfg.runtime.seed)
    return [
        _make_loader(ds, shuffle=False, cfg=cfg.dataloader, worker_init_fn=worker_init_fn)
        for ds in dl.test
    ]


def build_dataloaders(cfg: ExperimentConfig) -> DataLoaders:
    """Build train/val/test dataloaders from an experiment config.

    Requires the dataset root to be present on disk (raises
    :class:`DatasetUnavailableError` otherwise).

    Datasets are per-frame (``sequence_length == 1``) or fixed-length temporal
    windows (``sequence_length >= 2``, required by temporal models such as the
    MS-TCN).
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
    return DataLoaders(
        train=_make_loader(
            train_ds, shuffle=dl_cfg.shuffle, cfg=dl_cfg, worker_init_fn=worker_init_fn
        ),
        val=_make_loader(
            val_ds, shuffle=False, cfg=dl_cfg, worker_init_fn=worker_init_fn
        ),
        test=test_ds,
    )
