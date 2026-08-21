"""Dataset construction from config.

The current (and only) implemented dataset is CholecT50, provided by the
vendored reference loader :mod:`phase_classification.data.cholec_t50`. Future
datasets can register here, keyed on the leading token of ``dataset_variant``.
"""

from phase_classification.config.dataset import DatasetConfig
from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.data.video_loader import CholecT50Video


class DatasetRegistryError(Exception):
    """Raised when no dataset builder matches the configured variant."""


SUPPORTED_DATASET_PREFIX = "cholect"


def build_dataset(cfg: DatasetConfig) -> CholecT50:
    """Construct the dataset object for the given dataset config.

    The returned object exposes ``.train_dataset``, ``.val_dataset``,
    ``.test_dataset`` and ``.build() -> (train, val, test)`` torch Datasets,
    plus ``.train_records/.val_records/.test_records``.

    ``cfg.loader`` selects the style:
      - "frame": per-frame (or per-window) datasets (default)
      - "video": one video per item (:class:`CholecT50Video`)
    """
    if not cfg.dataset_variant.startswith(SUPPORTED_DATASET_PREFIX):
        raise DatasetRegistryError(
            f"No dataset builder for variant '{cfg.dataset_variant}' "
            f"(expected a '{SUPPORTED_DATASET_PREFIX}*' variant)."
        )

    cls = CholecT50Video if cfg.loader == "video" else CholecT50
    return cls(
        dataset_dir=str(cfg.root_dir),
        dataset_variant=cfg.dataset_variant,
        test_fold=cfg.test_fold,
        augmentation_list=list(cfg.augmentations),
        normalize=cfg.normalize,
        sequence_length=cfg.sequence_length,
    )
