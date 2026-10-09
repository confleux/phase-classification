"""Dataset construction from config.

CholecT50 is provided by the vendored reference loader, while HeiChole uses a
native phase-label loader. Dataset variants are selected by their leading
token.
"""

from phase_classification.config.dataset import DatasetConfig
from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.data.heichole import HeiChole


class DatasetRegistryError(Exception):
    """Raised when no dataset builder matches the configured variant."""


SUPPORTED_DATASET_PREFIXES = ("cholect", "heichole")


def build_dataset(cfg: DatasetConfig) -> CholecT50 | HeiChole:
    """Construct the dataset object for the given dataset config.

    The returned object exposes ``.train_dataset``, ``.val_dataset``,
    ``.test_dataset`` and ``.build() -> (train, val, test)`` torch Datasets,
    plus ``.train_records/.val_records/.test_records``.

    Per-frame (or per-window, when ``sequence_length > 1``) datasets are used;
    temporal models such as the MS-TCN consume fixed-length windows.
    """
    if cfg.dataset_variant.startswith("heichole"):
        return HeiChole(
            dataset_dir=str(cfg.root_dir),
            test_fold=cfg.test_fold,
            augmentation_list=list(cfg.augmentations),
            normalize=cfg.normalize,
            sequence_length=cfg.sequence_length,
        )

    if not cfg.dataset_variant.startswith("cholect"):
        raise DatasetRegistryError(
            f"No dataset builder for variant '{cfg.dataset_variant}' "
            f"(expected one of {', '.join(SUPPORTED_DATASET_PREFIXES)}*)."
        )

    return CholecT50(
        dataset_dir=str(cfg.root_dir),
        dataset_variant=cfg.dataset_variant,
        test_fold=cfg.test_fold,
        augmentation_list=list(cfg.augmentations),
        normalize=cfg.normalize,
        sequence_length=cfg.sequence_length,
    )
