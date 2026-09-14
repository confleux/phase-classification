"""Dataset configuration, CholecT50-aware.

The enum values mirror the reference CholecT50_dataloader.py (CAMMA Rendezvous):
dataset variants, augmentations, and test folds. Frames are resized to the
loader's fixed 256x448; ``sequence_length > 1`` selects temporal frame windows
(requires >= 2, matching T50Sequence).
"""

from pathlib import Path
from typing import Literal

from pydantic import Field

from phase_classification.config.base import StrictModel

CHOLEC50_VARIANTS = Literal[
    "cholect45-crossval",
    "cholect50-crossval",
    "cholect50-challenge",
    "cholect50",
    "cholect45",
    "cholect50-subset",
    "cholect50-small",
]

DataAugmentation = Literal[
    "original",
    "vflip",
    "hflip",
    "contrast",
    "rot90",
    "brightness",
    "random_contrast",
]


class DatasetConfig(StrictModel):
    """Dataset and preprocessing configuration.

    For cross-validation variants a ``test_fold`` in [1, 5] selects the held-out
    fold, matching the reference loader.
    """

    dataset_variant: CHOLEC50_VARIANTS = "cholect50-crossval"
    root_dir: Path
    test_fold: int = Field(default=1, ge=1, le=5)
    sequence_length: int = Field(default=1, ge=1)
    augmentations: list[DataAugmentation] = Field(default_factory=lambda: ["original"])
    normalize: bool = True
