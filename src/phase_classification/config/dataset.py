"""Dataset configuration, CholecT50-aware.

The enum values mirror the reference CholecT50_dataloader.py (CAMMA Rendezvous):
dataset variants, augmentations, test folds, and the temporal-window rule.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

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
CHOLEC50_VARIANT_STR = (
    "cholect45-crossval",
    "cholect50-crossval",
    "cholect50-challenge",
    "cholect50",
    "cholect45",
    "cholect50-subset",
    "cholect50-small",
)

DataAugmentation = Literal[
    "original",
    "vflip",
    "hflip",
    "contrast",
    "rot90",
    "brightness",
    "random_contrast",
]
DATA_AUGMENTATION_STR = (
    "original",
    "vflip",
    "hflip",
    "contrast",
    "rot90",
    "brightness",
    "random_contrast",
)

CROSSVAL_VARIANTS = ("cholect45-crossval", "cholect50-crossval")

LoaderStyle = Literal["frame", "video"]


class DatasetConfig(StrictModel):
    """Dataset and preprocessing configuration.

    For cross-validation variants a ``test_fold`` in [1, 5] selects the held-out
    fold, matching the reference loader. ``sequence_length > 1`` selects temporal
    frame windows (requires >= 2, matching T50Sequence).
    """

    dataset_variant: CHOLEC50_VARIANTS = "cholect50-crossval"
    root_dir: Path
    test_fold: int = Field(default=1, ge=1, le=5)
    sequence_length: int = Field(default=1, ge=1)
    augmentations: list[DataAugmentation] = Field(default_factory=lambda: ["original"])
    image_size: tuple[int, int] = (256, 448)
    normalize: bool = True
    video_ids: list[int] | None = None
    loader: LoaderStyle = "frame"

    @model_validator(mode="after")
    def _crossval_test_fold(self) -> DatasetConfig:
        if "crossval" in self.dataset_variant and not (1 <= self.test_fold <= 5):
            raise ValueError(
                f"test_fold must be in [1, 5] for cross-validation variant "
                f"'{self.dataset_variant}', got {self.test_fold}."
            )
        return self

    @model_validator(mode="after")
    def _temporal_window(self) -> DatasetConfig:
        if self.sequence_length > 1 and self.sequence_length < 2:
            raise ValueError(
                "sequence_length >= 2 required for temporal frame windows, "
                f"got {self.sequence_length}."
            )
        return self

    @model_validator(mode="after")
    def _image_size(self) -> DatasetConfig:
        h, w = self.image_size
        if h <= 0 or w <= 0:
            raise ValueError(
                f"image_size must be (H, W) with positive dims, got {self.image_size=}."
            )
        return self
