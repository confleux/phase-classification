"""DataLoader configuration."""

from pydantic import Field

from phase_classification.config.base import StrictModel


class DataLoaderConfig(StrictModel):
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    shuffle: bool = True
    pin_memory: bool = True
    prefetch_factor: int | None = Field(default=2, ge=1)
    persistent_workers: bool = True
    drop_last: bool = False
