"""Data layer: build datasets and dataloaders for the training loop.

The CholecT50 implementation lives in :mod:`phase_classification.data.cholec_t50`
(a vendored, unmodified copy of the CAMMA Rendezvous reference loader).
"""

from phase_classification.data.builders import build_dataset
from phase_classification.data.dataloaders import DataLoaders, build_dataloaders

__all__ = ["DataLoaders", "build_dataloaders", "build_dataset"]
