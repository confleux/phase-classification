"""Backward-compatibility shim for checkpoints pickled against the old layout.

Before the networks were split into separate files (commit ``67a65e3``), all
models lived in ``phase_classification.models.net``. Pickled full-model
checkpoints reference classes by that module path, so ``torch.load`` needs this
module to exist to unpickle them. Re-export the current implementations here.
"""

from phase_classification.models.backbones import (
    ConvNeXtFeatureExtractor,
    SwinFeatureExtractor,
)
from phase_classification.models.ms_tcn import (
    SingleStageTCN,
    SwinMSTCN,
    TemporalResidualBlock,
)
from phase_classification.models.resnet18 import ResNet18Classifier
from phase_classification.models.swin_classifier import SwinClassifier

__all__ = [
    "ConvNeXtFeatureExtractor",
    "ResNet18Classifier",
    "SingleStageTCN",
    "SwinClassifier",
    "SwinFeatureExtractor",
    "SwinMSTCN",
    "TemporalResidualBlock",
]