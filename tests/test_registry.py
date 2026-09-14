"""Tests for the model registry."""

from __future__ import annotations

import pytest

from phase_classification.config.model import ModelConfig
from phase_classification.models import ModelRegistryError, build_model
from phase_classification.models.ms_tcn import ConvNeXtMSTCN, SwinMSTCN
from phase_classification.models.resnet18 import ResNet18Classifier
from phase_classification.models.swin_classifier import SwinClassifier


class TestBuildModel:
    def test_resnet18(self, num_classes):
        model = build_model(
            ModelConfig(name="resnet18", num_classes=num_classes), heads=["phase"]
        )
        assert isinstance(model, ResNet18Classifier)
        assert set(model.heads_block) == {"phase"}

    def test_resnet18_params(self, num_classes):
        model = build_model(
            ModelConfig(
                name="resnet18",
                num_classes=num_classes,
                params={"pretrained": False, "features": 64},
            ),
            heads=["phase"],
        )
        assert model.project.out_features == 64

    def test_resnet18_unknown_param(self, num_classes):
        with pytest.raises(ModelRegistryError, match="bogus"):
            build_model(
                ModelConfig(name="resnet18", num_classes=num_classes, params={"bogus": 1})
            )

    def test_unknown_name(self, num_classes):
        with pytest.raises(ModelRegistryError, match="No model builder"):
            build_model(ModelConfig(name="bogus", num_classes=num_classes))

    def test_alias_ms_tcn(self, num_classes):
        model = build_model(
            ModelConfig(name="ms-tcn", num_classes=num_classes), heads=["phase"]
        )
        assert isinstance(model, ConvNeXtMSTCN)

    def test_convnext_ms_tcn_stage_channels(self, num_classes):
        model = build_model(
            ModelConfig(
                name="convnext-ms-tcn",
                num_classes=num_classes,
                params={"num_stages": 2, "num_f_maps": 16},
            ),
            heads=["phase"],
        )
        assert model.stages[0].in_conv.in_channels == 768
        assert model.stages[1].in_conv.in_channels == 768 + 7

    def test_swin_ms_tcn(self, num_classes):
        model = build_model(
            ModelConfig(name="swin-ms-tcn", num_classes=num_classes), heads=["phase"]
        )
        assert isinstance(model, SwinMSTCN)

    def test_swin_classifier_alias(self, num_classes):
        model = build_model(
            ModelConfig(name="swin-classifier", num_classes=num_classes), heads=["phase"]
        )
        assert isinstance(model, SwinClassifier)

    def test_default_heads_all(self, num_classes):
        model = build_model(ModelConfig(name="resnet18", num_classes=num_classes))
        assert set(model.heads_block) == set(num_classes)
