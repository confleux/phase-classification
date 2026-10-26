"""Tests for cached SurgCLIP feature dataloading and the feature MS-TCN."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from conftest import make_label_row

from phase_classification.data import surgclip_features
from phase_classification.heads import HEAD_TUPLE_INDEX
from phase_classification.losses import MSTCNLoss
from phase_classification.models.ms_tcn import CachedFeatureMSTCN


def _video(tmp_path: Path, name: str, phase_ids: tuple[int, int]):
    return SimpleNamespace(
        img_dir=str(tmp_path / "videos" / name),
        frames=["2", "1"],
        label_data={
            "1": [make_label_row(phase=phase_ids[0])],
            "2": [make_label_row(phase=phase_ids[1])],
        },
    )


def test_build_feature_dataloaders_aligns_flat_cache_by_video_and_frame(
    tmp_path, monkeypatch
):
    train = _video(tmp_path, "VID01", (0, 1))
    val = _video(tmp_path, "VID02", (2, 3))
    test = _video(tmp_path, "VID03", (4, 5))
    split = SimpleNamespace(
        train_dataset=SimpleNamespace(datasets=[train]),
        val_dataset=SimpleNamespace(datasets=[val]),
        test_dataset=[test],
    )
    monkeypatch.setattr(surgclip_features, "CholecT50", lambda **kwargs: split)

    paths = [
        "/cache/VID01/000002.png",
        "/cache/VID01/000001.png",
        "/cache/VID02/000002.png",
        "/cache/VID02/000001.png",
        "/cache/VID03/000002.png",
        "/cache/VID03/000001.png",
    ]
    features = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    cache = tmp_path / "features.pt"
    torch.save(
        {
            "features": features,
            "paths": paths,
            "variant": "cholect50-crossval",
            "test_fold": 1,
        },
        cache,
    )

    loaders = surgclip_features.build_feature_dataloaders(
        tmp_path, cache, variant="cholect50-crossval", test_fold=1
    )
    batch_features, labels = next(iter(loaders.train))
    assert loaders.feature_dim == 3
    assert loaders.videos == {
        "train": ["VID01"],
        "validation": ["VID02"],
        "test": ["VID03"],
    }
    torch.testing.assert_close(
        batch_features[0], torch.tensor([[3, 4, 5], [0, 1, 2]], dtype=torch.float32)
    )
    assert labels[HEAD_TUPLE_INDEX["phase"]].argmax(dim=-1).tolist() == [[0, 1]]
    assert len(loaders.test) == 1


def test_cache_reader_accepts_per_video_diffusion_artifact():
    cache = {
        "splits": {
            "train": [
                {"video_id": "VID01", "embeddings": torch.ones(2, 4)},
                {"video_id": "VID02", "embeddings": torch.zeros(3, 4)},
            ]
        }
    }
    result = surgclip_features._cache_video_features(cache)
    assert result["VID01"].shape == (2, 4)
    assert result["VID02"].shape == (3, 4)


def test_cached_feature_mstcn_uses_shared_mstcn_loss():
    model = CachedFeatureMSTCN(
        feature_dim=8,
        num_classes={"phase": 7},
        heads=["phase"],
        num_stages=2,
        num_layers=2,
        num_f_maps=16,
        dropout=0.5,
    )
    features = torch.randn(1, 12, 8)
    phase_labels = torch.nn.functional.one_hot(torch.randint(0, 7, (1, 12)), 7).float()
    labels = (
        torch.zeros(1, 12, 100),
        torch.zeros(1, 12, 6),
        torch.zeros(1, 12, 10),
        torch.zeros(1, 12, 15),
        phase_labels,
    )

    logits = model(features)
    loss = MSTCNLoss(model, ["phase"])(logits, labels)["total"]
    loss.backward()
    assert logits["phase"].shape == (1, 12, 7)
    assert len(model.stage_logits["phase"]) == 2
    assert any(
        isinstance(layer, torch.nn.Dropout) and layer.p == 0.5
        for layer in model.modules()
    )
    assert torch.isfinite(loss)
