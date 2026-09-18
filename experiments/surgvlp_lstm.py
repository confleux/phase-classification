#!/usr/bin/env python3
"""Extract SurgVLP/PeskaVLP frame embeddings, run zero-shot phase recognition,
and optionally train a per-frame LSTM head.

Usage:
    python surgvlp_lstm.py --dataset-dir /data/cholect50 [options]

Pipeline
    Stage 1 (embeddings): loads the official CAMMA SurgVLP vision tower
    (ResNet-50 + global projection to 768 dims) from the auto-downloaded (or
    --weights) checkpoint, runs every frame of the chosen CholecT50 split
    through it, and caches {"features": [N, 768], "paths": [...]} on disk.

    Stage 2 (zero-shot, --zero-shot): classifies the test frames with the
    pretrained text tower (Bio_ClinicalBERT) + the 7 class prompts, i.e. no
    training at all. Requires ``transformers`` (pip install transformers) and
    downloads Bio_ClinicalBERT on first use. Exits after evaluation.

    Stage 3 (PeskaVLP + BiLSTM + classification head, default): builds contiguous frame windows
    (--sequence-length) over the cached PeskaVLP embeddings, trains a bidirectional LSTM with
    a per-frame classification head on the phase class, trained with class-balanced cross-entropy, and reports
    metrics computed with ``phase_classification.metrics.MetricAccumulator``
    (edit_score is the mean over per-video test sequences).

The model is the self-contained vision tower from CAMMA-public/SurgVLP
(MedIA'25 / HecVL MICCAI'24 / PeskaVLP NeurIPS'24). Only the ``backbone_img.*``
weights are used for extraction, so the LSTM stage needs no mmengine/transformers
dependency. The zero-shot stage additionally ports the ``backbone_text.*``
(BertEncoder) tower.

Example
    python surgvlp_lstm.py --dataset-dir /data/cholect50 --model peskavlp
    python surgvlp_lstm.py --dataset-dir /data/cholect50 --zero-shot
    python surgvlp_lstm.py --dataset-dir /data/cholect50 \
        --embeddings-pt outputs/embeddings/peskavlp-2026-09-18/peskavlp_features.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torchvision import models as vision_models
from torchvision import transforms
from tqdm import tqdm


# --- Make the project's src/ package importable regardless of cwd -----------
def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` to the folder containing src/phase_classification."""
    for parent in [start, *start.parents]:
        if (parent / "src" / "phase_classification").is_dir():
            return parent
    raise RuntimeError(
        "Could not locate the phase_classification repo root "
        "(no src/phase_classification found above cwd)."
    )


REPO_ROOT = _find_repo_root(Path.cwd())
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from phase_classification.data.cholec_t50 import CholecT50
from phase_classification.engine import resolve_device
from phase_classification.heads import DEFAULT_NUM_CLASSES, HEAD_TUPLE_INDEX
from phase_classification.metrics import MetricAccumulator

EMBED_DIM = 768

# CholecT50 phase names (index == phase class id), used for zero-shot output.
PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

# One class prompt per phase, verbatim from CAMMA-public/SurgVLP tests/class_prompt.txt.
CLASS_PROMPTS = [
    "In preparation phase I insert trocars to patient abdomen cavity",
    "In calot triangle dissection phase I use grasper to hold gallbladder and use hook to expose the hepatic triangle area and cystic duct and cystic artery",
    "In clip and cut phase I use clipper to clip the cystic duct and artery then use scissor to cut them",
    "In dissection phase I use the hook to dissect the connective tissue between gallbladder and liver",
    "In packaging phase I put the gallbladder into the specimen bag",
    "In clean and coagulation phase I use suction and irrigation to clear the surgical field and coagulate bleeding vessels",
    "In retraction phase I grasp the specimen bag and remove it from trocar",
]

MODEL_DOWNLOADS = {
    "surgvlp": "https://seafile.unistra.fr/f/93757ace1bfc47248e1e/?dl=1",
    "hecvl": "https://seafile.unistra.fr/f/3b9b9207068a4b03bc2a/?dl=1",
    "peskavlp": "https://seafile.unistra.fr/f/65a2b1bf113e428280d0/?dl=1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, help="CholecT50 root with videos/ and labels/")
    p.add_argument("--variant", default="cholect50-crossval", help="CholecT50 split variant")
    p.add_argument("--test-fold", type=int, default=1, help="Cross-validation fold for test")
    p.add_argument(
        "--model",
        default="peskavlp",
        choices=sorted(MODEL_DOWNLOADS),
        help="Pretrained SurgVLP variant (same ResNet-50 vision tower for all)",
    )
    p.add_argument("--weights", default=None, help="Local .pth checkpoint (skips auto-download)")
    p.add_argument("--download-root", default=str(Path.home() / ".cache" / "surgvlp"))
    p.add_argument(
        "--embedding-dir",
        default=str(REPO_ROOT / "outputs" / "embeddings"),
        help="Where extracted embeddings are saved",
    )
    p.add_argument(
        "--embeddings-pt", default=None, help="Reuse an existing features .pt (skips extraction)"
    )
    p.add_argument("--extract-only", action="store_true", help="Stop after embedding extraction")
    p.add_argument(
        "--zero-shot",
        action="store_true",
        help="Run zero-shot phase classification (pretrained text tower; needs transformers), then exit",
    )
    p.add_argument(
        "--class-prompt",
        default=None,
        help="Path to a text file with one class prompt per line (default: embedded prompts)",
    )
    p.add_argument("--sequence-length", type=int, default=16, help="LSTM window size")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=128, help="LSTM hidden size")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--no-bidirectional", action="store_true", help="Disable bidirectional LSTM")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--heads", default="phase", help="Comma-separated heads (default: phase)")
    p.add_argument(
        "--metrics",
        default="jaccard,f1_macro,mean_accuracy,edit_score",
        help="Comma-separated metric names",
    )
    p.add_argument("--device", default="auto", help="auto|cpu|cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "surgvlp-lstm"))
    return p.parse_args()


# --- Stage 1: SurgVLP vision tower -----------------------------------------
def _build_resnet50() -> nn.Module:
    """torchvision resnet50 without pretrained weights (pretrained= was removed)."""
    try:
        return vision_models.resnet50(weights=None)
    except TypeError:
        return vision_models.resnet50(pretrained=False)


class SurgVLPImageEncoder(nn.Module):
    """Vision tower of CAMMA SurgVLP: ResNet-50 -> pool -> Linear(2048, 768).

    Mirrors ``surgvlp/codes/models/backbones/img_backbones.py::ImageEncoder``
    with the modern torchvision API. The official checkpoint's ``backbone_img.*``
    keys load straight into this module.
    """

    def __init__(self, num_classes: int = EMBED_DIM, img_norm: bool = False):
        super().__init__()
        self.model = _build_resnet50()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.global_embedder = nn.Linear(in_features, num_classes)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.norm = img_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        emb = self.global_embedder(x)
        if self.norm:
            emb = emb / emb.norm(dim=1, keepdim=True)
        return emb


def download_weights(model_name: str, root: str) -> str:
    """Fetch and unpack the official weights (mirrors ``surgvlp._download``)."""
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, f"{model_name}.zip")
    if not os.path.isfile(target):
        print(f"Downloading {model_name} weights -> {target}")
        subprocess.run(
            ["wget", "--content-disposition", "-P", root, MODEL_DOWNLOADS[model_name]],
            check=True,
        )
    zips = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".zip")]
    if zips and zipfile.is_zipfile(zips[0]):
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(root)
    pths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".pth")]
    if not pths:
        raise RuntimeError(f"No .pth found under {root} after download.")
    named = [p for p in pths if Path(p).stem.lower() == model_name]
    return (named or pths)[0]


def load_surgvlp(weights_path: str, device: torch.device) -> SurgVLPImageEncoder:
    """Build the vision tower and load the official ``backbone_img.*`` weights."""
    encoder = SurgVLPImageEncoder().to(device)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    prefixed = [k for k in state if k.startswith("backbone_img.")]
    if prefixed:
        state = {k[len("backbone_img."):]: v for k, v in state.items() if k.startswith("backbone_img.")}
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys loading SurgVLP weights: {missing[:10]}")
    if unexpected:
        print(f"Ignored {len(unexpected)} non-vision keys (e.g. backbone_text.*).")
    encoder.eval()
    print(f"Loaded SurgVLP vision tower from {weights_path} "
          f"({sum(p.numel() for p in encoder.parameters()):,} params)")
    return encoder


def surgvlp_preprocess() -> transforms.Compose:
    """The exact input transform used by the SurgVLP zero-shot eval."""
    return transforms.Compose(
        [
            transforms.Resize((360, 640)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# --- Stage 2: zero-shot evaluation (requires transformers) -------------------
# Faithful port of the text tower from CAMMA-public/SurgVLP
# (surgvlp/codes/models/backbones/text_backbones.py::BertEncoder), CC BY-NC-SA 4.0,
# non-commercial research use. The official checkpoint's ``backbone_text.*``
# keys load straight into this module.
class SurgVLPTextEncoder(nn.Module):
    """Bio_ClinicalBERT encoder with the SurgVLP token-sum aggregation."""

    def __init__(
        self,
        bert_type: str = "emilyalsentzer/Bio_ClinicalBERT",
        last_n_layers: int = 4,
        aggregate_method: str = "sum",
        norm: bool = False,
        embedding_dim: int = EMBED_DIM,
        freeze_bert: bool = False,
        agg_tokens: bool = True,
    ):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.bert_type = bert_type
        self.last_n_layers = last_n_layers
        self.aggregate_method = aggregate_method
        self.norm = norm
        self.embedding_dim = embedding_dim
        self.freeze_bert = freeze_bert
        self.agg_tokens = agg_tokens
        self.model = AutoModel.from_pretrained(self.bert_type, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_type)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        if self.freeze_bert:
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):
        _, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []
        for embs, caption_id in zip(embeddings, caption_ids):
            agg_embs = []
            token_bank = []
            words = []
            word_bank = []
            for word_emb, word_id in zip(embs, caption_id):
                word = self.idxtoword[word_id.item()]
                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))
                    agg_embs.append(word_emb)
                    words.append(word)
                    break
                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))
                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    token_bank.append(word_emb)
                    word_bank.append(word[2:])
            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size
            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)
        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids=None, attn_mask=None, token_type=None, input_embedding=None):
        if input_embedding is not None:
            outputs = self.model(inputs_embeds=input_embedding, attention_mask=attn_mask)
        else:
            outputs = self.model(ids, attn_mask, token_type)
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(all_embeddings[-self.last_n_layers :])
            embeddings = embeddings.permute(1, 0, 2, 3)
            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]
            sent_embeddings = embeddings.mean(axis=2)
            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1)
                sent_embeddings = sent_embeddings.sum(axis=1)
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                raise RuntimeError(f"Aggregation method not implemented: {self.aggregate_method}")
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]
            sents = None
        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)
        if self.norm:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)
        return word_embeddings, sent_embeddings, sents


def tokenize_surgvlp(texts: list[str], device: torch.device, max_length: int = 77) -> dict:
    """Tokenize class prompts exactly like ``surgvlp.tokenize``."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    processed = [
        tokenizer(
            t, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length
        )
        for t in texts
    ]
    caption_ids = torch.stack([x["input_ids"] for x in processed]).squeeze().to(device)
    attention_mask = torch.stack([x["attention_mask"] for x in processed]).squeeze().to(device)
    token_type_ids = torch.stack([x["token_type_ids"] for x in processed]).squeeze().to(device)
    return {
        "input_ids": caption_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def load_surgvlp_text(weights_path: str, device: torch.device) -> SurgVLPTextEncoder:
    """Build the text tower and load the official ``backbone_text.*`` weights."""
    text_encoder = SurgVLPTextEncoder().to(device)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    prefixed = [k for k in state if k.startswith("backbone_text.")]
    if not prefixed:
        raise RuntimeError(f"No 'backbone_text.*' keys found in {weights_path}.")
    state = {
        k[len("backbone_text."):]: v
        for k, v in state.items()
        if k.startswith("backbone_text.")
    }
    missing, unexpected = text_encoder.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys loading SurgVLP text tower: {missing[:10]}")
    if unexpected:
        print(f"Ignored {len(unexpected)} unexpected text-tower keys.")
    text_encoder.eval()
    print(f"Loaded SurgVLP text tower from {weights_path} "
          f"({sum(p.numel() for p in text_encoder.parameters()):,} params)")
    return text_encoder


def _read_prompts(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _require_transformers() -> None:
    """Raise a clear error when ``transformers`` (zero-shot dependency) is absent."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "--zero-shot requires the 'transformers' package: pip install transformers "
            "(also downloads Bio_ClinicalBERT on first run)."
        )


def run_zero_shot(
    features: torch.Tensor,
    paths: list[str],
    dataset_dir: str,
    test_records: list[str],
    weights_path: str,
    device: torch.device,
    class_prompts: list[str],
    metrics: list[str],
    output_dir: Path,
) -> None:
    """Zero-shot phase classification over the CholecT50 test split.

    Uses the cached image embeddings (L2-normalised externally, like the
    official ``zero_shot.py``) and the pretrained text tower on the class
    prompts. Metrics are computed per video with ``MetricAccumulator`` and
    merged (edit_score = mean over videos); the repo-style mean per-video
    accuracy is printed as well.
    """
    heads = ["phase"]
    num_classes = {h: DEFAULT_NUM_CLASSES[h] for h in heads}

    text_encoder = load_surgvlp_text(weights_path, device)
    with torch.no_grad():
        tokens = tokenize_surgvlp(class_prompts, device)
        _, text_feats, _ = text_encoder(
            tokens["input_ids"], tokens["attention_mask"], tokens["token_type_ids"]
        )
    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.to(device)

    lookup: dict[tuple[str, str], int] = {}
    for idx, path in enumerate(paths):
        p = Path(path)
        lookup[(p.parent.name, p.stem)] = idx

    per_video_acc = []
    accumulators = []
    for video in test_records:
        label_file = os.path.join(dataset_dir, "labels", f"{video}.json")
        label_data = _load_labels(label_file)
        frame_keys = sorted(label_data, key=int)
        probs, targets = [], []
        correct = 0
        for frame in frame_keys:
            idx = lookup[(video, str(frame).zfill(6))]
            emb = features[idx].to(device)
            p = (100.0 * emb @ text_feats.T).softmax(dim=-1)
            target = torch.tensor(_phase_onehot(label_data[frame])).bool().int()
            probs.append(p.cpu())
            targets.append(target)
            if p.argmax().item() == target.argmax().item():
                correct += 1
        probs = torch.stack(probs)
        targets = torch.stack(targets)
        per_video_acc.append(correct / len(frame_keys))

        acc = MetricAccumulator(heads, metrics, num_classes)
        acc.add_batch(
            {"phase": probs.numpy()},
            {"phase": targets.numpy()},
        )
        accumulators.append(acc)

    merged = MetricAccumulator.merged(accumulators).compute()
    mean_video_acc = float(np.mean(per_video_acc))

    print(f"\nZero-shot ({len(test_records)} test videos)")
    print(f"Mean per-video accuracy: {mean_video_acc:.4f}")
    for key, value in sorted(merged.items()):
        print(f"  {key}: {value:.4f}")
    if any(k.startswith(f"{heads[0]}_f1_class_") for k in merged):
        print("Per-class F1:")
        for i, name in enumerate(PHASE_NAMES):
            key = f"{heads[0]}_f1_class_{i}"
            if key in merged:
                print(f"  {name}: {merged[key]:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "zero_shot_metrics.json", "w") as f:
        json.dump(
            {
                "test_records": test_records,
                "class_prompts": class_prompts,
                "mean_video_accuracy": mean_video_acc,
                "metrics": {k: float(v) for k, v in merged.items()},
            },
            f,
            indent=2,
        )
    print("Saved zero-shot metrics:", output_dir / "zero_shot_metrics.json")


@torch.no_grad()
def extract_embeddings(
    video_datasets: list[Dataset],
    encoder: nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[str]]:
    """Embed every frame of every video in ``video_datasets`` in order."""
    features: list[torch.Tensor] = []
    paths: list[str] = []
    for ds in tqdm(video_datasets, desc="extract"):
        frames = sorted(ds.frames, key=int)
        buffer: list[torch.Tensor] = []
        for frame in frames:
            basename = f"{str(frame).zfill(6)}.png"
            img_path = os.path.join(ds.img_dir, basename)
            paths.append(img_path)
            buffer.append(preprocess(Image.open(img_path).convert("RGB")))
            if len(buffer) == batch_size:
                emb = encoder(torch.stack(buffer).to(device))
                features.append(emb.detach().cpu())
                buffer = []
        if buffer:
            emb = encoder(torch.stack(buffer).to(device))
            features.append(emb.detach().cpu())
    return torch.cat(features, dim=0), paths


# --- Stage 2: sequence feature dataset (mirrors the DINOv2 notebook) --------
def _load_labels(label_file: str) -> dict:
    with open(label_file, "rb") as f:
        return json.load(f)["annotations"]


def _phase_onehot(label_rows: list) -> np.ndarray:
    """7-class phase one-hot vector from the raw 15-column label rows."""
    phase_label = np.zeros([7])
    for label in label_rows:
        phase = label[14:15]
        if phase[0] != -1.0:
            phase_label[phase[0]] += 1
    return phase_label


class CholecT50Feature:
    """Feature-based CholecT50 loader building per-video datasets over cached
    embeddings. ``paths`` maps each feature row to its ``.../VIDxx/000001.png``.
    """

    def __init__(
        self,
        dataset_dir: str,
        features: torch.Tensor,
        paths: list[str],
        train_records: list[str],
        val_records: list[str],
        test_records: list[str],
        sequence_length: int = 1,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.dataset_dir = dataset_dir
        self.features = features
        self.sequence_length = sequence_length
        self.lookup: dict[tuple[str, str], int] = {}
        for idx, path in enumerate(paths):
            p = Path(path)
            self.lookup[(p.parent.name, p.stem)] = idx
        self.train_records = list(train_records)
        self.val_records = list(val_records)
        self.test_records = list(test_records)

    def _make_video_dataset(self, video: str) -> Dataset:
        label_file = os.path.join(self.dataset_dir, "labels", f"{video}.json")
        if self.sequence_length > 1:
            return T50SequenceFeature(
                video_id=video,
                features=self.features,
                lookup=self.lookup,
                label_file=label_file,
                window_size=self.sequence_length,
            )
        return T50Feature(
            video_id=video,
            features=self.features,
            lookup=self.lookup,
            label_file=label_file,
        )

    def _build_split(self, records: list[str]) -> ConcatDataset:
        datasets = [self._make_video_dataset(v) for v in records]
        datasets = [d for d in datasets if len(d) > 0]
        if not datasets:
            raise ValueError("Split dataset is empty (check videos and sequence_length).")
        return ConcatDataset(datasets)

    def build(self) -> tuple[ConcatDataset, ConcatDataset, list[Dataset]]:
        train = self._build_split(self.train_records)
        val = self._build_split(self.val_records)
        test = [self._make_video_dataset(v) for v in self.test_records]
        test = [d for d in test if len(d) > 0]
        return train, val, test


class T50Feature(Dataset):
    """Single-frame feature dataset for one video."""

    def __init__(self, video_id, features, lookup, label_file):
        self.label_data = _load_labels(label_file)
        self.frames = list(self.label_data.keys())
        self.video_id = video_id
        self.features = features
        self.lookup = lookup

    def __len__(self):
        return len(self.frames)

    def get_binary_labels(self, labels):
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])
        phase_label = np.zeros([7])
        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0:
                triplet_label[triplet[0]] += 1
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] += 1
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] += 1
            target = label[8:14]
            if target[0] != -1.0:
                target_label[target[0]] += 1
            phase = label[14:15]
            if phase[0] != -1.0:
                phase_label[phase[0]] += 1
        return (triplet_label, tool_label, verb_label, target_label, phase_label)

    def __getitem__(self, index):
        frame_key = self.frames[index]
        labels = self.label_data[frame_key]
        feature = self.features[self.lookup[(self.video_id, str(frame_key).zfill(6))]]
        return feature, tuple(
            torch.tensor(x).bool().int() for x in self.get_binary_labels(labels)
        )


class T50SequenceFeature(Dataset):
    """Contiguous feature windows from one video for temporal models (LSTM)."""

    def __init__(self, video_id, features, lookup, label_file, window_size):
        if window_size < 2:
            raise ValueError("T50SequenceFeature requires window_size >= 2")
        self.label_data = _load_labels(label_file)
        self.frames = sorted(self.label_data.keys(), key=int)
        self.video_id = video_id
        self.features = features
        self.lookup = lookup
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.frames) - self.window_size + 1)

    def get_binary_labels(self, labels):
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])
        phase_label = np.zeros([7])
        for label in labels:
            triplet = label[0:1]
            if triplet[0] != -1.0:
                triplet_label[triplet[0]] += 1
            tool = label[1:7]
            if tool[0] != -1.0:
                tool_label[tool[0]] += 1
            verb = label[7:8]
            if verb[0] != -1.0:
                verb_label[verb[0]] += 1
            target = label[8:14]
            if target[0] != -1.0:
                target_label[target[0]] += 1
            phase = label[14:15]
            if phase[0] != -1.0:
                phase_label[phase[0]] += 1
        return (triplet_label, tool_label, verb_label, target_label, phase_label)

    def __getitem__(self, index):
        frame_features = []
        stacked_components: list[list[torch.Tensor]] = [[] for _ in range(5)]
        for j in range(self.window_size):
            frame_key = self.frames[index + j]
            labels = self.label_data[frame_key]
            feature = self.features[self.lookup[(self.video_id, str(frame_key).zfill(6))]]
            frame_features.append(feature)
            for k, tensor in enumerate(self.get_binary_labels(labels)):
                stacked_components[k].append(torch.tensor(tensor).bool().int())
        features = torch.stack(frame_features, dim=0)
        labels = tuple(torch.stack(parts, dim=0) for parts in stacked_components)
        return features, labels


# --- Stage 3: PeskaVLP embeddings -> BiLSTM -> classification head ----------
class ClassificationHead(nn.Module):
    """Per-frame classification head over the BiLSTM outputs."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.dropout(x))


class PeskaVLPBiLSTM(nn.Module):
    """Frozen PeskaVLP frame embeddings -> BiLSTM -> classification head.

    The PeskaVLP vision tower (ResNet-50 + global projection to 768 dims)
    is run once at extraction time (Stage 1) and cached; its weights are frozen.
    This module consumes the cached per-frame embeddings [B, L, 768] and
    emits per-frame logits [B, L,,num_classes].
    """

    def __init__(
        self,
        feat_dim: int,
        lstm_hidden: int,
        num_layers: int,
        bidirectional: bool,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            feat_dim,
            lstm_hidden,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = ClassificationHead(
            in_features=lstm_hidden * (2 if bidirectional else 1),
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.lstm(x)
        return self.head(seq)


def collate_fn(batch):
    """(B, L, feat_dim) features and list of 5 stacked (B, L, K) label tensors."""
    feats, labels = zip(*batch)
    feats = torch.stack(list(feats))
    label_tensors = [torch.stack([l[i] for l in labels]) for i in range(5)]
    return feats, label_tensors


class VideoBatchSampler(Sampler):
    """Yield batches where every sample comes from the SAME video.

    Each sub-dataset in the ConcatDataset is one video, so this groups indices
    by video to keep temporal coherence within a batch.
    """

    def __init__(self, concat_ds, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.video_ranges = []
        start = 0
        for size in concat_ds.cumulative_sizes:
            self.video_ranges.append((start, size))
            start = size
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        for start, end in self.video_ranges:
            idxs = list(range(start, end))
            if self.shuffle:
                np.random.shuffle(idxs)
            for i in range(0, len(idxs), self.batch_size):
                batches.append(idxs[i : i + self.batch_size])
        if self.shuffle:
            np.random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _phase_counts(concat_ds, num_phases: int) -> torch.Tensor:
    counts = torch.zeros(num_phases)
    for sub in concat_ds.datasets:
        label_data = sub.label_data
        for fk in sub.frames:
            for label in label_data[fk]:
                p = label[14:15]
                if p[0] != -1.0:
                    counts[int(p[0])] += 1
    return counts


def run_epoch(
    loader,
    model: nn.Module,
    criterion,
    optimizer,
    device: torch.device,
    head_index: int,
    num_classes: int,
    accumulator: MetricAccumulator,
    train: bool,
    desc: str = "",
) -> tuple[float, int]:
    model.train() if train else model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, desc=desc, leave=False)
        for feats, label_tensors in pbar:
            feats = feats.to(device)
            phase = label_tensors[head_index].to(device)
            logits = model(feats)
            loss = criterion(
                logits.reshape(-1, num_classes),
                phase.argmax(dim=-1).reshape(-1),
            )
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * feats.size(0)
            n_samples += feats.size(0)
            accumulator.add_batch(
                {"phase": logits.detach().cpu().numpy()},
                {"phase": phase.detach().cpu().numpy()},
            )
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / n_samples, n_samples


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print("Device:", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not heads or not metrics:
        raise ValueError("--heads and --metrics must be non-empty")
    if any(h not in DEFAULT_NUM_CLASSES for h in heads):
        raise ValueError(f"Unknown head(s): {heads}. Available: {sorted(DEFAULT_NUM_CLASSES)}")
    num_classes = {h: DEFAULT_NUM_CLASSES[h] for h in heads}

    # --- Stage 1: embeddings ------------------------------------------------
    split = CholecT50(
        dataset_dir=args.dataset_dir,
        dataset_variant=args.variant,
        test_fold=args.test_fold,
        augmentation_list=["original"],
        normalize=True,
        sequence_length=1,
    )

    if args.embeddings_pt:
        data = torch.load(args.embeddings_pt, map_location="cpu", weights_only=False)
        features, paths = data["features"], list(data["paths"])
        print(f"Loaded cached embeddings: {tuple(features.shape)} from {args.embeddings_pt}")
    else:
        weights_path = args.weights or download_weights(args.model, args.download_root)
        encoder = load_surgvlp(weights_path, device)
        preprocess = surgvlp_preprocess()

        video_datasets = (
            list(split.train_dataset.datasets)
            + list(split.val_dataset.datasets)
            + list(split.test_dataset)
        )
        seen, unique = set(), []
        for ds in video_datasets:
            if getattr(ds, "img_dir", "") in seen:
                continue
            seen.add(ds.img_dir)
            unique.append(ds)
        print(
            f"Extracting embeddings over {len(unique)} videos "
            f"({args.model}, {args.variant}, fold {args.test_fold})"
        )
        features, paths = extract_embeddings(
            unique, encoder, preprocess, device, args.batch_size
        )
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        out_dir = Path(args.embedding_dir) / f"{args.model}-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_pt = out_dir / f"{args.model}_features.pt"
        torch.save(
            {
                "features": features,
                "paths": paths,
                "model": args.model,
                "variant": args.variant,
                "test_fold": args.test_fold,
            },
            out_pt,
        )
        print(f"Saved embeddings: {out_pt} ({tuple(features.shape)})")
        if args.extract_only:
            return

    if features.shape[-1] != EMBED_DIM:
        print(f"Note: embedding dim is {features.shape[-1]} (SurgVLP default is {EMBED_DIM}).")

    # --- Stage 2: zero-shot evaluation (optional, exits) --------------------
    if args.zero_shot:
        _require_transformers()
        weights_path = args.weights or download_weights(args.model, args.download_root)
        prompts = _read_prompts(args.class_prompt) if args.class_prompt else list(CLASS_PROMPTS)
        run_zero_shot(
            features,
            paths,
            args.dataset_dir,
            split.test_records,
            weights_path,
            device,
            prompts,
            metrics,
            Path(args.output_dir),
        )
        return

    # --- Stage 3: LSTM over the cached features -----------------------------
    feat_ds = CholecT50Feature(
        dataset_dir=args.dataset_dir,
        features=features,
        paths=paths,
        train_records=split.train_records,
        val_records=split.val_records,
        test_records=split.test_records,
        sequence_length=args.sequence_length,
    )
    train_ds, val_ds, test_ds = feat_ds.build()
    print(f"train: {len(train_ds)} val: {len(val_ds)} test: {len(test_ds)}")

    model = PeskaVLPBiLSTM(
        feat_dim=features.shape[-1],
        lstm_hidden=args.hidden,
        num_layers=args.layers,
        bidirectional=not args.no_bidirectional,
        num_classes=num_classes[heads[0]],
        dropout=args.dropout,
    ).to(device)
    print(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    phase_counts = _phase_counts(train_ds, num_classes[heads[0]])
    phase_freq = (phase_counts / phase_counts.sum()).clamp(min=1e-9)
    class_weights = 1.0 / phase_freq
    class_weights = class_weights / class_weights.mean()
    class_weights = class_weights.to(device)
    print("Per-class phase counts:", phase_counts.int().tolist())
    print("Used class weights:    ", [f"{w:.2f}" for w in class_weights.cpu().tolist()])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    head_idx = HEAD_TUPLE_INDEX[heads[0]]

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_sampler=VideoBatchSampler(ds, args.batch_size, shuffle=shuffle),
            collate_fn=collate_fn,
            num_workers=0,
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    best_primary, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_acc = MetricAccumulator(heads, metrics, num_classes)
        val_acc = MetricAccumulator(heads, metrics, num_classes)
        train_loss, _ = run_epoch(
            train_loader, model, criterion, optimizer, device, head_idx, num_classes[heads[0]],
            train_acc, train=True, desc=f"train {epoch}/{args.epochs}",
        )
        val_loss, _ = run_epoch(
            val_loader, model, criterion, optimizer, device, head_idx, num_classes[heads[0]],
            val_acc, train=False, desc=f"val   {epoch}/{args.epochs}",
        )
        train_metrics, val_metrics = train_acc.compute(), val_acc.compute()
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        print(f"  train: {train_metrics}")
        print(f"  val:   {val_metrics}")
        primary = val_metrics.get(f"{heads[0]}_{metrics[0]}", -1.0)
        if primary > best_primary:
            best_primary = primary
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    test_losses, test_accs = [], []
    for video_ds in test_ds:
        loader = DataLoader(
            video_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        acc = MetricAccumulator(heads, metrics, num_classes)
        loss, _ = run_epoch(
            loader, model, criterion, optimizer, device, head_idx, num_classes[heads[0]],
            acc, train=False, desc="test",
        )
        test_losses.append(loss)
        test_accs.append(acc)
    test_metrics = MetricAccumulator.merged(test_accs).compute()
    print(f"\nTest loss={np.mean(test_losses):.4f}")
    print(f"Test metrics: {test_metrics}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, out_dir / "best.pt")
        print("Saved best checkpoint:", out_dir / "best.pt")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "test_loss": float(np.mean(test_losses)),
                "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            },
            f,
            indent=2,
        )
    print("Saved metrics:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()