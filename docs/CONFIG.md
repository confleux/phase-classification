# Configuration Specification

Every configuration is a **YAML file** (see `configs/default.yaml`). It is parsed
by `src/phase_classification/loader.py`, schema-validated against the pydantic
models in `src/phase_classification/config/*.py`, and passed to the CLI.

**Schema is strict** (`extra="forbid"`): adding a field that isn't defined below
raises a validation error.

## How to use a config

```bash
# Validate schema only (no disk access)
uv run phase-cls validate-config configs/my.yaml

# Validate schema + verify dataset paths on disk
uv run phase-cls validate-config configs/my.yaml --check-paths

# Inspect the resolved config, optionally in a different format
uv run phase-cls inspect configs/my.yaml [--format yaml|json|mlflow]

# Print dataset split sizes + a first batch shape (needs real data)
uv run phase-cls build-dataloaders configs/my.yaml

# Train + validate + test in one run, logging to MLflow (needs real data)
uv run phase-cls train configs/my.yaml
```

You only need to write the fields you want to override. Everything else falls
back to the schema defaults below.

> **Note.** Every field below is wired into the code — there are no
> schema-only placeholders. Setting an unknown field raises a validation
> error (strict schema).

---

## Top-level sections

The config has 8 top-level sections. All are optional except `dataset`.

| Section | Purpose |
|---|---|
| `project` | run metadata & local outputs |
| `dataset` | data source & preprocessing |
| `dataloader` | torch DataLoader knobs |
| `training` | optimizer / loss / heads / epochs |
| `model` | architecture name + knobs + label space |
| `validation` | which metrics to compute |
| `mlflow` | tracking server & run identity |
| `runtime` | device / precision / seeding |

---

## `project`

| Field | Type | Default | Allowed / notes |
|---|---|---|---|
| `name` | str | `phase-classification` | free-form |
| `output_dir` | Path | `outputs` | best checkpoint → `outputs/checkpoints/best.pt` |
| `save_checkpoints` | bool | `true` | if `false`, skips local checkpoint + `mlflow.pytorch.log_model` |

---

## `dataset` (only required section)

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `dataset_variant` | str | `cholect50-crossval` | `cholect45-crossval`, `cholect50-crossval`, `cholect50-challenge`, `cholect50`, `cholect45`, `cholect50-subset`, `cholect50-small`, `heichole` |
| `root_dir` | Path | — (**required**) | dataset root; expected subdirectories and files depend on the variant (see below) |
| `feature_cache` | Path \| null | `null` | optional SurgCLIP `.pt` feature cache; enables frozen-feature training instead of RGB decoding |
| `test_fold` | int | `1` | `1..5`; official CholecT50 fold, or HeiChole's deterministic local five-fold split |
| `sequence_length` | int | `1` | `1` = single frame; `>=2` = contiguous video-frame windows |
| `augmentations` | list[str] | `["original"]` | `original`, `vflip`, `hflip`, `contrast`, `rot90`, `brightness`, `random_contrast` |
| `normalize` | bool | `true` | ImageNet normalization |

Frames are resized to the loader's fixed **256x448** (`image_size` is not
configurable) for image-based models. Video decoding for HeiChole requires
`opencv-python-headless` to be installed in the runtime environment.

When `feature_cache` is set, use `surgclip-ms-tcn`, `surgclip-ms-tcn-diffusion`,
or `surgclip-ms-tcnpp-diffusion`
with `training.heads: [phase]` and `dataloader.batch_size: 1`. The cache can be
the flat `features`/`paths` artifact or the per-video `splits` artifact produced
by the SurgCLIP scripts. The loader checks video/frame alignment against
CholecT50 labels and batches one complete video at a time. Images and the
SurgCLIP encoder are not loaded during these frozen-feature experiments.

`cholect*` variants use the vendored CholecT50 loader and its existing data
layout. `heichole` uses this local layout (the challenge's data page describes
the videos and annotations, but does not prescribe extracted archive filenames):

```text
<root_dir>/
  videos/<case-id>.mp4
  labels/<case-id>_Phase.csv
```

HeiChole accepts OpenCV-supported video extensions and also accepts
`labels/<case-id>.csv`. For this adapter, CSVs must have `Frame,Phase` columns,
zero-based frame indices, and phase IDs `0..6`. The challenge documents this
column format for phase results, but does not prescribe the downloaded training
archive's file layout; rename or convert annotations if needed. The loader
assigns matched cases to deterministic five folds: `test_fold` is held out, the
next fold is used for validation, and the other three folds are used for
training. This is a local split, not an official HeiChole
train/validation/test partition. The current adapter supervises the seven-class
`phase` head only; use `training.heads: [phase]` (the default). Action,
instrument, and skill annotations are not currently loaded.

The HeiChole challenge describes at least 30 laparoscopic cholecystectomy
videos from three hospitals, with framewise phase labels and optional
action/instrument/skill labels. Its public training data was released through
[Synapse](https://www.synapse.org/#!Synapse:syn18824884/wiki/592586); the
challenge's test set is not publicly released. The official phase CSV format
is documented on the [submission page](https://www.synapse.org/#!Synapse:syn18824884/wiki/592587).

---

## `dataloader`

| Field | Type | Default | Notes |
|---|---|---|---|
| `batch_size` | int | `32` | **the effective mini-batch size** |
| `num_workers` | int | `4` | DataLoader workers |
| `shuffle` | bool | `true` | train only; val/test forced to `false` |
| `pin_memory` | bool | `true` | |
| `prefetch_factor` | int \| null | `2` | ignored when `num_workers == 0` |
| `persistent_workers` | bool | `true` | effective only when `num_workers > 0` |
| `drop_last` | bool | `false` | |

---

## `training`

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `epochs` | int | `50` | `>= 1` (mini-batch size comes from `dataloader.batch_size`) |
| `optimizer` | str | `adamw` | `sgd`, `adam`, `adamw` |
| `learning_rate` | float | `0.001` | `> 0` |
| `weight_decay` | float | `0.0` | `>= 0` |
| `momentum` | float | `0.9` | `0..1` (used by `sgd`) |
| `scheduler` | str | `none` | `none`, `step`, `cosine`, `plateau` |
| `step_size` | int | `10` | `step` LR drop interval / `plateau` patience |
| `gamma` | float | `0.1` | factor for `step`/`plateau` |
| `loss` | str | `cross_entropy` | `cross_entropy`, `focal` |
| `class_weighting` | str | `none` | `none`; `inverse` = per-class weights ∝ 1/frequency computed from the train split (weighted CE, `pos_weight` in BCE) |
| `heads` | list[str] | `["phase"]` | subset of `phase`, `tool`, `verb`, `target`, `triplet` |
| `grad_clip_norm` | float \| null | `null` | max grad norm; disabled when `null` |

Loss semantics (`losses.py`):
- single-label heads (`phase`, `verb`) → `CE` (or `focal`)
- multi-label heads (`tool`, `target`, `triplet`) → always `BCEWithLogits`
- with `class_weighting: inverse`, weights are computed once from the train
  split at run start (`compute_class_weights`) and applied as CE `weight` /
  BCE `pos_weight` in both `MultiHeadLoss` and `MSTCNLoss`

Schedulers are stepped once per epoch; `plateau` tracks the primary val metric
and uses `mode=max` for accuracy-like metrics.

---

## `model`

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `name` | str | `resnet18` | `resnet18`, `convnext-ms-tcn` (alias `ms-tcn`), `swin-ms-tcn`, `surgclip-ms-tcn` (alias `feature-ms-tcn`), `surgclip-ms-tcn-diffusion`, `surgclip-ms-tcnpp-diffusion`, `swin-cls` (alias `swin-classifier`) |
| `params` | dict | `{}` | architecture-specific parameters; registry models reject unsupported keys |
| `num_classes` | dict[str,int] | see below | keys must be in `phase, tool, verb, target, triplet` |

Default `num_classes` (CholecT50 label space):
```yaml
model:
  num_classes:
    phase: 7
    tool: 6
    verb: 10
    target: 15
    triplet: 100
```

The set of heads that actually get an output layer is `training.heads`. The
`name` is modular through `models/registry.py`.

### `model.params` per architecture

`resnet18` (ResNet18 backbone + classifier heads in `models/resnet18.py`):

| Param | Type | Default | Notes |
|---|---|---|---|
| `in_channels` | int | `3` | replaced `conv1` stem if not 3 |
| `pretrained` | bool | `false` | ImageNet weights (`torchvision.models.ResNet18_Weights.DEFAULT`) |
| `freeze_backbone` | bool | `false` | freeze all backbone weights; only `project` + classifier heads train |
| `features` | int | `512` | feature dim before the classifier heads |

Example (frozen pretrained backbone, phase + tool heads):
```yaml
model:
  name: resnet18
  params: { pretrained: true, freeze_backbone: true }
  num_classes: { phase: 7, tool: 6 }
training:
  heads: [phase, tool]
```

`convnext-ms-tcn` (ConvNeXt frame encoder + multi-stage MS-TCN temporal
head in `models/ms_tcn.py`; **requires `dataset.sequence_length >= 2`**):

| Param | Type | Default | Notes |
|---|---|---|---|
| `variant` | str | `tiny` | ConvNeXt variant: `tiny`/`small`/`base`/`large` |
| `pretrained` | bool | `false` | ImageNet weights for the ConvNeXt encoder |
| `freeze_backbone` | bool | `false` | freeze the ConvNeXt encoder; only the temporal head trains |
| `num_stages` | int | `4` | number of MS-TCN stages |
| `num_layers` | int | `10` | dilated temporal layers per stage |
| `num_f_maps` | int | `64` | hidden channels of the temporal convs |
| `kernel_size` | int | `3` | temporal conv kernel size |

Produces **per-frame** predictions `(B, L, K)`; trained with the MS-TCN loss
(per-stage CE + truncated-MSE smoothing via `MSTCNLoss`). Metrics are computed
per frame. See `configs/ms-tcn.yaml` for a full example.

`swin-ms-tcn` (Swin Transformer frame encoder + multi-stage MS-TCN temporal
head in `models/ms_tcn.py`; **requires `dataset.sequence_length >= 2`**):

| Param | Type | Default | Notes |
|---|---|---|---|
| `variant` | str | `t` | Swin variant: `t`/`s`/`b` |
| `pretrained` | bool | `false` | ImageNet weights for the Swin encoder |
| `freeze_backbone` | bool | `false` | freeze the Swin encoder; only the temporal head trains |
| `num_stages` | int | `4` | number of MS-TCN stages |
| `num_layers` | int | `10` | dilated temporal layers per stage |
| `num_f_maps` | int | `64` | hidden channels of the temporal convs |
| `kernel_size` | int | `3` | temporal conv kernel size |

Same per-frame `(B, L, K)` output and `MSTCNLoss` training as
`convnext-ms-tcn`, but with a Swin encoder. See `configs/swin-ms-tcn.yaml`.

`surgclip-ms-tcn` (alias `feature-ms-tcn`) applies the shared temporal stages
to cached SurgCLIP feature sequences; it has **no image encoder** and requires
`dataset.feature_cache`:

| Param | Type | Default | Notes |
|---|---|---|---|
| `feature_dim` | int | `256` | must match the final dimension in the feature cache |
| `num_stages` | int | `3` | number of MS-TCN stages |
| `num_layers` | int | `8` | dilated temporal layers per stage |
| `num_f_maps` | int | `64` | hidden channels of the temporal convolutions |
| `kernel_size` | int | `3` | temporal convolution kernel size |

Input is one full video of cached features `(1, T, D)`. The model uses the
project's shared `Trainer`, `MSTCNLoss`, `MetricAccumulator`, checkpointing, and
MLflow run lifecycle. Only the `phase` head is currently supported for cached
SurgCLIP probing.

`surgclip-ms-tcn-diffusion` and `surgclip-ms-tcnpp-diffusion` are two-stage
custom run modes in `run.py`. Each trains a cached-feature temporal model as a
coarse phase predictor, then conditions the existing residual diffusion
refiner in `surgclip_diffusion.py` on that model's logits and SurgCLIP features.
Diffusion learns a correction to the coarse phase probabilities; this does
**not** use Meta-SurDiff's bilevel meta-weight objective. Both coarse and
refined metrics are logged to MLflow.

`surgclip-ms-tcnpp-diffusion` uses an MS-TCN++ prediction-generation stage with
dual-dilated layers, followed by probability-only refinement stages. Its
defaults follow the paper's four total stages (one generation stage, three
refinement stages), 11 generation layers, 10 refinement layers, 64 feature maps,
dropout 0.5, and smoothing weight 0.15 on log probabilities. The diffusion
settings are kept the same as the MS-TCN experiment for a more direct
architecture comparison.

In addition to the MS-TCN parameters above, `model.params` accepts:

| Param | Default | Notes |
|---|---:|---|
| `refiner_epochs` / `refiner_steps_per_epoch` | `20` / `100` | Diffusion training schedule |
| `refiner_batch_size` / `refiner_crop_length` | `8` / `256` | Video-crop batch and temporal length |
| `refiner_width` / `refiner_layers` | `96` / `6` | Temporal denoiser size |
| `diffusion_timesteps` / `sampling_steps` | `1000` / `25` | Forward and reverse diffusion steps |
| `refiner_learning_rate` / `refiner_weight_decay` | `0.0002` / `0.0001` | AdamW settings |
| `correction_scale` / `auxiliary_ce_weight` | `1.0` / `1.0` | Residual strength and auxiliary classification loss |

See [`configs/surgclip-ms-tcn-diffusion.yaml`](../configs/surgclip-ms-tcn-diffusion.yaml)
and [`configs/surgclip-ms-tcnpp-diffusion.yaml`](../configs/surgclip-ms-tcnpp-diffusion.yaml)
for runnable examples. The supplied centered SurgCLIP cache and symmetric
temporal convolutions make both experiments **offline**.

`swin-cls` (Swin Transformer encoder + per-head classifier in
`models/swin_classifier.py`; alias `swin-classifier`):

| Param | Type | Default | Notes |
|---|---|---|---|
| `variant` | str | `t` | Swin variant: `t`/`s`/`b` |
| `pretrained` | bool | `false` | ImageNet weights for the Swin encoder |
| `freeze_backbone` | bool | `false` | freeze the Swin encoder; only `project` + classifier heads train |
| `features` | int | `512` | feature dim before the classifier heads |

Like `resnet18`, accepts single frames or temporal windows (mean-pooled over
time) and emits per-head logits `(B, K)`.

---

## `validation`

| Field | Type | Default | Notes |
|---|---|---|---|
| `metrics` | list[str] | all six below | `jaccard`, `f1_macro`, `f1_per_class`, `mean_accuracy`, `accuracy_per_class`, `edit_score` |

Each metric is computed per head; aggregates are reported as `<head>_<metric>`
(e.g. `phase_jaccard`) while per-class metrics expand to
`<head>_<metric>_<i>` per class (e.g. `phase_f1_class_0`). Exactly one aggregate
metric should be listed first to serve as the "primary val metric" used for
best-checkpoint selection.

> **Primary metric caveat.** The first metric in `metrics` is used for
> best-checkpoint selection via the key `val_<head>_<metric>`. Per-class metrics
> (`f1_per_class`, `accuracy_per_class`) emit per-class keys
> (`val_<head>_f1_class_<i>`), so they **cannot** serve as the primary metric —
> if listed first, best-checkpoint selection silently never fires. Put an
> aggregate metric (e.g. `jaccard`, `f1_macro`) first.

---

## `mlflow`

| Field | Type | Default | Notes |
|---|---|---|---|
| `tracking_uri` | str \| null | `null` | e.g. `http://localhost:5000`; falls back to local file store |
| `experiment_name` | str | `phase_classification` | |
| `run_name` | str \| null | `null` | auto-generated if not set |
| `tags` | dict[str,str] | `{}` | set as MLflow run tags |
| `log_artifacts_to` | Path \| null | `null` | also write `model.pt` here + log as artifact |
| `s3_endpoint_url` | str \| null | `null` | MinIO/S3 endpoint for **client-side** artifact upload, e.g. `http://localhost:9000` |
| `aws_access_key_id` | str \| null | `null` | S3 access key (client-side); **excluded from logged params** |
| `aws_secret_access_key` | str \| null | `null` | S3 secret key (client-side); **excluded from logged params** |
| `aws_region` | str \| null | `null` | e.g. `us-east-1`; `AWS_DEFAULT_REGION` fallback |

The `s3_*`/`aws_*` fields configure the **training client** so it can upload the
model artifact to the S3 artifact store (the docker-compose `.env` only sets
credentials for the MLflow *server* container). Values are injected as
environment variables at run start; existing environment variables take
precedence, and `aws_access_key_id`/`aws_secret_access_key` are **never logged**
as MLflow params.

Every run automatically logs:
- all config params (`<section>.<field>` → value),
- the tag `gitcommithash` = current git HEAD,
- per-epoch metrics (`train_loss`, `train_*`, `val_*`),
- test metrics (`test_*`),
- the best model via `mlflow.pytorch.log_model`.

---

## `runtime`

| Field | Type | Default | Allowed values / notes |
|---|---|---|---|
| `device` | str | `auto` | `auto`, `cpu`, `cuda` |
| `cuda_device` | int \| null | `null` | requires `device` in `auto`/`cuda` |
| `mixed_precision` | str | `fp32` | `fp32`, `fp16` (AMP) |
| `seed` | int | `42` | global torch seed + DataLoader worker seeding |

---

## Example: minimal

Only `dataset` is required; everything else defaults:

```yaml
dataset:
  dataset_variant: cholect50-small
  root_dir: /data/cholect50
```

## Example: SurgCLIP feature MS-TCN with MLflow

Save this as a YAML file and run `uv run phase-cls train <config.yaml>`. The
cache must match the dataset variant and test fold. This configuration follows
the paper's temporal-probing head defaults; the cache's existing feature
dimension and clip construction are used as-is.

```yaml
project:
  name: surgclip-ms-tcn
  output_dir: outputs/surgclip-ms-tcn/fold1
  save_checkpoints: true

dataset:
  dataset_variant: cholect50-crossval
  root_dir: /data/cholect50
  feature_cache: /path/to/surgclip_embeddings.pt
  test_fold: 1

dataloader:
  batch_size: 1
  num_workers: 0
  shuffle: true
  pin_memory: false
  persistent_workers: false

model:
  name: surgclip-ms-tcn
  params:
    feature_dim: 256
    num_stages: 3
    num_layers: 8
    num_f_maps: 64
    kernel_size: 3
  num_classes: { phase: 7 }

training:
  heads: [phase]
  epochs: 100
  optimizer: adamw
  learning_rate: 0.01
  weight_decay: 0.01
  scheduler: cosine
  class_weighting: none

validation:
  metrics: [f1_macro, jaccard, mean_accuracy, edit_score]

mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: SurgCLIP CholecT50 probes
  run_name: surgclip-ms-tcn-fold1

runtime:
  device: cuda
  mixed_precision: fp32
  seed: 42
```

## Example: HeiChole phase recognition

```yaml
dataset:
  dataset_variant: heichole
  root_dir: /data/heichole
  test_fold: 1
training:
  heads: [phase]
```

## Example: phase classification (matches `configs/default.yaml`)

```yaml
project:
  output_dir: outputs
dataset:
  dataset_variant: cholect50-crossval
  root_dir: /data/cholect50
  test_fold: 1
  sequence_length: 1
  augmentations: [original]
model:
  name: resnet18
  params: {}
  num_classes: { phase: 7, tool: 6, verb: 10, target: 15, triplet: 100 }
training:
  epochs: 50
  optimizer: adamw
  learning_rate: 0.001
  loss: cross_entropy
  heads: [phase]
validation:
  metrics: [jaccard, f1_macro, f1_per_class, mean_accuracy, accuracy_per_class, edit_score]
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: phase_classification
runtime:
  device: auto
  mixed_precision: fp32
```

## Example: multi-head (phase + tool + verb, frozen pretrained backbone) with fp16

```yaml
dataset:
  dataset_variant: cholect50-crossval
  root_dir: /data/cholect50
model:
  name: resnet18
  params: { pretrained: true, freeze_backbone: true, features: 512 }
  num_classes: { phase: 7, tool: 6, verb: 10 }
training:
  epochs: 80
  optimizer: adamw
  learning_rate: 0.0005
  heads: [phase, tool, verb]
dataloader:
  batch_size: 64
validation:
  metrics: [jaccard, f1_macro, mean_accuracy, edit_score]
runtime:
  device: cuda
  mixed_precision: fp16
```
