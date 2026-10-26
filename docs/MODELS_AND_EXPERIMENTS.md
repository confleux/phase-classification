# Models and Current Experiments

This page is an overview of the model implementations and experiment
configurations currently represented in the repository. It is not a results
report: a YAML config records a reproducible setup, but does not establish that
the run completed. See [`CONFIG.md`](CONFIG.md) for the full parameter and data
specification, and [`METRICS.md`](METRICS.md) for metric definitions.

## Model implementations

| Model name (`model.name`) | Architecture / input | Notes |
|---|---|---|
| `resnet18` | ResNet-18 image encoder with a projection and one classifier per selected head. | Accepts individual frames or temporal windows; windows are mean-pooled. Supports ImageNet initialization and freezing the backbone. |
| `convnext-ms-tcn` (`ms-tcn`) | ConvNeXt frame encoder followed by multi-stage MS-TCN temporal prediction. | Requires `dataset.sequence_length >= 2`. Emits predictions for each frame in the input sequence. |
| `swin-ms-tcn` | Swin Transformer frame encoder followed by multi-stage MS-TCN temporal prediction. | Requires `dataset.sequence_length >= 2`. Emits predictions for each frame in the input sequence. |
| `surgclip-ms-tcn` (`feature-ms-tcn`) | Multi-stage MS-TCN over precomputed SurgCLIP feature sequences. | No image encoder is run during training; requires a compatible `dataset.feature_cache`. Currently phase-only, with one complete video per batch. |
| `swin-cls` (`swin-classifier`) | Swin Transformer image encoder with per-head classification layers. | Accepts individual frames or temporal windows; windows are mean-pooled. Available in the registry but no dedicated example config is currently included. |

The registered image models can use the configured label heads. The supplied
CholecT50 examples use the seven-class `phase` head, even though the default
label map also defines `tool`, `verb`, `target`, and `triplet`. The aliases in
the table select the same implementation as their canonical name.

### Custom SurgCLIP diffusion run modes

These names are handled by `run.py`, not by the standard model registry. Each
first trains a coarse temporal phase predictor on cached SurgCLIP features, then
trains a residual diffusion refiner conditioned on those features and coarse
scores. The refiner predicts corrections to the coarse phase probabilities.
Both stages are offline: no SurgCLIP encoder is run as part of these configs.

| Run-mode name | Coarse predictor | Distinguishing detail |
|---|---|---|
| `surgclip-ms-tcn-diffusion` | Cached-feature MS-TCN | Three-stage MS-TCN coarse predictor with the same diffusion-refiner setup used by the MS-TCN++ comparison. |
| `surgclip-ms-tcnpp-diffusion` | Cached-feature MS-TCN++ | Dual-dilated prediction-generation stage followed by probability-only refinement stages. Defaults/config use one generation stage and three refinement stages. |

Both custom modes currently require a compatible CholecT50 feature cache, the
seven-class phase head, and `dataloader.batch_size: 1`. The diffusion objective
is a residual refinement experiment; it is not Meta-SurDiff's bilevel
meta-weight objective.

## Experiment configurations in this repository

These configs are the current experiment examples. They target CholecT50
cross-validation fold 1 and select the phase head, unless noted. The dataset
roots and feature-cache paths are machine-specific placeholders/settings and
should be checked before launching on another environment.

| Experiment | Config | Setup summary |
|---|---|---|
| ResNet-18 image baseline | [`configs/resnet18.yaml`](../configs/resnet18.yaml) | Pretrained ResNet-18, fine-tuned; single-frame input; inverse-frequency class weighting. |
| ConvNeXt + MS-TCN | [`configs/ms-tcn.yaml`](../configs/ms-tcn.yaml) | Pretrained ConvNeXt-Tiny encoder and four-stage MS-TCN; 32-frame sequences; inverse-frequency class weighting. |
| Swin + MS-TCN | [`configs/swin-ms-tcn.yaml`](../configs/swin-ms-tcn.yaml) | Pretrained Swin-T encoder and four-stage MS-TCN; 32-frame sequences; inverse-frequency class weighting. |
| SurgCLIP feature probing | [`configs/surgclip-ms-tcn.yaml`](../configs/surgclip-ms-tcn.yaml) | Cached 256-dimensional features with a three-stage MS-TCN; 100 coarse-model epochs; cosine schedule. |
| SurgCLIP MS-TCN + diffusion | [`configs/surgclip-ms-tcn-diffusion.yaml`](../configs/surgclip-ms-tcn-diffusion.yaml) | Cached-feature MS-TCN coarse model plus residual diffusion refiner. |
| SurgCLIP MS-TCN++ + diffusion | [`configs/surgclip-ms-tcnpp-diffusion.yaml`](../configs/surgclip-ms-tcnpp-diffusion.yaml) | Cached-feature MS-TCN++ coarse model plus the same configured diffusion refiner, for comparison with the MS-TCN variant. |

`configs/default.yaml` is also available as a general ResNet-18 starting
example; unlike the named experiment configs, its dataset root is `/data/cholect50`.
There is currently no dedicated config for `swin-cls`.

The two diffusion configs share the refiner settings (20 epochs, 100 steps per
epoch, 1,000 diffusion timesteps, and 25 sampling steps). Their MS-TCN and
MS-TCN++ coarse predictors have different architectures and training defaults;
see the individual YAML files for the complete values. No measured scores are
listed here—use the corresponding MLflow runs/checkpoints for results.

## Running an experiment

From the repository root, validate the schema and then start training:

```bash
uv run phase-cls validate-config configs/surgclip-ms-tcn-diffusion.yaml
uv run phase-cls train configs/surgclip-ms-tcn-diffusion.yaml
```

Change local dataset/cache paths first. Use `--check-paths` on
`validate-config` to check dataset paths as well as the schema. For cached
feature experiments, confirm that the cache belongs to the selected dataset
and fold and that its feature dimension matches `model.params.feature_dim`.
