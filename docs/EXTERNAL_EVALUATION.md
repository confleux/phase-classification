# Cross-dataset phase evaluation

`phase-cls-external-eval` runs the combined SurgCLIP + cached-feature MS-TCN
checkpoint on Cholec80, M2CAI16 workflow, or AutoLaparo. It extracts the same
centered SurgCLIP clips used during training, evaluates at approximately 1 fps,
and writes per-video prediction CSVs and `metrics.json`. It does not modify
dataset files.

## Cholec80

The adapter reads the released layout:

```text
<root>/videos/videoXX.mp4
<root>/phase_annotations/videoXX-phase.txt  # Frame<TAB>Phase, 0-based frame IDs
```

Run the official Cholec80 evaluation subset, excluding all Cholec80 videos that
also appear in CholecT50 (to avoid evaluating on frames used by the current
model's supervised training/validation data):

```bash
uv run phase-cls-external-eval cholec80 \
  --root /mnt/nvme1/datasets/cholec80 \
  --split test \
  --config configs/surgclip-ms-tcnpp-diffusion.yaml \
  --checkpoint outputs/surgclip-ms-tcnpp-diffusion/fold1/checkpoints/best.pt \
  --device cuda
```

The tool reports how many overlapping videos it excluded. Pass
`--include-cholect50-overlap` only for a non-independent diagnostic evaluation.
Use `--max-videos 1` for a quick smoke run.

To run one named video, for example `video41`:

```bash
uv run phase-cls-external-eval cholec80 \
  --root /mnt/nvme1/datasets/cholec80 \
  --split test --video video41 \
  --config configs/surgclip-ms-tcnpp-diffusion.yaml \
  --checkpoint outputs/surgclip-ms-tcnpp-diffusion/fold1/checkpoints/best.pt
```

To score all 80 videos, including those overlapping CholecT50, use
`--split all --include-cholect50-overlap`; interpret that result as diagnostic,
not an independent test score.

## M2CAI16 workflow

The CAMMA layout is supported when the root contains `videos/` and
`annotations/workflow_video_XX.txt`; the annotation rows are frame-index and
phase-name pairs. Because M2CAI16 has an extra **Trocar Placement** phase, its
eight phases are mapped to the seven CholecT50 outputs by merging Trocar
Placement into Preparation. This mapping is recorded in the metrics JSON.

```bash
uv run phase-cls-external-eval m2cai16 \
  --root /path/to/m2cai16-workflow \
  --split all \
  --checkpoint outputs/surgclip-ms-tcnpp-diffusion/fold1/checkpoints/best.pt
```

## AutoLaparo

AutoLaparo is a hysterectomy dataset with seven **different** phase classes.
The current cholecystectomy model can be run to export predictions, but the
workflow deliberately omits accuracy/F1 because matching class IDs would be
misleading. Its adapter accepts per-video CSV/TSV/TXT sidecars in the common
`frame,phase` or `start,end,phase` form. For another release layout, provide a
manifest CSV with `video_id,video_path,annotation_path` columns:

```bash
uv run phase-cls-external-eval autolaparo \
  --root /path/to/autolaparo \
  --manifest /path/to/autolaparo/eval_manifest.csv \
  --annotation-fps 1 \
  --checkpoint outputs/surgclip-ms-tcnpp-diffusion/fold1/checkpoints/best.pt
```

To get meaningful AutoLaparo metrics, train or fine-tune a model with the
AutoLaparo taxonomy and a held-out video split; do not use a guessed class map.

## Output and requirements

Output defaults to `outputs/external-eval/<dataset>/<timestamp>/`, with
`metrics.json` and one prediction CSV per video. Video decoding requires the
project's optional OpenCV workflow dependency and the matching SurgCLIP model
weights to be available on the machine running the command.
