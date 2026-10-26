# Adaptive Edge-Guided RetinexFormer V2

## Architecture and integration

`Proposed_V2_arch.py` defines a separate V2 model. E0 and V1 remain in their
existing files. At each stage, the illumination estimator produces illumination
features `[B,C,H,W]` and a three-channel map. The lit-up RGB image is still
`image * illumination_map + image`. The unchanged three-channel Denoiser
embedding produces `F_rgb [B,C,H,W]`.

OpenCV Canny reads the **original low-light RGB model input** once, after the
dataset's crop and spatial augmentation. It converts `[0,1]` RGB to uint8,
then RGB to grayscale, and emits a detached float32 `[B,1,H,W]` map in `[0,1]`.
The defaults are 50/100 on the OpenCV 8-bit intensity scale. This is a
different Canny implementation and threshold scale from V1's PyTorch detector;
document that difference when attributing model changes.

The lightweight edge encoder outputs `F_edge [B,C,H,W]`. E3 predicts
`G=sigmoid(Conv1x1([F_rgb,F_edge]))`, then inserts
`F_rgb + G * F_edge` **after the embedding and before the first Transformer
encoder block**. E2 sets `G=1` for ungated addition, sharing the same edge
encoder. Illumination-guided attention, the encoder/decoder skip connections,
and `mapping(features) + lit_up` reconstruction remain in place. For multiple
stages, the same original edge map is reused throughout.

`model(image)` returns enhanced RGB. `model(image, return_intermediates=True)`
returns `(enhanced, details)` with edge, gate, lit-up and feature tensors.
`forward_with_intermediate(..., return_dict=True)` supports the existing
visualization workflow.

## Configuration and Colab

`official_lolv2_v2_config.json` retains the V1 200-epoch, batch-8,
256-patch Adam/L1/warmup-cosine training settings. Its `gated` flag selects E3
(`true`) or E2 (`false`). Use distinct experiment names and output directories.
`Adaptive_Edge_V2_Colab.ipynb` has independent setup, preflight, quick smoke,
three-epoch smoke, full training, resume, evaluation and visualization cells.
Set the dataset and Google Drive paths in its configuration cell. Full training
is only started by its dedicated cell.

Local/Colab CLI examples:

```bash
python v2_synthetic_smoke.py --config official_lolv2_v2_config.json --output-dir /path/to/new-synthetic-output
python train.py --config official_lolv2_v2_config.json --output-root /path/to/drive --experiment-name V2_E3 --mode smoke
python smoke_test.py --config official_lolv2_v2_config.json --output-dir /path/to/new-smoke-output
python train.py --config official_lolv2_v2_config.json --output-root /path/to/drive --experiment-name V2_E3 --mode full
python train.py --config official_lolv2_v2_config.json --output-root /path/to/drive --experiment-name V2_E3 --mode full --resume
python test.py --checkpoint /path/to/best.pth --dataset-root /path/to/Real_captured --output-dir /path/to/new-test-output
```

For optional fresh initialization from a baseline or V1 checkpoint, add
`--initialize-from /path/to/checkpoint.pth --initialize-source baseline` (or
`v1`) to a fresh `train.py` run. Shared keys and tensor shapes are checked
strictly. V1's fourth edge embedding channel is discarded explicitly; the new
edge encoder and gate retain fresh initialization. Source optimizer state is
never restored. A V2 resume uses only its own complete matching checkpoint,
including model, optimizer, scheduler, scaler and RNG states.

## Evaluation and ablations

`compare_v2.py` accepts a JSON object with `E0`, `E1`, `E2`, `E3` entries,
each containing `config` and `checkpoint` paths. E0's config uses
`"version":"baseline"`; E1 uses `v1`; E2 and E3 use `v2` with `gated=false`
and `true`. It scores the same 100 LOL-v2 Real Test pairs with the shared
PSNR/SSIM implementation and writes per-image and summary CSVs, parameter
counts, and time per image on one fixed Test sample. Runtime includes
preprocessing and OpenCV CPU transfers; hardware and sample name are recorded.
FLOPs/MACs are not currently exported, since available profiler counts omit
some custom attention operations. Do not report partial counts as full-model
cost.

The inherited official protocol uses all 689 Train pairs for optimization and
all 100 Test pairs for validation, checkpoint selection and the later test
export. Those Test results are **not independent held-out final estimates**.
Reserve a separate untouched set or redesign the split consistently for all
four models before making final generalization claims.

## Verification and remaining work

`python -m unittest tests.test_v2 -v` checks Canny shape/source, fusion
shapes, gate bounds, E2/E3 gradients, finite output, exact checkpoint reload,
multistage edge reuse, and strict baseline initialization. The existing V1
suite checks for regressions. `v2_synthetic_smoke.py` runs two actual optimizer
steps, metrics, visualization and checkpoint reload without the dataset.
A dataset-backed smoke run, A100 mixed precision,
full training, E0–E3 comparison, and visual/metric conclusions require the
LOL-v2 Real data and Colab environment. A full-scale run is not launched by
this repository change.
