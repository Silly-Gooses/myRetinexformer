# Model D-Pro Phase 1 — RetinexFormer Research

This branch preserves the completed D-Pro pilot as reusable Python modules.
The historical result is **21.1574 dB / 0.736978 SSIM** on the original 69-image
validation split, versus Model D's **21.0716 dB / 0.734275 SSIM**: approximately
**+0.0858 dB**. Best checkpoint: update **900**; completed updates: **1,200**.
These are archived Colab measurements, **not newly reproduced refactor results**.
No D-Pro Test score has been measured. **23.1 dB Test PSNR is an unverified target.**

Open [the thin Phase 1 Colab notebook](notebooks/Model_DPro_Phase1_Colab.ipynb).
Training and full validation are disabled by default. Historical evidence and
source hashes are in [the Phase 1 archive](experiments/model_d_pro/phase1/README.md).

## D-Pro architecture and recipe

D-Pro retains the trained Model B backbone, eight NAF mixers, Canny guidance,
and trained epoch-17 HVI head. It adds illumination modulation, three brightness
priors, bottleneck attention over at most 64 pooled tokens, and intensity/HV
branches with channel cross-attention. New correction projections start at zero.
`SafeAtan2` preserves the forward value and bounds the derivative near neutral H/V.
The backbone-to-head path is differentiable float32; Model D's internal uint8
boundary is deliberately absent. Final metrics still use the original uint8 RGB
conversion. This is a research adaptation, not an official CIDNet implementation.

The unchanged Phase 1 recipe uses seed 42, patch 128, microbatch 2, accumulation 4,
effective batch 8, and 1,200 successful optimizer updates. For the first 100 updates,
only new parameters train. Learning rates are `1e-5` (backbone), `2e-5` (legacy
head), and `1e-4` (new modules), with the original 25-update new-module ramp and
cosine schedule. AdamW has zero weight decay; AMP, clipping at 1.0, nonfinite
handling, and update-indexed stateless crop sampling are preserved.

The loss is `L1_RGB + 0.10 L1_intensity + 0.05 L1_HV + 0.02 L1_backbone +
0.002 residual_penalty`. Full validation runs every 300 updates and durable
checkpoints every 100. Early stopping requires three validations without a
0.01 dB patience improvement after at least update 900. The worker defaults to
45 minutes with a 90-second saving reserve and an outer terminate/kill watchdog.
These are budget controls, not evidence of model convergence.

## D-Pro commands

Clone `model_d_pro`, install the existing requirements, and run from the repository
root. Colab setup leaves its CUDA PyTorch in place and installs only missing small
dependencies. The package has separate model, dataset, loss, training, evaluation,
checkpoint, verification, runtime, and CLI modules. No new ML dependencies are needed.

```bash
git clone --branch model_d_pro https://github.com/Silly-Gooses/myRetinexformer.git
cd myRetinexformer
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_model_d*.py' -v
python -m model_d_pro.cli smoke --synthetic --device cpu --output-dir results/dpro_cpu_smoke
python -m model_d_pro.cli infer --checkpoint /path/to/pilot_01/best.pth --input /path/to/low.png --device cuda --output-dir results/dpro_infer
python -m model_d_pro.cli verify --checkpoint /path/to/pilot_01/best.pth --input /path/to/validation.png --device cuda --output-dir results/dpro_parity
python -m model_d_pro.cli evaluate --checkpoint /path/to/pilot_01/best.pth --data-root /path/to/Real_captured --device cuda --compare-historical --output-dir results/dpro_val
```

Use a new or empty output directory for each check. `infer --input` also accepts
an image directory. Replace `--checkpoint` with `--model-b /path/to/best.pth
--head /path/to/best_model_d_head.pth` to initialize from the original trained pair.
Supplying a D-Pro checkpoint does not require the original files. Original-pair
loading validates both SHA256 hashes; all D-Pro weights load with `strict=True`.
No failed load silently substitutes untrained weights.

Verification compares identical weights and inputs against the hash-checked
executed script, reporting maximum absolute differences for every auxiliary
output (CPU tolerance `1e-6`, CUDA `2e-5`). Full validation requires CUDA AMP,
the frozen 69-image split and original content fingerprint, and reports per-image
RGB PSNR/SSIM with arithmetic means. No GT brightness adjustment is applied.
`--compare-historical` requires step-900 weights and reports per-image differences
plus aggregate tolerances of 0.002 dB PSNR and 0.0001 SSIM. Failures retain reports
and exit nonzero. The D-Pro CLI has no Test evaluation command.

Python loading and future weight initialization:

```python
from model_d_pro import ModelDPro, load_model_d_pro

model, provenance = load_model_d_pro(checkpoint="/path/to/pilot_01/best.pth", device="cuda")
output = model(low_rgb)  # Floating RGB [B, 3, H, W] in [0, 1]; original dimensions.
diagnostics = model(low_rgb, return_aux=True)
# Future Phase 2 must explicitly define its own optimizer and schedule.
```

## Drive paths, opt-in training, and resume

Defaults under `MyDrive/RetinexFormer_NAF_Screening/`:

- Model B: `model_B_optionA_batch8_patch128_lr1e-5_from_batch8best/low_lr_1e-5_1000steps/best.pth`.
- HVI head: `model_D_HVI_from_ModelB_OptionA_v1/best_model_d_head.pth`.
- Historical D-Pro: `model_DPro_budget_v1/pilot_01/best.pth` and `last.pth`.
- Split file: `model_B_v1_full_data_1000steps/split.json` (optional; must match if supplied).
- New modular outputs: `model_DPro_modular_phase1/phase1_01/`.
- Dataset: `MyDrive/datasets/LOLv2/Real_captured/Train/{Low,Normal}`.

Copy `model_d_pro/configs/phase1_pilot.json` to a configuration outside the source
package and edit paths as needed. Its numeric recipe matches the historical config;
only its output directory/run name and default `resume=false` differ. Explicitly
requesting `train` launches the original bounded recipe on T4:

```bash
python -m model_d_pro.cli train --config /path/to/my_phase1.json --session-minutes 45
python -m model_d_pro.cli train --config /path/to/my_phase1.json --resume --session-minutes 45
```

Controls include `--max-steps`, `--validate-every`, `--save-every`, `--output-root`,
`--run-name`, and `--session-minutes`. Recipe changes require a new run. The
notebook exposes the same controls, an explicit training toggle, and optional
runtime disconnection. Opening it or running its defaults does not train.

Historical `best.pth` and `last.pth` are supported for **weights only**. Their
original script/commit identity cannot be used for full-state modular resume.
Within a modular run, resume requires matching package/baseline source, recipe,
checkpoint origins, and dataset fingerprints. It restores optimizer, scaler,
RNG, update count, stopping state, and any pending validation. A completed run
does not add updates. Original checkpoints and historical directories are never
overwritten. Future Phase 2 is not implemented or started by this migration.

Set `MODEL_D_BACKBONE`, `MODEL_D_HEAD`, and `MODEL_D_PRO_CHECKPOINT` to opt into
real-checkpoint tests. Missing artifacts are explicitly skipped. CPU tests use
synthetic weights to measure source parity and strict schema loading, and cannot
establish the real pilot score. See the archive README for provenance and the
original T4 software versions.

---

## Preserved Model D v1 documentation

Official preservation branch for the MSc thesis on low-light image enhancement.
Model D combines the original trained Model B backbone with its trained,
intensity-preserving HVI-inspired color correction head.

- **Backbone:** RetinexFormer Model B
- **Feature mixer:** NAFNet-inspired
- **Edge guidance:** Canny
- **Color correction:** trained HVI-inspired head
- **Reported validation PSNR:** **21.0716 dB**
- **Reported test PSNR:** **18.8959 dB**

These are historical results from the original v1 report, **not newly reproduced
Model D measurements**. The original head checkpoint is required to complete
trained Model D verification. See [checkpoint setup](checkpoints/README.md).

## Architecture

```text
Low-light RGB → Model B (illumination estimator + Canny + NAF mixer/IGAB)
             → enhanced RGB → uint8 rounding → float32 RGB
             → HVI conversion → bounded HV residual + unchanged intensity
             → inverse HVI → enhanced Model D RGB
```

The 8-bit boundary reproduces the original notebook's Model B PNG cache. Removing
it changes the experiment. Model B uses replicate padding to multiples of four,
CUDA AMP, and cropping to the original image size. HVI/head arithmetic uses
float32 with autocast disabled. This custom HVI-inspired transform is not the
full CIDNet architecture.

The fixed backbone has 3 input/output channels, 40 features, one stage, and
blocks `[1, 2, 2]`. The head uses four 3×3 convolutions with width 32, ReLU between
convolutions, and a zero-initialized final two-channel projection. Its correction
is `0.12 * tanh(residual)` with `HVI_K=1.0`; intensity is preserved.

## Setup and use

Use Python with CUDA-enabled PyTorch for historical evaluation. CPU supports
architecture tests and smoke inference, but does not reproduce the AMP protocol.

```bash
git clone --branch model_d https://github.com/Silly-Gooses/myRetinexformer.git
cd myRetinexformer
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p test_model_d.py -v
```

For Colab, open [Model_D_v1_Colab.ipynb](notebooks/Model_D_v1_Colab.ipynb).
It mounts Drive, loads the original checkpoint pair, runs verification and
validation, then permits official Test evaluation. It never trains either model.

Local inference (replace the checkpoint and image paths):

```bash
python -m model_d.cli infer --model-b /path/to/best.pth --head /path/to/best_model_d_head.pth --input /path/to/image.png --output-dir results/inference --device cuda
```

`--input` also accepts an image directory. Add `--intermediates` to save NPZ tensors
for input RGB, Canny edges, lit-up image, Model B RGB, quantized head input, HV
residual, and final RGB. Use a new or empty output directory for every command.

Python interface:

```python
from model_d import load_model_d

model, checkpoint_info = load_model_d("best.pth", "best_model_d_head.pth", device="cuda")
# x: floating-point [B, 3, H, W] RGB in [0, 1], on the same device.
output = model(x)
diagnostics = model.forward_with_intermediate(x)
```

The wrapper preserves input dimensions and returns only RGB during ordinary
inference. The loader fails on missing or incompatible weights; it never falls
back to an untrained model. `ModelD` construction alone is for architecture tests;
use `load_model_d` for trained inference.

## Dataset and evaluation

Use the original LOL-v2-real directories:

```text
Real_captured/
  Train/Low/       Train/Normal/       # 689 paired images
  Test/Low/        Test/Normal/        # 100 paired images
```

The original ordered 620/69 split is preserved in
[reference.json](model_d/reference.json). An optional `--split-file` must match
the original `split.json` exactly; no random split is generated.

```bash
python -m model_d.cli verify --model-b /path/to/best.pth --head /path/to/best_model_d_head.pth --input /path/to/validation.png --output-dir results/parity --device cuda
python -m model_d.cli evaluate --model-b /path/to/best.pth --head /path/to/best_model_d_head.pth --data-root /path/to/Real_captured --split val --output-dir results/validation --device cuda
python -m model_d.cli evaluate --model-b /path/to/best.pth --head /path/to/best_model_d_head.pth --data-root /path/to/Real_captured --split test --validation-report results/validation/report.json --output-dir results/test --device cuda --save-images
```

Validation and Test use full RGB images, the original rounding and PSNR/SSIM
implementations, and arithmetic means over images. Reports contain per-image
metrics, checkpoint hashes, dataset fingerprints, notebook parity, and runtime
versions. A mismatch beyond 0.002 dB PSNR or 0.0001 SSIM is reported as a failure,
with results retained for investigation. These are numerical reproduction
tolerances, not evidence of a research improvement.

Test requires successful CUDA validation of the same checkpoint pair and
validation data. Never tune the model using Test results.

## Historical results and reproducibility

The original `ModelD_Comparison_Report.zip` reports:

- **Model B:** validation 21.0367 dB / 0.735457 SSIM; Test 18.8533 dB / 0.774117 SSIM.
- **Model D v1:** validation 21.0716 dB / 0.734276 SSIM; Test 18.8959 dB / 0.773454 SSIM.

PSNR increased slightly, while SSIM decreased slightly. Full-precision historical
values and provenance are preserved in `reference.json`. The selected head is
from **epoch 17**, not the final epoch checkpoint.

Original head training settings were seed 42, patch 128, batch 8, 30 epochs,
AdamW at `1e-4`, zero weight decay, and
`L1_RGB + 0.10 * L1_HV + 0.01 * mean(abs(HV residual))`. Model B was frozen;
only the head was trained on crops of cached full-image Model B outputs.
These settings document the original experiment; this branch does not retrain it.

The backbone and metric source is pinned to
`ad33e6bf5b338bc36e4a14102781e78a03397b3b`. The original notebook is preserved
byte-for-byte as [a reference fixture](tests/reference_Model_D_v1.ipynb).
Verification loads only its hash-checked model/conversion definitions, never its
setup, cache-generation, or training cells. Do not run that archival notebook as
the verification workflow; use the thin Colab notebook above.

To enable real-checkpoint tests, set `MODEL_D_BACKBONE`, `MODEL_D_HEAD`, and
optionally `MODEL_D_IMAGE` before running the test command. Missing checkpoint
tests are explicitly skipped as unverified. Architecture tests use synthetic
weights and cannot establish reproduction of trained Model D results.

Local reconstruction checks on 6 November 2026 strictly loaded the original
Model B checkpoint and matched its independent notebook forward pass on a
400×600 validation image. Model B evaluation across all 69 validation images
produced **21.036669 dB PSNR / 0.735455 SSIM** using PyTorch `2.14.1+cu126` on a
GTX 1650. The 14 available Model D tests passed; the original trained-head test
was explicitly skipped. These checks do not reproduce trained Model D metrics.

## Project layout and baseline policy

`model_d/model.py` contains the architecture and loaders; `model_d/cli.py` handles
inference, verification, and evaluation. Shared backbone/Canny/metric files stay
unchanged. `tests/test_model_d.py` covers the new behavior. Existing v1 training
scripts and documents remain historical utilities, not the Model D workflow.

**All future improvement experiments must begin with the verified Model D v1
architecture and trained checkpoint pair.** Model B alone must not silently
replace Model D as the baseline. Keep experiment branches, checkpoints, and
results separate; preserve the split, document the intended change, and report
PSNR, SSIM, matched visual comparisons, and computational cost before promotion.
This baseline introduces no Laplacian, FFT, frequency refinement, or adaptive HVI
gating. Until trained-head verification succeeds, its verification status remains
pending.
