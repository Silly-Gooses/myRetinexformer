# Model D v1 — RetinexFormer Research

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
