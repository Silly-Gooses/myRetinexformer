# Proposed v1 epoch training

Use the existing `Lightweight_RetinexFormer_Colab_Full_Training_v2.ipynb` on branch
`v1`. This is now a proposed-model notebook: setup → Drive/configuration → split
→ fresh model → training → curves/visuals → final test evaluation. The old
iteration training and baseline comparison cells have been removed. Architecture,
Canny defaults, Adam betas `(0.9, 0.999)`, batch size 8, patch size 256, and L1 loss
remain unchanged. AMP remains off by default.

## Schedule

All settings are editable in the configuration cell:

| Setting | Default |
| --- | --- |
| `NUM_EPOCHS` | 200 (includes warm-up) |
| `WARMUP_EPOCHS` | 3 |
| `WARMUP_START_LR` | `1e-6` |
| `PEAK_LR` | `1e-4` |
| `MIN_LR` | `1e-6` |
| `SEED` | 42 |

LR is constant during an epoch. Epochs 1–3 use `1e-6`, `5.05e-5`, `1e-4`.
For epoch `e > 3`, LR is
`1e-6 + (1e-4 - 1e-6) * (1 + cos(pi * (e - 3) / 197)) / 2`.
Epoch 200 uses exactly `1e-6`. `EpochWarmupCosine.set_epoch(e)` runs before
training batches; do not also call an iteration scheduler. `global_step` counts
batches for reporting, not schedule control. The final partial batch is included.

## Data separation and reproducibility

Sorted training filenames are shuffled using a local Python RNG with seed 42.
`ceil(0.1 * N)` pairs are held out; the remainder train with existing random
crops and geometric augmentation. Full-image validation has no augmentation.
The exact split is stored in `split.json`; `config.json` stores training settings.
Both train and validation subsets must be nonempty.

The official test set is loaded only in final evaluation after the configured
training run completes. Best-checkpoint selection uses validation PSNR, never
test PSNR. Equal validation PSNR retains the earlier best. Validation L1 is the
mean per-image loss on unclamped model predictions; PSNR/SSIM use the repository's
existing saved-8-bit-pixel implementation. Training loss is sample-weighted.

Python, NumPy, PyTorch, CUDA, worker seeds, and separate loader generators are
controlled. Worker persistence is disabled so an epoch-boundary resume can
recreate worker augmentation streams. cuDNN benchmarking is off and deterministic
algorithms are required. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA work,
as the first notebook cell does. An unsupported deterministic operation raises
rather than silently becoming nondeterministic. Exact numerical reproduction
across different hardware or PyTorch versions is not promised.

## Checkpoints and resume

The default experiment directory on Drive is:

```text
RetinexFormer_full_training/Lightweight_RetinexFormer_v1_LOLv2_real_Epoch200/
  config.json
  split.json
  checkpoints/best.pth
  checkpoints/last.pth
  logs/training_metrics.csv
  logs/training_curves.png
  visuals/epoch_0001/...
  final_test_outputs/...
```

A fresh run refuses to overwrite a nonempty experiment directory. Change
`EXPERIMENT_NAME` for an independent run. To resume the existing run, use:

```python
RESUME_FROM = CKPT_DIR / "last.pth"
```

Re-run configuration, split, model, and training cells with unchanged settings.
Resume restores model, optimizer, scheduler, scaler, completed epoch, global
step, best PSNR/associated SSIM/epoch, complete metric history, and RNG/generator
states. It continues at the next epoch and does not restart warm-up. An interrupted
partial epoch is repeated from the preceding completed epoch. Old iteration
checkpoints cannot resume this workflow. Increasing `NUM_EPOCHS` on resume is
rejected because it changes the saved cosine schedule; this implementation does
not silently reinterpret an existing run.

Files are saved via a temporary file in the same directory and replaced only
once written. This avoids overwriting a valid local checkpoint with a partial
write; it does not guarantee Google Drive's remote synchronization during a
runtime failure. `last.pth` is the authoritative committed epoch. Its history
rebuilds CSV on resume, removing stale/duplicate rows. `best.pth` is written before
`last.pth`; a crash between these writes is recovered by replaying that epoch.
Checkpoints use tensors and primitive types and load with `weights_only=True`.

## Outputs and checks

Every epoch logs train/validation L1, validation PSNR/SSIM, LR actually used,
global step, epoch/cumulative processing time, and best flag. Times include
validation/visual generation but exclude checkpoint I/O and disconnected time.
Fixed validation examples are saved at epoch 1, every 10 epochs, and the final
epoch: Low / Edge / Lit-up / Output / GT, both individual PNGs and a five-panel
grid. The curve cell plots loss, validation metrics and LR. Final test evaluation
loads `best.pth`, exports all test images and per-image/average metrics.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

Tests compare uninterrupted versus resumed synthetic training, including worker
processes, exact model/optimizer states and augmentation inputs; time columns are
expected to differ. Other checks cover LR boundaries, split membership, final
partial batches, best/last ties, invalid resume, CSV repair, interrupted writes,
validation and five-panel Canny exports. Full training is not run by these tests.

### Local verification (2026-10-09)

- Eight epoch-training tests passed, including exact uninterrupted/resumed
  equivalence with zero and two DataLoader workers.
- All 15 existing runnable Canny/analysis tests passed; two CUDA tests skipped.
- Executed the actual notebook's configuration, split, full v1 model, training,
  plotting and final-export cells with temporary synthetic data: four epochs
  including three warm-up epochs, nine training pairs, two validation pairs and
  two test pairs. Checkpoints, CSV, curves, five-panel visuals and test exports
  were created successfully. This was a CPU smoke run, not the 200-epoch experiment.
- Python/notebook syntax and `git diff --check` passed. GPU epoch training remains
  unverified on this CPU-only environment.
