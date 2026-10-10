# Proposed v1 epoch training

Use the existing `Lightweight_RetinexFormer_Colab_Full_Training_v2.ipynb` on branch
`v1`. This is now a proposed-model notebook: setup → Drive/configuration → split
→ fresh model → training → curves/visuals → final test evaluation. The old
iteration training and baseline comparison cells have been removed. Architecture,
Canny defaults, Adam betas `(0.9, 0.999)`, batch size 8, patch size 256, and L1 loss
remain unchanged. AMP remains off by default.

## Smoke check or full training

Choose `RUN_MODE` in the notebook configuration cell. The default is `"smoke"`;
the separate smoke/full training cells execute only the selected mode, including
when using Run all.

- Smoke runs three complete epochs using the unchanged 200-epoch schedule, so
  all three are warm-up epochs. It saves under `<EXPERIMENT_NAME>_Smoke3`, with
  best/last checkpoints, an epoch-3 CSV row, curves and validation visuals at
  epochs 1 and 3. The curves mark individual points so the single CSV row is visible.
- Full runs all configured epochs under the original `<EXPERIMENT_NAME>` folder.
  After inspecting smoke results, set `RUN_MODE="full"`, `RESUME_FROM=None`, and
  re-run configuration, split and model setup, then the full training cell. This
  resets the seed, loaders, model and optimizer for independent training from scratch.
- Resume requires the same mode/folder and `RESUME_FROM=CKPT_DIR / "last.pth"`.
  Fresh runs never overwrite existing folders; change `EXPERIMENT_NAME` for another
  smoke check or independent full experiment. Existing full-run checkpoints retain
  their original path and configuration compatibility.

Use validation metrics and the saved Low / Edge / Lit-up / Output / GT grids to
check smoke results. The final official-test cell skips smoke mode. A three-epoch
warm-up run checks execution and saved artifacts, not final restoration quality.

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
rebuilds CSV at the selected reporting interval on resume, removing stale/duplicate rows. `best.pth` is written before
`last.pth`; a crash between these writes is recovered by replaying that epoch.
Checkpoints use tensors and primitive types and load with `weights_only=True`.

## Outputs and checks

Validation and checkpoint updates still run every epoch. The notebook uses
`LOG_EVERY=3` for CSV/console output: epochs 3, 6, ..., 198, and 200 (67 rows).
Each row reports that epoch, not an average of the preceding three epochs.
Checkpoint history retains every epoch for resume. `LOG_EVERY` is a presentation
setting, outside the strict training configuration, so existing checkpoints can
resume with the new reporting interval. The helper defaults to `log_every=1`
for compatibility; the notebook passes its explicit setting.

Reported fields are train/validation L1, train/validation PSNR and SSIM, LR actually used,
global step, epoch/cumulative processing time, and best flag. Times include
validation/visual generation but exclude checkpoint I/O and disconnected time.
Fixed validation examples are saved at epoch 1, every 10 epochs, and the last
epoch of the selected run (including an early smoke stop): Low / Edge / Lit-up / Output / GT, both individual PNGs and a five-panel
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


### Training PSNR/SSIM and existing runs

`train_psnr` and `train_ssim` are mean per-image scores over all augmented crops
seen in that epoch, including the final partial batch. Each score uses the
prediction from that batch's training forward, before its optimizer update.
The metrics detach the predictions and use the same clamped/rounded 8-bit RGB
PSNR/SSIM functions as validation. There is no additional model inference pass;
CPU image conversion and SSIM calculation do add measurement overhead. Crops
must be at least 11×11 for the existing SSIM window.

Validation uses full images with the final model state for that epoch in eval
mode. Thus train/validation curves share a metric definition but differ in data,
augmentation and model state; they are diagnostic curves, not accuracy percentages
or a controlled full-image generalization-gap measurement. Best selection remains
based only on validation PSNR.

Existing epoch checkpoints can resume without changing training configuration.
Historical epochs that did not record training PSNR/SSIM keep blank CSV fields;
those values cannot be reconstructed from the last checkpoint. New epochs record
both metrics, and notebook plots show gaps for unavailable history rather than
inventing zeros. CSV/console still report every 3 epochs plus the final epoch;
checkpoint history keeps all epoch metrics for recovery.

### Logging/metric verification (2026-10-10)

- Full regression suite: 30 tests run, 28 passed, two CUDA tests skipped because
  CUDA is unavailable. This includes exact uninterrupted/resumed worker training,
  per-image training metric weighting, unchanged optimizer updates, legacy metric
  history recovery, and three-epoch logging with best selection between log rows.
- Re-executed the notebook workflow on temporary synthetic data for four epochs:
  CSV/console contained epochs 3 and 4, paired training/validation quality curves
  rendered, and checkpoints, five-panel visuals and final test exports succeeded.
- Python/notebook syntax and `git diff --check` passed. No full training was run.

### Smoke/full mode verification (2026-10-10)

- All 13 single-process epoch-training tests passed, including the new three-epoch
  smoke test for unchanged 200-epoch scheduling and final smoke visuals.
- Executed the actual notebook cells with the full v1 architecture and synthetic
  images: smoke stopped at epoch 3, exported epoch-3 visuals and skipped official
  test loading. Switching modes without setup was rejected; re-running full setup
  reproduced the initial model weights with an empty optimizer and a separate folder.
- A shortened four-epoch full-mode check completed training, plots and test export
  without modifying the smoke checkpoint. Syntax and diff whitespace checks passed.
  This follow-up used CPU only; no real-data/full 200-epoch or GPU run was performed.
