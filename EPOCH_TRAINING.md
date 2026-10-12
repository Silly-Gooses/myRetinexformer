# Proposed v1 epoch training

Use the existing `Lightweight_RetinexFormer_Colab_Full_Training_v2.ipynb` on branch
`v1`. This is a proposed-model notebook: setup → Drive/configuration →
`train.py` launch → curves/visuals → final test evaluation. The script owns split,
loader, model, optimizer and scheduler setup; `training_utils.py` remains the shared
implementation for training, validation and checkpoint handling. The old
iteration training and baseline comparison cells have been removed. Architecture,
Canny defaults, Adam betas `(0.9, 0.999)`, batch size 8, patch size 256, and L1 loss
remain unchanged. AMP remains off by default.

## Smoke check or full training

Choose `RUN_MODE` in the notebook configuration cell. The default is `"smoke"`;
the separate smoke/full training cells execute only the selected mode, including
when using Run all.

- Smoke runs three complete epochs using the unchanged 200-epoch schedule, so
  all three are warm-up epochs. It saves under `<EXPERIMENT_NAME>_Smoke3`, with
  best/last checkpoints, CSV rows at epochs 0 and 3, curves and validation visuals at
  epochs 0, 1 and 3. The curves mark individual points so the single CSV row is visible.
- Full runs all configured epochs under the original `<EXPERIMENT_NAME>` folder.
  After inspecting smoke results, set `RUN_MODE="full"`, `RESUME_FROM=None`, and
  re-run configuration, then the full training cell. Each `train.py` invocation
  starts a new process and resets seed, loaders, model and optimizer for independent
  training from scratch. The launcher-definition cell only needs re-running after
  a runtime restart or notebook code update.
- Resume requires the same mode/folder and `RESUME_FROM=CKPT_DIR / "last.pth"`.
  Fresh runs never overwrite existing folders; change `EXPERIMENT_NAME` for another
  smoke check or independent full experiment. Existing full-run checkpoints retain
  their original path and configuration compatibility.

Use validation metrics and the saved Low / Edge / Lit-up / Output / GT grids to
check smoke results. The final official-test cell skips smoke mode. A three-epoch
warm-up run checks execution and saved artifacts, not final restoration quality.

## train.py entry point and Colab launcher

`train.py` takes the existing training configuration as JSON and exposes:

- `--mode smoke|full` (default smoke); smoke stops at epoch 3 without shortening
  the saved full schedule. Its experiment directory receives the `_Smoke3` suffix.
- `--output-root` and `--experiment-name` select the experiment folder.
- `--resume` uses that run's `checkpoints/last.pth`; `--resume-from PATH` accepts
  an explicit path to the same checkpoint. These options are mutually exclusive.
- `--log-every 3` controls reporting without changing strict checkpoint config.
- `--device auto|cpu|cuda`; AMP still requires CUDA and defaults off in config.
- `--evaluate-only --mode full` loads the best checkpoint after completed full
  training and exports official test results without running any training.

For a standalone launch, save the notebook configuration as JSON (or reuse an
existing run's `config.json` when its settings match):

```python
Path("run_config.json").write_text(json.dumps(CONFIG, indent=2))
```

```bash
python train.py --config run_config.json --output-root experiments --mode smoke
python train.py --config run_config.json --output-root experiments --mode full
python train.py --config run_config.json --output-root experiments --mode full --resume
python train.py --config run_config.json --output-root experiments --mode full --evaluate-only
```

The notebook performs this handoff automatically using a temporary JSON file and
an argument list (paths with spaces work). It streams the child process output,
raises on failure, and terminates the child if the cell is interrupted. Permanent
configuration and manifests remain inside the experiment directory. Curves and
saved validation panels are viewed in subsequent notebook cells; no in-memory
training model is needed. The final evaluation cell invokes the same script in
`--evaluate-only` mode and skips smoke runs.

The script sets deterministic CUDA configuration before initializing CUDA.
Configuration schema, experiment paths and checkpoint format are unchanged, so
existing notebook epoch checkpoints can resume. Dataset, architecture, batch size
and schedule mismatches are still rejected. Optional correctness tests and GPU
benchmarks are opt-in in the notebook. Colab's Git setup requires this script and
updated notebook to be available on remote branch `v1`; no commit/push is automatic.

## Epoch 0: initial validation

Fresh smoke/full runs evaluate all validation images before any training update.
The existing full-image padding, floating-point L1 and saved-8-bit PSNR/SSIM
implementations are used in evaluation/inference mode. Epoch 0 always appears in
CSV and console, regardless of `LOG_EVERY`. `global_step=0`, `is_best=False`;
training loss/PSNR/SSIM and LR are `None` (blank CSV, N/A console, gaps in curves).
Epoch-0 timing includes validation/visualization and contributes to elapsed time.

With visualization enabled, the fixed validation samples also produce five-panel
images in `visuals/epoch_0000/`. RNG and loader-generator states are restored after
initial diagnostics, so training data order and augmentation do not change.

After successful initial validation, `last.pth` is atomically saved with
`completed_epoch=0` and scheduler epoch 0. Resume starts epoch 1 without repeating
initial evaluation. No `best.pth` is created at epoch 0: only trained epochs
participate in best selection, even if their PSNR is below the initial score.
The scheduler stays at its initial state until epoch 1 sets the warm-up LR.

Fresh checkpoints retain history `0..completed_epoch`; older checkpoints with
history `1..completed_epoch` still load without backfilling initial metrics.
Configuration schema and checkpoint format identifier are unchanged. New
checkpoints require the updated loader; older code may reject epoch-0 history.
The 200 training epochs and three warm-up epochs remain unchanged. Final test
evaluation still requires completion of the full schedule and uses `best.pth`.

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
  logs/training.log
  logs/training_curves.png
  visuals/epoch_0001/...
  final_test_outputs/...
```

A fresh run refuses to overwrite a nonempty experiment directory. Change
`EXPERIMENT_NAME` for an independent run. To resume the existing run, use:

```python
RESUME_FROM = CKPT_DIR / "last.pth"
```

Re-run configuration and the matching training cell with unchanged settings.
After a runtime restart, also run the launcher-definition cell.
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
`LOG_EVERY=3` for CSV/console output: epochs 0, 3, 6, ..., 198, and 200 (68 rows for fresh runs).
Each row reports that epoch, not an average of the preceding three epochs.
Checkpoint history retains every epoch for resume. `LOG_EVERY` is a presentation
setting, outside the strict training configuration, so existing checkpoints can
resume with the new reporting interval. The helper defaults to `log_every=1`
for compatibility; the notebook passes its explicit setting.

Reported fields are train/validation L1, train/validation PSNR and SSIM, LR actually used,
global step, epoch/cumulative processing time, and best flag. Times include
validation/visual generation but exclude checkpoint I/O and disconnected time.
Fixed validation examples are saved at epochs 0 and 1, every 10 epochs, and the last
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

### train.py entry-point verification (2026-10-10)

- Full regression suite: 34 tests run, 32 passed, two CUDA checks skipped because
  CUDA is unavailable. Three new tests cover process-level notebook launching,
  old-helper checkpoint resume, CLI final evaluation, and invalid run requests.
- A checkpoint created with the former notebook setup resumes through `train.py`
  with exactly matching model, optimizer, scheduler, RNG state and non-timing
  history versus uninterrupted training. Changed configuration and premature
  final evaluation are rejected without replacing checkpoints.
- Executed the updated notebook launcher, smoke/full cells, plots and final export
  with the full v1 architecture on synthetic CPU data. Smoke used three epochs of
  the 200-epoch schedule; the full-mode check used a shortened four-epoch schedule.
  Separate folders, final smoke visuals and subprocess error propagation passed.
- Default CONFIG matches the previous notebook exactly. Python/notebook syntax
  and `git diff --check` passed. No Colab GPU or real-data 200-epoch run was made.

### Epoch-0 verification (2026-10-12)

- Full suite: 36 tests run, 34 passed, two CUDA tests skipped (CUDA unavailable).
- Initial checkpoint preserves model/optimizer/RNG/generator state; resume from
  epoch 0 matches uninterrupted training without repeating initial validation.
  Initial PSNR is excluded from best selection even when higher than trained scores.
- Legacy histories still resume without backfilling epoch 0. Inconsistent steps,
  scheduler state and empty history are rejected. Worker resume checks passed.
- Actual notebook/CLI smoke and shortened full runs on synthetic CPU data produced
  epoch-0 visuals, CSV rows 0/3 (smoke) and 0/3/4 (full check), valid curves with
  blank training/LR values, and final test exports. Syntax/whitespace checks passed.
  No real-data full training, commit or push was performed for this update.

## Console transcript

`train.py` mirrors Python stdout/stderr to `logs/training.log` while retaining
console output in Colab and terminal runs. Writes are flushed promptly. Each fresh,
resume or final-test evaluation session has a UTC timestamp and label. Resume and
evaluation append to the same file; they never truncate previous output. The log
includes training configuration, progress, warnings and session error tracebacks.

Logging begins after experiment/split validation succeeds, preserving fresh-run
overwrite protection. Earlier argument/configuration/dataset-validation errors
remain console-only. Native code or child workers writing directly to OS file
descriptors are not intercepted. A forcibly terminated runtime may not leave an
error traceback; flush does not guarantee remote Drive synchronization. Metrics
CSV and checkpoint/resume behavior are unchanged.

Verification (2026-10-12): all four CLI/launcher tests passed on CPU, including
live log visibility, warnings, failure/interruption tracebacks, append-on-resume,
overwrite rejection, unchanged checkpoint equivalence and notebook smoke output.
Python/notebook syntax and diff whitespace checks passed. No GPU/full-data run.
