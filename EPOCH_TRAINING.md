# Canny V1: official LOL-v2-real workflow (branch v1.1)

Train from scratch on all **689** `Real_captured/Train` pairs. Validate on all
**100** `Real_captured/Test` pairs during training. Independently test a selected
checkpoint on those same 100 pairs. No images are moved or randomly partitioned.

**The official Test set selects the best checkpoint. Its final score is therefore
not an independent held-out test evaluation.** Describe it as the official
RetinexFormer validation/testing protocol in thesis comparisons.

## What changed

| Area | Previous epoch workflow | Branch 1.1 |
| --- | --- | --- |
| Training membership | 620 of 689 pairs | All 689 Train pairs |
| Validation membership | 69 pairs reserved from Train | All 100 official Test pairs |
| Validation timing | Every epoch | Every 1,000 batch steps, plus completion |
| Training augmentation | Paired crop/flips/rotations; no MixUp | Upstream padding/crop/eight transforms; paired Beta(1.2, 1.2) MixUp with identity |
| Gradient clipping | None | Norm 0.01, after AMP unscaling |
| Validation PSNR | Rounded 8-bit images | Clamped, unquantized RGB, float64 MSE |
| Testing | Legacy hardcoded baseline script or completion-gated export | Explicit selected Canny V1 checkpoint; no training dependency |
| Checkpoints | Replaced best/last | Immutable periodic/best snapshots plus best/last aliases |
| Membership record | Random `split.json` | Official `dataset_manifest.json` with decoded image hashes |

The Canny architecture and thresholds are unchanged. Edges are extracted inside
the model from the actual low-light tensor, **after** crop/augmentation/MixUp
for training, or after full-image padding for evaluation. No precomputed edge
maps are mixed or reused from a different input. The four required qualitative
panels (input, lit-up, enhanced, GT) and the additional Canny panel are retained.

Implementation files:

| Files | Responsibility |
| --- | --- |
| `train.py`, `official_workflow.py`, `training_utils.py` | Official membership, epoch execution, validation events, MixUp/clipping, logs and recovery |
| `dataLoader.py`, `protocol_metrics.py` | Official preprocessing and separate float-PSNR/uint8-SSIM evaluation/export |
| `test.py`, `smoke_test.py`, `official_lolv2_config.json` | Selected-checkpoint testing, two-update verification and editable defaults |
| `Lightweight_RetinexFormer_Colab_Full_Training_v2.ipynb` | Branch-1.1 launch, Drive preflight, curves and independent testing |
| `tests/official_fixtures.py`, `tests/test_official_workflow.py`, `tests/test_train_cli.py` | Synthetic protocol and integration regressions |
| `EPOCH_TRAINING.md`, `EVALUATION.md`, `V1_CANNY.md` | Commands, numerical conventions, compatibility and remaining differences |

`RetinexFormer_arch.py`, `RetinexFormer_arch_og.py`, `canny.py`, legacy analysis
helpers and dependency requirements are unchanged.

## Install and configure

From the repository root:

```bash
python -m pip install -r requirements.txt
cp official_lolv2_config.json run_config.json
```

Edit `dataset_root` in `run_config.json` to the directory containing `Train` and
`Test`, for example `/content/drive/MyDrive/datasets/LOLv2/Real_captured` in Colab.
Pair names must match exactly between each `Low` and `Normal` directory. Matching
filenames across Train/Test are allowed; shared paths or identical decoded low
or normal images across the two splits are rejected. Every pair must decode,
have matching dimensions, and support the 11×11 SSIM window.

Defaults preserved from the notebook: 200 epochs; 3 warm-up epochs;
LR `1e-6 → 1e-4 → 1e-6`; Adam, betas `(0.9, 0.999)`, zero weight decay; mean L1;
batch size 8; patch size 256; seed 42; 2 training workers; AMP disabled;
`n_feat=40`, one stage, blocks `[1,2,2]`; unchanged Canny settings.
Validation/testing use full images, batch size 1, no shuffle or augmentation,
and no evaluation autocast. Existing dependencies suffice.

New required configuration fields are `dataset_protocol`, `val_freq_steps`,
`mixup_beta`, and `grad_clip_norm`; the example supplies their official defaults.
`val_fraction` is invalid in the new trainer. Retain the same configuration on
resume. Old V1 model weights remain testable but old random-split histories
cannot resume this protocol. Start a **new experiment**, as requested.

## Preflight and lightweight smoke

```bash
python train.py --config run_config.json --output-root experiments --preflight
python smoke_test.py --config run_config.json \
  --output-dir experiments/Proposed_V1_QuickSmoke --device auto
```

Preflight checks all 689/100 pairs, shapes, pairing and cross-split content
isolation. Quick smoke uses the configured production architecture/patch size,
performs two training updates, asserts that weights changed, validates all 100
pairs without changing weights, saves and reloads weights, and independently
exports all 100 test predictions/metrics and qualitative panels. It does not run
200 epochs. Choose a new output directory for each smoke invocation. Quick-smoke
weights are inference snapshots, not production resume states.

To also verify checkpoint storage on mounted Google Drive:

```bash
python train.py --config run_config.json --output-root experiments --preflight \
  --check-storage /content/drive/MyDrive/RetinexFormer_full_training
```

The probe atomically writes and reloads a temporary tensor checkpoint and cleans
it up. Both the trainer and smoke script support `--check-storage PATH`.
This verifies the mounted filesystem, not remote Drive synchronization after
a runtime disappears. Reopen the saved checkpoint after reconnecting to confirm
cross-runtime persistence. The notebook writes experiment outputs directly to
Drive, preserving its existing backup/persistence behavior.

For the previous, longer three-epoch smoke (isolated `_Smoke3` directory):

```bash
python train.py --config run_config.json --output-root experiments --mode smoke
```

This traverses all training pairs for three epochs, keeps the 200-epoch LR
schedule, and runs a clearly labeled `smoke_final` validation at step 261.
Neither smoke mode silently truncates the configured production schedule.

## Full training and resume

```bash
python train.py --config run_config.json --output-root experiments --mode full
python train.py --config run_config.json --output-root experiments --mode full --resume
```

The default experiment is `Proposed_RetinexFormer_V1_Official_Epoch200`.
Use `--experiment-name NAME` for another fresh experiment. Fresh runs refuse
nonempty output directories. `--resume-from` accepts only that experiment's
`checkpoints/last.pth`.

With `drop_last=False`, `len(train_loader)=ceil(689/8)=87`: 86 complete batches
and one single-pair batch. Every pair contributes once per epoch. Singleton
MixUp is naturally the identity. The 200-epoch schedule comprises **17,400 batch
steps**. Counters advance inside the actual loop; AMP-skipped optimizer updates,
if AMP is enabled, still count as batch iterations, as in upstream.

Validation runs exactly at steps 1000, 2000, …, 17000 and finally 17400. The
interval is 1000/87 ≈ **11.4943 epochs**, first occurring at epoch 12, batch 43.
It is not rounded to an epoch frequency. Initial step-0 validation is diagnostic
and cannot select best. A final event is not duplicated when it lands exactly
on an interval. `val_freq_steps` may be changed for an explicitly different run;
small-interval regression tests also cover multiple validations within an epoch.

The scheduler still advances only at the start of an epoch. Validation disables
gradients, restores model mode and preserves training RNG/loader-generator state.
Training loss and online crop PSNR/SSIM are sample-weighted epoch means; they
are not an additional final-model pass over the training set.

## Checkpoints and outputs

```text
experiments/Proposed_RetinexFormer_V1_Official_Epoch200/
  config.json
  dataset_manifest.json
  checkpoints/
    last.pth                     # Full state, each completed epoch (including 0)
    best.pth                     # Alias to selected immutable model snapshot
    step_<step>_<reason>_<id>.pth # Immutable periodic/best model snapshots
  logs/
    training.log                 # Streaming console/session log
    train_log.csv                # Every epoch, including diagnostic epoch 0
    training_metrics.csv         # Compatibility CSV filtered by --log-every
    validation_log.csv           # Every actual validation event
  validation/metrics.csv         # Per-image metrics with event/step/protocol
  visuals/epoch_<epoch>/          # Low, edge, lit_up, output, GT, and grids
  testing/<evaluation-name>/
    enhanced_images/
    per_image_metrics.csv
    summary_metrics.csv
    provenance.json
    visuals/
```

`last.pth` contains model, optimizer, epoch scheduler, AMP scaler, RNG and loader
states, official membership/content manifest, histories, and committed best
selection. Resume begins after its last completed epoch; an interrupted partial
epoch is replayed. CSV files and the best alias are reconciled from committed
state. Uncommitted immutable snapshots remain preserved but are not silently
adopted as the best checkpoint. Periodic model snapshots are saved every 1,000
batch steps; improving validation also saves an immutable snapshot. These model
snapshots and `best.pth` support testing, while `last.pth` supports full resume.
Only aliases inside the new experiment are replaced. Existing experiments and
immutable snapshots are not overwritten.

`--log-every` affects console reporting and the compatibility CSV only. Canonical
training/validation logs retain every event. Epoch CSV validation cells are blank
unless validation occurred at that epoch's final step; use `validation_log.csv`
for the complete curve. Checkpoint selection uses PSNR only and retains the first
checkpoint on a tie. Visuals are saved at epoch 0, 1, every 10, and completion.

## Independent testing

```bash
python test.py \
  --checkpoint experiments/Proposed_RetinexFormer_V1_Official_Epoch200/checkpoints/best.pth \
  --dataset-root /path/to/Real_captured \
  --output-dir experiments/Proposed_RetinexFormer_V1_Official_Epoch200/testing/best_run \
  --device auto
```

Any compatible selected V1 checkpoint can be tested, including `last.pth` or
an immutable snapshot, before epoch 200. Only `Test` is required on disk. Model
and Canny configuration come from checkpoint metadata. Supply `--config PATH`
for bare/older wrappers without metadata. Strict loading rejects baseline
three-channel weights instead of silently disabling Canny. Existing output
folders are rejected. `train.py --mode full --evaluate-only` remains an alias
for independent testing of `best.pth`, using a fresh timestamped output folder.

## Metric protocol and remaining differences

Reference sources inspected for this implementation:

- [Official LOL-v2-real options](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Options/RetinexFormer_LOL_v2_real.yml)
- [Dataset](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/data/paired_image_dataset.py), [transforms](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/data/transforms.py), and [DataLoader](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/data/__init__.py)
- [Training loop](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/train.py) and [restoration/validation model](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/models/image_restoration_model.py)
- [Validation metrics](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/metrics/psnr_ssim.py), [tensor conversion](https://github.com/caiyuanhao1998/Retinexformer/blob/master/basicsr/utils/img_util.py), [testing](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Enhancement/test_from_dataset.py), and [testing metrics](https://github.com/caiyuanhao1998/Retinexformer/blob/master/Enhancement/utils.py)

Validation uses RGB PSNR, zero border crop, no Y conversion, and float64 MSE on
clamped float predictions. Upstream `use_image=false` scores CPU tensors; its
preceding `tensor2img` call clamps that shared CPU storage in place, even though
the rounded images are not used for PSNR. This implementation clamps explicitly
without mutating the caller's prediction. Exact validation matches yield infinity.

Testing clamps float32 RGB output, removes reflection padding, and computes
float32 PSNR before PNG quantization (exact match: 100 dB). Both workflows use
the testing script's RGB-channel-averaged 11×11, sigma-1.5 Gaussian SSIM on rounded
uint8 images. SSIM is a thesis addition to validation, whose official config only
requests PSNR. Dataset means average image scores, not pooled pixel errors.
A saved PNG's recomputed PSNR may differ slightly from the reported float PSNR.

The restoration reproduces dataset/evaluation behavior, **not the complete
original training budget**. Deliberate differences remain: Canny V1; 200 epochs
(17,400 batches) instead of 150,000 iterations; the existing 3-epoch warm-up and
epoch cosine schedule with 1e-4 peak instead of the upstream cyclic schedule;
seed 42 and deterministic training; two training workers; no dropped final
partial batch; extra initial diagnostics, SSIM, visualizations and immutable
checkpoint retention. Validation/testing use direct full-image inference for
LOL-v2-real; no GT-mean correction or self-ensemble. The existing tiny-image
replicate-padding fallback is retained where reflection padding is impossible.
The upstream high-resolution split-and-test branch is outside this dataset's
normal dimensions and is not introduced here. Legacy author/retrained analysis
continues to use its existing saved-pixel metrics, explicitly separate from this
protocol.

## Verification

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Tests use synthetic images (including fixtures with 689/100 counts) and reduced
models to exercise coverage, augmentation, metric formulas, Canny input, weight
updates, read-only evaluation, exact validation events, interrupted replay,
checkpoint protection, notebook/CLI integration and output files. These do not
substitute for running the preflight/smoke on your real dataset. CUDA tests skip
when CUDA is unavailable; multi-worker tests need working shared memory.

No full training, actual LOL-v2 scores, or live Google Drive verification is
claimed from local synthetic tests. Run the provided preflight and quick smoke
in Colab before the full experiment.

Local verification on 2026-10-14 (Python 3.14, PyTorch 2.12, CPU): **46 tests
run, 44 passed, 2 CUDA tests skipped**. The multi-worker resume regression passed
outside the sandbox (the sandbox itself blocks `torch_shm_manager`). The nine
new protocol tests also passed after the final selected-checkpoint read/provenance
fix. No real dataset, trained production checkpoint, CUDA device or mounted
Google Drive was available locally.
