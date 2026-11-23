# Joint Regularized Luminance–Chrominance Fusion v2

This implementation packages the supplied research notebook without changing its numerical method. It fits two scalar fusion parameters while keeping both official pretrained backbones frozen. The complete inference pipeline still executes **both networks** on every low-light image; two parameters do not imply a two-parameter end-to-end model.

## Background and attribution

[RetinexFormer](https://github.com/caiyuanhao1998/Retinexformer/tree/2dd9ae05f34a34e2df3b955f743215ac3cda0f82) and [HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet/tree/eb43d7d91e9a336c66856824ff9e4603ae41f408) provide complementary low-light enhancement outputs. RetinexFormer uses illumination-guided restoration; CIDNet separates intensity and color processing through its HVI representation. Here their RGB predictions are combined after inference.

The project's FusionNet motivation is output-level fusion of complementary enhancement estimates. The specific FusionNet citation remains pending; no particular paper's architecture or claims are attributed here. Linear fusion itself is an established general idea. This experiment studies a particular luminance/chrominance parameterization and Train-only regularized selection protocol, rather than claiming invention of linear fusion.

## Architecture and numerical conventions

For low-light input `L`, compute frozen predictions `R = RetinexFormer(L)` and `H = HVI-CIDNet(L)`. Each is cropped to the original size and clipped to `[0, 1]`. Convert their float32 RGB arrays using OpenCV `COLOR_RGB2YCrCb`; the ordering is **Y, Cr, Cb**.

```text
Y_fused  = alpha_y * Y_R  + (1 - alpha_y) * Y_H
Cr_fused = alpha_c * Cr_R + (1 - alpha_c) * Cr_H
Cb_fused = alpha_c * Cb_R + (1 - alpha_c) * Cb_H
output   = clip(OpenCV_YCrCb_to_RGB(fused), 0, 1)
```

Weights are RetinexFormer contributions. The proposed method shares one pair across all three datasets, with no CNN, attention module, selector, or backbone fine-tuning. Different datasets still use different pretrained checkpoint pairs.

RetinexFormer uses `in_channels=3`, `out_channels=3`, `n_feat=40`, `stage=1`, `num_blocks=[1,2,2]`. Right/bottom reflection padding makes dimensions divisible by four. If either original dimension is at least 3000, width is padded to a multiple of eight and the two interleaved column subsets are inferred separately, exactly as in the source notebook. HVI pads to multiples of eight. Both run float32 in evaluation/inference mode. No mixed precision, GT-mean correction, or ×8 TTA is introduced.

OpenCV's float conversion is approximately invertible: endpoint and equal-weight RGB equivalence tests use the notebook's `5e-4` absolute tolerance. Final RGB is always clipped.

## Train-only optimization

The objective maximizes the equal-dataset average of mean per-image float RGB PSNR, minus

```text
lambda * ((alpha_y - 0.5)^2 + (alpha_c - 0.5)^2).
```

Each dataset contributes one third, independently of its image count. This is a mean of image PSNR values, not PSNR computed from a pooled MSE. The symmetric regularizer favors neutral weights without using a Test-derived anchor.

The inverse color transform gives an affine RGB decomposition `base + alpha_y*dy + alpha_c*dc`. Cache its six per-image quadratic MSE coefficients, plus three RGB-baseline coefficients and standalone scores. Preserve the notebook's float32 subtraction followed by float64 accumulation. **Fitting scores use the unclipped affine approximation; final evaluation clips RGB.** Replacing this with clipped fitting would change the experiment.

- Default Train sizes: LOL-v1 485, LOL-v2 Real 689, LOL-v2 Synthetic 900. All pairs are used unless `--fit-train-count` is explicitly set.
- Exact matching filenames, sorted order, and distinct resolved Train/Test paths are required. Basenames may repeat across different split folders. Test file names are checked for protocol bookkeeping; fitting never reads Test pixels or GT.
- Three deterministic folds use seed `20261123 + sum(map(ord, dataset_name))` with NumPy's generator and the notebook's permutation/modulo assignment. Optional subset sampling uses the original Python `random.Random` scheme.
- Lambda candidates: `[0.0, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0]`. Choose the strongest lambda within `0.005 dB` of the best mean held-out fold score. Joint CV holds out the corresponding fold from every dataset.
- Bound both parameters to `[0,1]`. L-BFGS-B starts at `(0,0)`, `(1,1)`, `(0,1)`, `(1,0)`, `(.5,.5)`, `(.2,.8)`, `(.8,.2)`, with `maxiter=220`, `ftol=1e-12`; starting points also remain candidates.
- The scalar RGB baseline uses bounded scalar optimization with `xatol=1e-7` and explicit endpoint candidates.
- Refit the selected settings on all chosen Train rows and write full-precision JSON before Test evaluation. Evaluators cannot invoke the fitter.

Protocol locks include filenames, checkpoint/source identities, numerical configuration, dataset roots, and implementation hashes. Changed settings or incompatible caches require a new output directory. Existing selected JSON is compared exactly and is never silently replaced. Dependency changes can change optimization at floating-point precision; manifests record the actual runtime.

## Datasets, sources, and checkpoints

Default relative layouts under `--data-root`:

| Dataset identifier | Dataset root | Train input / GT | Test input / GT | Test count |
|---|---|---|---|---:|
| `LOLv1` | `LOLv1` | `Train/input`, `Train/target` | `Test/input`, `Test/target` | 15 |
| `LOLv2_Real` | `LOLv2` | `Real_captured/Train/Low`, `Real_captured/Train/Normal` | `Real_captured/Test/Low`, `Real_captured/Test/Normal` | 100 |
| `LOLv2_Synthetic` | `LOLv2` | `Synthetic/Train/Low`, `Synthetic/Train/Normal` | `Synthetic/Test/Low`, `Synthetic/Test/Normal` | 100 |

Use repeatable `--dataset-root DATASET=PATH` to override a dataset root while preserving its internal layout. The Python API also accepts explicit layout dictionaries. Images must be large enough for reflection padding; evaluation SSIM requires at least 11×11 pixels.

| Dataset | RetinexFormer file | HVI file relative to its weight directory | HVI setting |
|---|---|---|---|
| LOL-v1 | `LOL_v1.pth` | `LOLv1/w_perc.pth` | `gated=True`, `gated2=False` |
| LOL-v2 Real | `LOL_v2_real.pth` | `LOLv2_real/best_PSNR.pth` | `gated=False`, `gated2=True`, `alpha=0.8` |
| LOL-v2 Synthetic | `LOL_v2_synthetic.pth` | `LOLv2_syn/wo_perc.pth` | both gates false |

The required source revisions are RetinexFormer `2dd9ae05f34a34e2df3b955f743215ac3cda0f82` and HVI-CIDNet `eb43d7d91e9a336c66856824ff9e4603ae41f408`. Source checks reject changed architecture files. State dictionaries load with `weights_only=True` and `strict=True`; backbone parameters are frozen.

RetinexFormer hashes are preserved from the notebook. HVI hashes, originally logged rather than pinned in the notebook, are now checked against the recovered reference report. All six hashes are recorded in the [reference manifest](../reference_results/joint_lumachroma_v2/report_manifest.json). Missing RetinexFormer files can be fetched from the author's `CaiYuanhao/Retinexformer` Hugging Face repository and must pass the same hashes. Supply the official HVI files locally; no checkpoints are committed.

## Local execution

Use Python 3.11 or later and install the appropriate PyTorch CPU/CUDA build for your machine. From the repository root:

```sh
python -m pip install -r requirements-joint-fusion.txt
git clone https://github.com/caiyuanhao1998/Retinexformer.git .fusion-local/Retinexformer-joint
git -C .fusion-local/Retinexformer-joint checkout --detach 2dd9ae05f34a34e2df3b955f743215ac3cda0f82
git clone https://github.com/Fediory/HVI-CIDNet.git .fusion-local/HVI-CIDNet
git -C .fusion-local/HVI-CIDNet checkout --detach eb43d7d91e9a336c66856824ff9e4603ae41f408
```

Place RetinexFormer checkpoints in `checkpoints/ret` and the HVI dataset folders in `checkpoints/hvi`. Commands below use all three datasets by default:

```sh
python scripts/fit_joint_fusion.py --data-root datasets --ret-repo .fusion-local/Retinexformer-joint --hvi-repo .fusion-local/HVI-CIDNet --ret-weights checkpoints/ret --hvi-weights checkpoints/hvi --output joint-fusion-runs/alltrain --device cuda
python scripts/evaluate_joint_fusion.py --data-root datasets --ret-repo .fusion-local/Retinexformer-joint --hvi-repo .fusion-local/HVI-CIDNet --ret-weights checkpoints/ret --hvi-weights checkpoints/hvi --weights-json joint-fusion-runs/alltrain/selected_weights_train_only.json --output joint-fusion-runs/alltrain --device cuda
```

For a pilot, fitting accepts `--fit-train-count 200`; evaluation accepts `--smoke-max-test 3`. Choose a different output directory for different protocols. `--datasets LOLv2_Real` limits a run, but fitting one dataset does not reproduce the three-dataset joint method. `--device cpu` is supported for functional use but full teacher inference can be slow.

To use the archived full-precision weights directly without fitting:

```sh
python scripts/run_joint_fusion.py --dataset LOLv2_Real --input example.png --ret-repo .fusion-local/Retinexformer-joint --hvi-repo .fusion-local/HVI-CIDNet --ret-weights checkpoints/ret --hvi-weights checkpoints/hvi --weights-json reference_results/joint_lumachroma_v2/selected_weights_train_only.json --output joint-fusion-runs/example --device cuda
```

Input can be a file or folder. `--compare` saves all seven methods; otherwise only joint regularized fusion is saved. Inference requires a new output directory and no ground truth. The evaluator can likewise take the reference JSON, but its outputs remain a new evaluation, separate from archived results.

## Google Colab

Open [the v2 notebook](../notebooks/RetinexFormer_HVI_RegularizedJoint_LuminanceChrominance_v2_Colab.ipynb) in Colab, select a GPU runtime, and execute top-to-bottom. It installs dependencies, mounts Drive interactively, clones the integration code and pinned teachers, checks data/weights, fits, freezes, evaluates, and displays comparisons.

Set storage, data, checkpoint, and output roots in configuration cell 2. Environment variables `JOINT_FUSION_STORAGE`, `JOINT_FUSION_DATA`, `JOINT_FUSION_RET_WEIGHTS`, `JOINT_FUSION_HVI_WEIGHTS`, and `JOINT_FUSION_OUTPUT` override generic defaults. Set `JOINT_FUSION_REVISION` to the published commit hash for an immutable implementation checkout. The default is this feature branch. The notebook imports the same Python implementation as the CLIs; it does not contain an independent copy of the fitting algorithm.

## Metrics and outputs

Float PSNR computes RGB MSE after casting prediction and GT to float64, with a floor of `1e-14` and peak 1. Identical inputs therefore score 140 dB. GT is RGB8 converted to float32 by division by 255 before this calculation.

For saved RGB8, RetinexFormer uses `np.round(clip(x)*255)`, while HVI and every fusion use truncating `astype(np.uint8)`. RGB8 PSNR uses float32 squared errors and denominator offset `1e-8`. RGB8 SSIM averages three channels using an 11×11 Gaussian window with sigma 1.5, the valid interior `[5:-5,5:-5]`, and constants based on 255. These intentionally differ from the legacy fixed-fusion branch's conventions.

Expected outputs:

- Per dataset: `train_fit_per_image_coefficients.csv` and `test_per_image.csv`.
- Root: `train_cross_validation_folds.csv`, `train_cross_validation_summary.csv`, `selected_weights_train_only.json`, `test_per_image_all.csv`, and `test_summary.csv`.
- Provenance: `protocol_lock.json`, `fit_manifest.json`, `evaluation_lock.json`, and `report_manifest.json`. Standalone inference writes its own lock and manifest.
- Full predictions for `ycc_regularized` and `ycc_joint_regularized`; the first three examples include all seven methods, input, and GT. Use `--no-save-predictions` or `--preview-count` to control evaluation output.
- A report ZIP containing CSV/JSON and previews, excluding full prediction directories. Keep it outside Git.

## Previously measured reference results

These are **previously measured results**, recovered from the supplied experiment report. They are not claimed as newly reproduced by this refactor. The archive provenance and original notebook SHA-256 are recorded in [provenance.json](../reference_results/joint_lumachroma_v2/provenance.json).

| Dataset | RetinexFormer float PSNR | HVI float PSNR | Proposed float PSNR | Proposed RGB8 SSIM | Gain vs best standalone |
|---|---:|---:|---:|---:|---:|
| LOL-v1 | 25.1554 | 23.8436 | 24.5604 | 0.86035 | −0.5950 dB |
| LOL-v2 Real | 22.7955 | 23.8905 | 24.3603 | 0.87383 | +0.4698 dB |
| LOL-v2 Synthetic | 25.6727 | 25.7199 | 26.3735 | 0.94392 | +0.6537 dB |

The archived shared parameters are `alpha_y=0.2992029777427523`, `alpha_c=0.40713060525512`, `lambda=0.5`. Their full precision is preserved in JSON; rounded values in prose are never used for evaluation.

All five previously measured ablations, float RGB PSNR / RGB8 SSIM:

| Method | LOL-v1 | LOL-v2 Real | LOL-v2 Synthetic |
|---|---:|---:|---:|
| Dataset RGB fit | 25.0403 / 0.86177 | 22.8936 / 0.84228 | 25.8053 / 0.94250 |
| Dataset YCrCb fit | 24.9024 / 0.85924 | 22.8273 / 0.84214 | 25.8064 / 0.94261 |
| Dataset regularized YCrCb | 24.9379 / 0.86077 | 22.8276 / 0.84225 | 25.8249 / 0.94274 |
| Joint YCrCb | 24.4785 / 0.86048 | 24.3425 / 0.87366 | 26.3066 / 0.94411 |
| Joint regularized YCrCb | 24.5604 / 0.86035 | 24.3603 / 0.87383 | 26.3735 / 0.94392 |

The complete reference CSVs include RGB8 PSNR, per-image scores, and CV selections. Do not retrospectively select a Test-winning ablation and call it an untouched-test improvement.

## Limitations and verification

LOL-v1 PSNR decreases despite improvements on the two LOL-v2 datasets. Neither regularization nor joint fitting guarantees improvement. Dataset-specific checkpoints prevent a claim of a universal unseen-domain backbone. Original Train images may have been seen by the pretrained teachers, including images held out during fusion CV. Prior experiments already inspected the official Test benchmarks, so this is not a fresh blind-test confirmation. Independent holdout data are needed for stronger generalization claims.

Run checkpoint-free tests and syntax checks with:

```sh
python -m unittest discover -s tests -v
python -m compileall -q fusion_model/joint_lumachroma scripts tests
git diff --check
```

See [implementation verification](joint_lumachroma_verification.md) for the actual local checks and their limits. Full three-dataset inference has not been re-run as part of publication.
