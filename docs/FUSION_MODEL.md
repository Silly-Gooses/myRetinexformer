# Reproducing the 25/75 ensemble

## Installation and weights

Use Git and Python 3.11 or later. Activate a virtual environment and install PyTorch for your CPU or CUDA runtime. Run the following from the fusion branch checkout.

```sh
python -m pip install -r requirements-fusion.txt
git clone https://github.com/Fediory/HVI-CIDNet.git .fusion-local/HVI-CIDNet
git -C .fusion-local/HVI-CIDNet checkout --detach eb43d7d91e9a336c66856824ff9e4603ae41f408
curl -fL https://huggingface.co/Fediory/HVI-CIDNet/resolve/main/weights/LOLv2_real/best_PSNR.pth -o .fusion-local/best_PSNR.pth
```

Download `LOL_v2_real.pth» from the Retinexformer authors and place it at `.fusion-local/LOL_v2_real.pth». The in-repo architecture comes from commit `17215ba8ce65417ee9f18df948996225c1d1a931». HVI runs from its external checkout under the upstream MIT license. [Author weights](https://drive.google.com/drive/folders/1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV?usp=drive_link) · [HVI checkpoint](https://huggingface.co/Fediory/HVI-CIDNet/blob/main/weights/LOLv2_real/best_PSNR.pth) · [HVI license](https://github.com/Fediory/HVI-CIDNet/blob/eb43d7d91e9a336c66856824ff9e4603ae41f408/LICENSE)

Both commands verify the following SHA-256 digests before deserializing the corresponding state on CPU and loading it strictly.

- RetinexFormer `LOL_v2_real.pth» — `539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b»
- HVI-CIDNet `best_PSNR.pth» — `44a7f92cf02a1da322c1a34b3aa940341dc93fd53840b7f19b49f461200b00a6»

## Inference

Pass `--device auto», `cpu», or `cuda». Auto selects CUDA when available. The CLI reads input as RGB8 and predicts in float32. Each invocation creates a new output directory. Use a fresh directory when repeating a run.

```sh
python scripts/run_fusion.py --input example.png --output fusion-runs/example --retinex-checkpoint .fusion-local/LOL_v2_real.pth --hvi-checkpoint .fusion-local/best_PSNR.pth --hvi-repo .fusion-local/HVI-CIDNet
python scripts/run_fusion.py --input images --compare --output fusion-runs/comparison --retinex-checkpoint .fusion-local/LOL_v2_real.pth --hvi-checkpoint .fusion-local/best_PSNR.pth --hvi-repo .fusion-local/HVI-CIDNet
```

RetinexFormer uses 40 features, one stage and block counts `[1, 2, 2]». It receives right/bottom reflection padding to a multiple of four. Each of eight passes applies a combination of height flip, width flip and 90° rotation. The inverse transforms run in reverse order before the float32 average. HVI receives reflection padding to a multiple of eight and gamma 1.0, with `gated=False», `gated2=True» and `alpha=0.8». Each prediction is cropped and clamped before the fixed RGB blend. The PNG conversion rounds the final floats with `numpy.rint».

The full-resolution forward pass must fit the selected device. Very small images can require more reflection padding than their dimensions allow. Inputs to the official metrics must be at least 11×11 for the SSIM window.

## Official Test evaluation

Download LOLv2 from the authors' dataset instructions and use the `Real_captured/Test» directory containing `Low» and `Normal». Extract `per_image_official_test.csv» from `RetinexFormer_x8_HVI_FixedFusion_Official100_Report.zip» into `.fusion-local/reference.csv» to compare against the original 25/75 experiment. The separate 50/50 experiment has different outputs. [Dataset instructions](https://github.com/caiyuanhao1998/Retinexformer#2-prepare-dataset)

```sh
python scripts/evaluate_fusion.py --test-root /data/LOLv2/Real_captured/Test --reference-csv .fusion-local/reference.csv --output fusion-runs/official100 --save-images --retinex-checkpoint .fusion-local/LOL_v2_real.pth --hvi-checkpoint .fusion-local/best_PSNR.pth --hvi-repo .fusion-local/HVI-CIDNet --device cuda
python -m unittest discover -s tests -v
```

The evaluator checks for 100 matching PNG filenames. It predicts from each low-light image before opening its matching ground truth and checking dimensions. PSNR uses clipped RGB float32, a peak of 1, and a value of 100 dB for identical images. SSIM uses rounded RGB8, an 11×11 Gaussian window with sigma 1.5, valid pixels, and the average across color channels. Both reported dataset scores are arithmetic means over the 100 per-image scores.

The run writes `per_image_official_test.csv», `summary.json», and `manifest.json». The manifest records the checkpoint hashes, source revisions, device, versions, and filenames. With `--reference-csv», the summary reports signed mean differences, mean absolute differences, worst absolute differences and filenames for all eight comparisons. A dataset-mean difference above 0.01 dB PSNR or 0.001 SSIM exits with an error after saving the report. Omit the reference argument to compute scores alone. Use `--save-images» to also retain the 400 full-resolution predictions.

## Reproduction

On 2026-11-16, both networks ran again on all 100 official pairs, from `00690.png» through `00789.png». The local run used a GeForce GTX 1650 Max-Q, Python 3.14.3, PyTorch 2.14.1+cu126, CUDA 12.6, cuDNN 91002, NumPy 2.5.4, OpenCV 5.0.0, einops 0.8.2 and huggingface_hub 2.4.0.

- **Original RetinexFormer** measured **22.795486 dB / 0.839669 SSIM**.
- **RetinexFormer ×8** measured **22.821172 dB / 0.849834 SSIM**.
- **HVI-CIDNet** measured **23.890488 dB / 0.865594 SSIM**.
- **Fixed fusion** measured **24.282168 dB / 0.873287 SSIM**, a gain of 0.391680 dB over HVI alone on these images.

The original report ZIP supplied 100 reference rows. All 800 comparisons by filename and metric were within 0.01 dB or 0.001 SSIM. Absolute differences below give the mean across images followed by the maximum and its filename for each metric.

- **Original RetinexFormer** — PSNR 4.54e-7 / 1.78e-6 dB (`00690.png»). SSIM 1.92e-7 / 7.05e-7 (`00789.png»).
- **RetinexFormer ×8** — PSNR 4.71e-7 / 1.59e-6 dB (`00739.png»). SSIM 1.40e-7 / 8.25e-7 (`00704.png»).
- **HVI-CIDNet** — PSNR 9.20e-7 / 4.44e-6 dB (`00722.png»). SSIM 1.16e-7 / 4.56e-7 (`00708.png»).
- **Fixed fusion** — PSNR 6.87e-7 / 3.18e-6 dB (`00722.png»). SSIM 1.32e-7 / 5.19e-7 (`00733.png»).

For the seven distinct filenames above, a second local pass used the original notebook's inference and metric functions. All 28 PNGs were pixel-identical to the CLI outputs and all 56 scores agreed within 1e-10. The original run used Colab. Earlier evaluation of the official Test set informed the fixed weighting, so this reproduction verifies agreement on the same 100 pairs. An independent holdout would be needed to assess generalization.
