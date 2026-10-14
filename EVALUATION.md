# Evaluation entry points

For the branch-1.1 Canny V1 official protocol, use `test.py --checkpoint PATH
--dataset-root ROOT --output-dir NEW_DIRECTORY`. See [EPOCH_TRAINING.md](EPOCH_TRAINING.md)
for runnable commands and numerical conventions. It evaluates all 100 official
Test pairs without loading Train or restarting training. PSNR is computed on
clamped float RGB; SSIM and PNGs use rounded uint8 RGB. Checkpoint selection also
uses Test, so this is not an independent held-out evaluation.

The author-versus-retrained analysis below is a separate diagnostic workflow.
Its existing metrics intentionally describe saved 8-bit pixels and have not been
silently changed to the new training/testing protocol.

# LOL-v2 Real author-versus-retrained analysis

This inference-only workflow compares the official author checkpoint and a
retrained checkpoint with the same low-light input. It does not train models or
change the normal `forward()` behavior.

Install dependencies after cloning:

```bash
pip install -r requirements.txt
```

Run a five-image smoke test with explicit paths:

```bash
python analysis/compare_author_vs_retrained.py \
  --author_ckpt /path/to/LOL_v2_real.pth \
  --retrained_ckpt /path/to/best_psnr.pth \
  --low_dir /path/to/LOLv2/Real_captured/Test/Low \
  --gt_dir /path/to/LOLv2/Real_captured/Test/Normal \
  --output_dir results/author_vs_retrained_smoke \
  --max_images 5
```

For the complete LOL-v2 Real test set, use the same command without
`--max_images`:

```bash
python analysis/compare_author_vs_retrained.py \
  --author_ckpt /path/to/LOL_v2_real.pth \
  --retrained_ckpt /path/to/best_psnr.pth \
  --low_dir /path/to/LOLv2/Real_captured/Test/Low \
  --gt_dir /path/to/LOLv2/Real_captured/Test/Normal \
  --output_dir results/author_vs_retrained
```

The script requires exact low/GT filename pairs and processes them in sorted
order. It supports direct state dictionaries and common `params`, `model`,
`state_dict`, and nested `checkpoint` wrappers. A uniform DataParallel
`module.` prefix is removed; all model weights are then loaded strictly.

The output directory contains lossless PNGs in `low/`, `author_litup/`,
`retrained_litup/`, `author_output/`, `retrained_output/`, `gt/`, and `grids/`,
as well as `per_image_analysis.csv` and `summary.csv`.

Final outputs are scored against ground truth using PSNR/SSIM. The output group
uses retrained-minus-author PSNR: Improved at `≥ +0.5 dB`, Degraded at
`≤ −0.5 dB`, otherwise Similar. The Lit-up image is
`input × illumination_map + input`; it has no dedicated ground truth target.
It is therefore analyzed neutrally through luminance, dark/highlight ratios,
MAE, and Author-vs-Retrained SSIM instead of Lit-up-to-GT quality scores.

`per_image_analysis.csv` also includes `author_litup_psnr`,
`retrained_litup_psnr`, `litup_psnr_delta`, `author_litup_ssim`,
`retrained_litup_ssim`, and `litup_ssim_delta`. These are Author Lit-up vs GT,
Retrained Lit-up vs GT, and retrained-minus-author deltas. They use the same
saved 8-bit RGB metric implementation as final-output metrics, but are
explicitly diagnostic-only because GT is not a true Lit-up-stage target.
