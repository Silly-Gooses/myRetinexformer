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
