# LOLv2 Real checkpoint comparison

This analysis uses the existing test dataset and checkpoints. It does not run
training or modify model weights. The standard `forward()` method and the
training notebook remain unchanged.

In Colab, mount Google Drive and make this repository the working directory.
Run the three-image smoke test first:

```bash
python compare_models.py \
  --output-dir /content/drive/MyDrive/retinexformer_analysis
```

Inspect `smoke/review.md` and its comparison grids. Check that the low and
ground-truth panels match, that the light-up and output panels have sensible
brightness and colors, and that `smoke/summary.json` contains finite metrics.
Then run the full dataset:

```bash
python compare_models.py \
  --output-dir /content/drive/MyDrive/retinexformer_analysis \
  --full
```

The default paths match the Colab training notebook. Override any location with
`--dataset-root`, `--author-checkpoint`, or `--ours-checkpoint`. The dataset root
must contain `Test/Low` and `Test/Normal`. The author checkpoint must contain
`params`; the trained checkpoint must contain `model`. Both are loaded strictly
into the notebook's single-stage architecture.

The output directory contains `author/metrics.csv`, `ours/metrics.csv`,
`comparison.csv`, `summary.json`, and `review.md`. Each model's `images/`
directory contains its low, ground truth, final light-up, final output, and
per-stage images. The `improved/`, `similar/`, and `degraded/` folders contain
six-panel comparison grids and matching source images for every test pair.
Results under `smoke/` are separate from full-set results.

PSNR and SSIM use the repository's existing 8-bit RGB metric functions on the
clipped, rounded pixels saved in PNGs. The default grouping threshold is
`±0.5 dB`; change it with `--threshold`. Group labels are an analysis
convention. `review.md` lists representative cases and prompts for visual
inspection; it makes no unverified claims about image quality.
