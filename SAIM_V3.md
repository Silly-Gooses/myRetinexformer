# SAIM v3 experiment

`v3` starts from `main`. The original `RetinexFormer_arch.py` is unchanged. Relative to `RetinexFormer_arch_og.py`, the `main` architecture only adds `forward_with_intermediate` analysis methods. Its estimator returns `illu_fea` and `illu_map`; the latter brightens the RGB input, while `illu_fea` passes through the denoiser's encoder, bottleneck, and decoder guidance attention. In `IG_MSA`, illumination multiplies the value tensor. SAIM modulates `illu_fea` once per stage, immediately before the denoiser, so every attention level receives the modulated guidance while the RGB illumination map and all baseline weights remain unchanged.

The gate uses a depthwise 3×3 convolution, GELU, pointwise 1×1 convolution, and sigmoid. `I_modulated = I + alpha × gate(I) × I`; `alpha` begins at zero. Consequently, gate weights first get nonzero gradients after alpha starts moving. No edge, wavelet, or extra input channels are used.

## Checkpoint and reproducibility

The expected author checkpoint is `D:\retinexformer_improvement\pretrained_weights\LOL_v2_real.pth`; edit `CHECKPOINT_PATH` in the notebook if stored elsewhere. The default architecture is RGB, one stage, 40 features, and blocks `[1, 2, 2]`. The loader accepts direct weights or `params`, `model`, `state_dict`, and nested `checkpoint` wrappers. It verifies every original key and shape, permits only SAIM keys to be initialized, then strictly loads the complete state. Baseline and SAIM outputs are compared at initialization. The manifest records the source checkpoint SHA-256, dataset membership, settings, and compatibility report. A missing or incompatible checkpoint stops the experiment.

LOL-v2 Real `Train/Low` and `Train/Normal` are paired by filename. Seed 42 selects a 10% validation holdout; Test remains untouched until final evaluation. A stateless per-update sampler produces the same image, 128×128 crop, flip, and rotation for both arms, including after resume. Both use batch size 1, L1, Adam (`2e-4`, betas `0.9/0.999`), cosine decay to `1e-6` across the selected 100, 1,000, or 5,000 updates, AMP on CUDA, and gradient clipping at 1.0. SAIM parameters use the same learning rate. Validation runs at step 0 and every 100 updates for the smoke run or every 500 updates for longer runs, plus completion. Mean validation PSNR selects the best checkpoint; ties keep the earlier checkpoint. Metrics use full-resolution inference with padding to multiples of four, clamp/round to uint8 RGB, and the repository's PSNR/SSIM functions. This metric convention differs from `protocol_metrics.py`'s float PSNR convention; comparisons within this experiment share one convention.

Each `last.pth` contains both models, optimizers, schedulers, AMP scalers, RNG states, full metric history, best checkpoints, and completed paired update count. Resuming requires the same manifest and restarts at the next paired update. The CSV files are rebuilt from checkpoint history to avoid duplicate rows. `progress.log` and training console lines show only a single `TOTAL_ETA` for both arms together. The run folder contains `learning_curves.csv`, `comparison_by_step.csv`, `per_image_results.csv`, `best_checkpoint_summary.csv`, `training_diagnostics.csv`, `progress.log`, `run_manifest.json`, best and last checkpoints, and loss/PSNR/SSIM plots. Final Test results are appended to `per_image_results.csv` and summarized in `final_test_summary.json`.

## Run

In PowerShell, from the cloned `v3` repository under `D:\retinexformer_improvement`:

```powershell
py -3.14 -m pip install -r requirements.txt jupyter
py -3.14 -m unittest tests.test_saim -v
py -3.14 -m jupyter notebook SAIM_Experiment.ipynb
```

Use the notebook configuration cell to point to the dataset and checkpoint. Run 100 updates first, then use distinct 1,000 and 5,000 update runs. Set `RESUME=True` to continue a matching interrupted run. The final Test cell should be run only after training. The repository contains no local dataset, pretrained weights, or CUDA device, so real PSNR/SSIM must be measured on the Windows machine.
