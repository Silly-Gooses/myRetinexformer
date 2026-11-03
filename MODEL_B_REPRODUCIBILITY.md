# Model B Option A reproducibility

The implementation is ported from `RetinexFormer_v1_ModelB_OptionA_Batch8_Patch128_LR1e-5_Colab.ipynb`. Its reference code checked out v1 commit `ad33e6bf5b338bc36e4a14102781e78a03397b3b`. The shared architecture, Canny, and metric helper files on the `v2.1` base match that commit. Model B lives in `model_b/`; existing RetinexFormer models are unchanged.

## Reference identity and protocol

- Source: prior Batch-8 / Patch-128 / LR `2e-5` `best.pth`, local step 1000, expected validation PSNR `20.9486`. Its expected SHA256 is checked before training: `a835284a8f72d16d131ddac6b05224c7afc2e4024c636beab9405c04e4403141`.
- Architecture: edge-guided single-stage RetinexFormer with 40 features and blocks `[1, 2, 2]`. Every `IG_MSA` is replaced by the notebook's NAF-inspired mixer. The original `state_dict` keys and shapes are preserved.
- Data: exact frozen 620/69 split from the original experiment's `split.json`; paired `Train/Low` and `Train/Normal` RGB files. The official Test set is unused. Train samples use a random 128-pixel square crop, then independent 50% horizontal flip, vertical flip, and 90-degree rotation.
- Fine-tune: start from source **model weights**, seed 42, fresh Adam with betas `(0.9, 0.999)`, zero weight decay, learning rate `1e-5`, batch 8, 1,000 updates, CUDA AMP, L1 loss, DataLoader shuffle, `drop_last=False`, zero workers. Validate every 100 updates, log every 25, and save `last.pth` every 100. Select `best.pth` on strictly higher mean validation PSNR, including the step-0 source result.
- Validation: full RGB images are padded on the right/bottom to a multiple of four with replicate padding, evaluated with CUDA autocast, cropped back, clamped and rounded to uint8. Per-image PSNR and SSIM use `analysis_utils.image_metrics`; the reported aggregates are arithmetic means across 69 images.
- Resume: optimizer, scaler, model, and RNG are restored from this run's `last.pth`. The original notebook did **not** store the sampler cursor, so an interrupted run is not guaranteed to resume the exact data sequence.

Reference Option A best: **21.036645189367817 dB PSNR**, **0.7354556825534054 SSIM**, local step 1000. `model_b/reference/` contains the frozen split and small CSV fixtures from the reference run. The aggregate comparison uses tolerances of 0.02 dB and 0.001 SSIM and reports every per-image delta.

## Colab execution

Open `notebooks/ModelB_GitHub_Batch8_Patch128_LR1e5_Colab.ipynb` on a CUDA runtime and run top to bottom. It clones GitHub branch `model-b`, records its resolved commit, mounts Drive, and expects:

```text
MyDrive/datasets/LOLv2/Real_captured/Train/{Low,Normal}/
MyDrive/RetinexFormer_NAF_Screening/model_B_v1_full_data_1000steps/split.json
MyDrive/RetinexFormer_NAF_Screening/model_B_batch8_patch128_from_best1500/batch_8_patch_128_lr2e-5_1000steps/best.pth
```

The original Option A best checkpoint at `MyDrive/RetinexFormer_NAF_Screening/model_B_optionA_batch8_patch128_lr1e-5_from_batch8best/low_lr_1e-5_1000steps/best.pth` is optional for direct evaluation. If absent, the notebook states that this part of Test A remains unresolved. The new run writes under `MyDrive/RetinexFormer_NAF_Screening/model_B_github_reproduction/low_lr_1e-5_1000steps/`, away from all reference checkpoints.

Outputs include `config.json`, `best.pth`, `last.pth`, `steps.csv`, `validation.csv`, `training.log`, source and best per-image CSVs, evaluation comparisons, training comparisons, and `reproducibility_summary.json`. The notebook prints live loss, validation PSNR/SSIM, peak allocated GPU memory, and an ETA that includes observed validation time after the first validation.

## Independent verification

Test A runs locally with the original notebook as an independent architecture builder:

```powershell
python -m model_b.verify_reference `
  --data-root 'D:\retinexformer_v1\datasets' `
  --split-file 'D:\retinexformer_v1\results\model_B_v1_full_data_1000steps\split.json' `
  --source 'D:\retinexformer_v1\results\model_B_batch8_patch128_from_best1500\batch_8_patch_128_lr2e-5_1000steps\best.pth' `
  --option-a 'D:\retinexformer_v1\results\model_B_optionA_batch8_patch128_lr1e-5_from_batch8best\low_lr_1e-5_1000steps\best.pth' `
  --reference-notebook 'D:\retinexformer_v1\research_lab\experiments\RetinexFormer_v1_ModelB_OptionA_Batch8_Patch128_LR1e-5_Colab.ipynb' `
  --output-dir "$env:TEMP\model_b_test_a"
```

It checks strict loading for both checkpoints, compares a real-image forward pass with the original notebook builder, evaluates the full 69-image split, and writes aggregate and per-image comparisons. Test B is the separate Colab rerun; hardware and nondeterministic operations may change loss curves and final metrics. The comparison reports show those differences without attributing them to a different training configuration.

The local Test A run loaded both checkpoints strictly and produced **zero** maximum forward difference against the original notebook builder on a `400×600` image. Source evaluation was `20.9486147563` PSNR / `0.7316439764` SSIM; original Option A best evaluation was `21.0366529819` PSNR / `0.7354565024` SSIM. The largest per-image deviations were under `0.000431` dB PSNR and `0.000027` SSIM. These small differences are consistent with numerical differences between the local CUDA/PyTorch environment and the original Colab run. Test B results remain pending the requested Colab execution.
