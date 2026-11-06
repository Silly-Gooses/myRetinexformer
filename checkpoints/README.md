# Model D v1 checkpoint pair

Both original trained checkpoints are required. Keep them outside Git; `.pth`
and `.pt` files are ignored. The source files are read-only during verification.

## Original Drive locations

Relative to `/content/drive/MyDrive/RetinexFormer_NAF_Screening/`:

```text
model_B_optionA_batch8_patch128_lr1e-5_from_batch8best/
  low_lr_1e-5_1000steps/best.pth
model_D_HVI_from_ModelB_OptionA_v1/
  best_model_d_head.pth
```

The corresponding split is `model_B_v1_full_data_1000steps/split.json`.

## Identity and loading

- **Model B SHA256:**
  `1cb495fc19ce1560a1b50bd9eef16e10e3c3184351e9ac6fe03ccc31e1b91ae6`.
  Confirmed against the local file and the original Model D report. The loader
  checks this hash before reading the checkpoint's `model` state dictionary.
- **HVI head SHA256:** pending access to the original Drive checkpoint; not
  included in the downloaded comparison ZIP. Every verification/evaluation
  records the actual SHA256. No substitute checkpoint is accepted as a fallback.
- The original head checkpoint contains `head`, `epoch`, `best_val_psnr`,
  `optimizer`, and `config`. Its best epoch is 17. Loading checks the frozen
  architecture, original training configuration/split, and recorded backbone
  hash, then applies `load_state_dict(..., strict=True)`.
- Model B requires `weights_only=False` for its saved RNG metadata, after its
  known SHA256 is verified. The head uses `weights_only=True`. Neither optimizer
  is restored and no weights are modified.

Metadata/shape agreement alone cannot authenticate previously unknown head
weights. Retain the SHA256 from the original Drive artifact with the resulting
notebook parity and validation report. Do not claim trained Model D reproduction
from architecture tests or historical PSNR alone.

Use [the Colab verification notebook](../notebooks/Model_D_v1_Colab.ipynb) to access
the original Drive pair. Its outputs go to a separate `model_D_v1_verification`
directory. A missing checkpoint stops execution with a clear error; obtain the
original file instead of running training or substituting v1.1/v1.2/v2 weights.
