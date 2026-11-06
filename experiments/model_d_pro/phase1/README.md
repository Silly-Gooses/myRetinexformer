# Model D-Pro Phase 1 preservation

These are **historical Colab pilot artifacts**, copied from `DPro_Pilot_Report.zip`.
They are not results produced by the modular implementation.

- Original Model D: **21.071596 dB / 0.734275 SSIM** on 69 validation images.
- D-Pro best: **21.157359 dB / 0.736978 SSIM**, update **900**.
- Improvement: **+0.085763 dB** (approximately **+0.0858 dB**).
- Completed: 1,200 optimizer updates; best and final checkpoints differ.
- No D-Pro Test evaluation. **23.1 dB Test PSNR remains an unverified target.**

The JSON/CSV files are byte-preserved. `config.json` describes the historical
Drive run and its `resume=true`; it is an archive, not the safe default for new
runs. Use the package's `configs/phase1_pilot.json` for modular execution.
`val_00900.csv` contains the best checkpoint's 69 per-image measurements.
The original runtime was Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.4.6,
CUDA 13.0, Tesla T4. Different kernels can affect numerical reproduction.

## Sources and hashes

- Verified Model D base: `108b14cc0d84c070a8331772b99789359a4f3137`.
- Exact executed script: `tests/fixtures/dpro_pilot.py`, SHA256
  `52b1ca3bfd059fe2584875d79f108a835ebd05d2ea22e40afd487a2b5debb4da`.
- Inspected notebook: `Model_DPro_Budget_Pilot_T4.ipynb`, SHA256
  `9fa25e0ca33d9b558f8aa3c35ecd7cf59d35946a35366f2040e05503a9059784`.
  Its embedded script is byte-identical to the executed script.
- Model B checkpoint SHA256:
  `1cb495fc19ce1560a1b50bd9eef16e10e3c3184351e9ac6fe03ccc31e1b91ae6`.
- Epoch-17 HVI checkpoint SHA256:
  `cebbc55c59906fc2d0be453f16660b3d0dadc14bae150ef97c8fcce29c331ca0`.

The original script is an archival test fixture. Verification executes only its
imports and definitions, excluding `run_pilot`, `main`, and top-level actions.
Do not run the fixture as the supported workflow. The raw report ZIP, original
notebook, images, datasets, caches, and checkpoint binaries are not committed.

## Migration boundaries

Architecture, state-dictionary keys, tensor shapes, initialization order,
forward/backward operations, loss weights, stateless sampling, and training
schedule are preserved. Training wrappers now use modular source fingerprints,
safer output ownership checks, and a subprocess watchdog. Python/NumPy RNG state
is saved in addition to the original Torch/CUDA state. Baseline source hashes
are checked against the pinned commit; the actual working commit is recorded in
runtime metadata. These packaging changes intentionally create a new resume
identity; historical full-state resume is rejected.

Historical `best.pth` and `last.pth` support strict **model-weight** loading.
Full-state continuation is supported only for compatible modular runs. Future
Phase 2 must start from best weights and a separately defined optimizer/schedule.

Synthetic tests measure source parity and checkpoint-schema compatibility.
They do not verify the real D-Pro checkpoint files or reproduce the pilot score.
Run the optional Colab verification and full validation commands in the root
README before claiming historical reproduction. No GPU training or Test
evaluation is part of this migration.

## Local migration checks — 6 November 2026

The combined Model D / D-Pro suite ran **34 tests: 31 passed, 3 skipped**.
The real, hash-verified Model B checkpoint test passed. The three skips require
the unavailable trained HVI head (two tests) or historical D-Pro best checkpoint
(one test). Synthetic strict best/last weight loading, initialized/nonzero-weight
forward/loss/gradient parity, deterministic crops, 69-image synthetic metric
protocol parity, warmup/scheduling, and interrupted validation passed. A small
CPU optimizer test matched uninterrupted versus restored optimizer/scaler/RNG
execution exactly; this does not establish historical CUDA resume compatibility.

End-to-end CPU CLI inference preserved a 17×19 image. Independent script
verification measured **0.0 maximum absolute difference** for all six auxiliary
outputs using identical synthetic weights with nonzero corrections. CPU smoke
reported finite, nonzero gradients in all three parameter groups without an
optimizer update. Python compile/import checks, notebook schema and code-cell
checks, and byte-preservation checks passed. The local runtime was Python 3.14.3,
PyTorch 2.14.1+cu126, and NumPy 2.5.4.

No D-Pro GPU training, real D-Pro checkpoint validation, historical score
reproduction, or Test evaluation was performed. The optional Colab commands
remain necessary for those real-checkpoint verification gaps.
