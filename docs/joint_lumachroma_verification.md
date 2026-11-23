# Implementation verification — 2026-11-23

These checks validate the repository integration. They do **not** constitute a full new evaluation on all three datasets. Previously measured results remain in `reference_results/joint_lumachroma_v2/`.

## CPU tests and notebook parity

Using the existing local Python runtime, `python -m unittest discover -s tests -v` passed **32 tests**: 13 existing tests and 19 new joint-fusion tests. Coverage includes color ordering, endpoint/equal-weight tolerances, clipping, analytic coefficients, deterministic folds, macro weighting, optimizer settings, CV/export, lambda ties, hash/source rejection, strict frozen loading, dataset gates, padding/interleaved splitting, cache locks, inference CLI outputs, and evaluation with no refitting. A fitting test uses unreadable Test input and GT files and succeeds using only Train pixels.

Ten numerical comparisons against functions parsed directly from the original supplied notebook were exact. The original notebook's optimizer, fold assignments, and complete CV workflow were also replayed on the recovered Train coefficient tables (485, 689, and 900 rows). The extracted implementation and original notebook produced **identical selected JSON and all 126 CV rows** on those same saved tables.

Compared with the historical report, the maximum fitted-parameter difference was `1.6955574733135137e-7`; the maximum CV PSNR difference was `1.1630586485011918e-6 dB`. Both implementations exhibit the same differences when replaying the saved coefficients in this environment. This check cannot isolate CSV serialization from historical runtime differences. All selected lambdas agree. The shared regularized pair was `0.29920297774275234`, `0.40713060525512`, with lambda `0.5`; the original reference JSON remains unchanged at its recorded precision.

Python compilation, notebook schema/code parsing, CLI help entry points, and Git whitespace checks are included in publication checks. No format/lint configuration exists in the original repository.

## GPU smoke test actually executed

The pinned official architectures and verified LOL-v2 Real checkpoints ran on an **NVIDIA GeForce GTX 1650 with Max-Q Design (4 GB)**. The first three sorted official Test pairs were `00690.png`, `00691.png`, and `00692.png`. The test used the original report's frozen full-precision weights, not weights fitted on Test.

For each of the seven methods, predictions were compared with the original notebook's loading, inference, fusion, conversion, and metric functions:

| Check | Observed maximum absolute difference |
|---|---:|
| 21 float RGB array comparisons | 0 |
| Saved preview RGB8 pixels | 0 |
| Evaluator CSV metrics vs notebook | 0 |
| Float PSNR vs historical report rows | `2.4116935222195934e-6 dB` |
| RGB8 SSIM vs historical report rows | `5.182702440542641e-7` |

The evaluator wrote per-image/summary CSVs, frozen parameters, an evaluation lock, runtime manifest, two sets of regularized predictions, all-method previews, and the compact report ZIP. These generated images, caches, and archives remain local and are not committed.

Runtime: Python 3.14.3; PyTorch 2.14.1+cu126; CUDA 12.6; cuDNN 91002; NumPy 2.5.4; SciPy 1.18.1; pandas 3.1.0; OpenCV distribution 5.0.0.93; Pillow 13.0.0; einops 0.8.2; huggingface_hub 2.4.0.

## Limits

- No fresh all-Train backbone coefficient generation or full three-dataset Test inference was performed for this publication.
- The Colab notebook was schema/syntax checked and its shared implementation tested locally; an interactive Colab session was not executed here.
- Source and checkpoint identities are pinned, but the original notebook did not pin its package versions. Record manifests and inspect differences when changing runtimes.
- The official Test sets had been inspected before this experiment. These verification checks do not establish blind-test generalization.
- The specific FusionNet bibliography entry remains pending, as agreed.
