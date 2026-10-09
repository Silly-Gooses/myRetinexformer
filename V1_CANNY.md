# RetinexFormer v1 — Canny edge guidance

Version: `v1`. Baseline: the original RetinexFormer architecture.

The single architectural change is a Canny edge channel at the restoration
input. Illumination estimation, illumination-guided attention, feature widths,
RGB correction, losses, dataset, and training hyperparameters remain unchanged.

## Actual architecture and changes

`RetinexFormer.forward` receives low-light RGB. Each single stage calls the
illumination estimator, whose `conv2` predicts the three-channel illumination
map; its depthwise features continue to guide every IGAB. The stage computes
`lit_up = img * illu_map + img`. Originally, that RGB tensor went directly into
`Denoiser.embedding`, and `Denoiser.mapping` predicted a three-channel correction
before the denoiser added its RGB input.

V1 computes edges once from the original model input and reuses them in every
stage. The embedding becomes `Conv2d(4, n_feat, 3)`; the rest of the denoiser is
unchanged. The residual tensor is now supplied separately:

```python
edge = canny(original_low_light)                    # [B, 1, H, W]
lit_up = img * illu_map + img                       # [B, 3, H, W]
denoiser_input = torch.cat([lit_up, edge], dim=1)    # [B, 4, H, W]
output = denoiser(denoiser_input, illu_fea, residual=lit_up)
# Inside the denoiser: output = mapping(features) + residual
```

The edge map is never added to RGB. For the notebook's `n_feat=40`, stage=1
model, the change adds only 360 trainable weights. Canny has no trainable weights.
Raw network spatial sizes still need to be multiples of four; existing
full-image evaluation helpers continue to pad and crop. Edge extraction then
operates on that padded low-light input. Standalone single-stage calls extract
edges from their own input if an edge map is not provided.

## Canny definition

Pure PyTorch; no new dependency. The steps follow the
[standard Canny algorithm](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html):

- Clamp a detached copy to `[0,1]`; RGB luminance weights `(0.299, 0.587, 0.114)`.
- Gaussian kernel size 5, sigma 1.0, replicate border padding.
- Unnormalized 3×3 Sobel derivatives and L2 gradient magnitude.
- Four-direction non-maximum suppression, with asymmetric ties to thin ridges.
- Fixed low/high thresholds 0.1/0.2, in unnormalized Sobel-magnitude units.
- Full eight-connected hysteresis until convergence; disconnected weak edges
  are discarded. Output is binary, in the input dtype and device.

All edge computation uses float32 with autocast disabled and no autograd graph.
Image tensors remain on device. Hysteresis convergence uses `torch.equal`, which
requires a host-visible scalar on CUDA; long weak-edge chains can increase
runtime. This is device-native processing, not a synchronization-free algorithm.
OpenCV pixel-for-pixel equivalence is not promised because border/tie/precision
conventions differ. Fixed thresholds can miss very faint boundaries; they have
not been tuned against validation/test results.

## Model and diagnostic APIs

```python
from RetinexFormer_arch import RetinexFormer

model = RetinexFormer(n_feat=40, stage=1, num_blocks=[1, 2, 2])  # v1 default
baseline = RetinexFormer(n_feat=40, stage=1, num_blocks=[1, 2, 2],
                         edge_guidance=False)
# Optional settings override; defaults remain fixed for the v1 experiment.
custom = RetinexFormer(n_feat=40, stage=1, num_blocks=[1, 2, 2],
                       canny_config={"low_threshold": 0.1, "high_threshold": 0.2,
                                     "kernel_size": 5, "sigma": 1.0,
                                     "luminance_weights": (0.299, 0.587, 0.114)})
output = model(low)  # RGB tensor, unchanged training interface
lightups, outputs = model.forward_with_intermediate(low)  # existing API
details = model.forward_with_intermediate(low, return_dict=True)
# input, edge, lit_up, output, lit_up_stages, output_stages
```

`lit_up` and `output` identify the final stage; stage lists retain stage order.
Baseline diagnostics use `edge=None`. Single-stage diagnostics also support
`return_dict=True` with the four tensor keys, while retaining the original
`(lit_up, output)` default. Diagnostics retain autograd behavior; use
`torch.inference_mode()` for inspection.

The original constructor's `n_feat=31` default remains unchanged, but the
illumination estimator requires a feature count divisible by four. Use the
existing experiment's explicit `n_feat=40` (tests use 8).

## Checkpoints

Ordinary strict loading intentionally rejects three-channel baseline embedding
weights in a v1 model. Explicit initialization expands only the embedding at
each stage, copying RGB weights exactly and zero-initializing the edge slice:

```python
import torch
from analysis_utils import extract_checkpoint_state_dict

checkpoint = torch.load("baseline.pth", map_location="cpu", weights_only=True)
converted_keys = model.load_baseline_state_dict(extract_checkpoint_state_dict(checkpoint))
print(converted_keys)  # Also reported by a warning during conversion.
```

All other keys and shapes must match. Native v1 checkpoints load directly with
`model.load_state_dict(...)`. Canny configuration is not part of the tensor
state dict: preserve it with the checkpoint and reconstruct the model with the
same settings. Notebook checkpoints record version, architecture, Canny configuration, and the
complete epoch-training state. Resume validates the saved configuration and split.

Fresh v1 models also start with zero edge-channel weights. These remain trainable
and receive gradients. Baseline conversion preserves initial baseline output to
floating-point tolerance. Use a fresh optimizer after conversion: baseline
optimizer states have incompatible embedding dimensions.

The notebook trains from scratch on branch `v1` using 200 epochs, including
3 warm-up epochs, in `Lightweight_RetinexFormer_v1_LOLv2_real_Epoch200`.
It reserves 10% of training pairs for validation and uses the official test set
only for final evaluation. See [EPOCH_TRAINING.md](EPOCH_TRAINING.md) for configuration,
logging, and resume. Commit/push separately before using remote Colab setup.
Baseline comparison is no longer in the notebook; standalone comparison scripts
and `test.py` still explicitly use baseline mode.

## Verification

```bash
python -m unittest discover -s tests -v
python tests/benchmark_canny_v1.py --device cpu --size 64 --batch-size 2 --steps 5
# On Colab with CUDA:
python tests/benchmark_canny_v1.py --device cuda --size 256 --batch-size 8 --steps 10
python tests/benchmark_canny_v1.py --device cuda --size 256 --batch-size 8 --steps 10 --amp
```

Tests cover Canny thinning/connectivity/thresholding, constant and diagonal
images, a long weak chain, dtype/device preservation, RGB residual identity,
forward/L1/backward and edge learning, multi-stage reuse, original-baseline
equivalence, strict checkpoint conversion/round trips, padded inference, and
existing metrics/PNG exports. CUDA tests skip when unavailable. Timing reports
median warmed-up forward/L1/backward and Canny-only wall time, including
convergence synchronization; it excludes optimizer steps. Synthetic timing is
not a substitute for measuring real batches on the training GPU.

No full training or PSNR/SSIM improvement claim is made by these smoke tests.

### Local results (2026-10-08)

PyTorch 2.12.0, CPU only: 17 tests discovered, 15 passed, 2 CUDA tests skipped.
The notebook configuration (`n_feat=40`, stage=1, blocks `[1,2,2]`) also passed
a separate 256×256 forward/L1/backward smoke check:

| Tensor | Shape |
| --- | --- |
| Input | `[1,3,256,256]` |
| Edge | `[1,1,256,256]` |
| Lit-up | `[1,3,256,256]` |
| Denoiser input | `[1,4,256,256]` |
| Output | `[1,3,256,256]` |

Synthetic L1 was 0.432757; all parameter gradients were finite and the edge
weight gradient norm was 0.071761. Total trainable parameters: 1,606,061.
The warmed-up CPU benchmark at batch=2, size=64, warmup=2, steps=5 measured
624.267 ms baseline versus 633.256 ms v1 (1.44% overhead), with 3.230 ms
Canny-only time. GPU timing and CUDA mixed precision remain unverified locally.
