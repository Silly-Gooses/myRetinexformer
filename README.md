# Joint regularized luminance–chrominance fusion v2

The notebook-faithful v2 implementation adds Train-only fitting of two shared fusion weights across LOL-v1, LOL-v2 Real, and LOL-v2 Synthetic, with frozen dataset-specific RetinexFormer/HVI checkpoints. See the [method and reproduction instructions](docs/joint_lumachroma_fusion.md), [Colab notebook](notebooks/RetinexFormer_HVI_RegularizedJoint_LuminanceChrominance_v2_Colab.ipynb), and [previously measured reference results](reference_results/joint_lumachroma_v2/README.md). The existing fixed-fusion experiment is unchanged.

# RetinexFormer ×8 + HVI-CIDNet fixed fusion

This branch enhances low-light RGB images with a training-free ensemble of two pretrained networks. RetinexFormer runs on eight orientations of the same image. Its averaged prediction contributes 25% of each output pixel, and HVI-CIDNet contributes 75%. Both networks keep their original learned weights.

```text
Low-light RGB ──┬── RetinexFormer ×8 ── 0.25 ──┐
                └── HVI-CIDNet ───────── 0.75 ──┴── RGB sum → enhanced PNG
```

## Run the pipeline

Use Python 3.11 or later and a PyTorch build for your device. Install dependencies with `python -m pip install -r requirements-fusion.txt». Follow the setup guide to obtain the two verified checkpoints and the pinned HVI checkout. [Setup and evaluation guide](docs/FUSION_MODEL.md)

From the repository root, run a single-image enhancement with the paths from the guide. Passing a folder to `--input» enhances each image in that folder. Add `--compare» to save the single-pass RetinexFormer, RetinexFormer ×8, HVI, and fused PNGs.

```sh
python scripts/run_fusion.py --input example.png --output fusion-runs/example --retinex-checkpoint .fusion-local/LOL_v2_real.pth --hvi-checkpoint .fusion-local/best_PSNR.pth --hvi-repo .fusion-local/HVI-CIDNet
```

For Colab, select a GPU runtime, set the Drive paths in the first cell, and run the notebook. The local and Colab entry points use the same predictor and evaluator. [Colab notebook](notebooks/RetinexFormer_HVI_FixedFusion_Colab.ipynb)

## Research context

The GTX 1650 reproduction on 100 LOLv2 Real Test images measured **24.282168 dB PSNR and 0.873287 SSIM** for fixed fusion. HVI alone measured 23.890488 dB and 0.865594 SSIM. Earlier project experiments on this Test set informed the choice to give HVI the greater weight. Interpret the comparison as exploratory. [Reproduction findings and protocol](docs/FUSION_MODEL.md#reproduction)

This work uses the pretrained networks from Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement* (ICCV 2023), and Yan et al., *HVI: A New Color Space for Low-light Image Enhancement* (CVPR 2025). Obtain weights from the original authors and follow their distribution terms. [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) · [HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet)
