"""Compare official and retrained LOL-v2 Real RetinexFormer checkpoints.

This inference-only script saves identical-input intermediate and final outputs,
then reports final-output quality and neutral Lit-up-image diagnostics.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RetinexFormer_arch import RetinexFormer
from analysis_utils import (
    classify_litup_diagnostics,
    classify_output_delta,
    find_paired_images,
    image_metrics,
    infer_full_image_stages,
    litup_diagnostics,
    load_checkpoint_strict,
    save_six_panel_grid,
    save_rgb,
    tensor_to_uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author_ckpt", "--author-ckpt", required=True, type=Path)
    parser.add_argument("--retrained_ckpt", "--retrained-ckpt", required=True, type=Path)
    parser.add_argument("--low_dir", "--low-dir", required=True, type=Path)
    parser.add_argument("--gt_dir", "--gt-dir", required=True, type=Path)
    parser.add_argument("--output_dir", "--output-dir", required=True, type=Path)
    parser.add_argument("--max_images", "--max-images", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output_threshold_db", "--output-threshold-db", type=float,
                        default=0.5, help="Improved/degraded final-output PSNR threshold.")
    parser.add_argument("--brightness_delta", "--brightness-delta", type=float,
                        default=0.02, help="Neutral Lit-up brightness grouping tolerance.")
    parser.add_argument("--ratio_delta", "--ratio-delta", type=float, default=0.02,
                        help="Dark/highlight diagnostic change flag tolerance.")
    parser.add_argument("--dark_threshold", "--dark-threshold", type=float, default=0.10)
    parser.add_argument("--highlight_threshold", "--highlight-threshold", type=float,
                        default=0.98)
    return parser.parse_args()


def _validate_args(args):
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max_images must be positive")
    for name in ("output_threshold_db", "brightness_delta", "ratio_delta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be a positive finite number")
    for name in ("dark_threshold", "highlight_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"--{name} must be between 0 and 1")
    if args.dark_threshold >= args.highlight_threshold:
        raise ValueError("--dark_threshold must be smaller than --highlight_threshold")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def _tensor_from_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if min(array.shape[:2]) < 11:
        raise ValueError(f"Image is too small for SSIM: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255.0)


def _output_name(filename):
    path = Path(filename)
    return path.name if path.suffix.lower() == ".png" else f"{path.stem}.png"


def _create_output_dirs(output_dir):
    folders = {
        name: output_dir / name for name in (
            "low", "author_litup", "retrained_litup", "author_output",
            "retrained_output", "gt", "grids",
        )
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    return folders


def _model(device):
    # This is the LOL-v2 Real architecture used by this repository's notebook.
    return RetinexFormer(
        in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2]
    ).to(device)


def analyze(args):
    """Run the full comparison and return per-image result dictionaries."""
    _validate_args(args)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    pairs = find_paired_images(args.low_dir, args.gt_dir)
    if args.max_images is not None:
        pairs = pairs[:args.max_images]
    print(f"Found {len(pairs)} test images")

    author_model, author_message = load_checkpoint_strict(
        _model(device), args.author_ckpt, device, "Author"
    )
    print(author_message)
    retrained_model, retrained_message = load_checkpoint_strict(
        _model(device), args.retrained_ckpt, device, "Retrained"
    )
    print(retrained_message)

    folders = _create_output_dirs(args.output_dir)
    rows = []
    for index, (filename, low_path, gt_path) in enumerate(pairs, 1):
        print(f"Processing image {index}/{len(pairs)}: {filename}")
        low, gt = _tensor_from_image(low_path), _tensor_from_image(gt_path)
        if tuple(low.shape[-2:]) != tuple(gt.shape[-2:]):
            raise ValueError(f"Image size mismatch: {low_path} and {gt_path}")
        low_batch = low.unsqueeze(0).to(device)
        with torch.inference_mode():
            author_litups, author_outputs = infer_full_image_stages(author_model, low_batch)
            retrained_litups, retrained_outputs = infer_full_image_stages(retrained_model, low_batch)
        images = {
            "low": tensor_to_uint8(low),
            "author_litup": tensor_to_uint8(author_litups[-1][0]),
            "retrained_litup": tensor_to_uint8(retrained_litups[-1][0]),
            "author_output": tensor_to_uint8(author_outputs[-1][0]),
            "retrained_output": tensor_to_uint8(retrained_outputs[-1][0]),
            "gt": tensor_to_uint8(gt),
        }
        output_name = _output_name(filename)
        for key, image in images.items():
            save_rgb(image, folders[key] / output_name)
        save_six_panel_grid(images, folders["grids"] / output_name)

        author_psnr, author_ssim = image_metrics(images["author_output"], images["gt"])
        retrained_psnr, retrained_ssim = image_metrics(
            images["retrained_output"], images["gt"]
        )
        # GT is the final normal-light target, not a true Lit-up target. These
        # values are retained only to diagnose stage-wise model differences.
        author_litup_psnr, author_litup_ssim = image_metrics(
            images["author_litup"], images["gt"]
        )
        retrained_litup_psnr, retrained_litup_ssim = image_metrics(
            images["retrained_litup"], images["gt"]
        )
        diagnostics = litup_diagnostics(
            images["author_litup"], images["retrained_litup"], args.dark_threshold,
            args.highlight_threshold,
        )
        delta_psnr, delta_ssim = retrained_psnr - author_psnr, retrained_ssim - author_ssim
        output_group = classify_output_delta(delta_psnr, args.output_threshold_db)
        litup_group = classify_litup_diagnostics(
            diagnostics, args.brightness_delta, args.ratio_delta
        )
        rows.append({
            "filename": output_name,
            "author_output_psnr": author_psnr,
            "retrained_output_psnr": retrained_psnr,
            "output_psnr_delta": delta_psnr,
            "author_output_ssim": author_ssim,
            "retrained_output_ssim": retrained_ssim,
            "output_ssim_delta": delta_ssim,
            "author_litup_psnr": author_litup_psnr,
            "retrained_litup_psnr": retrained_litup_psnr,
            "litup_psnr_delta": retrained_litup_psnr - author_litup_psnr,
            "author_litup_ssim": author_litup_ssim,
            "retrained_litup_ssim": retrained_litup_ssim,
            "litup_ssim_delta": retrained_litup_ssim - author_litup_ssim,
            **diagnostics,
            "output_group": output_group,
            "litup_group": litup_group,
            "combined_group": f"{litup_group} / Output {output_group}",
        })
    write_reports(rows, args)
    print(f"Saved analysis to {args.output_dir}")
    return rows


PER_IMAGE_FIELDS = (
    "filename", "author_output_psnr", "retrained_output_psnr", "output_psnr_delta",
    "author_output_ssim", "retrained_output_ssim", "output_ssim_delta",
    "author_litup_psnr", "retrained_litup_psnr", "litup_psnr_delta",
    "author_litup_ssim", "retrained_litup_ssim", "litup_ssim_delta",
    "author_litup_mean_luminance", "retrained_litup_mean_luminance",
    "litup_mean_luminance_delta", "author_litup_dark_ratio",
    "retrained_litup_dark_ratio", "author_litup_highlight_ratio",
    "retrained_litup_highlight_ratio", "author_vs_retrained_litup_mae",
    "author_vs_retrained_litup_ssim", "output_group", "litup_group", "combined_group",
)


def write_reports(rows, args):
    if not rows:
        raise ValueError("No images were analyzed")
    output_dir = Path(args.output_dir)
    with (output_dir / "per_image_analysis.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PER_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    means = {
        field: float(np.mean([row[field] for row in rows])) for field in (
            "author_output_psnr", "retrained_output_psnr", "output_psnr_delta",
            "author_output_ssim", "retrained_output_ssim", "output_ssim_delta",
            "author_litup_psnr", "retrained_litup_psnr", "litup_psnr_delta",
            "author_litup_ssim", "retrained_litup_ssim", "litup_ssim_delta",
            "author_litup_mean_luminance", "retrained_litup_mean_luminance",
            "litup_mean_luminance_delta", "author_litup_dark_ratio",
            "retrained_litup_dark_ratio", "author_litup_highlight_ratio",
            "retrained_litup_highlight_ratio", "author_vs_retrained_litup_mae",
            "author_vs_retrained_litup_ssim",
        )
    }
    summary = [("number_of_images", len(rows)), *sorted(means.items())]
    for group in ("Improved", "Similar", "Degraded"):
        summary.append((f"output_group_{group.lower()}_count",
                        sum(row["output_group"] == group for row in rows)))
    summary.extend((
        ("output_psnr_threshold_db", args.output_threshold_db),
        ("brightness_delta", args.brightness_delta),
        ("ratio_delta", args.ratio_delta),
        ("dark_threshold", args.dark_threshold),
        ("highlight_threshold", args.highlight_threshold),
    ))
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("metric", "value"))
        writer.writerows(summary)


def main():
    analyze(parse_args())


if __name__ == "__main__":
    main()
