"""Compare author and trained RetinexFormer checkpoints on LOLv2 Real.

Run without --full first to inspect three smoke-test grids. Pass --full after
reviewing them; it repeats the checks before evaluating the complete dataset.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from RetinexFormer_arch import RetinexFormer
from analysis_utils import (
    compare_analysis_results,
    evaluate_analysis_model,
    infer_full_image,
    infer_full_image_stages,
    write_analysis_metrics,
)
from dataLoader import paired_image_Dataset


DEFAULT_DATASET = Path("/content/drive/MyDrive/datasets/LOLv2/Real_captured")
DEFAULT_AUTHOR = Path("/content/drive/MyDrive/RetinexFormer_weights/LOL_v2_real.pth")
DEFAULT_OURS = Path(
    "/content/drive/MyDrive/RetinexFormer_full_training/"
    "Lightweight_RetinexFormer_LOLv2_real_Full/checkpoints/best_psnr.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--author-checkpoint", type=Path, default=DEFAULT_AUTHOR)
    parser.add_argument("--ours-checkpoint", type=Path, default=DEFAULT_OURS)
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"))
    parser.add_argument("--smoke-count", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Absolute PSNR delta in dB for improved/degraded groups")
    parser.add_argument("--representative-count", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--full", action="store_true",
                        help="Run the complete dataset after the smoke checks")
    return parser.parse_args()


def validate_dataset(dataset):
    """Reject missing, unmatched, duplicate, or size-mismatched image pairs."""
    if not dataset.low_paths:
        raise ValueError(f"No supported test images in {dataset.low_dir}")
    high_dir = Path(dataset.normal_dir)
    low_names = [Path(path).name for path in dataset.low_paths]
    if len(set(name.lower() for name in low_names)) != len(low_names):
        raise ValueError("Duplicate low-light filenames, ignoring case")
    high_names = {path.name for path in high_dir.iterdir() if path.is_file() and
                  path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")}
    if set(low_names) != high_names:
        missing = sorted(set(low_names) - high_names)
        extra = sorted(high_names - set(low_names))
        raise ValueError(f"Image-pair mismatch; missing GT: {missing}; extra GT: {extra}")
    for low_path, high_path in zip(dataset.low_paths, dataset.high_paths):
        with Image.open(low_path) as low, Image.open(high_path) as high:
            if low.size != high.size:
                raise ValueError(f"Image size mismatch: {low_path} and {high_path}")
            if min(low.size) < 11:
                raise ValueError(f"Image is too small for project SSIM: {low_path}")


def load_model(checkpoint_path, state_key, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or state_key not in checkpoint:
        raise ValueError(f"{checkpoint_path} lacks the '{state_key}' state dictionary")
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40,
                          stage=1, num_blocks=[1, 2, 2])
    model.load_state_dict(checkpoint[state_key], strict=True)
    return model.to(device).eval()


def verify_forward_matches(model, sample, device):
    low = sample["low"].unsqueeze(0).to(device)
    with torch.inference_mode():
        normal = infer_full_image(model, low)
        _, outputs = infer_full_image_stages(model, low)
    if not torch.allclose(normal, outputs[-1], rtol=1e-5, atol=1e-6):
        raise AssertionError("Intermediate path differs from normal forward output")


def evaluate_both(author_model, ours_model, dataset, device, root, limit=None,
                  threshold=0.5, representative_count=3, provenance=None):
    author_dir, ours_dir = root / "author", root / "ours"
    author_rows = evaluate_analysis_model(author_model, dataset, device, author_dir, limit)
    ours_rows = evaluate_analysis_model(ours_model, dataset, device, ours_dir, limit)
    author_avg = write_analysis_metrics(author_rows, author_dir)
    ours_avg = write_analysis_metrics(ours_rows, ours_dir)
    comparisons, selected = compare_analysis_results(
        author_rows, ours_rows, author_dir, ours_dir, root,
        threshold=threshold, representative_count=representative_count
    )
    if len(comparisons) != len(author_rows) or len(comparisons) != len(ours_rows):
        raise AssertionError("Comparison row count differs from model metrics")
    for row in comparisons:
        if not math.isclose(row["delta_psnr"], row["ours_psnr"] - row["author_psnr"], abs_tol=1e-9):
            raise AssertionError(f"Incorrect PSNR delta for {row['filename']}")
        if not math.isclose(row["delta_ssim"], row["ours_ssim"] - row["author_ssim"], abs_tol=1e-9):
            raise AssertionError(f"Incorrect SSIM delta for {row['filename']}")
    summary = {
        "image_count": len(comparisons),
        "author_average": author_avg,
        "ours_average": ours_avg,
        "average_delta": {"psnr": ours_avg["psnr"] - author_avg["psnr"],
                          "ssim": ours_avg["ssim"] - author_avg["ssim"]},
        "threshold_db": threshold,
        "group_counts": {group: sum(r["group"] == group for r in comparisons)
                         for group in ("improved", "similar", "degraded")},
        "representatives": {key: [r["filename"] for r in rows]
                            for key, rows in selected.items()},
    }
    if provenance is not None:
        summary["inputs"] = provenance
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    args = parse_args()
    if (args.smoke_count < 1 or not math.isfinite(args.threshold) or
            args.threshold <= 0 or args.representative_count < 1):
        raise ValueError("Smoke count, threshold, and representative count must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    low_dir = args.dataset_root / "Test" / "Low"
    high_dir = args.dataset_root / "Test" / "Normal"
    if not low_dir.is_dir() or not high_dir.is_dir():
        raise FileNotFoundError(f"Expected LOLv2 Real Test/Low and Test/Normal under {args.dataset_root}")
    dataset = paired_image_Dataset(str(low_dir), str(high_dir), split="test",
                                   crop_size=None, augment=False)
    validate_dataset(dataset)
    author_model = load_model(args.author_checkpoint, "params", device)
    ours_model = load_model(args.ours_checkpoint, "model", device)
    verify_forward_matches(author_model, dataset[0], device)
    verify_forward_matches(ours_model, dataset[0], device)

    provenance = {
        "dataset_root": str(args.dataset_root.resolve()),
        "author_checkpoint": str(args.author_checkpoint.resolve()),
        "ours_checkpoint": str(args.ours_checkpoint.resolve()),
        "model": {"n_feat": 40, "stage": 1, "num_blocks": [1, 2, 2]},
        "metric_pixels": "saved clipped and rounded 8-bit RGB PNG",
    }

    smoke_root = args.output_dir / "smoke"
    smoke_summary = evaluate_both(
        author_model, ours_model, dataset, device, smoke_root,
        limit=args.smoke_count, threshold=args.threshold,
        representative_count=args.representative_count, provenance=provenance
    )
    print(f"Smoke test passed for {smoke_summary['image_count']} images. Inspect: {smoke_root / 'review.md'}")
    if not args.full:
        print("After inspecting the smoke grids, rerun with --full for the complete dataset.")
        return
    full_summary = evaluate_both(
        author_model, ours_model, dataset, device, args.output_dir,
        threshold=args.threshold, representative_count=args.representative_count,
        provenance=provenance
    )
    print(json.dumps(full_summary, indent=2))
    print(f"Full comparison: {args.output_dir / 'review.md'}")


if __name__ == "__main__":
    main()
