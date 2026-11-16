"""Evaluate the four fixed methods on the 100 paired LOLv2 Real Test PNGs."""

import argparse
import csv
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_model.checkpoint_utils import HVI_COMMIT, HVI_SHA, RETINEX_COMMIT, RETINEX_SHA
from fusion_model.cli import add_model_arguments, build_predictor, create_output, read_rgb, save_prediction
from fusion_model.metrics import author_ssim, float_psnr, rgb8


METHODS = ("normal", "tta8", "hvi", "fusion")
TOLERANCES = {"psnr_float": 0.01, "ssim_author": 0.001}
FIELDS = [f"{m}_{metric}" for m in METHODS for metric in TOLERANCES]


def paired_names(root):
    low, normal = [sorted(p.name for p in (root / d).glob("*.png") if p.is_file())
                   for d in ("Low", "Normal")]
    if len(low) != 100 or low != normal:
        raise ValueError(f"Expected exactly 100 paired PNG filenames in {root}/Low and Normal")
    return low


def read_reference(path, names):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if sorted(r["image"] for r in rows) != names:
        raise ValueError("Reference filenames do not match the 100 test pairs")
    reference = {r["image"]: {f: float(r[f]) for f in FIELDS} for r in rows}
    if not all(math.isfinite(v) for row in reference.values() for v in row.values()):
        raise ValueError("Non-finite reference metric")
    return reference


def summarize(rows, reference):
    summary = {"images": len(rows), "models": {}}
    comparison = {}
    within = True
    for method in METHODS:
        summary["models"][method] = {}
        for metric, tolerance in TOLERANCES.items():
            field = f"{method}_{metric}"
            summary["models"][method][metric] = float(np.mean([r[field] for r in rows]))
            if reference is None:
                continue
            deltas = {r["image"]: r[field] - reference[r["image"]][field] for r in rows}
            values = np.array(list(deltas.values()))
            worst = max(deltas, key=lambda name: abs(deltas[name]))
            mean_delta = float(values.mean())
            within = within and abs(mean_delta) <= tolerance
            comparison[field] = {
                "mean_delta": mean_delta, "mean_abs_delta": float(np.abs(values).mean()),
                "max_abs_delta": abs(deltas[worst]), "worst_image": worst,
                "images_over_tolerance": [n for n, d in deltas.items() if abs(d) > tolerance],
            }
    if reference is not None:
        summary.update(comparison=comparison, dataset_means_within_tolerance=within)
    return summary


def write_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def evaluate(args):
    names = paired_names(args.test_root)
    reference = read_reference(args.reference_csv, names) if args.reference_csv else None
    predictor = build_predictor(args)
    create_output(args.output)
    device = predictor.device
    write_json(args.output / "manifest.json", {
        "retinex_commit": RETINEX_COMMIT, "hvi_commit": HVI_COMMIT,
        "retinex_sha256": RETINEX_SHA, "hvi_sha256": HVI_SHA,
        "fusion_weights": {"tta8": 0.25, "hvi": 0.75},
        "hvi": {"gated": False, "gated2": True, "alpha": 0.8, "gamma": 1.0},
        "precision": "float32", "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "numpy": np.__version__, "opencv": cv2.__version__,
        "einops": version("einops"), "huggingface_hub": version("huggingface_hub"),
        "test_image_names": names, "reference_csv": args.reference_csv.name if reference else None,
    })
    rows = []
    with (args.output / "per_image_official_test.csv").open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["image", "height", "width", *FIELDS])
        writer.writeheader()
        for name in names:
            low = read_rgb(args.test_root / "Low" / name)
            predictions = predictor.predict(low, compare=True)
            gt8 = read_rgb(args.test_root / "Normal" / name)
            if gt8.shape != low.shape:
                raise ValueError(f"Input and reference dimensions differ for {name}")
            gt = gt8.astype(np.float32) / 255.0
            row = {"image": name, "height": gt.shape[0], "width": gt.shape[1]}
            for method in METHODS:
                pred = predictions[method]
                row[f"{method}_psnr_float"] = float_psnr(gt, pred)
                row[f"{method}_ssim_author"] = author_ssim(gt8, rgb8(pred))
                if args.save_images:
                    save_prediction(args.output / method / name, pred)
            rows.append(row)
            writer.writerow(row)
            out.flush()
            if len(rows) % 10 == 0:
                print(f"{len(rows)}/{len(names)} images complete", flush=True)
    summary = summarize(rows, reference)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary["models"], indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--test-root", type=Path, required=True, help="Directory containing Low and Normal")
    parser.add_argument("--reference-csv", type=Path, help="Compare the eight primary metric columns by filename")
    parser.add_argument("--save-images", action="store_true", help="Write all 400 PNG predictions")
    summary = evaluate(parser.parse_args(argv))
    if summary.get("dataset_means_within_tolerance") is False:
        raise SystemExit("Reference tolerance exceeded. Inspect summary.json and per_image_official_test.csv.")


if __name__ == "__main__":
    main()
