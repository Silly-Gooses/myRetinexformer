"""Inference, reference verification, and frozen-split evaluation for Model D."""

import argparse
import ast
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from RetinexFormer_arch import IG_MSA, RetinexFormer
from analysis_utils import find_paired_images, image_metrics, tensor_to_uint8
from .model import (ARCH, HEAD_WIDTH, HVI_K, MAX_DELTA, REFERENCE,
                    load_model_d, sha256)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_NOTEBOOK = ROOT / "tests" / "reference_Model_D_v1.ipynb"
PROTOCOL = "model_d_v1_png_rgb_amp_v1"
PSNR_TOLERANCE = 0.002
SSIM_TOLERANCE = 0.0001


def read_rgb(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.0


def notebook_definitions(path):
    """Execute only named definitions from the hash-pinned original notebook."""
    if sha256(path) != REFERENCE["source_notebook_sha256"]:
        raise ValueError("Reference notebook SHA256 mismatch")
    wanted = {"NAFInspiredMixer", "build_model_b", "rgb_to_hvi", "hsv_to_rgb",
              "hvi_to_rgb", "HVIColorHead"}
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    definitions = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(node for node in tree.body
                               if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                               and node.name in wanted)
    if {node.name for node in definitions} != wanted or len(definitions) != len(wanted):
        raise ValueError("Original notebook definitions are incomplete or ambiguous")
    namespace = dict(torch=torch, nn=nn, F=F, math=math, ARCH=ARCH,
                     HEAD_WIDTH=HEAD_WIDTH, HVI_K=HVI_K, MAX_DELTA=MAX_DELTA,
                     RetinexFormer=RetinexFormer, IG_MSA=IG_MSA)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def verify_notebook(model, image, notebook_path=REFERENCE_NOTEBOOK):
    """Compare independently constructed notebook models with identical weights."""
    definitions = notebook_definitions(notebook_path)
    legacy_b, _ = definitions["build_model_b"]()
    legacy_head = definitions["HVIColorHead"]()
    legacy_b.load_state_dict(model.backbone.state_dict(), strict=True)
    legacy_head.load_state_dict(model.head.state_dict(), strict=True)
    device = image.device
    legacy_b, legacy_head = legacy_b.to(device).eval(), legacy_head.to(device).eval()
    model.eval()
    h, w = image.shape[-2:]
    padded = F.pad(image, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
    with torch.autocast(device.type, enabled=device.type == "cuda"):
        prediction = legacy_b(padded)
    prediction = prediction[..., :h, :w].float().clamp(0, 1)
    # Independent recreation of generate_cache -> rgb_tensor in the notebook.
    cached = torch.stack([
        torch.from_numpy(tensor_to_uint8(sample)).permute(2, 0, 1).float() / 255.0
        for sample in prediction
    ]).to(device)
    with torch.autocast(device.type, enabled=False):
        expected, expected_delta, _ = legacy_head(cached)
    actual = model.forward_with_intermediate(image)
    differences = dict(
        model_b=float((actual["model_b_rgb"] - prediction).abs().max()),
        head_input=float((actual["head_input"] - cached).abs().max()),
        hv_residual=float((actual["hv_residual"] - expected_delta).abs().max()),
        model_d=float((actual["output"] - expected).abs().max()),
    )
    if any(value > 1e-6 for value in differences.values()):
        raise AssertionError(f"Notebook parity failed: {differences}")
    return dict(status="passed", max_abs_difference=differences,
                notebook_sha256=sha256(notebook_path), shape=list(image.shape))


def load_pairs(data_root, split, split_file=None):
    """Use the original ordered split; never generate a replacement split."""
    config = REFERENCE["config"]
    frozen = dict(train=config["train_names"], val=config["val_names"])
    if split_file is not None:
        supplied = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if supplied != frozen:
            raise ValueError("split.json differs from the original 620/69 split")
    data_root = Path(data_root)
    if split == "val":
        pairs = find_paired_images(data_root / "Train/Low", data_root / "Train/Normal")
        indexed = {pair[0]: pair for pair in pairs}
        train, val = frozen["train"], frozen["val"]
        if (len(train) != 620 or len(val) != 69 or len(set(train)) != 620
                or len(set(val)) != 69 or set(train) & set(val)
                or set(train) | set(val) != set(indexed)):
            raise ValueError("Dataset does not match the frozen 620/69 split")
        return [indexed[name] for name in val]
    pairs = find_paired_images(data_root / "Test/Low", data_root / "Test/Normal")
    if len(pairs) != 100:
        raise ValueError("Expected exactly 100 official Test pairs")
    return pairs


def dataset_sha256(pairs):
    """Fingerprint actual ordered low/GT contents, not machine-specific paths."""
    import hashlib
    digest = hashlib.sha256()
    for name, low, gt in pairs:
        digest.update(json.dumps([name, sha256(low), sha256(gt)]).encode("utf-8"))
    return digest.hexdigest()


def runtime_info(device):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, cuda=torch.version.cuda, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                repository_commit=commit, reference_commit=REFERENCE["config"]["repo_commit"])


def check_validation_report(path, checkpoints, val_pairs):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if (report.get("protocol") != PROTOCOL or report.get("split") != "val"
            or report.get("count") != 69 or report.get("checkpoints") != checkpoints
            or report.get("historical_metrics_within_tolerance") is not True
            or report.get("notebook_parity", {}).get("status") != "passed"
            or report.get("runtime", {}).get("device", "").split(":")[0] != "cuda"
            or report.get("dataset_sha256") != dataset_sha256(val_pairs)):
        raise ValueError(
            "Test requires successful CUDA validation of this checkpoint pair and dataset")
    return sha256(path)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_output(path, protected=()):
    """Avoid mixing runs or overwriting source images and checkpoints."""
    path = Path(path).resolve()
    for source in protected:
        source = Path(source).resolve()
        if source == path or source.is_relative_to(path) or path.is_relative_to(source):
            raise ValueError(f"Output directory overlaps an input: {source}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@torch.inference_mode()
def evaluate(model, checkpoints, args, device):
    if device.type != "cuda":
        raise ValueError(
            "Historical evaluation requires CUDA AMP; CPU is for smoke inference/tests")
    val_pairs = load_pairs(args.data_root, "val", args.split_file)
    validation_hash = None
    if args.split == "test":
        if args.validation_report is None:
            raise ValueError("Run validation first and supply --validation-report")
        validation_hash = check_validation_report(args.validation_report, checkpoints, val_pairs)
    pairs = val_pairs if args.split == "val" else load_pairs(args.data_root, "test")
    parity = verify_notebook(model, read_rgb(val_pairs[0][1])[None].to(device),
                             args.reference_notebook)
    output = prepare_output(args.output_dir, [args.data_root, args.model_b, args.head])
    rows = []
    started = time.perf_counter()
    model.eval()
    if args.save_images:
        for name in ("ModelB", "ModelD"):
            (output / name).mkdir()
    for index, (name, low_path, gt_path) in enumerate(pairs, 1):
        low = read_rgb(low_path)[None].to(device)
        with Image.open(gt_path) as image:
            gt = np.asarray(image.convert("RGB")).copy()
        if tuple(low.shape[-2:]) != tuple(gt.shape[:2]):
            raise ValueError(f"Low/GT dimensions differ: {name}")
        prediction = model.forward_with_intermediate(low)
        row = dict(name=name)
        for key, tensor, folder in (("model_b", prediction["head_input"], "ModelB"),
                                    ("model_d", prediction["output"], "ModelD")):
            pixels = tensor_to_uint8(tensor[0])
            psnr, ssim = image_metrics(pixels, gt)
            if not math.isfinite(psnr) or not math.isfinite(ssim):
                raise FloatingPointError(f"Non-finite metrics: {name}")
            row[f"{key}_psnr"], row[f"{key}_ssim"] = psnr, ssim
            if args.save_images:
                Image.fromarray(pixels).save(output / folder / (Path(name).stem + ".png"))
        rows.append(row)
        if index == 1 or index % 10 == 0 or index == len(pairs):
            elapsed = time.perf_counter() - started
            print(f"{args.split}: {index}/{len(pairs)} ({elapsed:.1f}s)", flush=True)
    metrics = {name: {metric: float(np.mean([r[f"{name}_{metric}"] for r in rows]))
                      for metric in ("psnr", "ssim")} for name in ("model_b", "model_d")}
    historical = REFERENCE["metrics"][args.split]
    deltas = {name: {metric: metrics[name][metric] - historical[name][metric]
                     for metric in ("psnr", "ssim")} for name in metrics}
    matched = all(abs(values["psnr"]) <= PSNR_TOLERANCE
                  and abs(values["ssim"]) <= SSIM_TOLERANCE for values in deltas.values())
    report = dict(protocol=PROTOCOL, split=args.split, count=len(rows),
                  checkpoints=checkpoints, notebook_parity=parity,
                  dataset_sha256=dataset_sha256(pairs), runtime=runtime_info(device),
                  metrics=metrics, historical_metrics=historical, historical_delta=deltas,
                  tolerance=dict(psnr=PSNR_TOLERANCE, ssim=SSIM_TOLERANCE),
                  historical_metrics_within_tolerance=matched,
                  validation_report_sha256=validation_hash,
                  elapsed_seconds=time.perf_counter() - started)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "report.json", report)
    print(json.dumps(metrics, indent=2))
    if not matched:
        raise RuntimeError(
            f"Historical metrics outside tolerance; inspect {output / 'report.json'}")
    return report


@torch.inference_mode()
def infer(model, checkpoints, args, device):
    source = args.input
    paths = ([source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}))
    if not paths:
        raise ValueError("No input images found")
    if len({p.stem.casefold() for p in paths}) != len(paths):
        raise ValueError("Duplicate output PNG names in input directory")
    output = prepare_output(args.output_dir, [source, args.model_b, args.head])
    for path in paths:
        x = read_rgb(path)[None].to(device)
        if args.intermediates:
            values = model.forward_with_intermediate(x)
            np.savez_compressed(output / f"{path.stem}_intermediates.npz",
                                **{key: value.detach().float().cpu().numpy()
                                   for key, value in values.items()})
            prediction = values["output"]
        else:
            prediction = model(x)
        Image.fromarray(tensor_to_uint8(prediction[0])).save(output / f"{path.stem}.png")
    write_json(output / "inference.json", dict(
        protocol=PROTOCOL, count=len(paths), checkpoints=checkpoints,
        runtime=runtime_info(device), historical_protocol=device.type == "cuda"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("infer", "evaluate", "verify"):
        sub = commands.add_parser(command)
        sub.add_argument("--model-b", type=Path, required=True)
        sub.add_argument("--head", type=Path, required=True)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        if command in ("evaluate", "verify"):
            sub.add_argument("--reference-notebook", type=Path, default=REFERENCE_NOTEBOOK)
        if command == "infer":
            sub.add_argument("--input", type=Path, required=True)
            sub.add_argument("--intermediates", action="store_true")
        elif command == "evaluate":
            sub.add_argument("--data-root", type=Path, required=True)
            sub.add_argument("--split", choices=("val", "test"), required=True)
            sub.add_argument("--split-file", type=Path)
            sub.add_argument("--validation-report", type=Path)
            sub.add_argument("--save-images", action="store_true")
        else:
            sub.add_argument("--input", type=Path, required=True,
                             help="Real RGB image for notebook forward comparison")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    model, checkpoints = load_model_d(args.model_b, args.head, device=device)
    if args.command == "infer":
        infer(model, checkpoints, args, device)
    elif args.command == "evaluate":
        evaluate(model, checkpoints, args, device)
    else:
        result = verify_notebook(model, read_rgb(args.input)[None].to(device),
                                 args.reference_notebook)
        output = prepare_output(args.output_dir, [args.input, args.model_b, args.head])
        write_json(output / "verification.json", dict(
            notebook_parity=result, checkpoints=checkpoints, runtime=runtime_info(device)))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
