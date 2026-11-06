"""Inference, smoke/parity checks, validation, and explicitly requested training."""
import argparse
import csv
import json
import signal
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from analysis_utils import SUPPORTED_IMAGE_SUFFIXES, tensor_to_uint8
from model_d.cli import prepare_output, read_rgb
from model_d.model import REFERENCE
from .checkpoint import ROOT, EXPECTED_VAL_DATA, empty_model, load_model_d_pro
from .dataset import PairedStore
from .evaluate import evaluate
from .losses import training_loss
from .runtime import Budget, launch_training, request_stop, runtime_info, write_json
from .train import run_pilot
from .verify import verify_model

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "phase1_pilot.json"
HISTORY = ROOT / "experiments/model_d_pro/phase1"


def config_for(args):
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("data_root", "model_b", "head", "split_file", "output_root", "run_name",
                "max_steps", "validate_every", "save_every", "resume"):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value
    return cfg


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("infer", "smoke", "verify", "evaluate", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        sub.add_argument("--model-b", type=Path)
        sub.add_argument("--head", type=Path)
        if name == "train":
            for flag in ("data-root", "split-file", "output-root"):
                sub.add_argument("--" + flag, type=Path)
            sub.add_argument("--run-name")
            for flag in ("max-steps", "validate-every", "save-every"):
                sub.add_argument("--" + flag, type=int)
            sub.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
            sub.add_argument("--session-minutes", type=float, default=45)
            sub.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
        else:
            sub.add_argument("--checkpoint", type=Path)
            sub.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            sub.add_argument("--output-dir", type=Path, required=True)
            if name == "evaluate":
                sub.add_argument("--data-root", type=Path)
                sub.add_argument("--split-file", type=Path)
                sub.add_argument("--session-minutes", type=float, default=45)
                sub.add_argument("--compare-historical", action="store_true")
            else:
                sub.add_argument("--input", type=Path, required=name in {"infer", "verify"})
            if name == "smoke":
                sub.add_argument("--synthetic", action="store_true",
                                 help="Explicitly use untrained weights for CPU architecture checks")
    return result


def smoke(model, image, cfg, info):
    """Forward/backward only; never updates weights or reads the dataset."""
    model.set_phase(cfg["warmup_steps"], cfg["warmup_steps"])
    model.train()
    values = model(image, return_aux=True)
    target = torch.rand_like(image)
    loss, parts = training_loss(values, target)
    loss.backward()
    groups = {}
    for group in model.parameter_groups(cfg):
        gradients = [p.grad for p in group["params"] if p.grad is not None]
        groups[group["group_name"]] = dict(
            finite=all(bool(torch.isfinite(g).all()) for g in gradients),
            any_nonzero=any(bool(g.abs().max() > 0) for g in gradients))
    passed = (values["output"].shape == image.shape and bool(torch.isfinite(loss))
              and all(g["finite"] and g["any_nonzero"] for g in groups.values()))
    model.zero_grad(set_to_none=True)
    model.eval()
    return dict(status="passed" if passed else "failed", checkpoints=info,
                runtime=runtime_info(image.device), loss=float(loss.detach()),
                losses={k: float(v.detach()) for k, v in parts.items()}, gradient_groups=groups,
                shape=list(image.shape), optimizer_updates=0)


def compare_historical(result, csv_path):
    expected = json.loads((HISTORY / "pilot_summary.json").read_text())["best_validation"]
    with (HISTORY / "val_00900.csv").open(newline="") as stream:
        old = list(csv.DictReader(stream))
    with Path(csv_path).open(newline="") as stream:
        new = list(csv.DictReader(stream))
    if [r["name"] for r in new] != [r["name"] for r in old]:
        raise ValueError("Historical validation image ordering differs")
    deltas = [dict(name=a["name"], psnr=float(a["psnr"]) - float(b["psnr"]),
                   ssim=float(a["ssim"]) - float(b["ssim"])) for a, b in zip(new, old)]
    aggregate = {key: result[key] - expected[key] for key in ("psnr", "ssim")}
    passed = abs(aggregate["psnr"]) <= .002 and abs(aggregate["ssim"]) <= .0001
    return dict(status="passed" if passed else "failed", aggregate_delta=aggregate,
                per_image_deltas=deltas, psnr_tolerance=.002, ssim_tolerance=.0001)


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = config_for(args)
    if args.command == "train":
        if args.worker:
            signal.signal(signal.SIGTERM, request_stop)
            return run_pilot(cfg, ROOT, args.session_minutes)
        return launch_training(cfg, args.session_minutes)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if args.command == "evaluate" and device.type != "cuda":
        raise ValueError("Historical validation requires CUDA AMP; CPU is for smoke/parity checks")
    torch.manual_seed(cfg["seed"])
    if getattr(args, "synthetic", False):
        if args.checkpoint or args.model_b or args.head:
            raise ValueError("--synthetic cannot be combined with trained checkpoint arguments")
        model, info = empty_model().to(device), {"synthetic_untrained": True}
    elif args.checkpoint:
        model, info = load_model_d_pro(checkpoint=args.checkpoint, model_b=args.model_b,
                                      head=args.head, device=device)
    else:
        model, info = load_model_d_pro(model_b=cfg["model_b"], head=cfg["head"], device=device)
    protected = [cfg["model_b"], cfg["head"], cfg["data_root"]]
    if args.checkpoint:
        protected.append(args.checkpoint)
    if getattr(args, "input", None):
        protected.append(args.input)
    out = prepare_output(args.output_dir, protected)
    if args.command == "infer":
        images = sorted(p for p in args.input.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES) \
            if args.input.is_dir() else [args.input]
        if not images:
            raise ValueError("No input images")
        if len({p.stem for p in images}) != len(images):
            raise ValueError("Input image stems collide when exported as PNG")
        with torch.inference_mode():
            for path in images:
                x = read_rgb(path)[None].to(device)
                y = model(x)
                if y.shape != x.shape or not torch.isfinite(y).all():
                    raise FloatingPointError(f"Invalid output: {path}")
                Image.fromarray(tensor_to_uint8(y[0])).save(out / (path.stem + ".png"))
        report = dict(count=len(images), checkpoints=info, runtime=runtime_info(device))
    elif args.command in {"smoke", "verify"}:
        image = read_rgb(args.input)[None].to(device) if args.input else torch.rand(1, 3, 32, 32, device=device)
        if args.command == "smoke":
            image = image[..., :32, :32]
            report = smoke(model, image, cfg, info)
        else:
            report = verify_model(model, image, info)
    else:
        frozen = {k: REFERENCE["config"][k + "_names"] for k in ("train", "val")}
        split = Path(cfg["split_file"])
        if split.is_file() and json.loads(split.read_text()) != frozen:
            raise ValueError("split.json differs from the original 620/69 split")
        if not 5 <= args.session_minutes <= 120:
            raise ValueError("Session budget must be between 5 and 120 minutes")
        budget = Budget(args.session_minutes)
        store = PairedStore(cfg["data_root"], frozen["train"], frozen["val"], cfg["patch"], budget)
        if store.val_sha != EXPECTED_VAL_DATA:
            raise ValueError("Validation file contents differ from the pilot")
        if args.compare_historical and (not args.checkpoint or info.get("step") != 900):
            raise ValueError("Historical best comparison requires the step-900 best checkpoint")
        low = store.images[store.val_names[0]][0]
        image = torch.from_numpy(np.copy(low)).permute(2, 0, 1)[None].to(device).float() / 255
        parity = verify_model(model, image, info)
        write_json(out / "parity.json", parity)
        if parity["status"] != "passed":
            raise AssertionError("Original-script parity failed; see parity.json")
        report = evaluate(model, store, device, out / "validation", budget, "D-Pro modular validation")
        report.update(checkpoints=info, runtime=runtime_info(device), parity=parity)
        if args.compare_historical:
            comparison = compare_historical(report, out / "validation.csv")
            report["historical_comparison"] = comparison
            report["status"] = comparison["status"]
    write_json(out / "report.json", report)
    print(json.dumps(report, indent=2))
    return 1 if report.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
