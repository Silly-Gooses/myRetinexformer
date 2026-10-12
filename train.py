"""Run proposed v1 epoch training independently of the Colab notebook.

Example:
    python train.py --config run_config.json --output-root experiments --mode smoke
    python train.py --config run_config.json --output-root experiments --mode full
    python train.py --config run_config.json --output-root experiments --mode full --resume

The JSON configuration uses the existing notebook/checkpoint schema. Smoke runs
use a separate _Smoke3 directory and retain the full learning-rate schedule.
"""

import argparse
import json
import os
from pathlib import Path

# Must precede CUDA initialization, including when launched outside Colab.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from RetinexFormer_arch import RetinexFormer
from analysis_utils import export_test_set, find_paired_images
from canny import CannyEdgeDetector
from dataLoader import paired_image_Dataset
from training_utils import (
    EpochWarmupCosine, build_epoch_loaders, epoch_learning_rate,
    load_epoch_checkpoint, prepare_experiment, run_epoch_training,
    seed_everything, should_log_epoch,
)


DEFAULT_EXPERIMENT = "Lightweight_RetinexFormer_v1_LOLv2_real_Epoch200"
SCHEDULE_KEYS = ("num_epochs", "warmup_epochs", "warmup_start_lr", "peak_lr", "min_lr")


def experiment_directory(output_root, experiment_name, mode):
    """Keep existing full-run paths; isolate smoke checkpoints and exports."""
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    if not experiment_name or Path(experiment_name).name != experiment_name or experiment_name in (".", ".."):
        raise ValueError("experiment_name must be a single directory name")
    suffix = "_Smoke3" if mode == "smoke" else ""
    return Path(output_root) / (experiment_name + suffix)


def validate_config(config):
    """Reject unsupported settings before creating an experiment directory."""
    required = {
        "version", "training_mode", "seed", "dataset_root", "val_fraction",
        *SCHEDULE_KEYS, "batch_size", "patch_size", "num_workers", "use_amp",
        "vis_every", "fixed_vis_count", "optimizer", "loss", "edge_guidance",
        "architecture", "canny", "deterministic",
    }
    if not isinstance(config, dict) or required - config.keys():
        missing = sorted(required - config.keys()) if isinstance(config, dict) else sorted(required)
        raise ValueError(f"Training configuration must be an object with fields: {missing}")
    for key, expected in (("version", "v1"), ("training_mode", "epoch"),
                          ("loss", "L1"), ("edge_guidance", True), ("deterministic", True)):
        if config[key] != expected:
            raise ValueError(f"This proposed-model trainer requires {key}={expected!r}")
    for key, minimum in (("batch_size", 1), ("patch_size", 12), ("num_workers", 0),
                         ("vis_every", 1), ("fixed_vis_count", 1), ("seed", 0)):
        value = config[key]
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if config["patch_size"] % 4:
        raise ValueError("patch_size must be a multiple of 4 for this architecture")
    if config["seed"] >= 2**32 or not 0 < config["val_fraction"] < 1:
        raise ValueError("Require seed < 2**32 and 0 < val_fraction < 1")
    if type(config["use_amp"]) is not bool:
        raise ValueError("use_amp must be true or false")
    if not isinstance(config["dataset_root"], str) or not config["dataset_root"]:
        raise ValueError("dataset_root must be a nonempty path string")
    epoch_learning_rate(1, **{key: config[key] for key in SCHEDULE_KEYS})
    CannyEdgeDetector(**config["canny"])
    optimizer = config["optimizer"]
    if optimizer.get("name") != "Adam" or set(optimizer) != {"name", "betas", "weight_decay"}:
        raise ValueError("optimizer must specify Adam, betas and weight_decay")
    betas = optimizer["betas"]
    if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas) or not 0 <= optimizer["weight_decay"] < float("inf"):
        raise ValueError("Invalid Adam betas or weight_decay")
    arch = config["architecture"]
    if set(arch) != {"in_channels", "out_channels", "n_feat", "stage", "num_blocks"}:
        raise ValueError("Unexpected architecture fields")
    if arch["in_channels"] != 3 or arch["out_channels"] != 3:
        raise ValueError("Proposed v1 requires RGB input/output")
    if type(arch["n_feat"]) is not int or arch["n_feat"] < 4 or arch["n_feat"] % 4:
        raise ValueError("n_feat must be a positive multiple of 4")
    if type(arch["stage"]) is not int or arch["stage"] < 1:
        raise ValueError("stage must be a positive integer")
    if len(arch["num_blocks"]) != 3 or any(type(n) is not int or n < 0 for n in arch["num_blocks"]):
        raise ValueError("num_blocks must contain three nonnegative integers")


def resolve_device(name, use_amp):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if use_amp and name != "cuda":
        raise ValueError("use_amp=true requires CUDA")
    return torch.device(name)


def build_model(config, device):
    return RetinexFormer(**config["architecture"], edge_guidance=True,
                         canny_config=config["canny"]).to(device)


def train(config, root, device, *, mode="smoke", resume_from=None, log_every=3):
    """Build fresh training objects; the helper restores state on explicit resume."""
    validate_config(config)
    should_log_epoch(1, log_every, config["num_epochs"])
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    if mode == "smoke" and config["num_epochs"] < 3:
        raise ValueError("Smoke mode requires a schedule with at least 3 epochs")
    root = Path(root)
    seed_everything(config["seed"])
    data = Path(config["dataset_root"])
    low, normal = data / "Train/Low", data / "Train/Normal"
    pairs = find_paired_images(low, normal)
    split = prepare_experiment(root, config, [name for name, _, _ in pairs], resume_from)
    if resume_from is not None:
        checkpoint = load_epoch_checkpoint(resume_from, config, split)
        if mode == "smoke" and checkpoint["completed_epoch"] > 3:
            raise ValueError("Smoke mode cannot resume a checkpoint beyond epoch 3")
    train_loader, val_loader, generators = build_epoch_loaders(
        low, normal, split, batch_size=config["batch_size"], patch_size=config["patch_size"],
        num_workers=config["num_workers"], seed=config["seed"], pin_memory=(device.type == "cuda"))
    # Match the notebook's original initialization order and seed.
    seed_everything(config["seed"])
    model = build_model(config, device)
    settings = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["warmup_start_lr"],
                                 betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
    scheduler = EpochWarmupCosine(optimizer, **{key: config[key] for key in SCHEDULE_KEYS})
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
    print(f"Mode: {mode} | Device: {device} | Output: {root}", flush=True)
    print(f"Training pairs: {len(train_loader.dataset)} | Validation pairs: {len(val_loader.dataset)} | "
          f"Batches per epoch: {len(train_loader)}", flush=True)
    print("Trainable parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    print("Resume:", resume_from or "Fresh training", flush=True)
    history = run_epoch_training(
        model, train_loader, val_loader, optimizer, scheduler, scaler,
        device, config, split, root, generators, resume_from=resume_from,
        stop_after_epoch=3 if mode == "smoke" else None, log_every=log_every)
    print(f"{mode.capitalize()} training finished at epoch {history[-1]['epoch']}", flush=True)
    for label, path in (("Best", "checkpoints/best.pth"), ("Last", "checkpoints/last.pth"),
                        ("CSV", "logs/training_metrics.csv"), ("Visuals", "visuals")):
        print(f"{label}: {root / path}", flush=True)
    return history


def evaluate(config, root, device):
    """Export official test results only after the full schedule is complete."""
    validate_config(config)
    root = Path(root)
    split = json.loads((root / "split.json").read_text())
    last = load_epoch_checkpoint(root / "checkpoints/last.pth", config, split)
    if last["completed_epoch"] != config["num_epochs"]:
        raise ValueError("Complete the configured training run before final test evaluation")
    best = load_epoch_checkpoint(root / "checkpoints/best.pth", config, split)
    model = build_model(config, device)
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    data = Path(config["dataset_root"])
    low, normal = data / "Test/Low", data / "Test/Normal"
    find_paired_images(low, normal)
    dataset = paired_image_Dataset(str(low), str(normal), split="test", crop_size=None, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    output = root / "final_test_outputs"
    psnr, ssim = export_test_set(model, loader, device, output)
    print(f"Test images: {len(dataset)} | Best epoch: {best['best']['epoch']}")
    print(f"Test PSNR: {psnr:.4f} | SSIM: {ssim:.4f}")
    print("Outputs and metrics:", output)
    return psnr, ssim


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="JSON with the notebook/checkpoint training configuration")
    parser.add_argument("--output-root", required=True, type=Path, help="Parent directory containing experiment folders")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true", help="Resume this run's checkpoints/last.pth")
    resume.add_argument("--resume-from", type=Path, help="Explicit path to this run's checkpoints/last.pth")
    parser.add_argument("--log-every", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate best.pth after completed full training; no training")
    args = parser.parse_args(argv)
    if args.evaluate_only and (args.mode != "full" or args.resume or args.resume_from):
        parser.error("--evaluate-only requires --mode full and no resume arguments")
    config = json.loads(args.config.read_text())
    validate_config(config)
    device = resolve_device(args.device, config["use_amp"])
    root = experiment_directory(args.output_root, args.experiment_name, args.mode)
    seed_everything(config["seed"])
    if args.evaluate_only:
        return evaluate(config, root, device)
    resume_from = root / "checkpoints/last.pth" if args.resume else args.resume_from
    return train(config, root, device, mode=args.mode, resume_from=resume_from, log_every=args.log_every)


if __name__ == "__main__":
    main()
