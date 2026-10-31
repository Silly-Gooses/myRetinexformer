"""Run proposed v1 epoch training independently of the Colab notebook.

Example:
    python train.py --config run_config.json --output-root experiments --mode smoke
    python train.py --config run_config.json --output-root experiments --mode full
    python train.py --config run_config.json --output-root experiments --mode full --resume

Use official_lolv2_config.json as the branch-1.1 configuration template. Smoke runs
use a separate _Smoke3 directory and retain the full learning-rate schedule.
"""

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import filecmp
import sys
import traceback
import json
import math
import os
from pathlib import Path
import shutil

# Must precede CUDA initialization, including when launched outside Colab.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from RetinexFormer_arch import RetinexFormer
from Proposed_V2_arch import AdaptiveRetinexFormer, OpenCVCanny
from canny import CannyEdgeDetector
from training_utils import (
    EpochWarmupCosine, epoch_learning_rate,
    seed_everything, should_log_epoch,
)


from official_workflow import (PROTOCOL, inspect_dataset, build_loaders, prepare_run,
                               run_training, storage_probe, load_checkpoint)
from protocol_metrics import test_checkpoint

DEFAULT_EXPERIMENT = "Proposed_RetinexFormer_V1_Official_Epoch200"
SCHEDULE_KEYS = ("num_epochs", "warmup_epochs", "warmup_start_lr", "peak_lr", "min_lr")


class _TeeStream:
    """Mirror Python text output to the console and a promptly flushed log."""
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def training_log(root, session):
    """Append after experiment validation, never bypassing fresh-run protection.

    Python stdout/stderr (including warnings) are mirrored live. Tracebacks from
    failed sessions are saved before restoring streams and propagating the error.
    Native/worker writes directly to OS file descriptors are not intercepted.
    """
    path = Path(root) / "logs" / "training.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_TeeStream(sys.stdout, log)), redirect_stderr(_TeeStream(sys.stderr, log)):
            print(f"\n=== {datetime.now(timezone.utc).isoformat()} | {session} ===", flush=True)
            print("Console log:", path, flush=True)
            try:
                yield
            except BaseException:
                # The caller/interpreter prints the exception to the restored console.
                traceback.print_exc(file=log)
                log.flush()
                raise
            else:
                print(f"=== {datetime.now(timezone.utc).isoformat()} | session completed ===", flush=True)


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
        "version", "training_mode", "seed", "dataset_root", "dataset_protocol",
        "val_freq_steps", "mixup_beta", "grad_clip_norm",
        *SCHEDULE_KEYS, "batch_size", "patch_size", "num_workers", "use_amp",
        "vis_every", "fixed_vis_count", "optimizer", "loss", "edge_guidance",
        "architecture", "canny", "deterministic",
    }
    if not isinstance(config, dict) or required - config.keys():
        missing = sorted(required - config.keys()) if isinstance(config, dict) else sorted(required)
        raise ValueError(f"Training configuration must be an object with fields: {missing}")
    if config["version"] not in ("v1", "v2"):
        raise ValueError("version must be v1 or v2")
    if config["version"] == "v2" and type(config.get("gated")) is not bool:
        raise ValueError("V2 requires gated=true or false")
    for key, expected in (("training_mode", "epoch"),
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
    if config["seed"] >= 2**32:
        raise ValueError("Require seed < 2**32")
    if config["dataset_protocol"] != PROTOCOL or "val_fraction" in config:
        raise ValueError("Use the official protocol without val_fraction; old split runs require a fresh experiment")
    if type(config["val_freq_steps"]) is not int or config["val_freq_steps"] < 1:
        raise ValueError("val_freq_steps must be a positive integer")
    for key in ("mixup_beta", "grad_clip_norm"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if type(config["use_amp"]) is not bool:
        raise ValueError("use_amp must be true or false")
    if not isinstance(config["dataset_root"], str) or not config["dataset_root"]:
        raise ValueError("dataset_root must be a nonempty path string")
    epoch_learning_rate(1, **{key: config[key] for key in SCHEDULE_KEYS})
    (OpenCVCanny if config["version"] == "v2" else CannyEdgeDetector)(**config["canny"])
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
        raise ValueError("Proposed models require RGB input/output")
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
    if config.get("version") == "v2":
        return AdaptiveRetinexFormer(**config["architecture"],
                                    canny_config=config["canny"],
                                    gated=config.get("gated", True)).to(device)
    return RetinexFormer(**config["architecture"], edge_guidance=True,
                         canny_config=config["canny"]).to(device)


def preserve_epoch_checkpoint(root, checkpoint, config, manifest):
    """Keep a complete, immutable stage boundary before last.pth advances."""
    state = load_checkpoint(checkpoint, config, manifest)
    epoch = state["completed_epoch"]
    if epoch < config["num_epochs"]:
        return None
    destination = Path(root) / "checkpoints" / f"epoch_{epoch:04d}.pth"
    if destination.exists():
        if not filecmp.cmp(checkpoint, destination, shallow=False):
            raise ValueError(f"Stage checkpoint conflicts with {destination}")
        return destination
    with destination.open("xb") as output, Path(checkpoint).open("rb") as source:
        shutil.copyfileobj(source, output)
    return destination


def train(config, root, device, *, mode="smoke", resume_from=None, log_every=3,
          initialize_from=None, initialize_source=None, target_epochs=None):
    """Build fresh training objects; the helper restores state on explicit resume."""
    validate_config(config)
    should_log_epoch(1, log_every, config["num_epochs"])
    if mode not in ("smoke", "full"):
        raise ValueError("mode must be smoke or full")
    if mode == "smoke" and config["num_epochs"] < 3:
        raise ValueError("Smoke mode requires a schedule with at least 3 epochs")
    target_epochs = config["num_epochs"] if target_epochs is None else target_epochs
    if (type(target_epochs) is not int or target_epochs < config["num_epochs"]
            or (target_epochs > config["num_epochs"] and (resume_from is None or mode != "full"))):
        raise ValueError("Training extension requires full mode, a resume checkpoint, and target >= original epochs")
    root = Path(root)
    seed_everything(config["seed"])
    manifest, pairs = inspect_dataset(config["dataset_root"])
    prepare_run(root, config, manifest, resume_from)
    if resume_from is not None:
        preserve_epoch_checkpoint(root, resume_from, config, manifest)
    session = f"{mode} training | {'resume' if resume_from is not None else 'fresh'}"
    with training_log(root, session):
        print("Training configuration:", json.dumps(config, sort_keys=True), flush=True)
        train_loader, val_loader, generators = build_loaders(pairs, config, device)
        # Match the notebook's original initialization order and seed.
        seed_everything(config["seed"])
        model = build_model(config, device)
        if initialize_from is not None:
            if config.get("version") != "v2" or resume_from is not None:
                raise ValueError("Source initialization requires a fresh V2 run")
            from analysis_utils import extract_checkpoint_state_dict
            source_checkpoint = torch.load(initialize_from, map_location="cpu", weights_only=True)
            model.initialize_from(extract_checkpoint_state_dict(source_checkpoint),
                                  source=initialize_source)
        settings = config["optimizer"]
        optimizer = torch.optim.Adam(model.parameters(), lr=config["warmup_start_lr"],
                                     betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
        scheduler = EpochWarmupCosine(optimizer, **{key: config[key] for key in SCHEDULE_KEYS})
        scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
        print(f"Mode: {mode} | Device: {device} | Output: {root}", flush=True)
        print(f"Training pairs: {len(train_loader.dataset)} | Validation pairs: {len(val_loader.dataset)} | "
              f"Batches per epoch: {len(train_loader)}", flush=True)
        print(f"Total scheduled batches: {len(train_loader) * config['num_epochs']} | "
              f"Validation every {config['val_freq_steps']} batches "
              f"({config['val_freq_steps'] / len(train_loader):.4f} epochs)", flush=True)
        print("Trainable parameters:", sum(p.numel() for p in model.parameters()), flush=True)
        print("Resume:", resume_from or "Fresh training", flush=True)
        if target_epochs > config["num_epochs"]:
            print(f"Extension target: {target_epochs} | LR after epoch {config['num_epochs']}: "
                  f"{config['min_lr']:.8g} (fixed original cosine endpoint)", flush=True)
        history = run_training(
            model, train_loader, val_loader, optimizer, scheduler, scaler,
            device, config, manifest, root, generators, resume_from=resume_from,
            stop_after_epoch=3 if mode == "smoke" else None,
            target_epochs=target_epochs, log_every=log_every)
        if mode == "full" and history[-1]["epoch"] == target_epochs:
            archived = preserve_epoch_checkpoint(root, root / "checkpoints/last.pth", config, manifest)
            if archived is not None:
                print("Preserved stage checkpoint:", archived, flush=True)
        print(f"{mode.capitalize()} training finished at epoch {history[-1]['epoch']}", flush=True)
        for label, path in (("Best", "checkpoints/best.pth"), ("Last", "checkpoints/last.pth"),
                            ("CSV", "logs/training_metrics.csv"), ("Visuals", "visuals")):
            print(f"{label}: {root / path}", flush=True)
        return history


def evaluate(config, root, device):
    """Compatibility entry point; independent testing does not require completion."""
    root = Path(root)
    output = root / "testing" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return test_checkpoint(root / "checkpoints/best.pth", config["dataset_root"],
                           output, device, config)


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
    parser.add_argument("--initialize-from", type=Path,
                        help="Fresh V2 run: copy shared weights from an E0 or E1 checkpoint")
    parser.add_argument("--initialize-source", choices=("baseline", "v1"),
                        help="Source architecture for --initialize-from")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluate-only", action="store_true", help="Independently test best.pth without requiring completed training")
    parser.add_argument("--preflight", action="store_true", help="Audit all 689/100 official pairs without training")
    parser.add_argument("--check-storage", type=Path, help="Save/reload a temporary checkpoint in this mounted Drive/local directory")
    parser.add_argument("--target-epochs", type=int, help="Total epochs for an extended full resume; original schedule stays in config")
    args = parser.parse_args(argv)
    if bool(args.initialize_from) != bool(args.initialize_source):
        parser.error("--initialize-from and --initialize-source must be given together")
    if args.initialize_from and (args.resume or args.resume_from or args.evaluate_only or args.preflight):
        parser.error("Source initialization requires a fresh training run")
    if args.evaluate_only and (args.mode != "full" or args.resume or args.resume_from):
        parser.error("--evaluate-only requires --mode full and no resume arguments")
    config = json.loads(args.config.read_text())
    validate_config(config)
    device = resolve_device(args.device, config["use_amp"] and not (args.preflight or args.evaluate_only))
    root = experiment_directory(args.output_root, args.experiment_name, args.mode)
    seed_everything(config["seed"])
    if args.check_storage:
        print("Storage probe passed:", storage_probe(args.check_storage))
    if args.preflight:
        manifest, _ = inspect_dataset(config["dataset_root"])
        print(json.dumps({"protocol": PROTOCOL,
                          "counts": {k: len(v) for k, v in manifest["splits"].items()},
                          "validation_source": "Test", "overlap": False}, indent=2))
        return manifest
    if args.evaluate_only:
        return evaluate(config, root, device)
    resume_from = root / "checkpoints/last.pth" if args.resume else args.resume_from
    return train(config, root, device, mode=args.mode, resume_from=resume_from,
                 log_every=args.log_every, initialize_from=args.initialize_from,
                 initialize_source=args.initialize_source, target_epochs=args.target_epochs)


if __name__ == "__main__":
    main()
