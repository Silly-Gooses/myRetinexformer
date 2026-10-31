"""Read-only production checkpoint check with two isolated training updates."""

import argparse
from itertools import islice
import json
from pathlib import Path
import tempfile

import torch

from official_workflow import build_loaders, inspect_dataset, load_checkpoint
from protocol_metrics import evaluate_protocol, protocol_metrics, write_csv
from train import SCHEDULE_KEYS, build_model, resolve_device, validate_config
from training_utils import EpochWarmupCosine, restore_rng, seed_everything, train_one_epoch


def verify(config, checkpoint_path, target_epochs, device):
    validate_config(config)
    if type(target_epochs) is not int or target_epochs <= config["num_epochs"]:
        raise ValueError("target_epochs must exceed the original schedule length")
    manifest, pairs = inspect_dataset(config["dataset_root"])
    checkpoint = load_checkpoint(checkpoint_path, config, manifest)
    epoch, step = checkpoint["completed_epoch"], checkpoint["global_step"]
    if epoch >= target_epochs:
        raise ValueError("Checkpoint is already at or beyond the target epoch")
    seed_everything(config["seed"])
    train_loader, val_loader, generators = build_loaders(pairs, config, device)
    seed_everything(config["seed"])
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    assert all(torch.equal(v.cpu(), checkpoint["model"][k]) for k, v in model.state_dict().items())
    settings = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["warmup_start_lr"],
                                 betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if epoch and not optimizer.state:
        raise ValueError("Checkpoint has no Adam update state")
    scheduler = EpochWarmupCosine(optimizer, **{key: config[key] for key in SCHEDULE_KEYS})
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng(checkpoint["rng"], generators)
    lr = scheduler.set_epoch(epoch + 1)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    metrics, updates = train_one_epoch(
        model, train_loader, optimizer, scaler, device, config["use_amp"], True,
        mixup_beta=config["mixup_beta"], grad_clip_norm=config["grad_clip_norm"],
        max_steps=2, metric_fn=protocol_metrics)
    if updates != 2 or not any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise AssertionError("Two optimizer updates did not change the model")
    rows, summary = evaluate_protocol(model, islice(val_loader, 1), device)
    with tempfile.TemporaryDirectory(prefix="retinex_resume_smoke_") as temporary:
        root = Path(temporary)
        scratch = root / "scratch.pth"
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        completed_epoch=epoch, global_step=step + updates,
                        best=checkpoint["best"]), scratch)
        saved = torch.load(scratch, map_location="cpu", weights_only=True)
        if saved["global_step"] != step + 2 or saved["best"] != checkpoint["best"]:
            raise AssertionError("Scratch checkpoint round trip failed")
        write_csv(root / "validation.csv", ("image", "psnr", "ssim"), rows)
        if not (root / "validation.csv").is_file():
            raise AssertionError("Scratch validation log was not saved")
    print(f"Resume smoke passed: completed epoch {epoch}, next epoch {epoch + 1}, "
          f"global step {step} -> {step + 2}, LR {lr:.8g}")
    print(f"Two-update train loss {metrics['train_loss']:.6f}; one-image validation "
          f"PSNR {summary['psnr']:.4f}, SSIM {summary['ssim']:.4f}")
    print("Original checkpoint and experiment outputs were not modified")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-epochs", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    verify(config, args.checkpoint, args.target_epochs,
           resolve_device(args.device, config["use_amp"]))


if __name__ == "__main__":
    main()
