"""Epoch training, reproducible splits, and restartable checkpoints for v1."""

import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from analysis_utils import image_metrics, infer_full_image, tensor_to_uint8


CHECKPOINT_FORMAT = "retinexformer-epoch-v1"
METRIC_FIELDS = ("epoch", "global_step", "train_loss", "val_loss", "val_psnr",
                 "val_ssim", "lr", "epoch_seconds", "elapsed_seconds", "is_best")


def _canonical(value):
    return json.loads(json.dumps(value))


def seed_everything(seed):
    # Set before the first CUDA matrix multiplication in this process.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def epoch_learning_rate(epoch, num_epochs, warmup_epochs, warmup_start_lr,
                        peak_lr, min_lr):
    if not (isinstance(num_epochs, int) and isinstance(warmup_epochs, int)
            and num_epochs > warmup_epochs >= 1):
        raise ValueError("Require num_epochs > warmup_epochs >= 1")
    if not all(math.isfinite(x) and x > 0 for x in (warmup_start_lr, peak_lr, min_lr)):
        raise ValueError("Learning rates must be positive and finite")
    if max(warmup_start_lr, min_lr) > peak_lr:
        raise ValueError("Warm-up start and minimum LR cannot exceed peak LR")
    if not isinstance(epoch, int) or not 1 <= epoch <= num_epochs:
        raise ValueError("epoch must be between 1 and num_epochs")
    if epoch <= warmup_epochs:
        progress = (epoch - 1) / (warmup_epochs - 1) if warmup_epochs > 1 else 1.0
        return warmup_start_lr + progress * (peak_lr - warmup_start_lr)
    progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
    return min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2


class EpochWarmupCosine:
    """Set the LR explicitly at the start of each 1-based epoch.

    Checkpoints store the LR used by the completed epoch. Resume sets the next
    epoch once; there is no implicit scheduler.step() or off-by-one adjustment.
    """
    def __init__(self, optimizer, *, num_epochs, warmup_epochs, warmup_start_lr,
                 peak_lr, min_lr):
        self.optimizer = optimizer
        self.schedule = dict(num_epochs=num_epochs, warmup_epochs=warmup_epochs,
                             warmup_start_lr=warmup_start_lr, peak_lr=peak_lr,
                             min_lr=min_lr)
        epoch_learning_rate(1, **self.schedule)
        self.epoch = 0

    def set_epoch(self, epoch):
        lr = epoch_learning_rate(epoch, **self.schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.epoch = epoch
        return lr

    def state_dict(self):
        return {"epoch": self.epoch, "schedule": dict(self.schedule)}

    def load_state_dict(self, state):
        if state["schedule"] != self.schedule:
            raise ValueError("Resume scheduler settings differ from this run")
        self.set_epoch(state["epoch"])


def make_split(filenames, seed=42, val_fraction=0.1):
    names = sorted(filenames)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError("Dataset needs at least two uniquely named image pairs")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    count = math.ceil(val_fraction * len(names))
    if count >= len(names):
        raise ValueError("Training and validation subsets must both be nonempty")
    shuffled = names.copy()
    random.Random(seed).shuffle(shuffled)
    return {"seed": seed, "val_fraction": val_fraction,
            "train": sorted(shuffled[count:]), "validation": sorted(shuffled[:count])}


def validate_split(split, filenames):
    train, val = split["train"], split["validation"]
    if (not train or not val or len(set(train)) != len(train)
            or len(set(val)) != len(val) or set(train) & set(val)
            or sorted(train + val) != sorted(filenames)):
        raise ValueError("Dataset membership differs from the saved split, or split is invalid")


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(Path(temporary))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path, value):
    _atomic_write(path, lambda p: p.write_text(json.dumps(value, indent=2) + "\n"))


def prepare_experiment(root, config, filenames, resume_from=None):
    """Create a new run, or validate the existing manifest without replacing it."""
    root = Path(root)
    config = _canonical(config)
    if resume_from is None:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Run already exists: {root}. Set RESUME_FROM or choose a new name.")
        split = make_split(filenames, config["seed"], config["val_fraction"])
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "config.json", config)
        _write_json(root / "split.json", split)
    else:
        if Path(resume_from).resolve() != (root / "checkpoints" / "last.pth").resolve():
            raise ValueError("Resume this experiment from its checkpoints/last.pth")
        if not Path(resume_from).is_file():
            raise FileNotFoundError(resume_from)
        stored_config = json.loads((root / "config.json").read_text())
        if stored_config != config:
            changed = sorted(k for k in set(config) | set(stored_config)
                             if config.get(k) != stored_config.get(k))
            raise ValueError(f"Resume configuration differs: {changed}")
        split = json.loads((root / "split.json").read_text())
        validate_split(split, filenames)
    for folder in ("checkpoints", "logs", "visuals"):
        (root / folder).mkdir(exist_ok=True)
    return split


def build_epoch_loaders(low_dir, normal_dir, split, *, batch_size=8, patch_size=256,
                        num_workers=2, seed=42, pin_memory=False):
    from dataLoader import paired_image_Dataset

    train_base = paired_image_Dataset(str(low_dir), str(normal_dir), split="train",
                                     crop_size=(patch_size, patch_size), augment=True)
    val_base = paired_image_Dataset(str(low_dir), str(normal_dir), split="test",
                                   crop_size=None, augment=False)
    names = [Path(p).name for p in train_base.low_paths]
    validate_split(split, names)
    lookup = {name: i for i, name in enumerate(names)}
    train = Subset(train_base, [lookup[name] for name in split["train"]])
    validation = Subset(val_base, [lookup[name] for name in split["validation"]])
    generators = {"train": torch.Generator().manual_seed(seed),
                  "validation": torch.Generator().manual_seed(seed + 1)}
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=False, worker_init_fn=seed_worker, drop_last=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              generator=generators["train"], **common)
    val_loader = DataLoader(validation, batch_size=1, shuffle=False,
                            generator=generators["validation"], **common)
    return train_loader, val_loader, generators


def capture_rng(generators):
    numpy_state = np.random.get_state()
    # Store only tensors and primitive types, compatible with weights_only=True.
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {k: g.get_state() for k, g in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if set(generators) != set(state["generators"]):
        raise ValueError("Resume DataLoader generators differ")
    for key, generator in generators.items():
        generator.set_state(state["generators"][key].cpu())


def write_history(path, history):
    def write(temporary):
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
            writer.writeheader()
            writer.writerows(history)
    _atomic_write(path, write)


def load_epoch_checkpoint(path, config, split):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Expected an epoch-training checkpoint; iteration checkpoints cannot resume this run")
    if checkpoint["config"] != _canonical(config):
        raise ValueError("Resume checkpoint configuration differs (model, batch size, schedule, or seed)")
    if checkpoint["split"] != split:
        raise ValueError("Resume checkpoint split differs")
    epoch = checkpoint["completed_epoch"]
    history = checkpoint["history"]
    if (not 1 <= epoch <= config["num_epochs"] or len(history) != epoch
            or [row["epoch"] for row in history] != list(range(1, epoch + 1))
            or history[-1]["global_step"] != checkpoint["global_step"]
            or checkpoint["scheduler"]["epoch"] != epoch):
        raise ValueError("Inconsistent completed epoch, scheduler, or metric history")
    return checkpoint


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp=False):
    model.train()
    total, samples, steps = 0.0, 0, 0
    for batch in loader:
        low, high = batch["low"].to(device), batch["high"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=torch.device(device).type, enabled=use_amp):
            loss = F.l1_loss(model(low), high)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; epoch was not checkpointed")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        size = low.shape[0]
        total += float(loss.detach()) * size
        samples += size
        steps += 1
    if samples == 0:
        raise ValueError("Empty training loader")
    return total / samples, steps


@torch.inference_mode()
def validate_epoch(model, loader, device):
    was_training = model.training
    losses, psnrs, ssims = [], [], []
    model.eval()
    try:
        for batch in loader:
            prediction = infer_full_image(model, batch["low"].to(device))
            target = batch["high"].to(device)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite validation output")
            for output, gt in zip(prediction, target):
                losses.append(float(F.l1_loss(output, gt)))
                psnr, ssim = image_metrics(tensor_to_uint8(output), tensor_to_uint8(gt))
                psnrs.append(psnr)
                ssims.append(ssim)
    finally:
        model.train(was_training)
    if not losses:
        raise ValueError("Empty validation loader")
    return {"val_loss": float(np.mean(losses)), "val_psnr": float(np.mean(psnrs)),
            "val_ssim": float(np.mean(ssims))}


@torch.inference_mode()
def save_epoch_visuals(model, dataset, indices, device, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    try:
        for index in indices:
            sample = dataset[index]
            low = sample["low"].unsqueeze(0).to(device)
            h, w = low.shape[-2:]
            ph, pw = (-h) % 4, (-w) % 4
            mode = "reflect" if ph < h and pw < w else "replicate"
            padded = F.pad(low, (0, pw, 0, ph), mode=mode)
            details = model.forward_with_intermediate(padded, return_dict=True)
            panels = [("low", low[0]), ("edge", details["edge"][0, :, :h, :w]),
                      ("lit_up", details["lit_up"][0, :, :h, :w]),
                      ("output", details["output"][0, :, :h, :w]), ("gt", sample["high"])]
            grid = Image.new("RGB", (w * len(panels), h + 26), "white")
            draw = ImageDraw.Draw(grid)
            name = Path(sample["name"]).stem + ".png"
            for column, (label, tensor) in enumerate(panels):
                folder = directory / label
                folder.mkdir(exist_ok=True)
                if label == "edge":
                    pixels = (tensor[0].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
                    image = Image.fromarray(pixels)
                else:
                    image = Image.fromarray(tensor_to_uint8(tensor))
                image.save(folder / name)
                grid.paste(image.convert("RGB"), (column * w, 26))
                draw.text((column * w + 4, 5), label, fill="black")
            grid.save(directory / name)
    finally:
        model.train(was_training)


def run_epoch_training(model, train_loader, val_loader, optimizer, scheduler, scaler,
                       device, config, split, root, generators, resume_from=None,
                       stop_after_epoch=None, visualize=True):
    """Train complete epochs; optional stop_after_epoch supports short smoke runs."""
    root = Path(root)
    config = _canonical(config)
    history, global_step, start_epoch = [], 0, 1
    best = {"psnr": -float("inf"), "ssim": None, "epoch": 0}
    metrics_path = root / "logs" / "training_metrics.csv"
    if resume_from is not None:
        if Path(resume_from).resolve() != (root / "checkpoints" / "last.pth").resolve():
            raise ValueError("Resume requires this experiment's last.pth")
        checkpoint = load_epoch_checkpoint(resume_from, config, split)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, best = checkpoint["history"], checkpoint["best"]
        global_step = checkpoint["global_step"]
        start_epoch = checkpoint["completed_epoch"] + 1
        restore_rng(checkpoint["rng"], generators)
    elif (root / "checkpoints" / "last.pth").exists() or metrics_path.exists():
        raise FileExistsError("Training already started; resume last.pth instead of overwriting")
    # The committed checkpoint history is authoritative, even if a crash left CSV ahead.
    write_history(metrics_path, history)
    final_epoch = config["num_epochs"] if stop_after_epoch is None else stop_after_epoch
    if not 1 <= final_epoch <= config["num_epochs"]:
        raise ValueError("stop_after_epoch must be within the configured schedule")
    indices = np.linspace(0, len(val_loader.dataset) - 1,
                          min(config["fixed_vis_count"], len(val_loader.dataset)), dtype=int).tolist()
    elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, final_epoch + 1):
        started = time.perf_counter()
        lr = scheduler.set_epoch(epoch)
        train_loss, steps = train_one_epoch(model, train_loader, optimizer, scaler,
                                           device, config["use_amp"])
        global_step += steps
        metrics = validate_epoch(model, val_loader, device)
        improved = metrics["val_psnr"] > best["psnr"]
        if improved:
            best = {"psnr": metrics["val_psnr"], "ssim": metrics["val_ssim"], "epoch": epoch}
        if visualize and (epoch == 1 or epoch % config["vis_every"] == 0
                          or epoch == config["num_epochs"]):
            save_epoch_visuals(model, val_loader.dataset, indices, device,
                               root / "visuals" / f"epoch_{epoch:04d}")
        seconds = time.perf_counter() - started
        elapsed += seconds
        history.append(dict(epoch=epoch, global_step=global_step, train_loss=train_loss,
                            **metrics, lr=lr, epoch_seconds=seconds,
                            elapsed_seconds=elapsed, is_best=improved))
        checkpoint = {"format": CHECKPOINT_FORMAT, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "scaler": scaler.state_dict(), "completed_epoch": epoch,
                      "global_step": global_step, "best": best, "config": config,
                      "split": split, "history": history, "rng": capture_rng(generators)}
        # Write best before last; if interrupted, replaying this epoch restores consistency.
        if improved:
            _atomic_write(root / "checkpoints" / "best.pth", lambda p: torch.save(checkpoint, p))
        _atomic_write(root / "checkpoints" / "last.pth", lambda p: torch.save(checkpoint, p))
        write_history(metrics_path, history)
        print(f"Epoch {epoch:03d}/{config['num_epochs']} | step {global_step} | "
              f"LR {lr:.8g} | train L1 {train_loss:.6f} | val L1 {metrics['val_loss']:.6f} | "
              f"PSNR {metrics['val_psnr']:.4f} | SSIM {metrics['val_ssim']:.4f} | "
              f"{seconds:.1f}s" + (" | BEST" if improved else ""))
    return history
