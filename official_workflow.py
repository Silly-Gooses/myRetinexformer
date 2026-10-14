"""LOL-v2-real official membership with epoch scheduling and recoverable logs.

Legacy split helpers remain in training_utils only for old analysis/tests. This
module never generates a split and never resumes a legacy split checkpoint.
"""

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from analysis_utils import find_paired_images
from dataLoader import OfficialPairedDataset
from protocol_metrics import VALIDATION_PROTOCOL, evaluate_protocol, protocol_metrics, write_csv
from training_utils import (_atomic_write, _write_json, capture_rng,
                            restore_rng, seed_worker, save_epoch_visuals, train_one_epoch,
                            should_log_epoch, METRIC_FIELDS, write_history)

PROTOCOL = "lol-v2-real-official-v1.1"
FORMAT = "retinexformer-official-epoch-v1.1"
SNAPSHOT_FORMAT = "retinexformer-official-weights-v1.1"
VAL_FIELDS = ("event", "epoch", "batch", "global_step", "kind", "count", "psnr", "ssim",
              "protocol", "is_best")


def inspect_dataset(root, *, training=True):
    """Validate official counts, exact pairing, dimensions, and content isolation.

    Relative paths and decoded-pixel hashes let a Colab run move between mounts
    without changing membership. Filenames alone are not cross-split identities.
    Independent testing only opens Test; training audits both directories.
    """
    root = Path(root)
    entries, pairs, identities = {}, {}, {}
    for key, folder, expected in (("train", "Train", 689), ("test", "Test", 100)):
        if key == "train" and not training:
            continue
        found = find_paired_images(root / folder / "Low", root / folder / "Normal")
        if len(found) != expected:
            raise ValueError(f"{folder}: expected {expected} paired images, found {len(found)}")
        records, low_hashes, high_hashes, paths = [], set(), set(), set()
        for name, low, high in found:
            a, b = OfficialPairedDataset.read(low), OfficialPairedDataset.read(high)
            if a.shape != b.shape or min(a.shape[:2]) < 11:
                raise ValueError(f"Invalid paired dimensions (minimum 11 x 11): {folder}/{name}")
            def digest(image):
                return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            ah, bh = digest(a), digest(b)
            low_hashes.add(ah)
            high_hashes.add(bh)
            paths.update((str(low.resolve()), str(high.resolve())))
            records.append(dict(name=name, low=f"{folder}/Low/{name}",
                                normal=f"{folder}/Normal/{name}", shape=list(a.shape),
                                low_sha256=ah, normal_sha256=bh))
        entries[key], pairs[key] = records, found
        identities[key] = (low_hashes, high_hashes, paths)
    if training:
        train_ids, test_ids = identities["train"], identities["test"]
        if ((train_ids[0] | train_ids[1]) & (test_ids[0] | test_ids[1])
                or train_ids[2] & test_ids[2]):
            raise ValueError("Train/Test overlap: shared resolved paths or identical decoded images")
    manifest = dict(protocol=PROTOCOL, splits=entries, validation_source="test",
                    testing_source="test", expected_train=689, expected_test=100)
    return manifest, pairs


def build_loaders(pairs, config, device):
    generators = {"train": torch.Generator().manual_seed(config["seed"]),
                  "validation": torch.Generator().manual_seed(config["seed"] + 1)}
    train = DataLoader(OfficialPairedDataset(pairs["train"], config["patch_size"]),
                       batch_size=config["batch_size"], shuffle=True, drop_last=False,
                       num_workers=config["num_workers"], persistent_workers=False,
                       worker_init_fn=seed_worker, generator=generators["train"],
                       pin_memory=torch.device(device).type == "cuda")
    val = DataLoader(OfficialPairedDataset(pairs["test"]), batch_size=1, shuffle=False,
                     num_workers=0, generator=generators["validation"])
    return train, val, generators


def prepare_run(root, config, manifest, resume_from=None):
    root = Path(root)
    if resume_from is None:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Run already exists: {root}; choose a new name or resume last.pth")
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "config.json", config)
        _write_json(root / "dataset_manifest.json", manifest)
    else:
        if Path(resume_from).resolve() != (root / "checkpoints/last.pth").resolve():
            raise ValueError("Resume requires this experiment's checkpoints/last.pth")
        if not (root / "dataset_manifest.json").is_file():
            raise ValueError("Legacy split runs cannot resume the official protocol; start a fresh experiment")
        if json.loads((root / "config.json").read_text()) != config:
            raise ValueError("Resume configuration differs")
        if json.loads((root / "dataset_manifest.json").read_text()) != manifest:
            raise ValueError("Resume dataset membership or image contents differ")
        load_checkpoint(resume_from, config, manifest)
    for name in ("checkpoints", "logs", "validation", "visuals"):
        (root / name).mkdir(exist_ok=True)


def load_checkpoint(path, config, manifest):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("Resume requires a matching official-protocol epoch checkpoint")
    if state["config"] != config or state["manifest"] != manifest:
        raise ValueError("Checkpoint configuration or dataset manifest differs")
    epoch, step = state["completed_epoch"], state["global_step"]
    batches = math.ceil(len(manifest["splits"]["train"]) / config["batch_size"])
    if (type(epoch) is not int or not 0 <= epoch <= config["num_epochs"]
            or type(step) is not int or step != epoch * batches
            or state["scheduler"]["epoch"] != epoch
            or [r["epoch"] for r in state["history"]] != list(range(epoch + 1))
            or [r["global_step"] for r in state["history"]] != [e*batches for e in range(epoch+1)]
            or any(r["global_step"] > step for r in state["validation_history"])):
        raise ValueError("Inconsistent checkpoint epoch, step, scheduler, or history")
    return state


@contextmanager
def diagnostic_rng(generators):
    state = capture_rng(generators)
    try:
        yield
    finally:
        restore_rng(state, generators)


def save_weights(root, model, config, epoch, batch, global_step, reason):
    """Unique immutable snapshots also preserve abandoned partial-epoch attempts."""
    name = f"step_{global_step:08d}_{reason}_{uuid.uuid4().hex}.pth"
    path = Path(root) / "checkpoints" / name
    state = dict(format=SNAPSHOT_FORMAT, model=model.state_dict(), config=config,
                 epoch=epoch, batch=batch, global_step=global_step)
    # Exclusive creation: snapshots can never replace an existing file.
    with path.open("xb") as file:
        torch.save(state, file)
    return name


def update_best_alias(root, best):
    path = Path(root) / "checkpoints/best.pth"
    if best.get("snapshot"):
        source = Path(root) / "checkpoints" / best["snapshot"]
        _atomic_write(path, lambda temp: temp.write_bytes(source.read_bytes()))
    elif path.exists():
        # Only an uncommitted alias is removed; immutable snapshots remain.
        path.unlink()


def run_training(model, train_loader, val_loader, optimizer, scheduler, scaler,
                 device, config, manifest, root, generators, *, resume_from=None,
                 stop_after_epoch=None, visualize=True, log_every=3):
    root = Path(root)
    history, validations, per_image = [], [], []
    global_step, start_epoch, elapsed = 0, 1, 0.
    best = dict(psnr=-float("inf"), ssim=None, epoch=0, global_step=0, snapshot=None)
    if resume_from:
        cp = load_checkpoint(resume_from, config, manifest)
        model.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        scaler.load_state_dict(cp["scaler"])
        history, validations, per_image = cp["history"], cp["validation_history"], cp["per_image"]
        best, global_step, start_epoch = cp["best"], cp["global_step"], cp["completed_epoch"] + 1
        elapsed = history[-1]["elapsed_seconds"]
        restore_rng(cp["rng"], generators)
        update_best_alias(root, best)
    elif (root / "checkpoints/last.pth").exists():
        raise FileExistsError("Training already started; resume last.pth")
    final_epoch = config["num_epochs"] if stop_after_epoch is None else stop_after_epoch
    if not 1 <= final_epoch <= config["num_epochs"]:
        raise ValueError("Invalid final epoch")
    indices = np.linspace(0, len(val_loader.dataset)-1,
                          min(config["fixed_vis_count"], len(val_loader.dataset)), dtype=int)

    def logs():
        # The epoch checkpoint is authoritative after an interrupted write/replay.
        write_csv(root / "logs/train_log.csv", METRIC_FIELDS, history)
        write_history(root / "logs/training_metrics.csv", history, log_every, config["num_epochs"])
        write_csv(root / "logs/validation_log.csv", VAL_FIELDS, validations)
        write_csv(root / "validation/metrics.csv",
                  ("event", "epoch", "batch", "global_step", "kind", "image", "psnr", "ssim", "protocol"), per_image)

    def checkpoint(epoch):
        state = dict(format=FORMAT, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     completed_epoch=epoch, global_step=global_step, config=config,
                     manifest=manifest, history=history, validation_history=validations,
                     per_image=per_image, best=best, rng=capture_rng(generators))
        _atomic_write(root / "checkpoints/last.pth", lambda temp: torch.save(state, temp))

    def validation(epoch, batch, kind, snapshot=None):
        nonlocal best
        with diagnostic_rng(generators):
            rows, means = evaluate_protocol(model, val_loader, device)
        improved = kind != "initial" and means["psnr"] > best["psnr"]
        event = len(validations)
        common = dict(event=event, epoch=epoch, batch=batch, global_step=global_step, kind=kind)
        validations.append(dict(**common, **means, protocol=VALIDATION_PROTOCOL, is_best=improved))
        per_image.extend(dict(**common, **row, protocol=VALIDATION_PROTOCOL) for row in rows)
        if improved:
            snapshot = snapshot or save_weights(root, model, config, epoch, batch, global_step, "best")
            best = dict(psnr=means["psnr"], ssim=means["ssim"], epoch=epoch,
                        global_step=global_step, snapshot=snapshot)
            update_best_alias(root, best)
        logs()
        print(f"Validation {kind} | epoch {epoch} batch {batch} step {global_step} | "
              f"{means['count']} pairs | PSNR {means['psnr']:.4f} | SSIM {means['ssim']:.4f}"
              + (" | BEST" if improved else ""), flush=True)
        return means

    def visuals(epoch):
        if visualize:
            with diagnostic_rng(generators):
                save_epoch_visuals(model, val_loader.dataset, indices, device,
                                   root / "visuals" / f"epoch_{epoch:04d}")

    if not history:
        started = time.perf_counter()
        means = validation(0, 0, "initial")
        visuals(0)
        elapsed = time.perf_counter() - started
        history.append(dict(epoch=0, global_step=0, train_loss=None, train_psnr=None,
                            train_ssim=None, val_loss=None, val_psnr=means["psnr"],
                            val_ssim=means["ssim"], lr=None, epoch_seconds=elapsed,
                            elapsed_seconds=elapsed, is_best=False))
        checkpoint(0)
    logs()
    for epoch in range(start_epoch, final_epoch + 1):
        started = time.perf_counter()
        lr = scheduler.set_epoch(epoch)
        def after_step(batch):
            nonlocal global_step
            global_step += 1
            snapshot = None
            if global_step % 1000 == 0:
                snapshot = save_weights(root, model, config, epoch, batch, global_step, "periodic")
            if global_step % config["val_freq_steps"] == 0:
                validation(epoch, batch, "periodic", snapshot)
        metrics, steps = train_one_epoch(
            model, train_loader, optimizer, scaler, device, config["use_amp"], True,
            mixup_beta=config["mixup_beta"], grad_clip_norm=config["grad_clip_norm"],
            on_step=after_step, metric_fn=protocol_metrics)
        if epoch == final_epoch and validations[-1]["global_step"] != global_step:
            validation(epoch, steps, "final" if epoch == config["num_epochs"] else "smoke_final")
        if epoch == 1 or epoch % config["vis_every"] == 0 or epoch == final_epoch:
            visuals(epoch)
        seconds = time.perf_counter() - started
        elapsed += seconds
        event = validations[-1] if validations[-1]["global_step"] == global_step else None
        history.append(dict(epoch=epoch, global_step=global_step, **metrics, val_loss=None,
                            val_psnr=event["psnr"] if event else None,
                            val_ssim=event["ssim"] if event else None,
                            lr=lr, epoch_seconds=seconds, elapsed_seconds=elapsed,
                            is_best=bool(event and event["is_best"])))
        checkpoint(epoch)
        logs()
        if should_log_epoch(epoch, log_every, config["num_epochs"]) or epoch == final_epoch:
            print(f"Epoch {epoch:03d}/{config['num_epochs']} | step {global_step} | "
                  f"LR {lr:.8g} | train L1 {metrics['train_loss']:.6f} | {seconds:.1f}s", flush=True)
    return history


def storage_probe(directory):
    """Verify mounted Drive/local checkpoint persistence without touching a run."""
    import tempfile
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".checkpoint_probe_", dir=directory) as temp:
        path = Path(temp) / "probe.pth"
        value = {"probe": torch.arange(16)}
        _atomic_write(path, lambda p: torch.save(value, p))
        reloaded = torch.load(path, weights_only=True)
        if not torch.equal(value["probe"], reloaded["probe"]):
            raise IOError("Checkpoint storage readback failed")
    return str(directory.resolve())
