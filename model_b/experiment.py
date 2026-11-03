"""Exact Option A training sequence, with additional progress reporting."""

import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import TrainPairs, evaluate_model
from .model import ARCH, build_model


BATCH_SIZE = 8
PATCH_SIZE = 128
ADDITIONAL_STEPS = 1000
LEARNING_RATE = 1e-5
VAL_EVERY = 100
LOG_EVERY = 25
SAVE_EVERY = 100
SEED = 42
USE_AMP = True
EXPECTED_SOURCE_LOCAL_STEP = 1000
EXPECTED_SOURCE_PSNR = 20.9486
EXPECTED_SOURCE_SHA256 = "a835284a8f72d16d131ddac6b05224c7afc2e4024c636beab9405c04e4403141"
REFERENCE_BASE_COMMIT = "ad33e6bf5b338bc36e4a14102781e78a03397b3b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_checkpoint(path, train_names, val_names, *, kind="source"):
    """Load only a user-trusted reference checkpoint and verify its ancestry."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not all(
            key in checkpoint for key in ("model", "local_step", "config", "best_psnr")):
        raise ValueError("Unexpected Model B checkpoint schema")
    config = checkpoint["config"]
    if config["arch"] != ARCH or config["train_names"] != train_names or config["val_names"] != val_names:
        raise ValueError("Reference checkpoint architecture or frozen split mismatch")
    if kind == "source":
        if config.get("repo_commit") != REFERENCE_BASE_COMMIT:
            raise ValueError("Source checkpoint was not created from the pinned v1 commit")
        if (int(checkpoint["local_step"]) != EXPECTED_SOURCE_LOCAL_STEP
                or abs(float(checkpoint["best_psnr"]) - EXPECTED_SOURCE_PSNR) >= .005
                or config["batch"] != 8 or config["patch"] != 128
                or float(config["lr"]) != 2e-5 or int(config["source_step"]) != 1500):
            raise ValueError("Not the expected Batch-8 Patch-128 source best checkpoint")
        digest = sha256(path)
        if digest != EXPECTED_SOURCE_SHA256:
            raise ValueError(f"Source checkpoint SHA256 mismatch: {digest}")
    elif kind == "option_a":
        if (not 0 <= int(checkpoint["local_step"]) <= ADDITIONAL_STEPS
                or config["batch"] != BATCH_SIZE or config["patch"] != PATCH_SIZE
                or float(config["lr"]) != LEARNING_RATE
                or config["source_sha256"] != EXPECTED_SOURCE_SHA256):
            raise ValueError("Not the expected Option A best checkpoint")
    else:
        raise ValueError(f"Unknown checkpoint kind: {kind}")
    model, replaced = build_model()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint, replaced


def append_csv(path, row):
    with Path(path).open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


def store_ckpt(path, model, optimizer, scaler, local_step, best_psnr, config):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(dict(config=config, local_step=local_step,
                    source_local_step=EXPECTED_SOURCE_LOCAL_STEP,
                    best_psnr=best_psnr, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                    rng_py=random.getstate(), rng_np=np.random.get_state(),
                    rng_torch=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all()), temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def smoke_test(model):
    """Reference batch-size forward/backward check without optimizer update."""
    import gc
    scratch, _ = build_model()
    scratch.load_state_dict(model.state_dict(), strict=True)
    scratch = scratch.cuda().train()
    xb = torch.rand(BATCH_SIZE, 3, PATCH_SIZE, PATCH_SIZE, device="cuda")
    yb = torch.rand_like(xb)
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast("cuda", enabled=USE_AMP):
            pred = scratch(xb)
            loss = F.l1_loss(pred, yb)
        loss.backward()
        torch.cuda.synchronize()
        return dict(shape=list(pred.shape),
                    peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
    except torch.cuda.OutOfMemoryError as error:
        raise RuntimeError("Batch 8, patch 128 does not fit this GPU") from error
    finally:
        del scratch, xb, yb
        gc.collect()
        torch.cuda.empty_cache()


def run_option_a(source_path, run_dir, train_pairs, val_pairs, train_names,
                 val_names, dataset_root, repo_commit, *, resume=True):
    """Run the reference 1,000-step fine-tune; only ETA reporting differs."""
    if not torch.cuda.is_available():
        raise RuntimeError("Option A requires a CUDA GPU")
    source_path, run_dir = Path(source_path), Path(run_dir)
    source_model, source_ckpt, _ = load_reference_checkpoint(
        source_path, train_names, val_names, kind="source")
    source_state = source_ckpt["model"]
    source_digest = sha256(source_path)
    if source_path.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Run directory must not contain the source checkpoint")
    run_dir.mkdir(parents=True, exist_ok=True)
    config = dict(source_path=str(source_path), source_sha256=source_digest,
                  source_local_step=EXPECTED_SOURCE_LOCAL_STEP, arch=ARCH,
                  repo_commit=repo_commit, dataset_root=str(dataset_root),
                  train_names=train_names, val_names=val_names,
                  patch=PATCH_SIZE, batch=BATCH_SIZE, steps=ADDITIONAL_STEPS,
                  lr=LEARNING_RATE, seed=SEED, amp=USE_AMP,
                  val_every=VAL_EVERY, log_every=LOG_EVERY, save_every=SAVE_EVERY,
                  optimizer="Adam (fresh for Option-A low-LR fine-tuning)",
                  criterion="L1", model="NAF-inspired Model B")
    manifest = run_dir / "config.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run has a different config")
    else:
        manifest.write_text(json.dumps(config, indent=2), encoding="utf-8")
    seed_all(SEED)
    model, _ = build_model()
    model.load_state_dict(source_state, strict=True)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 betas=(.9, .999), weight_decay=0.)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    last_file, best_file = run_dir / "last.pth", run_dir / "best.pth"
    start, best = 0, -float("inf")
    if last_file.exists():
        if not resume:
            raise RuntimeError(f"Previous run exists: {last_file}")
        ck = torch.load(last_file, map_location="cpu", weights_only=False)
        if ck["config"] != config:
            raise ValueError("Incompatible saved training config")
        start, best = int(ck["local_step"]), float(ck["best_psnr"])
        if start > ADDITIONAL_STEPS:
            raise ValueError("Saved local step exceeds Option A length")
        model.load_state_dict(ck["model"], strict=True)
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if start and not optimizer.state:
            raise ValueError("Optimizer state missing; cannot resume")
        print(f"Option A: resuming local step {start + 1} (sampler cursor NOT saved)")
    elif (run_dir / "steps.csv").exists() or (run_dir / "validation.csv").exists():
        raise RuntimeError("CSV exists without last.pth; refusing to mix runs")
    if start == ADDITIONAL_STEPS:
        print("Already complete:", run_dir)
        return run_dir
    if start == 0 and not last_file.exists():
        result, rows = evaluate_model(model, val_pairs)
        best = result["psnr"]
        append_csv(run_dir / "validation.csv", dict(local_step=0, **result))
        write_rows(run_dir / "source_per_image.csv", rows)
        store_ckpt(best_file, model, optimizer, scaler, 0, best, config)
        store_ckpt(last_file, model, optimizer, scaler, 0, best, config)
        print(f"Option A source: PSNR={best:.4f}, SSIM={result['ssim']:.5f}")
    loader = DataLoader(TrainPairs(train_pairs, PATCH_SIZE), batch_size=BATCH_SIZE,
                        shuffle=True, drop_last=False, num_workers=0, pin_memory=True)
    iterator = iter(loader)
    if start > 0:
        random.setstate(ck["rng_py"])
        np.random.set_state(ck["rng_np"])
        torch.set_rng_state(ck["rng_torch"])
        torch.cuda.set_rng_state_all(ck["rng_cuda"])

    def log(message):
        print(message, flush=True)
        with (run_dir / "training.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    log(f"=== {datetime.now(timezone.utc).isoformat()} option=A patch={PATCH_SIZE} "
        f"start={start} additional_steps={ADDITIONAL_STEPS} lr={LEARNING_RATE} ===")
    wall, ema, val_total, val_count = time.monotonic(), None, 0., 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(start + 1, ADDITIONAL_STEPS + 1):
        tick = time.monotonic()
        try:
            low, high = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            low, high = next(iterator)
        low = low.cuda(non_blocking=True)
        high = high.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=USE_AMP):
            loss = F.l1_loss(model(low), high)
        if not torch.isfinite(loss).item():
            store_ckpt(last_file, model, optimizer, scaler, step - 1, best, config)
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        elapsed = time.monotonic() - tick
        ema = elapsed if ema is None else .9 * ema + .1 * elapsed
        remaining_validations = (ADDITIONAL_STEPS // VAL_EVERY) - (step // VAL_EVERY)
        eta_total = ((ADDITIONAL_STEPS - step) * ema
                     + remaining_validations * (val_total / val_count if val_count else 0)) / 60
        if step == start + 1 or step % LOG_EVERY == 0 or step == ADDITIONAL_STEPS:
            row = dict(local_step=step, loss=float(loss.item()),
                       lr=float(optimizer.param_groups[0]["lr"]),
                       elapsed_s=round(time.monotonic() - wall, 2),
                       eta_optimizer_min=round((ADDITIONAL_STEPS - step) * ema / 60, 2),
                       gpu_peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3))
            append_csv(run_dir / "steps.csv", row)
            log(f"patch={PATCH_SIZE} | {step}/{ADDITIONAL_STEPS} | loss={row['loss']:.5f} "
                f"| total ETA={eta_total:.1f}min | peak={row['gpu_peak_allocated_gb']:.2f}GB")
        if step % VAL_EVERY == 0 or step == ADDITIONAL_STEPS:
            validation_tick = time.monotonic()
            result, rows = evaluate_model(model, val_pairs)
            val_total += time.monotonic() - validation_tick
            val_count += 1
            append_csv(run_dir / "validation.csv", dict(local_step=step, **result))
            log(f"VAL patch={PATCH_SIZE} step={step} PSNR={result['psnr']:.4f} "
                f"SSIM={result['ssim']:.5f}")
            if result["psnr"] > best:
                best = result["psnr"]
                store_ckpt(best_file, model, optimizer, scaler, step, best, config)
                write_rows(run_dir / "best_per_image.csv", rows)
            model.train()
        if step % SAVE_EVERY == 0 or step == ADDITIONAL_STEPS:
            store_ckpt(last_file, model, optimizer, scaler, step, best, config)
    log(f"DONE patch={PATCH_SIZE} best_PSNR={best:.4f} "
        f"elapsed_min={(time.monotonic() - wall) / 60:.2f}")
    return run_dir
