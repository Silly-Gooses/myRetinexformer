"""Isolated low-LR fine-tuning of the official gated V2 checkpoint.

The original 100 Test pairs remain the validation set, as in the V2 run.
Scores selected on them are not independent held-out test results.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from analysis_utils import extract_checkpoint_state_dict
from diagnostic_analysis import select_checkpoints
from official_workflow import (PROTOCOL, build_loaders, diagnostic_rng, inspect_dataset,
                               storage_probe)
from protocol_metrics import VALIDATION_PROTOCOL, evaluate_protocol, protocol_metrics, write_csv
from train import build_model, resolve_device, validate_config
from training_utils import (_atomic_write, _write_json, capture_rng, restore_rng,
                            seed_everything, train_one_epoch)

FORMAT = "retinexformer-v2.1-finetune-v1"
FIELDS = ("experiment", "source_checkpoint", "source_sha256", "source_epoch",
          "source_batch", "source_global_step", "additional_epoch", "total_epoch",
          "global_step", "train_loss", "validation_psnr", "validation_ssim",
          "improvement_psnr", "learning_rate", "learning_rates", "best_validation_psnr",
          "metric_origin")
VALIDATION_FIELDS = ("additional_epoch", "total_epoch", "global_step", "image",
                     "psnr", "ssim", "protocol")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_info(original_root, device="cpu"):
    paths, final = select_checkpoints(original_root)
    source = paths["best"].resolve()
    state = torch.load(source, map_location="cpu", weights_only=True)
    config = final["config"]
    validate_config(config)
    if config["version"] != "v2" or config["gated"] is not True or config["num_epochs"] != 200:
        raise ValueError("Source must be the gated V2 original 200-epoch run")
    if state.get("config") != config or state.get("global_step") != final["best"]["global_step"]:
        raise ValueError("Best snapshot metadata differs from epoch-200 best record")
    model = build_model(config, device)
    model.load_state_dict(extract_checkpoint_state_dict(state), strict=True)
    epoch = state.get("epoch", state.get("completed_epoch"))
    batch = state.get("batch", 0)
    if epoch != final["best"]["epoch"] or not isinstance(batch, int):
        raise ValueError("Best snapshot epoch/batch is inconsistent")
    history = final["history"]
    if not 1 <= epoch < len(history) or history[epoch].get("lr") is None:
        raise ValueError("Original best epoch learning rate is unavailable")
    info = dict(source_checkpoint=str(source), source_sha256=sha256(source),
                source_format=state["format"], source_epoch=epoch, source_batch=batch,
                source_global_step=state["global_step"],
                source_effective_lr=history[epoch]["lr"],
                baseline_psnr=final["best"]["psnr"], baseline_ssim=final["best"]["ssim"],
                baseline_origin="epoch-200 checkpoint best metadata",
                reference_lr=config["peak_lr"], validation_protocol=VALIDATION_PROTOCOL)
    return info, config, final["manifest"], model


def cosine_lr(epoch, max_epochs, start_lr):
    if not 1 <= epoch <= max_epochs or max_epochs < 2 or start_lr <= 0:
        raise ValueError("Invalid fine-tuning schedule")
    return start_lr * (0.1 + 0.9 * (1 + math.cos(math.pi * (epoch - 1) / (max_epochs - 1))) / 2)


def settings_for(config, experiment, max_epochs=30, validation_interval=5,
                 patience=3, min_improvement=0.01):
    if experiment not in ("finetune_lr5", "finetune_lr10"):
        raise ValueError("Choose finetune_lr5 or finetune_lr10")
    if (max_epochs < 2 or validation_interval < 1 or patience < 0
            or not math.isfinite(min_improvement) or min_improvement < 0):
        raise ValueError("Invalid fine-tuning settings")
    scale = 0.2 if experiment == "finetune_lr5" else 0.1
    return dict(experiment=experiment, max_additional_epochs=max_epochs,
                validation_interval_epochs=validation_interval, patience=patience,
                min_improvement=min_improvement, lr_scale=scale,
                start_lr=config["peak_lr"] * scale)


def _save(root, state, best=False):
    _atomic_write(root / "checkpoints/latest.pth", lambda p: torch.save(state, p))
    if best:
        _atomic_write(root / "checkpoints/best.pth", lambda p: torch.save(state, p))


def _logs(root, history, per_image):
    write_csv(root / "logs/finetune.csv", FIELDS, history)
    write_csv(root / "logs/validation_per_image.csv", VALIDATION_FIELDS, per_image)


def _reconcile_best(root, saved, info):
    path = root / "checkpoints/best.pth"
    best_epoch = saved["best"]["additional_epoch"]
    if best_epoch == 0:
        if not path.is_file() or sha256(path) != info["source_sha256"]:
            _atomic_write(path, lambda p: p.write_bytes(Path(info["source_checkpoint"]).read_bytes()))
        return
    if path.is_file():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if (state.get("format") == FORMAT and state.get("source") == info
                and state.get("additional_epoch") == best_epoch):
            return
    if saved["additional_epoch"] != best_epoch:
        raise ValueError("Best alias is missing or inconsistent with latest checkpoint")
    _atomic_write(path, lambda p: torch.save(saved, p))


def _plots(root, history, info):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    for name, field, baseline in (("training_loss", "train_loss", None),
                                  ("validation_psnr", "validation_psnr", info["baseline_psnr"]),
                                  ("validation_ssim", "validation_ssim", info["baseline_ssim"]),
                                  ("learning_rate", "learning_rate", None)):
        points = [(row["additional_epoch"], row[field]) for row in history
                  if row[field] is not None and (field != "validation_psnr" and field != "validation_ssim"
                                                 or row["additional_epoch"] > 0)]
        if not points:
            continue
        figure = Figure(figsize=(7, 4))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(*zip(*points), marker="o", label="fine-tuning")
        if baseline is not None:
            axis.axhline(baseline, color="black", linestyle="--", label="original best")
            axis.legend()
        axis.set(xlabel="Additional epoch", ylabel=field.replace("_", " ").title())
        axis.grid(alpha=.25)
        figure.tight_layout()
        figure.savefig(root / "visuals" / f"{name}.png", dpi=150)


def run(original_root, output_root, experiment, device, *, resume=False,
        max_epochs=30, validation_interval=5, patience=3, min_improvement=0.01,
        smoke=False):
    info, config, source_manifest, model = source_info(original_root, device)
    settings = settings_for(config, experiment, max_epochs, validation_interval,
                            patience, min_improvement)
    root = Path(output_root) / "v2.1" / experiment
    if smoke:
        root = Path(output_root) / "v2.1" / "smoke" / experiment
        if resume:
            raise ValueError("Smoke test cannot resume a production run")
    if root.resolve() == Path(original_root).resolve() or Path(original_root).resolve() in root.resolve().parents:
        raise ValueError("Fine-tuning output must be outside the original run")
    if not resume and root.exists():
        raise FileExistsError(f"Run already exists: {root}; use --resume or a new output root")
    if resume and not (root / "checkpoints/latest.pth").is_file():
        raise FileNotFoundError(root / "checkpoints/latest.pth")
    manifest, pairs = inspect_dataset(config["dataset_root"])
    if manifest != source_manifest:
        raise ValueError("Dataset membership/content differs from the original checkpoint")
    seed_everything(config["seed"])
    train, val, generators = build_loaders(pairs, config, device)
    optimizer_settings = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["start_lr"],
                                 betas=tuple(optimizer_settings["betas"]),
                                 weight_decay=optimizer_settings["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
    history, per_image = [], []
    best = dict(psnr=info["baseline_psnr"], ssim=info["baseline_ssim"],
                additional_epoch=0, origin="original best checkpoint")
    start_epoch, global_step, stale, stopping_best = 1, info["source_global_step"], 0, best["psnr"]
    if resume:
        saved = torch.load(root / "checkpoints/latest.pth", map_location="cpu", weights_only=True)
        if (saved.get("format") != FORMAT or saved.get("settings") != settings
                or saved.get("source") != info or saved.get("manifest") != manifest):
            raise ValueError("Resume checkpoint differs from source, settings, or dataset")
        progress = saved.get("scheduler", {})
        if (progress.get("schedule") != "cosine-0.1"
                or progress.get("epoch") != saved.get("additional_epoch")
                or not 0 <= progress["epoch"] <= settings["max_additional_epochs"]):
            raise ValueError("Resume scheduler state is inconsistent")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        expected_lr = cosine_lr(progress["epoch"], settings["max_additional_epochs"],
                                settings["start_lr"]) if progress["epoch"] else settings["start_lr"]
        if any(not math.isclose(group["lr"], expected_lr, rel_tol=1e-9)
               for group in optimizer.param_groups):
            raise ValueError("Resume optimizer LR differs from saved scheduler")
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"], generators)
        history, per_image, best = saved["history"], saved["per_image"], saved["best"]
        global_step, stale, stopping_best = saved["global_step"], saved["stale"], saved["stopping_best"]
        start_epoch = saved["additional_epoch"] + 1
        _reconcile_best(root, saved, info)
        _logs(root, history, per_image)
        if saved.get("stopped_early"):
            print("Run already stopped early; no further epochs", flush=True)
            start_epoch = settings["max_additional_epochs"] + 1
    else:
        root.mkdir(parents=True)
        for name in ("checkpoints", "logs", "visuals"):
            (root / name).mkdir()
        _write_json(root / "config.json", settings)
        _write_json(root / "source.json", info)
        _write_json(root / "dataset_manifest.json", manifest)
        # Keep a local fallback alias without ever rewriting the original bytes.
        _atomic_write(root / "checkpoints/best.pth",
                      lambda p: p.write_bytes(Path(info["source_checkpoint"]).read_bytes()))
        history.append(dict(experiment=experiment, source_checkpoint=info["source_checkpoint"],
                            source_sha256=info["source_sha256"], source_epoch=info["source_epoch"],
                            source_batch=info["source_batch"], source_global_step=info["source_global_step"],
                            additional_epoch=0, total_epoch=info["source_epoch"],
                            global_step=global_step, train_loss=None,
                            validation_psnr=best["psnr"], validation_ssim=best["ssim"],
                            improvement_psnr=0, learning_rate=None, learning_rates=None,
                            best_validation_psnr=best["psnr"], metric_origin=info["baseline_origin"]))
        _logs(root, history, per_image)
        _save(root, dict(format=FORMAT, settings=settings, source=info, manifest=manifest,
                         model=model.state_dict(), optimizer=optimizer.state_dict(),
                         scaler=scaler.state_dict(), scheduler=dict(epoch=0, schedule="cosine-0.1"),
                         rng=capture_rng(generators), additional_epoch=0,
                         global_step=global_step, history=history, per_image=per_image,
                         best=best, stale=stale, stopping_best=stopping_best,
                         stopped_early=False))
    print(f"{experiment}: source {info['source_checkpoint']} sha256={info['source_sha256']}", flush=True)
    print(f"Source epoch {info['source_epoch']} batch {info['source_batch']} step {info['source_global_step']} | "
          f"recorded validation {info['baseline_psnr']:.4f} dB / {info['baseline_ssim']:.4f} SSIM | "
          f"effective source LR {info['source_effective_lr']:.8g}", flush=True)
    for epoch in range(start_epoch, settings["max_additional_epochs"] + 1):
        lr = cosine_lr(epoch, settings["max_additional_epochs"], settings["start_lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        actual_lrs = [group["lr"] for group in optimizer.param_groups]
        if not all(math.isclose(actual, lr, rel_tol=1e-12) for actual in actual_lrs):
            raise AssertionError("Optimizer LR differs from fine-tuning schedule")
        print(f"Additional epoch {epoch}: optimizer LRs {actual_lrs}", flush=True)
        if smoke:
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        def after_step(_):
            nonlocal global_step
            global_step += 1
        metrics, steps = train_one_epoch(model, train, optimizer, scaler, device,
                                         config["use_amp"], True,
                                         mixup_beta=config["mixup_beta"],
                                         grad_clip_norm=config["grad_clip_norm"],
                                         on_step=after_step, metric_fn=protocol_metrics,
                                         max_steps=2 if smoke else None)
        if smoke and (steps != 2 or not any(not torch.equal(before[k], v) for k, v in model.state_dict().items())):
            raise AssertionError("Smoke did not perform two weight-changing optimizer updates")
        validation = epoch % settings["validation_interval_epochs"] == 0 or epoch == settings["max_additional_epochs"] or smoke
        mean = None
        improved = False
        if validation:
            with diagnostic_rng(generators):
                rows, mean = evaluate_protocol(model, val, device)
            per_image.extend(dict(additional_epoch=epoch, total_epoch=info["source_epoch"] + epoch,
                                  global_step=global_step, **row, protocol=VALIDATION_PROTOCOL)
                             for row in rows)
            improved = mean["psnr"] > best["psnr"]
            if improved:
                best = dict(psnr=mean["psnr"], ssim=mean["ssim"],
                            additional_epoch=epoch, origin="measured fine-tuning validation")
            if mean["psnr"] >= stopping_best + settings["min_improvement"]:
                stopping_best, stale = mean["psnr"], 0
            else:
                stale += 1
        history.append(dict(experiment=experiment, source_checkpoint=info["source_checkpoint"],
                            source_sha256=info["source_sha256"], source_epoch=info["source_epoch"],
                            source_batch=info["source_batch"], source_global_step=info["source_global_step"],
                            additional_epoch=epoch, total_epoch=info["source_epoch"] + epoch,
                            global_step=global_step, train_loss=metrics["train_loss"],
                            validation_psnr=mean["psnr"] if mean else None,
                            validation_ssim=mean["ssim"] if mean else None,
                            improvement_psnr=mean["psnr"] - info["baseline_psnr"] if mean else None,
                            learning_rate=actual_lrs[0], learning_rates=json.dumps(actual_lrs),
                            best_validation_psnr=best["psnr"],
                            metric_origin="measured fine-tuning validation" if mean else ""))
        stopped_early = validation and settings["patience"] > 0 and stale >= settings["patience"] and not smoke
        state = dict(format=FORMAT, settings=settings, source=info, manifest=manifest,
                     model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scaler=scaler.state_dict(), scheduler=dict(epoch=epoch, schedule="cosine-0.1"),
                     rng=capture_rng(generators), additional_epoch=epoch,
                     global_step=global_step, history=history, per_image=per_image,
                     best=best, stale=stale, stopping_best=stopping_best,
                     stopped_early=stopped_early)
        _save(root, state, best=improved)
        _logs(root, history, per_image)
        if mean:
            print(f"Validation: PSNR {mean['psnr']:.4f}, SSIM {mean['ssim']:.4f}, "
                  f"delta {mean['psnr'] - info['baseline_psnr']:+.4f} dB", flush=True)
        if smoke or stopped_early:
            break
    if sha256(info["source_checkpoint"]) != info["source_sha256"]:
        raise RuntimeError("Original checkpoint changed during fine-tuning")
    if smoke:
        reloaded = torch.load(root / "checkpoints/latest.pth", map_location="cpu", weights_only=True)
        clone = build_model(config, device)
        clone.load_state_dict(reloaded["model"], strict=True)
        if not (root / "logs/finetune.csv").is_file() or not (root / "logs/validation_per_image.csv").is_file():
            raise AssertionError("Smoke logs missing")
    _write_json(root / "selection.json", dict(best=best, checkpoint=str(root / "checkpoints/best.pth"),
                                               source_sha256=info["source_sha256"],
                                               validation_protocol=VALIDATION_PROTOCOL,
                                               held_out=False))
    _plots(root, history, info)
    return root


def compare(output_root, original_root):
    info, _, _, _ = source_info(original_root)
    base = Path(output_root) / "v2.1"
    rows = [dict(model="Proposed V2 original best", additional_epochs=0,
                 best_psnr=info["baseline_psnr"], best_ssim=info["baseline_ssim"],
                 improvement_psnr=0, ssim_delta=0, best_learning_rate=info["source_effective_lr"],
                 train_loss_first=None, train_loss_last=None, early_stopped=None,
                 origin=info["baseline_origin"],
                 checkpoint=info["source_checkpoint"])]
    for experiment in ("finetune_lr5", "finetune_lr10"):
        path = base / experiment / "selection.json"
        if not path.is_file():
            continue
        selection = json.loads(path.read_text())
        if selection["source_sha256"] != info["source_sha256"]:
            raise ValueError(f"{experiment} used a different original checkpoint")
        best = selection["best"]
        with (base / experiment / "logs/finetune.csv").open(newline="") as file:
            logged = list(csv.DictReader(file))
        trained = [row for row in logged if row["train_loss"]]
        best_rows = [row for row in trained if int(row["additional_epoch"]) == best["additional_epoch"]]
        latest = torch.load(base / experiment / "checkpoints/latest.pth",
                            map_location="cpu", weights_only=True)
        rows.append(dict(model=experiment, additional_epochs=best["additional_epoch"],
                         best_psnr=best["psnr"], best_ssim=best["ssim"],
                         improvement_psnr=best["psnr"] - info["baseline_psnr"],
                         ssim_delta=best["ssim"] - info["baseline_ssim"],
                         best_learning_rate=float(best_rows[0]["learning_rate"]) if best_rows else None,
                         train_loss_first=float(trained[0]["train_loss"]) if trained else None,
                         train_loss_last=float(trained[-1]["train_loss"]) if trained else None,
                         early_stopped=latest["stopped_early"],
                         origin=best["origin"], checkpoint=selection["checkpoint"]))
    if len(rows) < 3:
        print("Comparison incomplete: run both experiments before final selection", flush=True)
    path = base / "comparison"
    path.mkdir(exist_ok=True)
    write_csv(path / "comparison.csv", tuple(rows[0]), rows)
    selected = max(rows, key=lambda row: row["best_psnr"])
    _write_json(path / "comparison.json", dict(rows=rows, selected=selected,
        psnr_improved=selected["best_psnr"] > info["baseline_psnr"],
        ssim_improved=selected["best_ssim"] > info["baseline_ssim"], held_out=False,
        note="Original Test pairs were used for validation and checkpoint selection."))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(8, 4))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.bar([row["model"] for row in rows], [row["best_psnr"] for row in rows])
    axis.set(ylabel="Best validation PSNR (dB)", title="Original best and V2.1 fine-tuning")
    axis.tick_params(axis="x", labelrotation=15)
    figure.tight_layout()
    figure.savefig(path / "comparison.png", dpi=150)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment", choices=("finetune_lr5", "finetune_lr10"))
    parser.add_argument("--max-additional-epochs", type=int, default=30)
    parser.add_argument("--validation-interval-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args(argv)
    device = resolve_device(args.device, False)
    if args.preflight:
        info, config, _, _ = source_info(args.original_root, device)
        print(json.dumps(info, indent=2))
        print("Drive storage probe:", storage_probe(args.output_root))
        return
    if args.compare:
        print(json.dumps(compare(args.output_root, args.original_root), indent=2))
        return
    if not args.experiment:
        parser.error("--experiment is required unless --preflight or --compare is used")
    result = run(args.original_root, args.output_root, args.experiment, device,
                 resume=args.resume, max_epochs=args.max_additional_epochs,
                 validation_interval=args.validation_interval_epochs,
                 patience=args.patience, min_improvement=args.min_improvement,
                 smoke=args.smoke)
    print("Output:", result)


if __name__ == "__main__":
    main()
