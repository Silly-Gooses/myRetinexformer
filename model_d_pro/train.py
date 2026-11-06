"""Executed Phase 1 recipe, extracted without changing numerical operations."""
import copy
import json
import math
import os
import platform
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .model import ModelDPro
from .losses import training_loss
from .dataset import PairedStore
from .evaluate import evaluate, smoke_check, make_comparisons
from .runtime import (Budget, BudgetStop, sha_file, write_json, append_csv, export_small_report)
from .checkpoint import (atomic_torch_save, make_identity, prepare_run, restore_training_state,
                         EXPECTED_B, EXPECTED_HEAD, EXPECTED_VAL_DATA, rng_metadata)

VERSION = "dpro_modular_phase1_v1"


def validate_config(cfg, session_minutes):
    if not 5 <= session_minutes <= 120:
        raise ValueError("Session budget must be between 5 and 120 minutes")
    name = cfg["run_name"]
    if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
        raise ValueError("run_name must be a single directory name")
    for key in ("patch", "micro_batch", "accumulation", "max_steps", "validate_every",
                "save_every", "patience_evals"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["warmup_steps"] < 0 or cfg["max_steps"] < cfg["warmup_steps"] or cfg["patch"] % 4:
        raise ValueError("Invalid warmup or patch configuration")
    if cfg["micro_batch"] * cfg["accumulation"] != 8:
        raise ValueError("Phase 1 effective batch must remain 8")
    if any(not math.isfinite(v) or v <= 0 for v in cfg["learning_rates"].values()):
        raise ValueError("Learning rates must be finite and positive")


def protect_output(run, inputs):
    run = Path(run).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if run == source or run.is_relative_to(source) or source.is_relative_to(run):
            raise ValueError(f"Output overlaps an input: {source}")


def pending_validation(state, cfg):
    done = state["completed_steps"]
    return (not state["finished"] and done > state.get("last_validated_step", 0)
            and (done % cfg["validate_every"] == 0 or done >= cfg["max_steps"]))


def set_learning_rates(optimizer, cfg, step):
    """Original schedule, indexed by successful optimizer updates."""
    progress = max(0.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
    factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
    for group in optimizer.param_groups:
        lr = cfg["learning_rates"][group["group_name"]] * factor
        if step < cfg["warmup_steps"] and group["group_name"] != "new":
            lr = 0.0
        elif group["group_name"] == "new":
            lr *= min(1.0, (step + 1) / 25)
        group["lr"] = lr


def validate_checkpoint(model, store, device, run, budget, state, baseline, cfg, identity):
    """Publish only a completed full validation; an interrupted pass changes no progress."""
    done = state["completed_steps"]
    prefix = f"val_{done:05d}"
    measured = evaluate(model, store, device, run / prefix, budget, f"D-Pro update {done}")
    state["last_validated_step"] = done
    state["full_validation_count"] += 1
    append_csv(run / "validation.csv", dict(step=done, psnr=measured["psnr"], ssim=measured["ssim"],
                                             delta_vs_model_d=measured["psnr"] - baseline["psnr"]))
    if measured["psnr"] > state["best"]["psnr"]:
        state.update(best=measured, best_step=done, best_prefix=prefix)
        atomic_torch_save(dict(model=model.state_dict(), identity=identity, metrics=measured, step=done),
                          run / "best.pth")
    if measured["psnr"] >= state["patience_reference"] + cfg["patience_min_delta"]:
        state["patience_reference"] = measured["psnr"]
        state["patience"] = 0
    else:
        state["patience"] += 1


def run_pilot(cfg, repo, session_minutes):
    import subprocess
    from model_d import load_model_d
    from model_d.model import REFERENCE
    budget = Budget(session_minutes)
    device = torch.device("cuda")
    run = Path(cfg["output_root"]) / cfg["run_name"]
    validate_config(cfg, session_minutes)
    protect_output(run, [cfg["data_root"], cfg["model_b"], cfg["head"]])
    state = dict(completed_steps=0, last_validated_step=0, best=None, best_step=0, best_prefix="step0",
                 patience=0, patience_reference=None, stop_reason="starting", finished=False,
                 total_train_seconds=0.0, full_validation_count=0)
    model = optimizer = scaler = store = None
    baseline = step0 = None
    identity = None
    last = run / "last.pth"
    run_owned = False
    start = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required. Select T4; CPU training is disabled.")
        if cfg["require_t4"] and "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This cost-controlled preset requires T4. Change the Colab GPU to T4.")
        if cfg["micro_batch"] * cfg["accumulation"] != 8:
            raise ValueError("This preset uses effective batch 8; change settings deliberately in a new run.")
        if cfg["max_steps"] < cfg["warmup_steps"] or cfg["patch"] % 4:
            raise ValueError("Invalid warmup or patch configuration")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        backbone_path, head_path = Path(cfg["model_b"]), Path(cfg["head"])
        for path, expected in [(backbone_path, EXPECTED_B), (head_path, EXPECTED_HEAD)]:
            if not path.is_file() or sha_file(path) != expected:
                raise ValueError(f"Missing/wrong ORIGINAL checkpoint (hash check failed): {path}")
        frozen = {"train": REFERENCE["config"]["train_names"], "val": REFERENCE["config"]["val_names"]}
        split = Path(cfg["split_file"])
        if split.is_file() and json.loads(split.read_text()) != frozen:
            raise ValueError("split.json differs from the preserved 620/69 split")
        if not split.is_file():
            print("External split.json absent; using the identical split embedded in the pinned repository.", flush=True)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        import cv2
        cv2.setNumThreads(1)
        store = PairedStore(cfg["data_root"], frozen["train"], frozen["val"], cfg["patch"], budget)
        if store.val_sha != EXPECTED_VAL_DATA:
            raise ValueError("Validation file CONTENTS differ from the previously reproduced 69 images")
        identity = make_identity(cfg, store, repo)
        prepare_run(run, identity, cfg["resume"])
        identity_path = run / "identity.json"
        run_owned = True
        write_json(identity_path, identity)
        write_json(run / "config.json", cfg)
        write_json(run / "split.json", frozen)
        write_json(run / "source_manifest.json", identity["sources"])
        runtime = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), commit=commit,
                       session_id=os.environ.get("DPRO_SESSION_ID"),
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        write_json(run / "runtime.json", runtime)
        original, checkpoint_info = load_model_d(backbone_path, head_path, device=device)
        if (run / "baseline.json").is_file():
            baseline = json.loads((run / "baseline.json").read_text())
            if baseline.get("dataset_sha256") != store.val_sha or baseline.get("count") != 69:
                raise ValueError("Stored baseline provenance mismatch")
            print(f"Reusing this run's checked full validation baseline: {baseline['psnr']:.6f} dB", flush=True)
        else:
            baseline = evaluate(original, store, device, run / "baseline", budget, "Original Model D")
        expected = REFERENCE["metrics"]["val"]["model_d"]
        if abs(baseline["psnr"] - expected["psnr"]) > 0.002 or abs(baseline["ssim"] - expected["ssim"]) > 0.0001:
            raise RuntimeError("Baseline reproduction FAILED. No training was started; inspect baseline.json.")
        model = ModelDPro(original).to(device)
        groups = model.parameter_groups(cfg)
        write_json(run / "parameters.json", {g["group_name"]: sum(p.numel() for p in g["params"]) for g in groups})
        if not last.is_file():
            budget.check()
            smoke = smoke_check(model, original, store, cfg, device)
            write_json(run / "gradient_smoke.json", smoke)
            print("Gradient smoke test passed:", json.dumps(smoke), flush=True)
        del original
        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0, foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        if last.is_file():
            payload = torch.load(last, map_location="cpu", weights_only=True)
            state = restore_training_state(payload, identity, model, optimizer, scaler, device)
            step0 = json.loads((run / "step0.json").read_text())
            print(f"Resumed at successful optimizer update {state['completed_steps']}.", flush=True)
            del payload
        else:
            step0 = evaluate(model, store, device, run / "step0", budget, "D-Pro FLOAT step 0")
            if abs(step0["psnr"] - baseline["psnr"]) > cfg["step0_max_abs_psnr_delta"]:
                raise RuntimeError("Unexpectedly large FLOAT step-0 PSNR shift. Inspect step0.json; training blocked.")
            state.update(best=step0, best_step=0, patience_reference=step0["psnr"],
                         full_validation_count=1)
            atomic_torch_save(dict(model=model.state_dict(), identity=identity, metrics=step0, step=0), run / "best.pth")
        def save_last():
            atomic_torch_save(dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scaler=scaler.state_dict(), state=copy.deepcopy(state),
                                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), **rng_metadata()), last)
        save_last()
        # A time cap can interrupt validation after an optimizer update. On
        # resume, complete that pending full-set evaluation before more updates.
        done = state["completed_steps"]
        if pending_validation(state, cfg):
            validate_checkpoint(model, store, device, run, budget, state, baseline, cfg, identity)
            if done >= cfg["min_early_stop_step"] and state["patience"] >= cfg["patience_evals"]:
                state.update(stop_reason="pilot_plateau", finished=True)
            save_last()
        if state["finished"]:
            print("This run already stopped/completed. No additional GPU training is launched.", flush=True)
        else:
            state["stop_reason"] = "running"
            state.pop("error", None)
            consecutive_bad = 0
            while state["completed_steps"] < cfg["max_steps"]:
                budget.check()
                step = state["completed_steps"]
                phase = model.set_phase(step, cfg["warmup_steps"])
                model.train()
                # Schedule is tied to successful updates, so resuming does not reset it.
                set_learning_rates(optimizer, cfg, step)
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                totals = {k: 0.0 for k in ["loss", "rgb", "intensity", "hv", "backbone", "residual"]}
                for micro in range(cfg["accumulation"]):
                    budget.check()
                    low, target = store.batch(step, micro, cfg, device)
                    values = model(low, return_aux=True)
                    loss, parts = training_loss(values, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss; no optimizer update applied")
                    scaler.scale(loss / cfg["accumulation"]).backward()
                    totals["loss"] += float(loss.detach()) / cfg["accumulation"]
                    for name, value in parts.items():
                        totals[name] += float(value.detach()) / cfg["accumulation"]
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    # GradScaler skips corrupt updates; retries use a lower scale.
                    scaler.update(new_scale=max(1.0, scaler.get_scale() / 2))
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_bad += 1
                    print(f"AMP nonfinite gradients; skipped update; scale={scaler.get_scale()}", flush=True)
                    if consecutive_bad >= 3:
                        raise FloatingPointError("Three consecutive nonfinite gradient attempts")
                    continue
                scaler.step(optimizer)
                scaler.update()
                consecutive_bad = 0
                state["completed_steps"] += 1
                seconds = time.monotonic() - tick
                state["total_train_seconds"] += seconds
                done = state["completed_steps"]
                row = dict(step=done, phase=phase, **totals, grad_norm=float(grad_norm),
                           lr_backbone=optimizer.param_groups[0]["lr"], lr_head=optimizer.param_groups[1]["lr"],
                           lr_new=optimizer.param_groups[2]["lr"], seconds=seconds)
                append_csv(run / "steps.csv", row)
                if done % 25 == 0 or done == 1:
                    eta = (cfg["max_steps"] - done) * (state["total_train_seconds"] / done)
                    print(f"step {done}/{cfg['max_steps']} | {phase} | loss {totals['loss']:.5f} | "
                          f"train-only ETA {eta/60:.1f} min | session budget left {budget.remaining()/60:.1f} min", flush=True)
                if done % cfg["save_every"] == 0:
                    save_last()
                if done % cfg["validate_every"] == 0 or done == cfg["max_steps"]:
                    validate_checkpoint(model, store, device, run, budget, state, baseline, cfg, identity)
                    save_last()
                    if done >= cfg["min_early_stop_step"] and state["patience"] >= cfg["patience_evals"]:
                        state.update(stop_reason="pilot_plateau", finished=True)
                        break
            if state["completed_steps"] >= cfg["max_steps"]:
                state.update(stop_reason="max_steps", finished=True)
            save_last()
    except BudgetStop as error:
        state.update(stop_reason=str(error), finished=False)
        print("Budget pause. Resume will continue, not restart.", flush=True)
    except KeyboardInterrupt:
        state.update(stop_reason="interrupted", finished=False)
    except Exception as error:
        state.update(stop_reason="error", error=f"{type(error).__name__}: {error}", finished=False)
        trace = traceback.format_exc()
        if run_owned:
            (run / "error.txt").write_text(trace)
        print(trace, flush=True)
    finally:
        if not run_owned:
            print("Existing run identity was not accepted; its checkpoints and summary were not overwritten.", flush=True)
        else:
            # Last known full validation chooses best; interrupted partial validations
            # never replace it. Preserve optimizer/sampler progress without saving grads.
            try:
                if model is not None and optimizer is not None and scaler is not None and state["best"] is not None:
                    atomic_torch_save(dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                           scaler=scaler.state_dict(), state=state, torch_rng=torch.get_rng_state(),
                                           cuda_rng=torch.cuda.get_rng_state_all(), **rng_metadata()), last)
                delta = state["best"]["psnr"] - baseline["psnr"] if state["best"] is not None and baseline else None
                ssim_delta = state["best"]["ssim"] - baseline["ssim"] if delta is not None else None
                learned_delta = state["best"]["psnr"] - step0["psnr"] if state["best"] is not None and step0 else None
                if delta is None:
                    decision = "NO_COMPLETE_CANDIDATE_VALIDATION"
                elif delta >= 0.05 and ssim_delta >= -0.002 and state["best_step"] > 0 and learned_delta is not None and learned_delta >= 0.02:
                    decision = "PROMISING_PILOT_REVIEW_BEFORE_EXTENDING"
                elif delta > 0.002:
                    decision = "SMALL_GAIN_OR_SSIM_TRADEOFF_REVIEW"
                else:
                    decision = "NO_CLEAR_GAIN_IN_THIS_BUDGET_NOT_A_FINAL_ARCHITECTURE_VERDICT"
                summary = dict(version=VERSION, inference_protocol="DPro_float_interstage_RGB_uint8_metrics",
                               baseline_protocol="ModelD_PNG_interstage_RGB_uint8_metrics",
                               stop_reason=state["stop_reason"], error=state.get("error"),
                               successful_updates=state["completed_steps"], target_updates=cfg["max_steps"],
                               completed=state["finished"], baseline=baseline, candidate_float_step0=step0,
                               best_validation=state["best"], best_step=state["best_step"],
                               delta_psnr_vs_model_d=delta, delta_ssim_vs_model_d=ssim_delta,
                               delta_psnr_vs_candidate_step0=learned_delta,
                               decision=decision, test_evaluated=False, full_training=False,
                               thresholds_are_heuristic_not_statistical_significance=True,
                               final_test_target_db=23.1, final_test_target_achieved=None,
                               session_elapsed_seconds=time.monotonic()-start,
                               total_train_seconds=state["total_train_seconds"], output=str(run))
                write_json(run / "pilot_summary.json", summary)
                if store is not None and state["best"] is not None:
                    make_comparisons(store, run, state["best_prefix"])
                artifact = export_small_report(run)
                print("\nRESULT:", json.dumps(summary, indent=2), flush=True)
                print("Report ZIP:", artifact, flush=True)
                print("Checkpoint files stay on Drive; they are NOT included in the small report ZIP.", flush=True)
            except Exception:
                print("Final export/save error:\n" + traceback.format_exc(), flush=True)
    return 1 if not run_owned or state["stop_reason"] == "error" else 0
