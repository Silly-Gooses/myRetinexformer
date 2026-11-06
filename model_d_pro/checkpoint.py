"""Strict checkpoint loading and migration-aware run identities."""
import os
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

from model_d.model import HVIColorHead, ModelD, build_model_b, load_model_d
from .model import ModelDPro
from .runtime import sha_file

PINNED_COMMIT = "108b14cc0d84c070a8331772b99789359a4f3137"
EXPECTED_B = "1cb495fc19ce1560a1b50bd9eef16e10e3c3184351e9ac6fe03ccc31e1b91ae6"
EXPECTED_HEAD = "cebbc55c59906fc2d0be453f16660b3d0dadc14bae150ef97c8fcce29c331ca0"
EXPECTED_VAL_DATA = "c298929b9c8484c64d79f8aee9e80233ca437bfbe519c44a4a40c7b4bacc928f"
SOURCE_SHA256 = "52b1ca3bfd059fe2584875d79f108a835ebd05d2ea22e40afd487a2b5debb4da"
SCHEMA = "dpro_modular_phase1_v1"
ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_KEYS = (
    "seed", "patch", "micro_batch", "accumulation", "max_steps", "warmup_steps",
    "validate_every", "save_every", "learning_rates", "patience_evals",
    "min_early_stop_step", "patience_min_delta", "step0_max_abs_psnr_delta",
)
PROTECTED_FILES = (
    "model_d/__init__.py", "model_d/model.py", "model_d/reference.json",
    "RetinexFormer_arch.py", "canny.py", "analysis_utils.py", "utils.py",
)


def empty_model():
    """Construct the exact structure; callers must load trained weights strictly."""
    return ModelDPro(ModelD(build_model_b(), HVIColorHead()))


def check_pilot_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("Missing D-Pro checkpoint identity")
    for key, expected in (("backbone_sha256", EXPECTED_B), ("head_sha256", EXPECTED_HEAD),
                          ("val_sha256", EXPECTED_VAL_DATA)):
        if identity.get(key) != expected:
            raise ValueError(f"Checkpoint provenance mismatch: {key}")
    if identity.get("version") == "dpro_budget_v1.0":
        if identity.get("script_sha256") != SOURCE_SHA256 or identity.get("commit") != PINNED_COMMIT:
            raise ValueError("Historical pilot source identity mismatch")
    elif identity.get("version") != SCHEMA:
        raise ValueError("Unsupported D-Pro checkpoint version")


def load_model_d_pro(*, model_b=None, head=None, checkpoint=None, device="cpu"):
    """Load original trained initialization OR strict D-Pro weights, never both.

    Loading a best/last model does not restore optimizer, scaler, or progress.
    No original weight files are needed when loading a complete D-Pro checkpoint.
    """
    if checkpoint is not None:
        if model_b is not None or head is not None:
            raise ValueError("Choose a D-Pro checkpoint OR the original checkpoint pair")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise ValueError("Expected a D-Pro model/identity checkpoint")
        check_pilot_identity(payload.get("identity"))
        model = empty_model()
        model.load_state_dict(payload["model"], strict=True)
        info = dict(checkpoint_sha256=sha_file(checkpoint), identity=payload["identity"],
                    step=payload.get("step", payload.get("state", {}).get("completed_steps")),
                    weights_only_initialization=True)
    else:
        if model_b is None or head is None:
            raise ValueError("Supply both original Model B and epoch-17 HVI checkpoints")
        for path, expected in ((model_b, EXPECTED_B), (head, EXPECTED_HEAD)):
            if sha_file(path) != expected:
                raise ValueError(f"Original checkpoint SHA256 mismatch: {path}")
        original, info = load_model_d(model_b, head, device="cpu")
        model = ModelDPro(original)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError("Nonfinite D-Pro checkpoint tensors")
    return model.to(device).eval(), info


def source_manifest(repo=ROOT):
    """Stable across Windows/Colab line endings; fail on changed baseline code."""
    repo = Path(repo).resolve()
    protected = {}
    for name in PROTECTED_FILES:
        expected = subprocess.check_output(
            ["git", "show", f"{PINNED_COMMIT}:{name}"], cwd=repo)
        actual = (repo / name).read_bytes().replace(b"\r\n", b"\n")
        expected = expected.replace(b"\r\n", b"\n")
        if actual != expected:
            raise ValueError(f"Protected Model D source changed: {name}")
        protected[name] = hashlib.sha256(expected).hexdigest()
    package = {}
    for path in sorted((repo / "model_d_pro").rglob("*")):
        if path.suffix in {".py", ".json"}:
            package[path.relative_to(repo).as_posix()] = hashlib.sha256(
                path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return dict(protected=protected, package=package)


def make_identity(cfg, store, repo=ROOT):
    return dict(version=SCHEMA, baseline_commit=PINNED_COMMIT,
                original_script_sha256=SOURCE_SHA256, sources=source_manifest(repo),
                backbone_sha256=EXPECTED_B, head_sha256=EXPECTED_HEAD,
                train_sha256=store.train_sha, val_sha256=store.val_sha,
                recipe={key: cfg[key] for key in BEHAVIOR_KEYS})


def validate_resume_payload(payload, identity):
    if payload.get("identity", {}).get("version") != SCHEMA:
        raise ValueError("Historical last.pth cannot be fully resumed after extraction. "
                         "Load best.pth as model weights with a separately defined future recipe.")
    if payload["identity"] != identity:
        raise ValueError("Resume identity/recipe/source mismatch; use a new run")
    required = {"model", "optimizer", "scaler", "state", "torch_rng", "cuda_rng",
                "python_rng", "numpy_rng"}
    if not required.issubset(payload):
        raise ValueError("Incomplete modular training checkpoint")


def prepare_run(run, identity, resume):
    """Accept ownership only after all existing run metadata has been checked."""
    run = Path(run)
    if run.exists() and any(run.iterdir()):
        if not resume:
            raise FileExistsError("Existing run: enable resume or use a NEW run_name")
        path = run / "identity.json"
        if not path.is_file() or json.loads(path.read_text()) != identity:
            raise ValueError("Run identity/recipe/source changed; use a NEW run_name. "
                             "Historical pilot runs are weights-only inputs.")
        if not (run / "last.pth").is_file():
            raise ValueError("Existing run has no durable last.pth; use a new run")
        payload = torch.load(run / "last.pth", map_location="cpu", weights_only=True)
        validate_resume_payload(payload, identity)
        if not (run / "baseline.json").is_file() or not (run / "step0.json").is_file():
            raise ValueError("Resume requires the run's baseline and step0 reports")
    run.mkdir(parents=True, exist_ok=True)


def rng_metadata():
    state = np.random.get_state()
    return dict(python_rng=random.getstate(),
                numpy_rng=[state[0], state[1].tolist(), state[2], state[3], state[4]])


def restore_training_state(payload, identity, model, optimizer, scaler, device):
    validate_resume_payload(payload, identity)
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.device(device).type == "cuda":
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])
    state = payload["numpy_rng"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    for slots in optimizer.state.values():
        for key, value in slots.items():
            if torch.is_tensor(value) and key != "step":
                slots[key] = value.to(device)
    return payload["state"]


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_torch_save(value, path):
    # Only our own tensors/plain metadata are saved; originals are never written.
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(cpu_tree(value), temp)
    os.replace(temp, path)
