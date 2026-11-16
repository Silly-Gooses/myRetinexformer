"""Verify the original source and weight identities before inference."""

import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import torch


RETINEX_COMMIT = "17215ba8ce65417ee9f18df948996225c1d1a931"
HVI_COMMIT = "eb43d7d91e9a336c66856824ff9e4603ae41f408"
RETINEX_SHA = "539bd16c4da6179e45616329f249c4672951b1045193428e1d042c50d4b65a0b"
HVI_SHA = "44a7f92cf02a1da322c1a34b3aa940341dc93fd53840b7f19b49f461200b00a6"


def load_verified_state(model, path, expected_sha):
    with Path(path).open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected_sha:
            raise ValueError(f"Checkpoint SHA-256 mismatch for {path}: {actual}, expected {expected_sha}")
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("params", "params_ema", "state_dict", "model_state_dict", "model", "net"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Empty or invalid checkpoint state in {path}")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def load_cidnet_class(repo):
    repo = Path(repo).resolve(strict=True)
    git = ["git", "-C", str(repo)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output([*git, "status", "--porcelain", "--", "net"], text=True)
    if revision != HVI_COMMIT or changes.strip():
        raise ValueError(f"Expected clean HVI source at {HVI_COMMIT}, found {revision}: {changes}")
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("net.CIDNet")
        if Path(module.__file__).resolve() != repo / "net" / "CIDNet.py":
            raise ImportError("Another net.CIDNet is already imported. Use a fresh Python process.")
        return module.CIDNet
    finally:
        sys.path.pop(0)
