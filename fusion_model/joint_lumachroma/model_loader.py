"""Pinned official sources, verified checkpoint identities, strict frozen loading."""
import importlib.util
import subprocess
from pathlib import Path

import torch

from fusion_model.checkpoint_utils import load_cidnet_class
from .config import CFG, HVI_COMMIT, HVI_REFERENCE_HASH, RET_COMMIT, RET_EXPECTED_HASH
from .io_utils import digest


def verify_source(repo, commit, source_path):
    repo = Path(repo).resolve(strict=True)
    git = ["git", "-C", str(repo)]
    actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    changes = subprocess.check_output([*git, "status", "--porcelain", "--", source_path], text=True)
    if actual != commit or changes.strip():
        raise ValueError(f"Expected clean source {commit}, found {actual}: {changes}")
    return repo


def resolve_weights(datasets, ret_weights, hvi_weights):
    weights = {}
    for ds in datasets:
        c = CFG[ds]
        r = Path(ret_weights) / c["ret"]
        if not r.is_file():
            from huggingface_hub import hf_hub_download
            r = Path(hf_hub_download(repo_id="CaiYuanhao/Retinexformer", filename=c["ret"], repo_type="model"))
        h = Path(hvi_weights) / c["hvi"]
        rh, hh = digest(r), digest(h)
        if rh != RET_EXPECTED_HASH[c["ret"]]:
            raise ValueError(f"RetinexFormer SHA-256 mismatch: {r}")
        # The notebook logs HVI hashes; the recovered reference report supplies these identities.
        if hh != HVI_REFERENCE_HASH[ds]:
            raise ValueError(f"HVI SHA-256 mismatch against reference run: {h}")
        weights[ds] = {"ret": r, "hvi": h, "ret_sha256": rh, "hvi_sha256": hh}
    return weights


def unwrap_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint structure: {type(state)}")
    for key in ("params", "params_ema", "state_dict", "model_state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not state:
        raise ValueError("Empty checkpoint")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def configure_hvi(net, domain):
    c = CFG[domain]
    net.trans.gated = c["gate"] == "gated"
    net.trans.gated2 = c["gate"] == "gated2"
    if c["gate"] == "gated2":
        net.trans.alpha = c["hvi_alpha"]
    return net


def load_models(weights, domain, ret_repo, hvi_repo, device):
    ret_repo = verify_source(ret_repo, RET_COMMIT, "basicsr/models/archs")
    verify_source(hvi_repo, HVI_COMMIT, "net")
    spec = importlib.util.spec_from_file_location(
        "official_ret_arch_ycc_joint_v2", ret_repo / "basicsr/models/archs/RetinexFormer_arch.py")
    ret_arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ret_arch)
    ret = ret_arch.RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
    hvi = load_cidnet_class(hvi_repo)()
    # Recheck immediately before deserialization, including when called without resolve_weights.
    for kind, net, expected in (("ret", ret, RET_EXPECTED_HASH[CFG[domain]["ret"]]),
                                ("hvi", hvi, HVI_REFERENCE_HASH[domain])):
        if digest(weights[kind]) != expected:
            raise ValueError(f"{kind} SHA-256 mismatch")
        net.load_state_dict(unwrap_state(weights[kind]), strict=True)
        net.to(device).eval().requires_grad_(False)
    return ret, configure_hvi(hvi, domain)
