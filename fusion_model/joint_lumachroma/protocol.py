"""Run identity and runtime provenance; reference records live separately."""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

import torch

from .config import CV_FOLDS, HVI_COMMIT, LAMBDA_GRID, REGULARIZATION_CENTER, RET_COMMIT, SPLIT_SEED
from .io_utils import digest, write_json


def protocol_identity(configs, datasets, weights):
    return {
        "experiment": "reg_joint_ycc_v2",
        "ret_commit": RET_COMMIT, "hvi_commit": HVI_COMMIT,
        "split_seed": SPLIT_SEED, "cv_folds": CV_FOLDS,
        "lambda_grid": LAMBDA_GRID, "regularization_center": REGULARIZATION_CENTER,
        "selected_datasets": list(datasets),
        "inference": "single pass; dataset-specific frozen checkpoints; no GT-mean",
        "checkpoint_sha256": {ds: {"ret": weights[ds]["ret_sha256"], "hvi": weights[ds]["hvi_sha256"]}
                              for ds in datasets},
        "dataset_roots": {ds: str(configs[ds]["root"].resolve()) for ds in datasets},
        "implementation_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def write_manifest(output, lock, params, device, methods, smoke_max_test=None, stage="evaluation"):
    versions = {"python": platform.python_version()}
    for name in ("torch", "numpy", "scipy", "pandas", "opencv-python", "opencv-python-headless", "Pillow", "einops", "huggingface_hub"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    manifest = {
        "experiment": "Regularized + Joint Luminance-Chrominance Fusion v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result_status": "new_smoke_run" if smoke_max_test is not None else f"new_{stage}_run",
        "sources": {"RetinexFormer": RET_COMMIT, "HVI-CIDNet": HVI_COMMIT},
        "versions": versions, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "methods": methods, "parameter_selection": params,
        "protocol": lock, "smoke_max_test": smoke_max_test,
        "limitations": [
            "Teacher checkpoints may have seen original Train images, including CV holdouts.",
            "Train scoring is an unclipped affine approximation; final fusion is clipped.",
            "Official Test benchmarks were inspected in earlier exploratory work.",
            "Shared fusion weights still use dataset-specific pretrained backbones.",
            "A smoke run does not reproduce full-dataset reference results.",
        ],
    }
    write_json(Path(output) / ("report_manifest.json" if stage == "evaluation" else f"{stage}_manifest.json"), manifest)
