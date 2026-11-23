"""Shared file operations and immutable run records."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def save_png(path, arr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.png")
    Image.fromarray(arr).save(temporary, format="PNG")
    os.replace(temporary, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def locked_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Protocol or selected weights changed: choose a new output directory ({path.name})")
    else:
        write_json(path, value)


def load_parameters(path, datasets):
    params = json.loads(Path(path).read_text(encoding="utf-8"))
    if params.get("experiment") != "regjoint_v2":
        raise ValueError("Expected regjoint_v2 selected_weights_train_only.json")
    fields = ("alpha_y", "alpha_c", "alpha_y_regularized", "alpha_c_regularized")
    for group in [params["joint"], *(params["local"][ds] for ds in datasets)]:
        for key in fields:
            if not isinstance(group[key], (int, float)) or not np.isfinite(group[key]) or not 0 <= group[key] <= 1:
                raise ValueError(f"Invalid weight {key}")
        if not np.isfinite(group["lambda_regularized"]) or group["lambda_regularized"] < 0:
            raise ValueError("Invalid regularization strength")
    for ds in datasets:
        alpha = params["local"][ds]["alpha_rgb"]
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Invalid RGB fusion weight")
    return params
