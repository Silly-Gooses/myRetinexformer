"""Train-only teacher inference and sufficient-statistics cache."""
import gc

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .coefficients import mse_coefs
from .color_fusion import dual_mse_coeffs
from .datasets import discover_splits
from .inference import predict_hvi, predict_ret
from .io_utils import locked_json, read_rgb
from .metrics import psnr_float
from .model_loader import load_models
from .optimize_weights import fit_tables
from .protocol import protocol_identity, write_manifest

COEFFICIENT_COLUMNS = ["rgb_a", "rgb_b", "rgb_c", "Ayy", "Acc", "Ayc", "By", "Bc", "C",
                       "ret_psnr_float", "hvi_psnr_float"]


def validate_cache(df, domain, names):
    if len(df) != len(names) or df.filename.tolist() != names or not df.dataset.eq(domain).all():
        raise ValueError(f"{domain}: stale coefficient cache")
    if not set(COEFFICIENT_COLUMNS).issubset(df.columns) or not np.isfinite(df[COEFFICIENT_COLUMNS].to_numpy()).all():
        raise ValueError(f"{domain}: invalid coefficient cache")


def collect_coefficients(domain, pairs, weights, ret_repo, hvi_repo, device):
    ret, hvi = load_models(weights, domain, ret_repo, hvi_repo, device)
    records = []
    for low_file, gt_file in tqdm(pairs, desc=domain + " Train coefficient cache"):
        low, gt8 = read_rgb(low_file), read_rgb(gt_file)
        if low.shape != gt8.shape:
            raise ValueError(f"{domain}: dimensions differ for {low_file.name}")
        gt = gt8.astype(np.float32) / 255.
        rr, hh = predict_ret(ret, low, device), predict_hvi(hvi, low, device)
        ra, rb, rc = mse_coefs(rr, hh, gt)
        zz = dual_mse_coeffs(rr, hh, gt)
        records.append(dict(dataset=domain, filename=low_file.name, rgb_a=ra, rgb_b=rb, rgb_c=rc,
                            ret_psnr_float=psnr_float(rr, gt), hvi_psnr_float=psnr_float(hh, gt), **zz))
    del ret, hvi
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(records)


def fit_experiment(configs, datasets, weights, ret_repo, hvi_repo, device, output, fit_count=None):
    splits = discover_splits(configs, datasets, fit_count)
    lock = protocol_identity(configs, datasets, weights)
    lock.update(fit_count=fit_count,
                train_filenames={ds: [x[0].name for x in splits[ds]["fit"]] for ds in datasets},
                test_filenames={ds: [x[0].name for x in splits[ds]["test"]] for ds in datasets})
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "protocol_lock.json").exists() and any(output.glob("*/train_fit_per_image_coefficients.csv")):
        raise ValueError("Existing Train cache has no protocol lock; use a new output directory")
    locked_json(output / "protocol_lock.json", lock)
    tables = {}
    for ds in datasets:
        path = output / ds / "train_fit_per_image_coefficients.csv"
        names = lock["train_filenames"][ds]
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = collect_coefficients(ds, splits[ds]["fit"], weights[ds], ret_repo, hvi_repo, device)
            validate_cache(df, ds, names)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".temporary.csv")
            df.to_csv(temporary, index=False)
            temporary.replace(path)
        validate_cache(df, ds, names)
        tables[ds] = df
    params = fit_tables(tables, output)
    write_manifest(output, lock, params, device, [], stage="fit")
    return params
