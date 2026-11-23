"""Evaluate locked weights; this module never imports the fitter."""
import gc
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .config import CFG, METHODS
from .datasets import exact_pairs
from .inference import all_predictions, predict_hvi, predict_ret
from .io_utils import load_parameters, locked_json, read_rgb, save_png
from .metrics import psnr_float, psnr_rgb8, rgb8_hvi_style, rgb8_ret_style, ssim_rgb8
from .model_loader import load_models
from .protocol import protocol_identity, write_manifest


def evaluate_experiment(configs, datasets, weights, ret_repo, hvi_repo, device, output,
                        weights_json, smoke_max_test=None, save_predictions=True, preview_count=3):
    if smoke_max_test is not None and smoke_max_test < 1:
        raise ValueError("smoke_max_test must be positive")
    if preview_count < 0:
        raise ValueError("preview_count cannot be negative")
    params = load_parameters(weights_json, datasets)
    all_pairs = {ds: exact_pairs(configs[ds]["root"] / configs[ds]["test_low"],
                                configs[ds]["root"] / configs[ds]["test_gt"], configs[ds]["n_test"])
                 for ds in datasets}
    lock = protocol_identity(configs, datasets, weights)
    lock.update(parameter_selection=params, smoke_max_test=smoke_max_test,
                save_predictions=save_predictions, preview_count=preview_count,
                test_filenames={ds: [x[0].name for x in all_pairs[ds]] for ds in datasets})
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "evaluation_lock.json").exists() and any(output.glob("*/test_per_image.csv")):
        raise ValueError("Existing Test CSV has no evaluation lock; use a new output directory")
    locked_json(output / "evaluation_lock.json", lock)
    locked_json(output / "selected_weights_train_only.json", params)
    rows = []
    metric_columns = [m + "_" + k for m in METHODS for k in ("psnr_float", "psnr_rgb8", "ssim_rgb8")]
    for ds in datasets:
        pairs = all_pairs[ds][:smoke_max_test] if smoke_max_test is not None else all_pairs[ds]
        ddir = output / ds
        ddir.mkdir(parents=True, exist_ok=True)
        csvfile = ddir / "test_per_image.csv"
        if csvfile.exists():
            old = pd.read_csv(csvfile)
            if (len(old) != len(pairs) or old.filename.tolist() != [x[0].name for x in pairs]
                    or not old.dataset.eq(ds).all() or not set(metric_columns).issubset(old.columns)
                    or not np.isfinite(old[metric_columns].to_numpy()).all()):
                raise ValueError(f"{ds}: stale Test CSV")
            # A completed CSV is written only after all requested images are saved.
            for j, (low_file, _) in enumerate(pairs):
                expected = []
                if save_predictions:
                    expected += [ddir / "predictions" / m / (low_file.stem + ".png")
                                 for m in ("ycc_regularized", "ycc_joint_regularized")]
                if j < preview_count:
                    expected += [ddir / "previews" / (low_file.stem + "_" + m + ".png")
                                 for m in [*METHODS, "low", "gt"]]
                if any(not p.is_file() for p in expected):
                    raise ValueError("Incomplete saved images: use a new output directory")
            rows.extend(old.to_dict("records"))
            continue
        ret, hvi = load_models(weights[ds], ds, ret_repo, hvi_repo, device)
        records = []
        for j, (low_file, gt_file) in enumerate(tqdm(pairs, desc=ds + " Test")):
            low = read_rgb(low_file)
            rr, hh = predict_ret(ret, low, device), predict_hvi(hvi, low, device)
            preds = all_predictions(rr, hh, params["local"][ds], params["joint"])
            # Ground truth is read only after output generation; never passed to a predictor.
            gt8 = read_rgb(gt_file)
            if low.shape != gt8.shape or min(low.shape[:2]) < 11:
                raise ValueError(f"Invalid metric image dimensions: {ds}/{low_file.name}")
            gt = gt8.astype(np.float32) / 255.
            record = {"dataset": ds, "filename": low_file.name}
            for method, img in preds.items():
                img8 = rgb8_ret_style(img) if method == "ret" else rgb8_hvi_style(img)
                record[method + "_psnr_float"] = psnr_float(img, gt)
                record[method + "_psnr_rgb8"] = psnr_rgb8(img8, gt8)
                record[method + "_ssim_rgb8"] = ssim_rgb8(img8, gt8)
                if save_predictions and method in ("ycc_regularized", "ycc_joint_regularized"):
                    save_png(ddir / "predictions" / method / (low_file.stem + ".png"), img8)
                if j < preview_count:
                    save_png(ddir / "previews" / (low_file.stem + "_" + method + ".png"), img8)
            if j < preview_count:
                save_png(ddir / "previews" / (low_file.stem + "_low.png"), low)
                save_png(ddir / "previews" / (low_file.stem + "_gt.png"), gt8)
            records.append(record)
        temporary = csvfile.with_suffix(".temporary.csv")
        pd.DataFrame(records).to_csv(temporary, index=False)
        temporary.replace(csvfile)
        rows.extend(records)
        del ret, hvi
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    res = pd.DataFrame(rows)
    res.to_csv(output / "test_per_image_all.csv", index=False)
    summary = summarize(res, params, datasets, smoke_max_test)
    summary.to_csv(output / "test_summary.csv", index=False)
    write_manifest(output, lock, params, device, METHODS, smoke_max_test)
    archive_path = output / "RetinexFormer_HVI_RegularizedJoint_LumaChroma_Report.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in output.rglob("*"):
            if file.is_file() and "predictions" not in file.relative_to(output).parts and file.suffix.lower() in {".csv", ".json", ".png"}:
                archive.write(file, file.relative_to(output))
    return summary


def summarize(res, params, datasets, smoke_max_test=None):
    summary=[]
    for ds in datasets:
        z=res[res.dataset==ds]
        p=params["local"][ds]
        rec={'dataset':ds,'n_test':len(z),'evaluation_type':'smoke' if smoke_max_test is not None else 'official_test',
             'alpha_rgb':p['alpha_rgb'],'alpha_y':p['alpha_y'],'alpha_c':p['alpha_c'],
             'alpha_y_regularized':p['alpha_y_regularized'],'alpha_c_regularized':p['alpha_c_regularized'],
             'lambda_local':p['lambda_regularized'],'lambda_joint':params["joint"]['lambda_regularized']}
        for method in METHODS:
            for metric in ['psnr_float','psnr_rgb8','ssim_rgb8']:
                rec[method+'_'+metric]=float(z[method+'_'+metric].mean())
        baseline=max(rec['ret_psnr_float'],rec['hvi_psnr_float'])
        rec['best_standalone_psnr_float']=baseline
        for method in METHODS[2:]:
            rec[method+'_delta_vs_best_psnr_float']=rec[method+'_psnr_float']-baseline
            rec[method+'_better_than_both_image_count']=int(((z[method+'_psnr_float']>z.ret_psnr_float)&
                                                             (z[method+'_psnr_float']>z.hvi_psnr_float)).sum())
        rec['ret_delta_official_reference_float']=rec['ret_psnr_float']-CFG[ds]['ref_ret']
        rec['hvi_delta_reference_rgb8']=rec['hvi_psnr_rgb8']-CFG[ds]['ref_hvi']
        summary.append(rec)
    summary_df=pd.DataFrame(summary)
    return summary_df
