"""Machine-readable comparisons against the recorded Option A run."""

import csv
import json
from pathlib import Path

import numpy as np

from .data import REFERENCE_DIR


REFERENCE_SOURCE = dict(psnr=20.948619411618647, ssim=0.7316434587529961)
REFERENCE_BEST = dict(psnr=21.036645189367817, ssim=0.7354556825534054)
PSNR_TOLERANCE_DB = .02
SSIM_TOLERANCE = .001


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def compare_evaluation(actual, rows, *, reference="best", output_dir=None):
    reference_file = REFERENCE_DIR / ("best_per_image.csv" if reference == "best"
                                      else "source_per_image.csv")
    reference_aggregate = REFERENCE_BEST if reference == "best" else REFERENCE_SOURCE
    expected = {row["name"]: row for row in read_csv(reference_file)}
    observed = {row["name"]: row for row in rows}
    if len(expected) != 69 or set(expected) != set(observed):
        raise ValueError("Per-image comparison names differ from the reference 69")
    per_image = []
    for name in expected:
        row = observed[name]
        baseline = expected[name]
        per_image.append(dict(name=name, psnr=float(row["psnr"]),
                              ssim=float(row["ssim"]),
                              reference_psnr=float(baseline["psnr"]),
                              reference_ssim=float(baseline["ssim"]),
                              delta_psnr=float(row["psnr"]) - float(baseline["psnr"]),
                              delta_ssim=float(row["ssim"]) - float(baseline["ssim"])))
    delta_psnr = actual["psnr"] - reference_aggregate["psnr"]
    delta_ssim = actual["ssim"] - reference_aggregate["ssim"]
    report = dict(reference=reference, count=actual["count"],
                  actual_psnr=actual["psnr"], actual_ssim=actual["ssim"],
                  reference_psnr=reference_aggregate["psnr"],
                  reference_ssim=reference_aggregate["ssim"],
                  delta_psnr=delta_psnr, delta_ssim=delta_ssim,
                  max_abs_per_image_psnr_delta=float(max(abs(row["delta_psnr"]) for row in per_image)),
                  max_abs_per_image_ssim_delta=float(max(abs(row["delta_ssim"]) for row in per_image)),
                  aggregate_within_tolerance=(abs(delta_psnr) <= PSNR_TOLERANCE_DB
                                              and abs(delta_ssim) <= SSIM_TOLERANCE),
                  psnr_tolerance_db=PSNR_TOLERANCE_DB,
                  ssim_tolerance=SSIM_TOLERANCE)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"{reference}_per_image_comparison.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_image[0]))
            writer.writeheader()
            writer.writerows(per_image)
        (output_dir / f"{reference}_evaluation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_training(run_dir, *, output_dir=None):
    run_dir = Path(run_dir)
    history = read_csv(run_dir / "validation.csv")
    steps = read_csv(run_dir / "steps.csv")
    reference_history = {int(row["local_step"]): row for row in
                         read_csv(REFERENCE_DIR / "validation.csv")}
    reference_steps = {int(row["local_step"]): row for row in
                       read_csv(REFERENCE_DIR / "steps.csv")}
    history_by_step = {int(row["local_step"]): row for row in history}
    steps_by_step = {int(row["local_step"]): row for row in steps}
    validation = []
    for step, baseline in reference_history.items():
        current = history_by_step.get(step)
        validation.append(dict(local_step=step,
                               reference_psnr=float(baseline["psnr"]),
                               actual_psnr=float(current["psnr"]) if current else None,
                               reference_ssim=float(baseline["ssim"]),
                               actual_ssim=float(current["ssim"]) if current else None))
    loss = []
    for step, baseline in reference_steps.items():
        current = steps_by_step.get(step)
        loss.append(dict(local_step=step,
                         reference_loss=float(baseline["loss"]),
                         actual_loss=float(current["loss"]) if current else None))
    if 0 not in history_by_step or 1000 not in history_by_step:
        raise ValueError("Training comparison requires steps 0 and 1000")
    best_step = max(history_by_step, key=lambda step: float(history_by_step[step]["psnr"]))
    report = dict(start_psnr=float(history_by_step[0]["psnr"]),
                  start_ssim=float(history_by_step[0]["ssim"]),
                  best_local_step=best_step,
                  best_psnr=float(history_by_step[best_step]["psnr"]),
                  best_ssim=float(history_by_step[best_step]["ssim"]),
                  final_psnr=float(history_by_step[1000]["psnr"]),
                  final_ssim=float(history_by_step[1000]["ssim"]),
                  reference_best_local_step=1000,
                  reference_best_psnr=REFERENCE_BEST["psnr"],
                  reference_best_ssim=REFERENCE_BEST["ssim"],
                  reference_start_psnr=REFERENCE_SOURCE["psnr"],
                  reference_start_ssim=REFERENCE_SOURCE["ssim"],
                  loss_mae=float(np.mean([abs(row["actual_loss"] - row["reference_loss"])
                                          for row in loss if row["actual_loss"] is not None])))
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, rows in (("validation_comparison.csv", validation),
                               ("loss_comparison.csv", loss)):
            with (output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (output_dir / "training_comparison.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        summary = dict(model="Model B · Option A", batch=8, patch=128, lr=1e-5,
                       additional_steps=1000, source_local_step=1000,
                       original_source_best_psnr=20.9486,
                       initial_revalidated_psnr=report["start_psnr"],
                       initial_revalidated_ssim=report["start_ssim"],
                       best_psnr=report["best_psnr"], best_ssim=report["best_ssim"],
                       best_local_step=report["best_local_step"],
                       final_psnr=report["final_psnr"],
                       final_ssim=report["final_ssim"],
                       gain_vs_revalidated_start_db=report["best_psnr"] - report["start_psnr"],
                       gap_vs_original_v1_db=report["best_psnr"] - 21.637163312246877)
        with (output_dir / "optionA_summary.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary))
            writer.writeheader()
            writer.writerow(summary)
    return report
