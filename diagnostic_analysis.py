"""Read-only V2 checkpoint diagnostics for the Colab Analysis section."""

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import uuid

from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from analysis_utils import extract_checkpoint_state_dict, infer_full_image, tensor_to_uint8
from dataLoader import OfficialPairedDataset
from official_workflow import FORMAT, SNAPSHOT_FORMAT, inspect_dataset
from protocol_metrics import VALIDATION_PROTOCOL, protocol_metrics, write_csv
from train import build_model


def _checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint is not a mapping: {path}")
    return state


def select_checkpoints(experiment_root):
    """Find the original epoch-200 state and its immutable selected-best snapshot."""
    root = Path(experiment_root)
    archive = root / "checkpoints/epoch_0200.pth"
    latest = root / "checkpoints/last.pth"
    final_path = archive if archive.is_file() else latest
    final = _checkpoint(final_path)
    if final.get("format") != FORMAT or final.get("completed_epoch") != 200:
        raise ValueError("The epoch-200 full checkpoint is unavailable; expected "
                         "checkpoints/epoch_0200.pth or an epoch-200 checkpoints/last.pth")
    config = final.get("config")
    if not isinstance(config, dict) or config.get("version") != "v2":
        raise ValueError("The epoch-200 checkpoint is not a V2 training checkpoint")
    if final.get("global_step") != 200 * math.ceil(len(final["manifest"]["splits"]["train"])
                                                    / config["batch_size"]):
        raise ValueError("Epoch-200 checkpoint has inconsistent global step")
    best_info = final.get("best", {})
    snapshot_name = best_info.get("snapshot")
    if snapshot_name:
        if Path(snapshot_name).name != snapshot_name:
            raise ValueError("Unsafe best-snapshot name in checkpoint metadata")
        best_path = root / "checkpoints" / snapshot_name
    else:
        best_path = root / "checkpoints/best.pth"
    best = _checkpoint(best_path)
    if best.get("format") not in (SNAPSHOT_FORMAT, FORMAT):
        raise ValueError("Selected best checkpoint has an unsupported format")
    if best.get("config") != config:
        raise ValueError("Best/final checkpoint configurations differ")
    best_epoch = best.get("epoch", best.get("completed_epoch"))
    if best_epoch != best_info.get("epoch") or best.get("global_step") != best_info.get("global_step"):
        raise ValueError("Selected best snapshot does not match epoch-200 best metadata")
    return {"best": best_path, "final": final_path}, final


def read_logs(experiment_root, final_checkpoint):
    """Read recorded history; checkpoint history is a recovery fallback."""
    root = Path(experiment_root)
    def rows(path, fallback):
        if not path.is_file():
            return list(fallback), f"{path.name} missing; recovered from epoch-200 checkpoint"
        with path.open(newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file)), None
    training, training_note = rows(root / "logs/train_log.csv", final_checkpoint["history"])
    validation, validation_note = rows(root / "logs/validation_log.csv",
                                        final_checkpoint["validation_history"])
    # Only the original first 200 epochs belong in this checkpoint comparison.
    training = [row for row in training if int(row["epoch"]) <= 200]
    validation = [row for row in validation if int(row["global_step"]) <= final_checkpoint["global_step"]]
    if not training or not validation:
        raise ValueError("Training or validation history is empty")
    unavailable = []
    for field, title in (("val_loss", "Validation loss"), ("grad_norm", "Gradient norm")):
        if not any(row.get(field) not in (None, "") for row in training):
            unavailable.append(f"{title} was not recorded; no historical plot was created.")
    return training, validation, [note for note in (training_note, validation_note) if note] + unavailable


def save_stability_plots(training, validation, best_step, output_dir):
    output_dir = Path(output_dir)
    specs = (
        ("training_loss", training, "train_loss", "Training L1 loss"),
        ("validation_psnr", validation, "psnr", "Validation PSNR (dB)"),
        ("validation_ssim", validation, "ssim", "Validation SSIM"),
        ("learning_rate", training, "lr", "Learning rate"),
        ("gradient_norm", training, "grad_norm", "Gradient norm"),
    )
    written = []
    for filename, rows, field, label in specs:
        points = [(int(row["global_step"]), float(row[field])) for row in rows
                  if row.get(field) not in (None, "")]
        if not points:
            continue
        fig = Figure(figsize=(8, 4))
        FigureCanvasAgg(fig)
        axis = fig.subplots()
        axis.plot(*zip(*points))
        axis.axvline(best_step, color="tab:red", linestyle="--", label=f"Best step {best_step}")
        axis.set(xlabel="Global step", ylabel=label, title=label)
        axis.grid(alpha=.25)
        axis.legend()
        fig.tight_layout()
        path = output_dir / f"{filename}.png"
        fig.savefig(path, dpi=150)
        written.append(path)
    return written


def _pad(low):
    height, width = low.shape[-2:]
    ph, pw = (-height) % 4, (-width) % 4
    if ph or pw:
        mode = "reflect" if ph < height and pw < width else "replicate"
        low = F.pad(low, (0, pw, 0, ph), mode=mode)
    return low, height, width


def _score(image, target):
    psnr, ssim = protocol_metrics(image, target)
    return psnr, ssim


def _mean_clamped(tensor):
    return float(tensor.float().clamp(0, 1).mean())


def _load_model(path, config, device):
    saved = _checkpoint(path)
    if saved.get("config") != config:
        raise ValueError(f"Checkpoint config differs: {path}")
    model = build_model(config, device)
    model.load_state_dict(extract_checkpoint_state_dict(saved), strict=True)
    return model.eval()


def analyze_images(paths, final_checkpoint, dataset_root, max_images, device,
                   *, edge_ablation=True, visual_count=8):
    """Score the identical official validation images for both V2 checkpoints."""
    if type(max_images) is not int or not 1 <= max_images <= 100:
        raise ValueError("max_images must be an integer from 1 to 100")
    config = final_checkpoint["config"]
    inspected, pairs = inspect_dataset(dataset_root, training=False)
    if inspected["splits"]["test"] != final_checkpoint["manifest"]["splits"]["test"]:
        raise ValueError("Current Test images differ from the checkpoint dataset manifest")
    dataset = OfficialPairedDataset(pairs["test"])
    count = min(max_images, len(dataset))
    selected = list(range(count))
    visual_indices = set(np.linspace(0, count - 1, min(visual_count, count), dtype=int).tolist())
    metrics, edges, visuals = [], [], {}
    for label in ("best", "final"):
        model = _load_model(paths[label], config, device)
        for index in selected:
            sample = dataset[index]
            low = sample["low"].unsqueeze(0).to(device)
            gt = sample["high"].to(device)
            with torch.inference_mode():
                padded, height, width = _pad(low)
                output, details = model(padded, return_intermediates=True)
                output = output[0, :, :height, :width]
                lit_up = details["lit_up"][0, :, :height, :width]
                edge_map = details["edge_map"][0, :, :height, :width]
                if index < 3 and not torch.allclose(
                        output, infer_full_image(model, low)[0], atol=1e-6, rtol=1e-6):
                    raise AssertionError("Captured output differs from normal full-image inference")
                input_psnr, input_ssim = _score(low[0], gt)
                lit_psnr, lit_ssim = _score(lit_up, gt)
                out_psnr, out_ssim = _score(output, gt)
                correction = float((output.clamp(0, 1) - lit_up.clamp(0, 1)).abs().mean())
                feature = details["edge_features"][0, :, :height, :width]
                gate = details["gate_map"][0, :, :height, :width]
                contribution = gate * feature
                row = dict(checkpoint=label, image=sample["name"],
                           input_psnr=input_psnr, input_ssim=input_ssim,
                           lit_up_psnr=lit_psnr, lit_up_ssim=lit_ssim,
                           output_psnr=out_psnr, output_ssim=out_ssim,
                           lit_to_output_psnr_delta=out_psnr - lit_psnr,
                           lit_to_output_ssim_delta=out_ssim - lit_ssim,
                           mean_absolute_correction=correction,
                           lit_up_brightness=_mean_clamped(lit_up),
                           output_brightness=_mean_clamped(output))
                metrics.append(row)
                edge_row = dict(checkpoint=label, image=sample["name"],
                                edge_pixel_density=float((edge_map > 0).float().mean()),
                                edge_feature_mean_abs=float(feature.abs().mean()),
                                fusion_contribution_mean_abs=float(contribution.abs().mean()),
                                gate_mean=float(gate.mean()), gate_std=float(gate.std(unbiased=False)),
                                gate_min=float(gate.min()), gate_max=float(gate.max()),
                                sensitivity_test="inference-time zero-edge input")
                if edge_ablation:
                    handle = model.edge_extractor.register_forward_hook(
                        lambda module, inputs, result: torch.zeros_like(result))
                    try:
                        zero_output = infer_full_image(model, low)[0]
                    finally:
                        handle.remove()
                    zero_psnr, zero_ssim = _score(zero_output, gt)
                    edge_row.update(zero_edge_psnr=zero_psnr, zero_edge_ssim=zero_ssim,
                                    edge_psnr_delta=out_psnr - zero_psnr,
                                    edge_ssim_delta=out_ssim - zero_ssim)
                edges.append(edge_row)
                if index in visual_indices:
                    visuals[(label, sample["name"])] = {
                        "low": tensor_to_uint8(low[0]),
                        "lit_up": tensor_to_uint8(lit_up),
                        "output": tensor_to_uint8(output),
                        "gt": tensor_to_uint8(gt),
                        "edge": (edge_map[0].float().cpu().numpy() * 255).round().astype(np.uint8),
                        "psnr": out_psnr, "ssim": out_ssim,
                    }
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return metrics, edges, visuals


def compare_checkpoints(metrics):
    by_label = {label: {row["image"]: row for row in metrics if row["checkpoint"] == label}
                for label in ("best", "final")}
    if set(by_label["best"]) != set(by_label["final"]):
        raise ValueError("Best/final validation image IDs differ")
    rows = []
    for image in by_label["best"]:
        best, final = by_label["best"][image], by_label["final"][image]
        rows.append(dict(image=image, best_psnr=best["output_psnr"],
                         final_psnr=final["output_psnr"],
                         final_minus_best_psnr=final["output_psnr"]-best["output_psnr"],
                         best_ssim=best["output_ssim"], final_ssim=final["output_ssim"],
                         final_minus_best_ssim=final["output_ssim"]-best["output_ssim"],
                         best_lit_to_output_psnr_delta=best["lit_to_output_psnr_delta"],
                         final_lit_to_output_psnr_delta=final["lit_to_output_psnr_delta"]))
    return rows


def save_visuals(visuals, output_dir):
    directory = Path(output_dir) / "visuals"
    directory.mkdir()
    for (label, image), panels in visuals.items():
        arrays = [panels[key] for key in ("low", "lit_up", "output", "gt")]
        arrays.append(np.repeat(panels["edge"][..., None], 3, axis=2))
        error = np.abs(panels["output"].astype(np.float32)-panels["gt"].astype(np.float32)).mean(2)
        arrays.append((colormaps["inferno"](np.clip(error / 255., 0, 1))[..., :3] * 255).astype(np.uint8))
        height, width = arrays[0].shape[:2]
        figure = Image.new("RGB", (width*6, height+46), "white")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(figure)
        for i, (title, array) in enumerate(zip(("Low", "Lit-up", "Output", "GT", "Canny", "Abs error"), arrays)):
            figure.paste(Image.fromarray(array).convert("RGB"), (i*width, 46))
            draw.text((i*width+4, 4), title, fill="black")
        draw.text((4, 24), f"{label} | {image} | PSNR {panels['psnr']:.4f} | SSIM {panels['ssim']:.4f}", fill="black")
        safe_name = Path(image).stem
        figure.save(directory / f"{label}_{safe_name}.png")
    return directory


def create_run_directory(experiment_root, output_parent=None):
    root = Path(output_parent) if output_parent is not None else Path(experiment_root) / "analysis"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                        + "_" + uuid.uuid4().hex[:8])
    directory.mkdir(exist_ok=False)
    return directory


def export_results(output_dir, metrics, edges, comparisons, final_checkpoint,
                   notes, paths, *, training=(), validation=(), save_csv=True):
    output_dir = Path(output_dir)
    if save_csv:
        for name, rows in (("illumination_denoising_analysis.csv", metrics),
                           ("edge_analysis.csv", edges),
                           ("checkpoint_comparison.csv", comparisons)):
            if rows:
                write_csv(output_dir / name, tuple(dict.fromkeys(k for row in rows for k in row)), rows)
    summary = {}
    numeric = ("input_psnr", "input_ssim", "lit_up_psnr", "lit_up_ssim",
               "output_psnr", "output_ssim", "lit_to_output_psnr_delta",
               "lit_to_output_ssim_delta", "mean_absolute_correction",
               "lit_up_brightness", "output_brightness")
    for label in ("best", "final"):
        selected = [row for row in metrics if row["checkpoint"] == label]
        summary[label] = {key: dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                   min=float(np.min(values)), max=float(np.max(values)))
                          for key in numeric
                          for values in [[row[key] for row in selected]]}
        summary[label]["images_improved_over_lit_up_pct"] = 100 * sum(
            row["lit_to_output_psnr_delta"] > 0 for row in selected) / len(selected)
    with (output_dir / "aggregate_summary.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, allow_nan=False)
    count = len([row for row in metrics if row["checkpoint"] == "best"])
    best_info = final_checkpoint["best"]
    lines = ["# V2 diagnostic analysis", "",
             f"Protocol: {VALIDATION_PROTOCOL}. Official Test pairs were used for validation and checkpoint selection; these are not held-out scores.",
             f"Images analyzed per checkpoint: {count}. Results use the same image IDs.", "",
             "## Observed facts", "",
             f"Selected best: epoch {best_info['epoch']}, step {best_info['global_step']}; checkpoint {paths['best']}.",
             f"Epoch-200 final: step {final_checkpoint['global_step']}; checkpoint {paths['final']}."]
    for label in ("best", "final"):
        values = summary[label]
        lines.append(f"{label}: output PSNR {values['output_psnr']['mean']:.4f} dB, "
                     f"SSIM {values['output_ssim']['mean']:.4f}; "
                     f"{values['images_improved_over_lit_up_pct']:.1f}% of images improve in PSNR from lit-up to output.")
    lines.append(f"Final-minus-best mean output PSNR: "
                 f"{summary['final']['output_psnr']['mean'] - summary['best']['output_psnr']['mean']:+.4f} dB; "
                 f"SSIM: {summary['final']['output_ssim']['mean'] - summary['best']['output_ssim']['mean']:+.4f}.")
    recorded_losses = [float(row["train_loss"]) for row in training
                       if row.get("train_loss") not in (None, "")]
    if recorded_losses:
        lines.append(f"Recorded training loss: first {recorded_losses[0]:.6f}, "
                     f"last {recorded_losses[-1]:.6f}, minimum {min(recorded_losses):.6f}.")
    recorded_validation = [(int(row["global_step"]), float(row["psnr"])) for row in validation
                           if row.get("psnr") not in (None, "")]
    if recorded_validation:
        lines.append(f"Recorded validation PSNR: first {recorded_validation[0][1]:.4f} dB "
                     f"at step {recorded_validation[0][0]}, last {recorded_validation[-1][1]:.4f} dB "
                     f"at step {recorded_validation[-1][0]}.")
    if count == 100:
        for label, reference in (("best", (20.7297, .7833)),
                                 ("final", (19.9906, .7774))):
            measured = summary[label]
            psnr_gap = measured["output_psnr"]["mean"] - reference[0]
            ssim_gap = measured["output_ssim"]["mean"] - reference[1]
            status = "consistent within 0.02 dB / 0.002" if abs(psnr_gap) <= .02 and abs(ssim_gap) <= .002 else "DIFFERS from supplied reference"
            lines.append(f"{label} supplied full-set reference: {reference[0]:.4f} dB / "
                         f"{reference[1]:.4f}; measured-minus-reference {psnr_gap:+.4f} dB / "
                         f"{ssim_gap:+.4f} ({status}).")
    worst = sorted((row for row in metrics if row["checkpoint"] == "final"),
                   key=lambda row: row["lit_to_output_psnr_delta"])
    lines += ["", "Largest negative lit-up-to-output PSNR changes: " +
              ", ".join(f"{row['image']} ({row['lit_to_output_psnr_delta']:+.3f} dB)" for row in worst[:3]),
              "Largest positive changes: " +
              ", ".join(f"{row['image']} ({row['lit_to_output_psnr_delta']:+.3f} dB)" for row in worst[-3:][::-1])]
    measured_edges = [row["edge_psnr_delta"] for row in edges if row.get("checkpoint") == "final"
                      and "edge_psnr_delta" in row]
    for label in ("best", "final"):
        selected_edges = [row for row in edges if row["checkpoint"] == label]
        if selected_edges:
            lines.append(f"{label} edge diagnostics: mean Canny density "
                         f"{float(np.mean([row['edge_pixel_density'] for row in selected_edges])):.4f}, "
                         f"mean gate {float(np.mean([row['gate_mean'] for row in selected_edges])):.4f}, "
                         f"mean absolute fusion contribution "
                         f"{float(np.mean([row['fusion_contribution_mean_abs'] for row in selected_edges])):.4f}.")
    if measured_edges:
        lines.append(f"Inference-only zero-edge sensitivity: mean normal-minus-zero PSNR "
                     f"{float(np.mean(measured_edges)):+.4f} dB. This is not a retrained ablation.")
    lines += ["", "Possible training-instability indicator: " +
              ("validation PSNR declined from selected best to epoch 200."
               if summary["final"]["output_psnr"]["mean"] < summary["best"]["output_psnr"]["mean"]
               else "no best-to-final PSNR decline in this analyzed subset."),
              "Missing historical information: " + (" ".join(notes) if notes else "None detected."),
              "", "## Possible explanations", "",
              "Changes in validation scores may reflect optimization or image-specific behavior; the diagnostics alone do not identify a causal module.",
              "", "## Hypotheses for further experiments", "",
              "Confirm suspected module effects with controlled retraining and a separate held-out evaluation set."]
    (output_dir / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
