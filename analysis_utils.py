"""Reusable full-image evaluation and image exports for Colab and local runs."""

import csv
import math
import shutil
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from utils import PSNR, calculate_ssim


SUPPORTED_IMAGE_SUFFIXES = frozenset((".png", ".jpg", ".jpeg", ".bmp"))


def tensor_to_uint8(tensor):
    """Convert a CHW RGB tensor in [0, 1] to the pixels saved in PNG files."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected a CHW RGB tensor")
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def image_metrics(prediction, target):
    """Calculate PSNR and SSIM from matching uint8 RGB images."""
    return float(PSNR(prediction, target)), float(calculate_ssim(prediction, target))


def find_paired_images(low_dir, gt_dir):
    """Return sorted (filename, low_path, gt_path) triples with exact pairing."""
    low_dir, gt_dir = Path(low_dir), Path(gt_dir)
    if not low_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError("Both --low_dir and --gt_dir must be directories")

    def indexed_images(directory):
        paths = [path for path in directory.iterdir()
                 if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
        names = [path.name for path in paths]
        folded = [name.casefold() for name in names]
        if len(set(folded)) != len(folded):
            raise ValueError(f"Duplicate image filenames in {directory}, ignoring case")
        return {path.name: path for path in paths}

    low, gt = indexed_images(low_dir), indexed_images(gt_dir)
    if not low:
        raise ValueError(f"No supported images found in {low_dir}")
    if set(low) != set(gt):
        missing_gt = sorted(set(low) - set(gt))
        extra_gt = sorted(set(gt) - set(low))
        raise ValueError(
            f"Image-pair mismatch; missing GT: {missing_gt}; extra GT: {extra_gt}"
        )
    return [(name, low[name], gt[name]) for name in sorted(low)]


def extract_checkpoint_state_dict(checkpoint):
    """Extract a state dict from common checkpoint wrappers without guessing keys."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping or contain a state dictionary")
    if checkpoint and all(isinstance(key, str) and torch.is_tensor(value)
                          for key, value in checkpoint.items()):
        state_dict = dict(checkpoint)
    else:
        state_dict = None
        for key in ("params", "model", "state_dict", "checkpoint"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                try:
                    state_dict = extract_checkpoint_state_dict(candidate)
                    break
                except ValueError:
                    continue
        if state_dict is None:
            raise ValueError(
                "Checkpoint has no supported state dictionary; expected direct tensor keys "
                "or one of params, model, state_dict, checkpoint"
            )
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint_strict(model, checkpoint_path, device, label):
    """Strictly load a model and return a concise, user-visible load message."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Older PyTorch releases do not accept weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_checkpoint_state_dict(checkpoint)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError(
            f"{label} checkpoint is incompatible with the configured RetinexFormer: {error}"
        ) from error
    return model.to(device).eval(), (
        f"Loaded {label} checkpoint: {checkpoint_path} ({len(state_dict)} tensors)"
    )


def _luminance(image):
    """Return Rec. 709 luminance from an RGB uint8 image in the [0, 1] range."""
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def litup_diagnostics(author_litup, retrained_litup, dark_threshold=0.10,
                      highlight_threshold=0.98):
    """Return neutral brightness and direct Lit-up-image comparison statistics."""
    if author_litup.shape != retrained_litup.shape:
        raise ValueError("Lit-up images must have matching dimensions")
    author_luma, retrained_luma = _luminance(author_litup), _luminance(retrained_litup)
    author_ssim = calculate_ssim(author_litup, retrained_litup)
    return {
        "author_litup_mean_luminance": float(np.mean(author_luma)),
        "retrained_litup_mean_luminance": float(np.mean(retrained_luma)),
        "litup_mean_luminance_delta": float(np.mean(retrained_luma) - np.mean(author_luma)),
        "author_litup_dark_ratio": float(np.mean(author_luma < dark_threshold)),
        "retrained_litup_dark_ratio": float(np.mean(retrained_luma < dark_threshold)),
        "author_litup_highlight_ratio": float(np.mean(author_luma > highlight_threshold)),
        "retrained_litup_highlight_ratio": float(np.mean(retrained_luma > highlight_threshold)),
        "author_vs_retrained_litup_mae": float(np.mean(
            np.abs(np.asarray(author_litup, dtype=np.float64) -
                   np.asarray(retrained_litup, dtype=np.float64)) / 255.0
        )),
        "author_vs_retrained_litup_ssim": float(author_ssim),
    }


def classify_litup_diagnostics(metrics, brightness_delta=0.02, ratio_delta=0.02):
    """Use neutral labels for measurable Lit-up changes rather than quality claims."""
    luminance_delta = metrics["litup_mean_luminance_delta"]
    if luminance_delta >= brightness_delta:
        label = "Brighter"
    elif luminance_delta <= -brightness_delta:
        label = "Darker"
    else:
        label = "Similar brightness"
    highlight_delta = (metrics["retrained_litup_highlight_ratio"] -
                       metrics["author_litup_highlight_ratio"])
    dark_delta = (metrics["retrained_litup_dark_ratio"] -
                  metrics["author_litup_dark_ratio"])
    flags = []
    if highlight_delta >= ratio_delta:
        flags.append("more clipped highlights")
    elif highlight_delta <= -ratio_delta:
        flags.append("fewer clipped highlights")
    if dark_delta >= ratio_delta:
        flags.append("more dark regions")
    elif dark_delta <= -ratio_delta:
        flags.append("fewer dark regions")
    return "; ".join((label, *flags))


def classify_output_delta(delta_psnr, threshold_db=0.5):
    """Classify final-output change using the documented PSNR threshold."""
    if delta_psnr >= threshold_db:
        return "Improved"
    if delta_psnr <= -threshold_db:
        return "Degraded"
    return "Similar"


def save_six_panel_grid(images, path):
    """Save the required comparison grid without resizing any image panel."""
    panels = (
        ("low", "Low"), ("author_litup", "Author Lit-up"),
        ("retrained_litup", "Retrained Lit-up"),
        ("author_output", "Author Output"),
        ("retrained_output", "Retrained Output"), ("gt", "Ground Truth"),
    )
    width, height = images["low"].shape[1], images["low"].shape[0]
    grid = Image.new("RGB", (width * len(panels), height + 28), "white")
    draw = ImageDraw.Draw(grid)
    for index, (key, title) in enumerate(panels):
        image = np.asarray(images[key])
        if image.shape[:2] != (height, width):
            raise ValueError(f"Panel size mismatch for {key}")
        grid.paste(Image.fromarray(image), (index * width, 28))
        draw.text((index * width + 4, 7), title, fill="black")
    grid.save(path, format="PNG")


def infer_full_image(model, low, with_intermediate=False):
    """Run a BCHW image, padding to the model's four-pixel size multiple."""
    if low.ndim != 4:
        raise ValueError("Expected a BCHW input tensor")
    height, width = low.shape[-2:]
    pad_h, pad_w = (-height) % 4, (-width) % 4
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        low = F.pad(low, (0, pad_w, 0, pad_h), mode=mode)

    if with_intermediate:
        if getattr(model, "stage", None) != 1:
            raise ValueError("Intermediate images require a single-stage RetinexFormer")
        intermediate, output = model.body[0].forward_with_intermediate(low)
        return intermediate[..., :height, :width], output[..., :height, :width]

    return model(low)[..., :height, :width]


def evaluate(model, loader, device):
    """Return mean (PSNR, SSIM) over all full-size image pairs in a loader."""
    was_training = model.training
    model.eval()
    psnr_scores, ssim_scores = [], []
    try:
        with torch.inference_mode():
            for batch in loader:
                outputs = infer_full_image(model, batch["low"].to(device))
                for prediction, target in zip(outputs, batch["high"]):
                    psnr, ssim = image_metrics(
                        tensor_to_uint8(prediction), tensor_to_uint8(target)
                    )
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim)
    finally:
        model.train(was_training)

    if not psnr_scores:
        raise ValueError("Evaluation loader is empty")
    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def _png_name(name):
    # Always use lossless PNG so the metrics describe the pixels on disk.
    name = Path(str(name)).name
    return name if name.lower().endswith(".png") else name + ".png"


def _save_rgb(array, path):
    Image.fromarray(array).save(path, format="PNG")


def save_rgb(array, path):
    """Save an RGB uint8 array as a lossless PNG for analysis artifacts."""
    _save_rgb(array, path)


def save_comparisons(model, dataset, indices, device, output_dir):
    """Save low, intermediate, output, GT, and a four-panel grid for indices."""
    output_dir = Path(output_dir)
    folders = {key: output_dir / key for key in ("low", "input_img", "output", "gt", "grid")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index in indices:
                sample = dataset[index]
                low = sample["low"].unsqueeze(0).to(device)
                intermediate, output = infer_full_image(model, low, with_intermediate=True)
                images = {
                    "low": tensor_to_uint8(low[0]),
                    "input_img": tensor_to_uint8(intermediate[0]),
                    "output": tensor_to_uint8(output[0]),
                    "gt": tensor_to_uint8(sample["high"]),
                }
                filename = _png_name(sample["name"])
                for key, array in images.items():
                    _save_rgb(array, folders[key] / filename)

                height, width = images["low"].shape[:2]
                grid = Image.new("RGB", (width * 4, height + 26), "white")
                draw = ImageDraw.Draw(grid)
                for column, (key, title) in enumerate((
                    ("low", "Low"), ("input_img", "Input Img"),
                    ("output", "Output"), ("gt", "Ground Truth"),
                )):
                    grid.paste(Image.fromarray(images[key]), (column * width, 26))
                    draw.text((column * width + 4, 5), title, fill="black")
                grid.save(folders["grid"] / filename, format="PNG")
    finally:
        model.train(was_training)
    return output_dir


def export_test_set(model, loader, device, output_dir):
    """Save full test images and per-image/average PSNR and SSIM to CSV."""
    output_dir = Path(output_dir)
    folders = {key: output_dir / key for key in ("low", "output", "gt")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                low_batch = batch["low"].to(device)
                output_batch = infer_full_image(model, low_batch)
                names = batch["name"]
                for low, output, target, name in zip(
                    low_batch, output_batch, batch["high"], names
                ):
                    filename = _png_name(name)
                    low_img = tensor_to_uint8(low)
                    output_img = tensor_to_uint8(output)
                    gt_img = tensor_to_uint8(target)
                    psnr, ssim = image_metrics(output_img, gt_img)
                    rows.append((filename, psnr, ssim))
                    _save_rgb(low_img, folders["low"] / filename)
                    _save_rgb(output_img, folders["output"] / filename)
                    _save_rgb(gt_img, folders["gt"] / filename)
    finally:
        model.train(was_training)

    if not rows:
        raise ValueError("Test loader is empty")
    averages = (float(np.mean([r[1] for r in rows])), float(np.mean([r[2] for r in rows])))
    csv_path = output_dir / "per_image_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "psnr", "ssim"))
        writer.writerows(rows)
        writer.writerow(())
        writer.writerow(("AVERAGE", *averages))
    return averages


def infer_full_image_stages(model, low):
    """Run every stage on a full BCHW image and remove inference padding."""
    if low.ndim != 4 or low.shape[1] != 3:
        raise ValueError("Expected a BCHW RGB input tensor")
    height, width = low.shape[-2:]
    pad_h, pad_w = (-height) % 4, (-width) % 4
    if pad_h or pad_w:
        mode = "reflect" if pad_h < height and pad_w < width else "replicate"
        low = F.pad(low, (0, pad_w, 0, pad_h), mode=mode)
    lightups, outputs = model.forward_with_intermediate(low)
    if not lightups or len(lightups) != len(outputs):
        raise ValueError("Model returned mismatched or empty stage results")
    return ([x[..., :height, :width] for x in lightups],
            [x[..., :height, :width] for x in outputs])


def _checked_rgb(tensor, label, expected_size):
    if tensor.ndim != 3 or tensor.shape[0] != 3 or tuple(tensor.shape[-2:]) != expected_size:
        raise ValueError(f"{label} has an unexpected RGB shape: {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{label} contains NaN or infinity")
    return tensor_to_uint8(tensor)


def evaluate_analysis_model(model, dataset, device, output_dir, limit=None):
    """Export stages and score light-up and output PNG pixels against GT.

    Returns one metric dictionary per image. Existing evaluation helpers remain
    available for callers of the training notebook.
    """
    output_dir = Path(output_dir)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    count = len(dataset) if limit is None else min(len(dataset), limit)
    if count == 0:
        raise ValueError("Test dataset is empty")
    rows = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index in range(count):
                sample = dataset[index]
                filename = _png_name(sample["name"])
                image_dir = output_dir / "images" / filename[:-4]
                low = sample["low"]
                gt = sample["high"]
                expected_size = tuple(low.shape[-2:])
                if tuple(gt.shape[-2:]) != expected_size:
                    raise ValueError(f"Image pair has different sizes: {filename}")
                if low.min().item() < 0 or low.max().item() > 1 or gt.min().item() < 0 or gt.max().item() > 1:
                    raise ValueError(f"Input or ground truth is outside [0,1]: {filename}")
                lightups, outputs = infer_full_image_stages(model, low.unsqueeze(0).to(device))
                images = {
                    "low.png": _checked_rgb(low, "low", expected_size),
                    "gt.png": _checked_rgb(gt, "ground truth", expected_size),
                }
                for stage_index, (lightup, output) in enumerate(zip(lightups, outputs), 1):
                    images[f"lightup_stage{stage_index}.png"] = _checked_rgb(
                        lightup[0], "light-up", expected_size
                    )
                    images[f"output_stage{stage_index}.png"] = _checked_rgb(
                        output[0], "stage output", expected_size
                    )
                images["lightup.png"] = images[f"lightup_stage{len(lightups)}.png"]
                images["output.png"] = images[f"output_stage{len(outputs)}.png"]
                lightup_psnr, lightup_ssim = image_metrics(
                    images["lightup.png"], images["gt.png"]
                )
                output_psnr, output_ssim = image_metrics(
                    images["output.png"], images["gt.png"]
                )
                if not all(math.isfinite(value) for value in (
                    lightup_psnr, lightup_ssim, output_psnr, output_ssim
                )):
                    raise ValueError(f"Non-finite metric for {filename}")
                image_dir.mkdir(parents=True, exist_ok=True)
                for image_name, array in images.items():
                    _save_rgb(array, image_dir / image_name)
                rows.append({
                    "filename": filename,
                    "lightup_psnr": lightup_psnr,
                    "lightup_ssim": lightup_ssim,
                    "output_psnr": output_psnr,
                    "output_ssim": output_ssim,
                    "stage_count": len(outputs),
                })
    finally:
        model.train(was_training)
    return rows


def write_analysis_metrics(rows, output_dir):
    """Write per-image stage metrics and return per-stage arithmetic means."""
    if not rows:
        raise ValueError("No image metrics to write")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        fields = (
            "filename", "lightup_psnr", "lightup_ssim", "output_psnr",
            "output_ssim", "stage_count",
        )
        writer.writerow(fields)
        writer.writerows(tuple(row[field] for field in fields) for row in rows)
    return {
        "lightup": {
            "psnr": float(np.mean([row["lightup_psnr"] for row in rows])),
            "ssim": float(np.mean([row["lightup_ssim"] for row in rows])),
        },
        "output": {
            "psnr": float(np.mean([row["output_psnr"] for row in rows])),
            "ssim": float(np.mean([row["output_ssim"] for row in rows])),
        },
    }


def _six_panel_grid(images, path):
    panels = (("low", "Low"), ("author_lightup", "Author Light-up"),
              ("author_output", "Author Output"), ("ours_lightup", "Our Light-up"),
              ("ours_output", "Our Output"), ("gt", "Ground Truth"))
    width, height = images["low"].size
    grid = Image.new("RGB", (width * len(panels), height + 28), "white")
    draw = ImageDraw.Draw(grid)
    for column, (key, label) in enumerate(panels):
        if images[key].size != (width, height):
            raise ValueError(f"Panel size mismatch for {path}: {key}")
        grid.paste(images[key], (column * width, 28))
        draw.text((column * width + 4, 7), label, fill="black")
    grid.save(path, format="PNG")


def compare_analysis_results(author_rows, ours_rows, author_dir, ours_dir,
                             output_dir, threshold=0.5, representative_count=3):
    """Join metrics by filename, group images, and create visual review assets."""
    if not math.isfinite(threshold) or threshold <= 0 or representative_count < 1:
        raise ValueError("threshold and representative_count must be positive")
    author = {row["filename"]: row for row in author_rows}
    ours = {row["filename"]: row for row in ours_rows}
    if len(author) != len(author_rows) or len(ours) != len(ours_rows):
        raise ValueError("Duplicate image filenames in evaluation")
    if set(author) != set(ours):
        raise ValueError("The models were evaluated on different filename sets")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for filename in sorted(author):
        a, o = author[filename], ours[filename]
        lightup_delta_psnr = o["lightup_psnr"] - a["lightup_psnr"]
        lightup_delta_ssim = o["lightup_ssim"] - a["lightup_ssim"]
        output_delta_psnr = o["output_psnr"] - a["output_psnr"]
        output_delta_ssim = o["output_ssim"] - a["output_ssim"]
        group = "improved" if output_delta_psnr >= threshold else (
            "degraded" if output_delta_psnr <= -threshold else "similar")
        comparisons.append({
            "filename": filename,
            "author_lightup_psnr": a["lightup_psnr"],
            "ours_lightup_psnr": o["lightup_psnr"],
            "delta_lightup_psnr": lightup_delta_psnr,
            "author_lightup_ssim": a["lightup_ssim"],
            "ours_lightup_ssim": o["lightup_ssim"],
            "delta_lightup_ssim": lightup_delta_ssim,
            "author_output_psnr": a["output_psnr"],
            "ours_output_psnr": o["output_psnr"],
            "delta_output_psnr": output_delta_psnr,
            "author_output_ssim": a["output_ssim"],
            "ours_output_ssim": o["output_ssim"],
            "delta_output_ssim": output_delta_ssim,
            "group": group,
        })
    fields = (
        "filename", "author_lightup_psnr", "ours_lightup_psnr",
        "delta_lightup_psnr", "author_lightup_ssim", "ours_lightup_ssim",
        "delta_lightup_ssim", "author_output_psnr", "ours_output_psnr",
        "delta_output_psnr", "author_output_ssim", "ours_output_ssim",
        "delta_output_ssim", "group",
    )
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)

    author_dir, ours_dir = Path(author_dir), Path(ours_dir)
    for row in comparisons:
        stem = row["filename"][:-4]
        destination = output_dir / row["group"] / stem
        destination.mkdir(parents=True, exist_ok=True)
        source_dirs = {"author": author_dir / "images" / stem,
                       "ours": ours_dir / "images" / stem}
        for shared in ("low.png", "gt.png"):
            shutil.copy2(source_dirs["author"] / shared, destination / shared)
        for label, source in source_dirs.items():
            for image_path in source.glob("lightup_stage*.png"):
                shutil.copy2(image_path, destination / f"{label}_{image_path.name}")
            for image_path in source.glob("output_stage*.png"):
                shutil.copy2(image_path, destination / f"{label}_{image_path.name}")
            for key in ("lightup", "output"):
                shutil.copy2(source / f"{key}.png", destination / f"{label}_{key}.png")
        with_images = {}
        for key in ("low", "author_lightup", "author_output", "ours_lightup",
                    "ours_output", "gt"):
            with Image.open(destination / f"{key}.png") as image:
                with_images[key] = image.convert("RGB")
        _six_panel_grid(with_images, destination / "comparison.png")

    selected = {
        "highest_improvement": sorted((r for r in comparisons if r["group"] == "improved"),
                                      key=lambda r: (-r["delta_output_psnr"], r["filename"]))[:representative_count],
        "largest_degradation": sorted((r for r in comparisons if r["group"] == "degraded"),
                                      key=lambda r: (r["delta_output_psnr"], r["filename"]))[:representative_count],
        "similar_performance": sorted((r for r in comparisons if r["group"] == "similar"),
                                      key=lambda r: (abs(r["delta_output_psnr"]), r["filename"]))[:representative_count],
    }
    lines = ["# Visual comparison review", "",
             "Inspect the linked grids before writing visual conclusions. Metrics use saved 8-bit RGB PNG pixels.", ""]
    for heading, rows in selected.items():
        lines.extend((f"## {heading.replace('_', ' ').title()}", ""))
        if not rows:
            lines.extend(("No images met this group's threshold.", ""))
        for row in rows:
            grid = f"{row['group']}/{row['filename'][:-4]}/comparison.png"
            lines.extend((f"### {row['filename']}", "",
                          f"Light-up — PSNR: author {row['author_lightup_psnr']:.4f}, ours {row['ours_lightup_psnr']:.4f}, delta {row['delta_lightup_psnr']:+.4f} dB; "
                          f"SSIM: author {row['author_lightup_ssim']:.4f}, ours {row['ours_lightup_ssim']:.4f}, delta {row['delta_lightup_ssim']:+.4f}.",
                          f"Output — PSNR: author {row['author_output_psnr']:.4f}, ours {row['ours_output_psnr']:.4f}, delta {row['delta_output_psnr']:+.4f} dB; "
                          f"SSIM: author {row['author_output_ssim']:.4f}, ours {row['ours_output_ssim']:.4f}, delta {row['delta_output_ssim']:+.4f}.", "",
                          f"![Six-panel comparison]({grid})", "",
                          "- Illumination recovery:", "- Structure and fine detail:",
                          "- Noise:", "- Color fidelity:", "- Highlight clipping or halos:",
                          "- Light-up versus final output differences:", ""))
    (output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    return comparisons, selected


def write_analysis_report(comparisons, author_average, ours_average, output_dir):
    """Write aggregate and per-image Markdown tables for a checkpoint comparison."""
    if not comparisons:
        raise ValueError("No comparison metrics to report")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RetinexFormer checkpoint comparison", "",
        "Both stages are measured against the paired ground-truth image using "
        "the saved, clipped, rounded 8-bit RGB PNG pixels.", "",
        "## Aggregate metrics", "",
        "| Stage | Author PSNR (dB) | Retrained PSNR (dB) | Δ PSNR | "
        "Author SSIM | Retrained SSIM | Δ SSIM |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage, title in (("lightup", "Light-up"), ("output", "Output")):
        author_metric = author_average[stage]
        ours_metric = ours_average[stage]
        lines.append(
            f"| {title} | {author_metric['psnr']:.4f} | {ours_metric['psnr']:.4f} | "
            f"{ours_metric['psnr'] - author_metric['psnr']:+.4f} | "
            f"{author_metric['ssim']:.4f} | {ours_metric['ssim']:.4f} | "
            f"{ours_metric['ssim'] - author_metric['ssim']:+.4f} |"
        )
    lines.extend((
        "", "## Per-image metrics", "",
        "| Image | Author light-up PSNR | Retrained light-up PSNR | Δ light-up PSNR | "
        "Author light-up SSIM | Retrained light-up SSIM | Δ light-up SSIM | "
        "Author output PSNR | Retrained output PSNR | Δ output PSNR | "
        "Author output SSIM | Retrained output SSIM | Δ output SSIM | Group |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ))
    for row in comparisons:
        lines.append(
            "| {filename} | {author_lightup_psnr:.4f} | {ours_lightup_psnr:.4f} | "
            "{delta_lightup_psnr:+.4f} | {author_lightup_ssim:.4f} | "
            "{ours_lightup_ssim:.4f} | {delta_lightup_ssim:+.4f} | "
            "{author_output_psnr:.4f} | {ours_output_psnr:.4f} | "
            "{delta_output_psnr:+.4f} | {author_output_ssim:.4f} | "
            "{ours_output_ssim:.4f} | {delta_output_ssim:+.4f} | {group} |".format(**row)
        )
    lines.extend((
        "", "## Visual review", "",
        "See [review.md](review.md) for representative six-panel grids. "
        "A light-up difference indicates a divergence before denoising; an "
        "output-only difference indicates divergence after that point. These "
        "metrics identify where to inspect, not a causal model defect.", "",
    ))
    (output_dir / "results.md").write_text("\n".join(lines), encoding="utf-8")
