"""Reusable full-image evaluation and image exports for Colab and local runs."""

import csv
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from utils import PSNR, calculate_ssim


def tensor_to_uint8(tensor):
    """Convert a CHW RGB tensor in [0, 1] to the pixels saved in PNG files."""
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("Expected a CHW RGB tensor")
    array = tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def image_metrics(prediction, target):
    """Calculate PSNR and SSIM from matching uint8 RGB images."""
    return float(PSNR(prediction, target)), float(calculate_ssim(prediction, target))


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
    """Export each stage and score final PNG pixels against matching GT PNG pixels.

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
                psnr, ssim = image_metrics(images["output.png"], images["gt.png"])
                if not math.isfinite(psnr) or not math.isfinite(ssim):
                    raise ValueError(f"Non-finite metric for {filename}")
                image_dir.mkdir(parents=True, exist_ok=True)
                for image_name, array in images.items():
                    _save_rgb(array, image_dir / image_name)
                rows.append({"filename": filename, "psnr": psnr, "ssim": ssim,
                             "stage_count": len(outputs)})
    finally:
        model.train(was_training)
    return rows


def write_analysis_metrics(rows, output_dir):
    """Write per-image metrics and return the two arithmetic means."""
    if not rows:
        raise ValueError("No image metrics to write")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("filename", "psnr", "ssim"))
        writer.writerows((row["filename"], row["psnr"], row["ssim"]) for row in rows)
    return {"psnr": float(np.mean([row["psnr"] for row in rows])),
            "ssim": float(np.mean([row["ssim"] for row in rows]))}


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
        delta_psnr = o["psnr"] - a["psnr"]
        delta_ssim = o["ssim"] - a["ssim"]
        group = "improved" if delta_psnr >= threshold else (
            "degraded" if delta_psnr <= -threshold else "similar")
        comparisons.append({"filename": filename, "author_psnr": a["psnr"],
                            "ours_psnr": o["psnr"], "delta_psnr": delta_psnr,
                            "author_ssim": a["ssim"], "ours_ssim": o["ssim"],
                            "delta_ssim": delta_ssim, "group": group})
    fields = ("filename", "author_psnr", "ours_psnr", "delta_psnr",
              "author_ssim", "ours_ssim", "delta_ssim", "group")
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
                                      key=lambda r: (-r["delta_psnr"], r["filename"]))[:representative_count],
        "largest_degradation": sorted((r for r in comparisons if r["group"] == "degraded"),
                                      key=lambda r: (r["delta_psnr"], r["filename"]))[:representative_count],
        "similar_performance": sorted((r for r in comparisons if r["group"] == "similar"),
                                      key=lambda r: (abs(r["delta_psnr"]), r["filename"]))[:representative_count],
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
                          f"PSNR: author {row['author_psnr']:.4f}, ours {row['ours_psnr']:.4f}, delta {row['delta_psnr']:+.4f} dB. "
                          f"SSIM: author {row['author_ssim']:.4f}, ours {row['ours_ssim']:.4f}, delta {row['delta_ssim']:+.4f}.", "",
                          f"![Six-panel comparison]({grid})", "",
                          "- Illumination recovery:", "- Structure and fine detail:",
                          "- Noise:", "- Color fidelity:", "- Highlight clipping or halos:",
                          "- Light-up versus final output differences:", ""))
    (output_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    return comparisons, selected
