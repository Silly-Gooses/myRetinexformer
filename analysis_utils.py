"""Reusable full-image evaluation and image exports for Colab and local runs."""

import csv
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
