"""Official LOL-v2 evaluation conventions, separate from legacy pixel analysis.

Validation: BasicSR float64 RGB PSNR on clamped, unquantized predictions.
Testing: Enhancement/utils.py float32 PSNR; 100 dB for identical images.
Both: testing-script uint8, channel-averaged Gaussian SSIM (thesis addition
for validation). PNG quantization is NOT used to compute either PSNR.
"""

import csv
import hashlib
import io
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image

from analysis_utils import infer_full_image, tensor_to_uint8, extract_checkpoint_state_dict
from dataLoader import OfficialPairedDataset
from utils import calculate_ssim

VALIDATION_PROTOCOL = "official-val-float64-rgb-psnr_uint8-rgb-ssim"
TEST_PROTOCOL = "official-test-float32-rgb-psnr_uint8-rgb-ssim"


def protocol_metrics(output, target, *, testing=False):
    if output.shape != target.shape or output.ndim != 3 or output.shape[0] != 3:
        raise ValueError("Metrics require matching CHW RGB images")
    if min(output.shape[-2:]) < 11:
        raise ValueError("Gaussian SSIM requires images at least 11 x 11")
    if not torch.isfinite(output).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite evaluation image")
    # HWC matches the upstream NumPy reduction order as well as dtype.
    pred = output.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    gt = target.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    if testing:
        mse = np.mean((gt - pred) ** 2)
        psnr = 100. if mse == 0 else 10 * math.log10(1 / float(mse))
    else:
        mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
        psnr = float("inf") if mse == 0 else 20 * np.log10(1 / np.sqrt(mse))
    ssim = calculate_ssim(np.rint(pred * 255).astype(np.uint8),
                          np.rint(gt * 255).astype(np.uint8))
    return float(psnr), float(ssim)


def write_csv(path, fields, rows):
    from training_utils import _atomic_write
    def write(temp):
        with temp.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    _atomic_write(path, write)


@torch.inference_mode()
def evaluate_protocol(model, loader, device, *, testing=False, image_dir=None):
    """Score every pair without changing model state or its caller's mode."""
    was_training = model.training
    rows = []
    model.eval()
    try:
        for batch in loader:
            outputs = infer_full_image(model, batch["low"].to(device))
            for output, target, name in zip(outputs, batch["high"], batch["name"]):
                psnr, ssim = protocol_metrics(output, target.to(device), testing=testing)
                rows.append(dict(image=name, psnr=psnr, ssim=ssim))
                if image_dir is not None:
                    # Preserve the original extension in the stem to avoid collisions.
                    filename = name if name.lower().endswith(".png") else name + ".png"
                    Image.fromarray(tensor_to_uint8(output)).save(Path(image_dir) / filename)
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Empty evaluation dataset")
    return rows, dict(count=len(rows), psnr=float(np.mean([r["psnr"] for r in rows])),
                      ssim=float(np.mean([r["ssim"] for r in rows])))


def test_checkpoint(checkpoint_path, dataset_root, output_dir, device, config=None):
    """Independent testing needs only a checkpoint and the official Test pairs."""
    from official_workflow import inspect_dataset
    from train import build_model
    checkpoint_path, output_dir = Path(checkpoint_path), Path(output_dir)
    # Read once: an active trainer may atomically replace a best/last alias while
    # testing. Metadata, weights and provenance must refer to the same bytes.
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    stored = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if stored is not None:
        for key in ("architecture", "canny", "edge_guidance", "version", "gated"):
            if config is not None and config.get(key) != stored.get(key):
                raise ValueError(f"Explicit {key} differs from checkpoint metadata")
        config = stored
    if not isinstance(config, dict) or not all(k in config for k in ("architecture", "canny")):
        raise ValueError("Checkpoint has no model metadata; supply --config with architecture and canny")
    if config.get("edge_guidance") is not True and config.get("version") != "v2":
        raise ValueError("Independent testing requires Canny V1 or adaptive V2")
    manifest, pairs = inspect_dataset(dataset_root, training=False)
    model = build_model(config, device)
    state = extract_checkpoint_state_dict(checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint is incompatible with configured model: {error}") from error
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} ({len(state)} tensors)", flush=True)
    # Refuse overwrites, including incomplete previous exports.
    output_dir.mkdir(parents=True, exist_ok=False)
    images = output_dir / "enhanced_images"
    images.mkdir()
    loader = DataLoader(OfficialPairedDataset(pairs["test"]), batch_size=1,
                        shuffle=False, num_workers=0)
    rows, summary = evaluate_protocol(model, loader, device, testing=True, image_dir=images)
    write_csv(output_dir / "per_image_metrics.csv", ("image", "psnr", "ssim"), rows)
    summary.update(protocol=TEST_PROTOCOL, checkpoint=str(checkpoint_path.resolve()))
    write_csv(output_dir / "summary_metrics.csv", tuple(summary), [summary])
    from training_utils import _write_json, save_epoch_visuals
    provenance = dict(checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
                      checkpoint_format=checkpoint.get("format"),
                      global_step=checkpoint.get("global_step"),
                      architecture=config["architecture"], canny=config["canny"],
                      dataset=manifest, protocol=TEST_PROTOCOL,
                      training_protocol=config.get("dataset_protocol", "legacy-or-unspecified"),
                      selection_note="For official-v1.1 runs, Test is also used for validation/checkpoint selection; "
                                     "this is not an independent held-out evaluation.")
    _write_json(output_dir / "provenance.json", provenance)
    indices = np.linspace(0, len(loader.dataset)-1,
                          min(config.get("fixed_vis_count", 5), len(loader.dataset)), dtype=int)
    save_epoch_visuals(model, loader.dataset, indices, device, output_dir / "visuals")
    print(f"Test: {summary['count']} pairs | PSNR {summary['psnr']:.4f} | SSIM {summary['ssim']:.4f}")
    print("Results:", output_dir)
    return summary["psnr"], summary["ssim"]
