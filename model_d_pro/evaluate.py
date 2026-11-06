"""Original full-resolution validation and diagnostic smoke checks."""
import time
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .model import safe_hvi_to_rgb
from .losses import training_loss
from .runtime import write_json


@torch.inference_mode()
def evaluate(model, store, device, output_prefix, budget, label):
    from analysis_utils import image_metrics, tensor_to_uint8
    was_training = model.training
    model.eval()
    rows = []
    started = time.monotonic()
    prefix = Path(output_prefix)
    selected = set(store.val_names[i] for i in np.linspace(0, 68, 6, dtype=int))
    image_dir = prefix.parent / (prefix.name + "_images")
    image_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, name in enumerate(store.val_names, 1):
            budget.check()
            low, gt = store.images[name]
            x = torch.from_numpy(low).permute(2, 0, 1)[None].to(device).float() / 255
            out = model(x)
            if out.shape != x.shape or not torch.isfinite(out).all():
                raise FloatingPointError(f"Invalid full-image output: {name}")
            prediction = tensor_to_uint8(out[0])
            psnr, ssim = image_metrics(prediction, gt)
            rows.append(dict(name=name, psnr=psnr, ssim=ssim))
            if name in selected:
                Image.fromarray(prediction).save(image_dir / name)
            if index % 23 == 0:
                print(f"{label}: {index}/69 full RGB images", flush=True)
    finally:
        model.train(was_training)
    result = dict(label=label, count=69, psnr=float(np.mean([r["psnr"] for r in rows])),
                  ssim=float(np.mean([r["ssim"] for r in rows])),
                  elapsed_seconds=time.monotonic() - started,
                  dataset_sha256=store.val_sha, metric_protocol="original_RGB_uint8_per_image_mean",
                  test_evaluated=False)
    write_json(prefix.with_suffix(".json"), result)
    with prefix.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim"])
        w.writeheader()
        w.writerows(rows)
    print(f"{label}: PSNR {result['psnr']:.6f} | SSIM {result['ssim']:.6f}", flush=True)
    return result


def smoke_check(model, original, store, cfg, device):
    """Real-data forward/backward preflight, no optimizer updates."""
    from model_d.model import hvi_to_rgb
    name = store.train_names[0]
    a = store.images[name][0][:32, :32]
    x = torch.from_numpy(a.copy()).permute(2, 0, 1)[None].to(device).float() / 255
    model.eval()
    original.eval()
    with torch.no_grad():
        with torch.autocast(device.type, enabled=device.type == "cuda", dtype=torch.float16):
            direct = original.backbone(x)
        direct = direct.float().clamp(0, 1)
        float_reference = original.head(direct)[0]
        z = model(x)
        difference = float((z - float_reference).abs().max())
        # Adding zero residuals must match the original FLOAT path, not PNG path.
        if difference > 2e-5:
            raise AssertionError(f"Zero-start float-path parity failed: {difference}")
        hv = torch.randn(1, 2, 8, 8, device=device) * 0.02
        intensity = torch.rand(1, 1, 8, 8, device=device)
        inverse_difference = float((safe_hvi_to_rgb(hv, intensity) - hvi_to_rgb(hv, intensity)).abs().max())
        if inverse_difference > 1e-6:
            raise AssertionError("Safe inverse changed inference values")
    zero_hv = torch.zeros(1, 2, 8, 8, device=device, requires_grad=True)
    neutral_i = torch.full((1, 1, 8, 8), 0.2, device=device, requires_grad=True)
    safe_hvi_to_rgb(zero_hv, neutral_i).sum().backward()
    if not torch.isfinite(zero_hv.grad).all() or not torch.isfinite(neutral_i.grad).all():
        raise AssertionError("Nonfinite neutral-HVI gradients")
    model.set_phase(cfg["warmup_steps"], cfg["warmup_steps"])
    model.train()
    low, target = store.batch(0, 0, cfg, device)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    values = model(low, return_aux=True)
    loss, _ = training_loss(values, target)
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    gradients = {}
    for group in model.parameter_groups(cfg):
        found = [p.grad for p in group["params"] if p.grad is not None]
        finite = all(bool(torch.isfinite(g).all()) for g in found)
        nonzero = any(bool(g.abs().max() > 0) for g in found)
        gradients[group["group_name"]] = dict(finite=finite, any_nonzero=nonzero)
        if not finite or not nonzero:
            raise AssertionError(f"Gradient route failed: {group['group_name']}")
    report = dict(float_identity_max_abs=difference, inverse_max_abs=inverse_difference,
                  neutral_gradients_finite=True, gradient_groups=gradients,
                  micro_batch_seconds=time.monotonic() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None)
    model.zero_grad(set_to_none=True)
    return report


def make_comparisons(store, run, best_prefix):
    selected = [store.val_names[i] for i in np.linspace(0, 68, 6, dtype=int)]
    folder = run / "comparisons"
    folder.mkdir(exist_ok=True)
    for name in selected:
        base_file = run / "baseline_images" / name
        pro_file = run / (best_prefix + "_images") / name
        if not base_file.is_file() or not pro_file.is_file():
            continue
        low, gt = store.images[name]
        with Image.open(base_file) as f:
            baseline = np.asarray(f).copy()
        with Image.open(pro_file) as f:
            pro = np.asarray(f).copy()
        h, w = low.shape[:2]
        canvas = Image.new("RGB", (4 * w, h + 30), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (title, image) in enumerate(zip(["Low", "Verified Model D", "D-Pro pilot", "Ground truth"],
                                              [low, baseline, pro, gt])):
            canvas.paste(Image.fromarray(image), (i * w, 30))
            draw.text((i * w + 6, 7), title, fill="black")
        canvas.save(folder / name)
