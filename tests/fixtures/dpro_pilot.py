"""Model D-Pro budget pilot v1.

Research adaptation, NOT an official CIDNet/Multinex implementation.
Imports the pinned Model D repository; preserves its original checkpoints.
No Test images are read. See the companion notebook for configuration.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import signal
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw

VERSION = "dpro_budget_v1.0"
PINNED_COMMIT = "108b14cc0d84c070a8331772b99789359a4f3137"
EXPECTED_B = "1cb495fc19ce1560a1b50bd9eef16e10e3c3184351e9ac6fe03ccc31e1b91ae6"
EXPECTED_HEAD = "cebbc55c59906fc2d0be453f16660b3d0dadc14bae150ef97c8fcce29c331ca0"
EXPECTED_VAL_DATA = "c298929b9c8484c64d79f8aee9e80233ca437bfbe519c44a4a40c7b4bacc928f"
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("Stop requested: saving at the next safe boundary.", flush=True)


def sha_file(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def append_csv(path, row):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def atomic_torch_save(value, path):
    # Only our own tensors/plain metadata are saved; originals are never written.
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(cpu_tree(value), temp)
    os.replace(temp, path)


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, minutes, reserve_seconds=90):
        self.started = time.monotonic()
        self.deadline = self.started + minutes * 60 - reserve_seconds

    def check(self):
        if STOP_REQUESTED or time.monotonic() >= self.deadline:
            raise BudgetStop("session_time_budget_or_stop_request")

    def remaining(self):
        return max(0.0, self.deadline - time.monotonic())


class SafeAtan2(torch.autograd.Function):
    """Exact atan2 forward; bounded surrogate derivative near neutral H=V=0.

    This is an explicitly documented training stabilization, not a claim that
    hue is differentiable at the origin. Inference values are unchanged.
    """
    @staticmethod
    def forward(ctx, y, x):
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad):
        y, x = ctx.saved_tensors
        denominator = (x.square() + y.square()).clamp_min(1e-8)
        return grad * x / denominator, -grad * y / denominator


def safe_hvi_to_rgb(hv, intensity):
    from model_d.model import hsv_to_rgb
    intensity = intensity.float().clamp(0, 1)
    hv = hv.float()
    ck = torch.sin(math.pi * intensity / 2).clamp_min(1e-8)  # fixed k=1
    hue = torch.remainder(SafeAtan2.apply(hv[:, 1:2], hv[:, 0:1]) / (2 * math.pi), 1.0)
    sat = torch.linalg.vector_norm(hv, dim=1, keepdim=True) / ck
    return hsv_to_rgb(hue, sat, intensity)


def brightness_priors(rgb):
    weights = rgb.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    maximum = rgb.amax(1, keepdim=True)
    minimum = rgb.amin(1, keepdim=True)
    return torch.cat([(rgb * weights).sum(1, keepdim=True), maximum,
                      0.5 * (maximum + minimum)], 1)


def zero_conv(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class IlluminationMixer(nn.Module):
    """Keep the trained NAF mixer and add zero-start illumination modulation."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        dim = base.expand.in_channels
        hidden = min(32, max(8, dim // 4))
        self.norm = nn.LayerNorm(dim)
        self.guide = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                   zero_conv(nn.Conv2d(hidden, 2 * dim, 1)))

    def forward(self, x_in, illu_fea_trans=None):
        if illu_fea_trans is None:
            raise ValueError("Illumination features are required")
        original = self.base(x_in, illu_fea_trans)
        g = self.norm(illu_fea_trans.float()).permute(0, 3, 1, 2)
        gain, bias = self.guide(g).chunk(2, dim=1)
        residual = 0.1 * (torch.tanh(gain).permute(0, 2, 3, 1) * x_in
                          + torch.tanh(bias).permute(0, 2, 3, 1))
        return original + residual


class PooledContext(nn.Module):
    """64 pooled spatial tokens, not quadratic full-resolution attention."""
    def __init__(self, base, channels, width=32, heads=4, grid=8):
        super().__init__()
        self.base = base
        self.width, self.heads, self.grid = width, heads, grid
        self.reduce = nn.Conv2d(channels, width, 1)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = zero_conv(nn.Conv2d(width, channels, 1))

    def forward(self, x, illu_fea):
        x = self.base(x, illu_fea)
        h, w = x.shape[-2:]
        gh, gw = min(self.grid, h), min(self.grid, w)
        pooled = F.adaptive_avg_pool2d(self.reduce(x), (gh, gw))
        tokens = self.norm(pooled.flatten(2).transpose(1, 2).float())
        b, n, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(b, n, 3, self.heads, self.width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
        z = (attention.softmax(-1) @ v.float()).transpose(1, 2)
        z = z.reshape(b, n, self.width).transpose(1, 2).reshape(b, self.width, gh, gw)
        z = F.interpolate(z, size=(h, w), mode="bilinear", align_corners=False)
        return x + self.proj(z)


class ChannelCrossAttention(nn.Module):
    """Lightweight channel cross-attention; not the published LCA verbatim."""
    def __init__(self, width=24, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.q = nn.Conv2d(width, width, 1)
        self.kv = nn.Conv2d(width, 2 * width, 1)
        self.out = nn.Conv2d(width, width, 1)
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, query, context):
        b, c, h, w = query.shape
        q = self.q(query).reshape(b, self.heads, c // self.heads, h * w)
        k, v = self.kv(context).chunk(2, 1)
        k = k.reshape(b, self.heads, c // self.heads, h * w)
        v = v.reshape(b, self.heads, c // self.heads, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        a = (q @ k.transpose(-1, -2)) * self.temperature.clamp(0.1, 10.0)
        z = (a.softmax(-1) @ v).reshape(b, c, h, w)
        return query + self.out(z)


class DualBranchHead(nn.Module):
    def __init__(self, legacy, feature_channels=40, width=24):
        super().__init__()
        self.legacy = legacy
        self.feature_projection = nn.Conv2d(feature_channels, 8, 1)
        self.i_branch = nn.Sequential(nn.Conv2d(18, width, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(width, width, 3, padding=1, groups=width),
                                     nn.GELU(), nn.Conv2d(width, width, 1))
        self.hv_projection = nn.Conv2d(legacy.net[0].out_channels, width, 1)
        self.i_cross = ChannelCrossAttention(width)
        self.hv_cross = ChannelCrossAttention(width)
        self.i_output = zero_conv(nn.Conv2d(width, 1, 3, padding=1))
        self.hv_output = zero_conv(nn.Conv2d(width, 2, 3, padding=1))

    def forward(self, rgb, low, features):
        from model_d.model import rgb_to_hvi
        with torch.autocast(rgb.device.type, enabled=False):
            rgb, low, features = rgb.float(), low.float(), features.float()
            hv, intensity = rgb_to_hvi(rgb)
            z = torch.cat([hv, intensity], 1)
            # Original trained convolutions and exactly the original HV residual.
            for layer in list(self.legacy.net.children())[:-1]:
                z = layer(z)
            legacy_delta = self.legacy.max_delta * torch.tanh(self.legacy.net[-1](z))
            inputs = torch.cat([low, rgb, intensity, brightness_priors(low),
                                self.feature_projection(features)], 1)
            i_features = self.i_branch(inputs)
            hv_features = self.hv_projection(z)
            i_fused = self.i_cross(i_features, hv_features)
            hv_fused = self.hv_cross(hv_features, i_features)
            delta_i = 0.20 * torch.tanh(self.i_output(i_fused))
            delta_hv = 0.03 * torch.tanh(self.hv_output(hv_fused))
            corrected_i = (intensity + delta_i).clamp(0, 1)
            corrected_hv = hv + legacy_delta + delta_hv
            output = safe_hvi_to_rgb(corrected_hv, corrected_i)
            return dict(output=output, intensity=corrected_i, hv=corrected_hv,
                        delta_i=delta_i, delta_hv=delta_hv)


class ModelDPro(nn.Module):
    """Differentiable float pipeline, initialized from BOTH trained checkpoints."""
    def __init__(self, original):
        super().__init__()
        from model_d.model import NAFInspiredMixer
        self.backbone = copy.deepcopy(original.backbone)
        legacy = copy.deepcopy(original.head)
        self.backbone.requires_grad_(True)
        legacy.requires_grad_(True)
        self.original_backbone_ids = {id(p) for p in self.backbone.parameters()}
        self.original_head_ids = {id(p) for p in legacy.parameters()}
        if len(self.backbone.body) != 1:
            raise ValueError("This pilot requires the exact one-stage Model D backbone")
        count = 0
        for name, module in list(self.backbone.named_modules()):
            if isinstance(module, NAFInspiredMixer):
                parent_name, _, leaf = name.rpartition(".")
                parent = self.backbone.get_submodule(parent_name)
                wrapper = IlluminationMixer(module)
                if isinstance(parent, (nn.Sequential, nn.ModuleList)):
                    parent[int(leaf)] = wrapper
                else:
                    setattr(parent, leaf, wrapper)
                count += 1
        if count != 8:
            raise ValueError(f"Expected 8 original NAF mixers; found {count}")
        stage = self.backbone.body[0]
        dim = stage.denoiser.dim
        stage.denoiser.bottleneck = PooledContext(stage.denoiser.bottleneck,
                                                  dim * 2 ** stage.denoiser.level)
        self.prior_embed = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.GELU(),
                                         zero_conv(nn.Conv2d(16, dim, 1)))
        self.head = DualBranchHead(legacy, feature_channels=dim)

    def parameter_groups(self, cfg):
        groups = {"backbone": [], "legacy_head": [], "new": []}
        for p in self.parameters():
            key = ("backbone" if id(p) in self.original_backbone_ids else
                   "legacy_head" if id(p) in self.original_head_ids else "new")
            groups[key].append(p)
        return [dict(params=params, group_name=name, lr=cfg["learning_rates"][name])
                for name, params in groups.items()]

    def set_phase(self, completed, warmup_steps):
        original_on = completed >= warmup_steps
        for p in self.parameters():
            is_original = id(p) in self.original_backbone_ids or id(p) in self.original_head_ids
            p.requires_grad_(original_on if is_original else True)
        return "joint" if original_on else "new_modules_warmup"

    def forward(self, low, return_aux=False):
        h, w = low.shape[-2:]
        x = F.pad(low.float(), (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        stage = self.backbone.body[0]
        with torch.autocast(low.device.type, enabled=low.device.type == "cuda", dtype=torch.float16):
            illumination, illumination_map = stage.estimator(x)
            illumination = illumination + self.prior_embed(brightness_priors(x))
            lit_up = x * illumination_map + x
            edge = stage.edge_extractor(x)
            denoiser = stage.denoiser
            features = denoiser.embedding(torch.cat([lit_up, edge.to(lit_up.dtype)], 1))
            skips, illumination_skips = [], []
            for block, down, illu_down in denoiser.encoder_layers:
                features = block(features, illumination)
                skips.append(features)
                illumination_skips.append(illumination)
                features, illumination = down(features), illu_down(illumination)
            features = denoiser.bottleneck(features, illumination)
            for i, (up, fusion, block) in enumerate(denoiser.decoder_layers):
                features = fusion(torch.cat([up(features), skips[denoiser.level - 1 - i]], 1))
                features = block(features, illumination_skips[denoiser.level - 1 - i])
            enhanced = denoiser.mapping(features) + lit_up
        enhanced = enhanced[..., :h, :w].float().clamp(0, 1)
        # Intentional protocol change: NO internal uint8 conversion or detach.
        values = self.head(enhanced, low, features[..., :h, :w])
        values["backbone_rgb"] = enhanced
        return values if return_aux else values["output"]


def training_loss(values, target):
    from model_d.model import rgb_to_hvi
    target_hv, target_i = rgb_to_hvi(target)
    output_hv, _ = rgb_to_hvi(values["output"])
    parts = {
        "rgb": F.l1_loss(values["output"], target),
        "intensity": F.l1_loss(values["intensity"], target_i),
        "hv": F.l1_loss(output_hv, target_hv),
        "backbone": F.l1_loss(values["backbone_rgb"], target),
        "residual": values["delta_i"].abs().mean() + values["delta_hv"].abs().mean(),
    }
    loss = (parts["rgb"] + 0.10 * parts["intensity"] + 0.05 * parts["hv"]
            + 0.02 * parts["backbone"] + 0.002 * parts["residual"])
    return loss, parts


class PairedStore:
    """Read Train only, once into uint8 RAM (~1 GB for original 689 pairs)."""
    def __init__(self, root, train_names, val_names, patch, budget):
        from analysis_utils import find_paired_images
        root = Path(root)
        paired = find_paired_images(root / "Train/Low", root / "Train/Normal")
        indexed = {name: (low, gt) for name, low, gt in paired}
        names = train_names + val_names
        if (len(train_names) != 620 or len(val_names) != 69 or len(set(names)) != 689
                or set(names) != set(indexed)):
            raise ValueError("Dataset differs from the original 620/69 split")
        self.images, self.hashes = {}, {}
        def read_one(name):
            budget.check()
            arrays, hashes = [], []
            for path in indexed[name]:
                data = path.read_bytes()
                hashes.append(hashlib.sha256(data).hexdigest())
                with Image.open(io.BytesIO(data)) as im:
                    if im.mode != "RGB":
                        raise ValueError(f"Expected original 8-bit RGB file: {path}")
                    arrays.append(np.asarray(im).copy())
            if arrays[0].shape != arrays[1].shape or min(arrays[0].shape[:2]) < patch:
                raise ValueError(f"Bad shape/crop size: {name}")
            return name, arrays, hashes
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, (name, arrays, hashes) in enumerate(pool.map(read_one, names), 1):
                self.images[name] = arrays
                self.hashes[name] = hashes
                if i % 100 == 0 or i == len(names):
                    print(f"Data staging {i}/{len(names)} pairs (Train folders only)", flush=True)
        self.train_names, self.val_names = train_names, val_names
        self.train_sha = self.fingerprint(train_names)
        self.val_sha = self.fingerprint(val_names)
        self.bytes = sum(a.nbytes for pair in self.images.values() for a in pair)
        print(f"RGB data resident in RAM: {self.bytes / 2**30:.2f} GiB", flush=True)

    def fingerprint(self, names):
        d = hashlib.sha256()
        for name in names:
            d.update(json.dumps([name, *self.hashes[name]]).encode("utf-8"))
        return d.hexdigest()

    def batch(self, completed, micro, cfg, device):
        # Stateless sampling makes continuation independent of DataLoader workers.
        seed = cfg["seed"] * 1000003 + completed * 97 + micro
        rng = np.random.default_rng(seed)
        low_batch, gt_batch = [], []
        patch = cfg["patch"]
        for _ in range(cfg["micro_batch"]):
            name = self.train_names[int(rng.integers(len(self.train_names)))]
            low, gt = self.images[name]
            h, w = low.shape[:2]
            y, x = int(rng.integers(h - patch + 1)), int(rng.integers(w - patch + 1))
            low = low[y:y + patch, x:x + patch]
            gt = gt[y:y + patch, x:x + patch]
            k = int(rng.integers(4))
            flip = bool(rng.integers(2))
            low, gt = np.rot90(low, k), np.rot90(gt, k)
            if flip:
                low, gt = low[:, ::-1], gt[:, ::-1]
            low_batch.append(np.ascontiguousarray(low))
            gt_batch.append(np.ascontiguousarray(gt))
        def convert(batch):
            return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device).float() / 255
        return convert(low_batch), convert(gt_batch)


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


def export_small_report(run):
    target = run / "DPro_Pilot_Report.zip"
    temp = run / "DPro_Pilot_Report.zip.tmp"
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".csv", ".log", ".txt", ".py"}:
                z.write(path, path.relative_to(run))
        for path in sorted((run / "comparisons").glob("*.png")):
            z.write(path, path.relative_to(run))
    os.replace(temp, target)
    return target


def run_pilot(cfg, repo, session_minutes):
    import subprocess
    sys.path.insert(0, str(repo))
    from model_d import load_model_d
    from model_d.model import REFERENCE
    budget = Budget(session_minutes)
    device = torch.device("cuda")
    run = Path(cfg["output_root"]) / cfg["run_name"]
    run.mkdir(parents=True, exist_ok=True)
    state = dict(completed_steps=0, last_validated_step=0, best=None, best_step=0, best_prefix="step0",
                 patience=0, patience_reference=None, stop_reason="starting", finished=False,
                 total_train_seconds=0.0, full_validation_count=0)
    model = optimizer = scaler = store = None
    baseline = step0 = None
    identity = None
    last = run / "last.pth"
    run_owned = not (run / "identity.json").exists()
    start = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required. Select T4; CPU training is disabled.")
        if cfg["require_t4"] and "T4" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This cost-controlled preset requires T4. Change the Colab GPU to T4.")
        if cfg["micro_batch"] * cfg["accumulation"] != 8:
            raise ValueError("This preset uses effective batch 8; change settings deliberately in a new run.")
        if cfg["max_steps"] < cfg["warmup_steps"] or cfg["patch"] % 4:
            raise ValueError("Invalid warmup or patch configuration")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if commit != PINNED_COMMIT:
            raise ValueError("Repository commit differs from the reproduced baseline")
        if subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=repo, text=True).strip():
            raise ValueError("Tracked repository files were changed. Use a clean pinned checkout.")
        backbone_path, head_path = Path(cfg["model_b"]), Path(cfg["head"])
        for path, expected in [(backbone_path, EXPECTED_B), (head_path, EXPECTED_HEAD)]:
            if not path.is_file() or sha_file(path) != expected:
                raise ValueError(f"Missing/wrong ORIGINAL checkpoint (hash check failed): {path}")
        frozen = {"train": REFERENCE["config"]["train_names"], "val": REFERENCE["config"]["val_names"]}
        split = Path(cfg["split_file"])
        if split.is_file() and json.loads(split.read_text()) != frozen:
            raise ValueError("split.json differs from the preserved 620/69 split")
        if not split.is_file():
            print("External split.json absent; using the identical split embedded in the pinned repository.", flush=True)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        import cv2
        cv2.setNumThreads(1)
        store = PairedStore(cfg["data_root"], frozen["train"], frozen["val"], cfg["patch"], budget)
        if store.val_sha != EXPECTED_VAL_DATA:
            raise ValueError("Validation file CONTENTS differ from the previously reproduced 69 images")
        behavior_keys = ["seed", "patch", "micro_batch", "accumulation", "max_steps", "warmup_steps",
                         "validate_every", "save_every", "learning_rates", "patience_evals",
                         "min_early_stop_step", "patience_min_delta", "step0_max_abs_psnr_delta"]
        identity = dict(version=VERSION, commit=commit, script_sha256=sha_file(__file__),
                        backbone_sha256=EXPECTED_B, head_sha256=EXPECTED_HEAD,
                        train_sha256=store.train_sha, val_sha256=store.val_sha,
                        recipe={k: cfg[k] for k in behavior_keys})
        identity_path = run / "identity.json"
        if identity_path.is_file() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Run identity/recipe/source changed. Use a NEW run_name; do not mix results.")
        if not cfg["resume"] and last.is_file():
            raise FileExistsError("Existing run: enable resume or choose a NEW run_name")
        run_owned = True
        write_json(identity_path, identity)
        write_json(run / "config.json", cfg)
        write_json(run / "split.json", frozen)
        shutil.copy2(__file__, run / "dpro_pilot.py")
        runtime = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                       cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), commit=commit,
                       deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        write_json(run / "runtime.json", runtime)
        original, checkpoint_info = load_model_d(backbone_path, head_path, device=device)
        if (run / "baseline.json").is_file():
            baseline = json.loads((run / "baseline.json").read_text())
            if baseline.get("dataset_sha256") != store.val_sha or baseline.get("count") != 69:
                raise ValueError("Stored baseline provenance mismatch")
            print(f"Reusing this run's checked full validation baseline: {baseline['psnr']:.6f} dB", flush=True)
        else:
            baseline = evaluate(original, store, device, run / "baseline", budget, "Original Model D")
        expected = REFERENCE["metrics"]["val"]["model_d"]
        if abs(baseline["psnr"] - expected["psnr"]) > 0.002 or abs(baseline["ssim"] - expected["ssim"]) > 0.0001:
            raise RuntimeError("Baseline reproduction FAILED. No training was started; inspect baseline.json.")
        model = ModelDPro(original).to(device)
        groups = model.parameter_groups(cfg)
        write_json(run / "parameters.json", {g["group_name"]: sum(p.numel() for p in g["params"]) for g in groups})
        if not last.is_file():
            budget.check()
            smoke = smoke_check(model, original, store, cfg, device)
            write_json(run / "gradient_smoke.json", smoke)
            print("Gradient smoke test passed:", json.dumps(smoke), flush=True)
        del original
        torch.cuda.empty_cache()
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0, foreach=False)
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0)
        if last.is_file():
            payload = torch.load(last, map_location="cpu", weights_only=True)
            if payload["identity"] != identity:
                raise ValueError("Resume checkpoint identity mismatch")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            scaler.load_state_dict(payload["scaler"])
            state = payload["state"]
            torch.set_rng_state(payload["torch_rng"])
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
            # Copy optimizer slots to GPU after weights-only CPU loading.
            for slots in optimizer.state.values():
                for key, value in slots.items():
                    if torch.is_tensor(value) and key != "step":
                        slots[key] = value.to(device)
            step0 = json.loads((run / "step0.json").read_text())
            print(f"Resumed at successful optimizer update {state['completed_steps']}.", flush=True)
            del payload
        else:
            step0 = evaluate(model, store, device, run / "step0", budget, "D-Pro FLOAT step 0")
            if abs(step0["psnr"] - baseline["psnr"]) > cfg["step0_max_abs_psnr_delta"]:
                raise RuntimeError("Unexpectedly large FLOAT step-0 PSNR shift. Inspect step0.json; training blocked.")
            state.update(best=step0, best_step=0, patience_reference=step0["psnr"],
                         full_validation_count=1)
            atomic_torch_save(dict(model=model.state_dict(), identity=identity, metrics=step0, step=0), run / "best.pth")
        def save_last():
            atomic_torch_save(dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   scaler=scaler.state_dict(), state=copy.deepcopy(state),
                                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), last)
        save_last()
        # A time cap can interrupt validation after an optimizer update. On
        # resume, complete that pending full-set evaluation before more updates.
        done = state["completed_steps"]
        if (not state["finished"] and done > state.get("last_validated_step", 0)
                and (done % cfg["validate_every"] == 0 or done >= cfg["max_steps"])):
            prefix = f"val_{done:05d}"
            measured = evaluate(model, store, device, run / prefix, budget, f"Resumed validation {done}")
            state["last_validated_step"] = done
            state["full_validation_count"] += 1
            append_csv(run / "validation.csv", dict(step=done, psnr=measured["psnr"], ssim=measured["ssim"],
                                                     delta_vs_model_d=measured["psnr"] - baseline["psnr"]))
            if measured["psnr"] > state["best"]["psnr"]:
                state.update(best=measured, best_step=done, best_prefix=prefix)
                atomic_torch_save(dict(model=model.state_dict(), identity=identity, metrics=measured, step=done),
                                  run / "best.pth")
            if measured["psnr"] >= state["patience_reference"] + cfg["patience_min_delta"]:
                state["patience_reference"] = measured["psnr"]
                state["patience"] = 0
            else:
                state["patience"] += 1
            if done >= cfg["min_early_stop_step"] and state["patience"] >= cfg["patience_evals"]:
                state.update(stop_reason="pilot_plateau", finished=True)
            save_last()
        if state["finished"]:
            print("This run already stopped/completed. No additional GPU training is launched.", flush=True)
        else:
            state["stop_reason"] = "running"
            state.pop("error", None)
            consecutive_bad = 0
            while state["completed_steps"] < cfg["max_steps"]:
                budget.check()
                step = state["completed_steps"]
                phase = model.set_phase(step, cfg["warmup_steps"])
                model.train()
                # Schedule is tied to successful updates, so resuming does not reset it.
                progress = max(0.0, (step - cfg["warmup_steps"]) / max(1, cfg["max_steps"] - cfg["warmup_steps"]))
                factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))
                for group in optimizer.param_groups:
                    lr = cfg["learning_rates"][group["group_name"]] * factor
                    if step < cfg["warmup_steps"] and group["group_name"] != "new":
                        lr = 0.0
                    elif group["group_name"] == "new":
                        lr *= min(1.0, (step + 1) / 25)
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                totals = {k: 0.0 for k in ["loss", "rgb", "intensity", "hv", "backbone", "residual"]}
                for micro in range(cfg["accumulation"]):
                    budget.check()
                    low, target = store.batch(step, micro, cfg, device)
                    values = model(low, return_aux=True)
                    loss, parts = training_loss(values, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss; no optimizer update applied")
                    scaler.scale(loss / cfg["accumulation"]).backward()
                    totals["loss"] += float(loss.detach()) / cfg["accumulation"]
                    for name, value in parts.items():
                        totals[name] += float(value.detach()) / cfg["accumulation"]
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if not torch.isfinite(grad_norm):
                    # GradScaler skips corrupt updates; retries use a lower scale.
                    scaler.update(new_scale=max(1.0, scaler.get_scale() / 2))
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_bad += 1
                    print(f"AMP nonfinite gradients; skipped update; scale={scaler.get_scale()}", flush=True)
                    if consecutive_bad >= 3:
                        raise FloatingPointError("Three consecutive nonfinite gradient attempts")
                    continue
                scaler.step(optimizer)
                scaler.update()
                consecutive_bad = 0
                state["completed_steps"] += 1
                seconds = time.monotonic() - tick
                state["total_train_seconds"] += seconds
                done = state["completed_steps"]
                row = dict(step=done, phase=phase, **totals, grad_norm=float(grad_norm),
                           lr_backbone=optimizer.param_groups[0]["lr"], lr_head=optimizer.param_groups[1]["lr"],
                           lr_new=optimizer.param_groups[2]["lr"], seconds=seconds)
                append_csv(run / "steps.csv", row)
                if done % 25 == 0 or done == 1:
                    eta = (cfg["max_steps"] - done) * (state["total_train_seconds"] / done)
                    print(f"step {done}/{cfg['max_steps']} | {phase} | loss {totals['loss']:.5f} | "
                          f"train-only ETA {eta/60:.1f} min | session budget left {budget.remaining()/60:.1f} min", flush=True)
                if done % cfg["save_every"] == 0:
                    save_last()
                if done % cfg["validate_every"] == 0 or done == cfg["max_steps"]:
                    prefix = f"val_{done:05d}"
                    measured = evaluate(model, store, device, run / prefix, budget, f"D-Pro update {done}")
                    state["full_validation_count"] += 1
                    state["last_validated_step"] = done
                    append_csv(run / "validation.csv", dict(step=done, psnr=measured["psnr"], ssim=measured["ssim"],
                                                             delta_vs_model_d=measured["psnr"] - baseline["psnr"]))
                    if measured["psnr"] > state["best"]["psnr"]:
                        state.update(best=measured, best_step=done, best_prefix=prefix)
                        atomic_torch_save(dict(model=model.state_dict(), identity=identity, metrics=measured, step=done),
                                          run / "best.pth")
                    if measured["psnr"] >= state["patience_reference"] + cfg["patience_min_delta"]:
                        state["patience_reference"] = measured["psnr"]
                        state["patience"] = 0
                    else:
                        state["patience"] += 1
                    save_last()
                    if done >= cfg["min_early_stop_step"] and state["patience"] >= cfg["patience_evals"]:
                        state.update(stop_reason="pilot_plateau", finished=True)
                        break
            if state["completed_steps"] >= cfg["max_steps"]:
                state.update(stop_reason="max_steps", finished=True)
            save_last()
    except BudgetStop as error:
        state.update(stop_reason=str(error), finished=False)
        print("Budget pause. Resume will continue, not restart.", flush=True)
    except KeyboardInterrupt:
        state.update(stop_reason="interrupted", finished=False)
    except Exception as error:
        state.update(stop_reason="error", error=f"{type(error).__name__}: {error}", finished=False)
        trace = traceback.format_exc()
        (run / ("error.txt" if run_owned else f"preflight_error_{int(time.time())}.txt")).write_text(trace)
        print(trace, flush=True)
    finally:
        if not run_owned:
            print("Existing run identity was not accepted; its checkpoints and summary were not overwritten.", flush=True)
            return 1
        # Last known full validation chooses best; interrupted partial validations
        # never replace it. Preserve optimizer/sampler progress without saving grads.
        try:
            if model is not None and optimizer is not None and scaler is not None and state["best"] is not None:
                atomic_torch_save(dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       scaler=scaler.state_dict(), state=state, torch_rng=torch.get_rng_state(),
                                       cuda_rng=torch.cuda.get_rng_state_all()), last)
            delta = state["best"]["psnr"] - baseline["psnr"] if state["best"] is not None and baseline else None
            ssim_delta = state["best"]["ssim"] - baseline["ssim"] if delta is not None else None
            learned_delta = state["best"]["psnr"] - step0["psnr"] if state["best"] is not None and step0 else None
            if delta is None:
                decision = "NO_COMPLETE_CANDIDATE_VALIDATION"
            elif delta >= 0.05 and ssim_delta >= -0.002 and state["best_step"] > 0 and learned_delta is not None and learned_delta >= 0.02:
                decision = "PROMISING_PILOT_REVIEW_BEFORE_EXTENDING"
            elif delta > 0.002:
                decision = "SMALL_GAIN_OR_SSIM_TRADEOFF_REVIEW"
            else:
                decision = "NO_CLEAR_GAIN_IN_THIS_BUDGET_NOT_A_FINAL_ARCHITECTURE_VERDICT"
            summary = dict(version=VERSION, inference_protocol="DPro_float_interstage_RGB_uint8_metrics",
                           baseline_protocol="ModelD_PNG_interstage_RGB_uint8_metrics",
                           stop_reason=state["stop_reason"], error=state.get("error"),
                           successful_updates=state["completed_steps"], target_updates=cfg["max_steps"],
                           completed=state["finished"], baseline=baseline, candidate_float_step0=step0,
                           best_validation=state["best"], best_step=state["best_step"],
                           delta_psnr_vs_model_d=delta, delta_ssim_vs_model_d=ssim_delta,
                           delta_psnr_vs_candidate_step0=learned_delta,
                           decision=decision, test_evaluated=False, full_training=False,
                           thresholds_are_heuristic_not_statistical_significance=True,
                           final_test_target_db=23.1, final_test_target_achieved=None,
                           session_elapsed_seconds=time.monotonic()-start,
                           total_train_seconds=state["total_train_seconds"], output=str(run))
            write_json(run / "pilot_summary.json", summary)
            if store is not None and state["best"] is not None:
                make_comparisons(store, run, state["best_prefix"])
            artifact = export_small_report(run)
            print("\nRESULT:", json.dumps(summary, indent=2), flush=True)
            print("Report ZIP:", artifact, flush=True)
            print("Checkpoint files stay on Drive; they are NOT included in the small report ZIP.", flush=True)
        except Exception:
            print("Final export/save error:\n" + traceback.format_exc(), flush=True)
    return 1 if state["stop_reason"] == "error" else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--session-minutes", type=float, default=45)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    sys.exit(run_pilot(json.loads(args.config.read_text()), args.repo, args.session_minutes))


if __name__ == "__main__":
    main()
