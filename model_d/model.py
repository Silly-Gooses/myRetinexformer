"""Checkpoint-compatible Model D v1, ported from the original experiment."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from RetinexFormer_arch import IG_MSA, RetinexFormer
from analysis_utils import tensor_to_uint8


REFERENCE = json.loads(Path(__file__).with_name("reference.json").read_text())
ARCH = REFERENCE["config"]["arch"]
HEAD_WIDTH = REFERENCE["config"]["head_width"]
HVI_K = REFERENCE["config"]["hvi_k"]
MAX_DELTA = REFERENCE["config"]["max_delta"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class NAFInspiredMixer(nn.Module):
    """Exact notebook mixer; the enclosing IGAB supplies the residual."""

    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Conv2d(dim, 2 * dim, 1)
        self.dw = nn.Conv2d(2 * dim, 2 * dim, 3, padding=1, groups=2 * dim)
        self.sca = nn.Conv2d(dim, dim, 1)
        self.project = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, x_in, illu_fea_trans=None):
        x = x_in.permute(0, 3, 1, 2).contiguous()
        a, b = self.dw(self.expand(x)).chunk(2, dim=1)
        x = a * b
        x = x * self.sca(F.adaptive_avg_pool2d(x, 1))
        return self.project(x).permute(0, 2, 3, 1).contiguous()


def build_model_b():
    model = RetinexFormer(**ARCH, edge_guidance=True)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, IG_MSA):
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path)
            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(leaf)] = NAFInspiredMixer(module.dim)
            else:
                setattr(parent, leaf, NAFInspiredMixer(module.dim))
            replaced.append(name)
    if not replaced or any(isinstance(m, IG_MSA) for m in model.modules()):
        raise RuntimeError("Model B IGAB layout differs from the reference")
    return model


def rgb_to_hvi(rgb, k=HVI_K):
    """Original polarized hue/saturation mapping; float32, fixed k=1."""
    rgb = rgb.float().clamp(0, 1)
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    hi = rgb.max(1, keepdim=True).values
    lo = rgb.min(1, keepdim=True).values
    delta = hi - lo
    sat = torch.where(hi > 1e-8, delta / hi.clamp_min(1e-8), torch.zeros_like(hi))
    dr = (g - b) / delta.clamp_min(1e-8)
    dg = (b - r) / delta.clamp_min(1e-8) + 2
    db = (r - g) / delta.clamp_min(1e-8) + 4
    hue = torch.where(hi == r, dr, torch.where(hi == g, dg, db))
    hue = torch.remainder(hue / 6, 1.0)
    theta = 2 * math.pi * hue
    ck = torch.sin(math.pi * hi / 2).clamp_min(1e-8) ** (1 / k)
    hv = torch.cat([ck * sat * torch.cos(theta), ck * sat * torch.sin(theta)], 1)
    return hv, hi


def hsv_to_rgb(hue, sat, val):
    hue = torch.remainder(hue, 1.0)
    sat, val = sat.clamp(0, 1), val.clamp(0, 1)
    h6 = hue * 6
    c = val * sat
    x = c * (1 - torch.abs(torch.remainder(h6, 2) - 1))
    z = torch.zeros_like(c)
    i = torch.floor(h6).long().clamp(0, 5)
    r = torch.where(i == 0, c, torch.where(i == 1, x, torch.where(
        i == 2, z, torch.where(i == 3, z, torch.where(i == 4, x, c)))))
    g = torch.where(i == 0, x, torch.where(i == 1, c, torch.where(
        i == 2, c, torch.where(i == 3, x, z))))
    b = torch.where(i == 0, z, torch.where(i == 1, z, torch.where(
        i == 2, x, torch.where(i == 3, c, torch.where(i == 4, c, x)))))
    return torch.cat([r + val - c, g + val - c, b + val - c], 1).clamp(0, 1)


def hvi_to_rgb(hv, intensity, k=HVI_K):
    hv, intensity = hv.float(), intensity.float().clamp(0, 1)
    ck = torch.sin(math.pi * intensity / 2).clamp_min(1e-8) ** (1 / k)
    hue = torch.remainder(torch.atan2(hv[:, 1:2], hv[:, 0:1]) / (2 * math.pi), 1.0)
    sat = torch.linalg.vector_norm(hv, dim=1, keepdim=True) / ck.clamp_min(1e-8)
    return hsv_to_rgb(hue, sat, intensity)


class HVIColorHead(nn.Module):
    """Original four-convolution head, preserving checkpoint net.* names."""

    def __init__(self, width=HEAD_WIDTH, max_delta=MAX_DELTA):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(width, 2, 3, padding=1),
        )
        self.max_delta = max_delta
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rgb):
        # Disable enclosing autocast too: the reference head always uses FP32.
        with torch.autocast(rgb.device.type, enabled=False):
            hv, intensity = rgb_to_hvi(rgb)
            delta = self.max_delta * torch.tanh(self.net(torch.cat([hv, intensity], 1)))
            corrected = hv + delta
            return hvi_to_rgb(corrected, intensity), delta, corrected


def quantize_model_b(rgb):
    """Reproduce the notebook's PNG save/reload boundary, without disk I/O."""
    return torch.stack([
        torch.from_numpy(tensor_to_uint8(image)).permute(2, 0, 1)
        for image in rgb
    ]).to(device=rgb.device, dtype=torch.float32) / 255.0


class ModelD(nn.Module):
    """Full-image v1 inference. Use load_model_d for trained inference."""

    def __init__(self, backbone, head, *, backbone_amp=True):
        super().__init__()
        self.backbone, self.head = backbone, head
        self.backbone_amp = backbone_amp
        self.backbone.requires_grad_(False)

    def _forward(self, x, capture=False):
        if (x.ndim != 4 or x.shape[1] != 3 or not x.is_floating_point()
                or x.shape[0] < 1 or min(x.shape[-2:]) < 1):
            raise ValueError("Expected nonempty floating-point RGB [B, 3, H, W]")
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("Input RGB must be finite and in [0, 1]")
        h, w = x.shape[-2:]
        padded = F.pad(x.float(), (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        with torch.no_grad(), torch.autocast(
                x.device.type, enabled=self.backbone_amp and x.device.type == "cuda"):
            if capture:
                stages = self.backbone.forward_with_intermediate(padded, return_dict=True)
                enhanced = stages["output"]
            else:
                enhanced = self.backbone(padded)
        enhanced = enhanced[..., :h, :w].float().clamp(0, 1)
        head_input = quantize_model_b(enhanced)
        output, delta, _ = self.head(head_input)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Model D produced non-finite RGB")
        if capture:
            return dict(input=x, edge=stages["edge"][..., :h, :w],
                        lit_up=stages["lit_up"][..., :h, :w].float(),
                        model_b_rgb=enhanced, head_input=head_input,
                        hv_residual=delta, output=output)
        return output

    def forward(self, x):
        return self._forward(x)

    def forward_with_intermediate(self, x):
        return self._forward(x, capture=True)


def load_backbone(path):
    """Hash-check the original Option A checkpoint before unpickling it."""
    digest = sha256(path)
    if digest != REFERENCE["config"]["model_b_sha256"]:
        raise ValueError("Model B SHA256 differs from the original Model D v1 backbone")
    # Original Model B files contain NumPy/Python RNG state. Only the pinned,
    # hash-verified self-generated artifact is loaded with weights_only=False.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    for key in ("arch", "repo_commit", "train_names", "val_names"):
        if config.get(key) != REFERENCE["config"][key]:
            raise ValueError(f"Model B checkpoint config mismatch: {key}")
    model = build_model_b()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), dict(sha256=digest, local_step=checkpoint["local_step"])


def load_head(path, backbone_sha256):
    """Load the v1 best head; never substitute an initialized head on failure."""
    digest = sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Expected original Model D head/config checkpoint schema")
    config = checkpoint["config"]
    for key, expected in REFERENCE["config"].items():
        if config.get(key) != expected:
            raise ValueError(f"Model D v1 checkpoint config mismatch: {key}")
    if config["model_b_sha256"] != backbone_sha256:
        raise ValueError("Head was trained with a different Model B checkpoint")
    if (checkpoint.get("epoch") != REFERENCE["best_head_epoch"]
            or not math.isclose(float(checkpoint.get("best_val_psnr", float("nan"))),
                                REFERENCE["metrics"]["val"]["model_d"]["psnr"],
                                rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Expected the original v1 best head (epoch 17), not last.pth")
    head = HVIColorHead()
    head.load_state_dict(checkpoint["head"], strict=True)
    if any(not torch.isfinite(p).all() for p in head.parameters()):
        raise ValueError("Non-finite head checkpoint weights")
    return head.eval(), dict(sha256=digest, epoch=checkpoint["epoch"],
                             best_val_psnr=checkpoint["best_val_psnr"])


def load_model_d(backbone_path, head_path, *, device="cpu"):
    backbone, backbone_info = load_backbone(backbone_path)
    head, head_info = load_head(head_path, backbone_info["sha256"])
    model = ModelD(backbone, head).to(device).eval()
    return model, dict(model_b=backbone_info, head=head_info)
