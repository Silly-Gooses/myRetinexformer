"""Exact executed Phase 1 architecture; state-dictionary names are preserved."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


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
