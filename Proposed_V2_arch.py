"""Adaptive feature-level Canny guidance, independent of the V1 architecture."""

import warnings

import cv2
import numpy as np
import torch
from torch import nn

from RetinexFormer_arch import Denoiser, Illumination_Estimator


class OpenCVCanny(nn.Module):
    """Deterministic OpenCV Canny on the original RGB model input."""

    def __init__(self, low_threshold=50, high_threshold=100):
        super().__init__()
        if not (0 <= low_threshold < high_threshold <= 255):
            raise ValueError("OpenCV Canny thresholds must satisfy 0 <= low < high <= 255")
        self.config = dict(low_threshold=low_threshold, high_threshold=high_threshold)

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Canny expects floating-point RGB [B, 3, H, W]")
        array = (rgb.detach().float().clamp(0, 1).mul(255).round().byte()
                 .permute(0, 2, 3, 1).cpu().numpy())
        result = []
        for image in array:
            gray = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, self.config["low_threshold"],
                             self.config["high_threshold"])
            result.append(torch.from_numpy(edge.copy()).unsqueeze(0))
        return torch.stack(result).to(device=rgb.device, dtype=torch.float32).div_(255)


class AdaptiveEdgeFusion(nn.Module):
    """Encode one-channel edges and gate their contribution to RGB features."""

    def __init__(self, channels, gated=True):
        super().__init__()
        self.gated = gated
        self.edge_encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.gate = nn.Conv2d(2 * channels, channels, 1) if gated else None

    def forward(self, rgb_features, edge_map):
        edge_features = self.edge_encoder(edge_map.to(dtype=rgb_features.dtype))
        gate = (torch.sigmoid(self.gate(torch.cat((rgb_features, edge_features), 1)))
                if self.gated else torch.ones_like(edge_features))
        return rgb_features + gate * edge_features, edge_features, gate


class AdaptiveDenoiser(Denoiser):
    """Original Denoiser with fusion directly after its RGB embedding."""

    def __init__(self, *, gated=True, **kwargs):
        super().__init__(**kwargs)
        self.fusion = AdaptiveEdgeFusion(self.dim, gated=gated)

    def forward(self, x, illu_fea, edge_map, *, return_features=False):
        rgb_features = self.embedding(x)
        fea, edge_features, gate = self.fusion(rgb_features, edge_map)
        fea_encoder, illu_fea_list = [], []
        for block, feature_down, illumination_down in self.encoder_layers:
            fea = block(fea, illu_fea)
            fea_encoder.append(fea)
            illu_fea_list.append(illu_fea)
            fea = feature_down(fea)
            illu_fea = illumination_down(illu_fea)
        fea = self.bottleneck(fea, illu_fea)
        for i, (feature_up, fuse_skip, block) in enumerate(self.decoder_layers):
            fea = feature_up(fea)
            fea = fuse_skip(torch.cat((fea, fea_encoder[self.level - 1 - i]), 1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = block(fea, illu_fea)
        output = self.mapping(fea) + x
        if return_features:
            return output, dict(rgb_features=rgb_features, edge_features=edge_features,
                                fused_features=rgb_features + gate * edge_features,
                                gate_map=gate)
        return output


class AdaptiveSingleStage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2,
                 num_blocks=(1, 2, 2), gated=True):
        super().__init__()
        if in_channels != 3 or out_channels != 3:
            raise ValueError("Adaptive edge fusion requires RGB input and output")
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = AdaptiveDenoiser(in_dim=3, out_dim=3, dim=n_feat,
                                         level=level, num_blocks=num_blocks, gated=gated)

    def forward(self, image, edge_map, *, capture=False):
        illu_fea, illu_map = self.estimator(image)
        lit_up = image * illu_map + image
        result = self.denoiser(lit_up, illu_fea, edge_map, return_features=capture)
        if capture:
            output, features = result
            return output, lit_up, features
        return result


class AdaptiveRetinexFormer(nn.Module):
    """E2 (gated=False) and E3 (gated=True) share the same edge encoder."""

    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=1,
                 num_blocks=(1, 2, 2), canny_config=None, gated=True):
        super().__init__()
        if not isinstance(stage, int) or stage < 1:
            raise ValueError("stage must be a positive integer")
        self.stage = stage
        self.gated = gated
        self.edge_extractor = OpenCVCanny(**(canny_config or {}))
        self.canny_config = dict(self.edge_extractor.config)
        self.body = nn.ModuleList([
            AdaptiveSingleStage(in_channels, out_channels, n_feat, 2, num_blocks, gated)
            for _ in range(stage)
        ])

    def _forward(self, image, *, capture=False):
        original = image
        # Extract once, before any illumination or residual stage changes the image.
        edge_map = self.edge_extractor(original).float()
        lightups, outputs, features = [], [], []
        for stage in self.body:
            if capture:
                image, lit_up, details = stage(image, edge_map, capture=True)
                lightups.append(lit_up)
                outputs.append(image)
                features.append(details)
            else:
                image = stage(image, edge_map)
        if capture:
            return dict(input=original, edge=edge_map, edge_map=edge_map,
                        lit_up=lightups[-1], output=image, enhanced=image,
                        gate_map=features[-1]["gate_map"],
                        rgb_features=features[-1]["rgb_features"],
                        edge_features=features[-1]["edge_features"],
                        fused_features=features[-1]["fused_features"],
                        lit_up_stages=lightups, output_stages=outputs,
                        feature_stages=features)
        return image

    def forward(self, image, return_intermediates=False):
        if return_intermediates:
            details = self._forward(image, capture=True)
            return details["output"], details
        return self._forward(image)

    def forward_with_intermediate(self, image, return_dict=False):
        details = self._forward(image, capture=True)
        return details if return_dict else (details["lit_up_stages"], details["output_stages"])

    def initialize_from(self, state_dict, *, source="baseline"):
        """Load a matching E0 or E1 state dict; keep new fusion weights initialized.

        Optimizer state from another architecture must never be resumed here.
        """
        if source not in ("baseline", "v1"):
            raise ValueError("source must be baseline or v1")
        expected = self.state_dict()
        new_keys = {key for key in expected if ".fusion." in key}
        old_keys = set(expected) - new_keys
        if set(state_dict) != old_keys:
            raise RuntimeError(f"Source keys differ: missing={sorted(old_keys-set(state_dict))}, "
                               f"unexpected={sorted(set(state_dict)-old_keys)}")
        converted = dict(expected)
        embeddings = {f"body.{i}.denoiser.embedding.weight" for i in range(self.stage)}
        for key in old_keys:
            source_value = state_dict[key]
            if key in embeddings and source == "v1":
                if tuple(source_value.shape) != (expected[key].shape[0], 4, *expected[key].shape[2:]):
                    raise RuntimeError(f"V1 embedding shape mismatch: {key}")
                source_value = source_value[:, :3]
            if not torch.is_tensor(source_value) or source_value.shape != expected[key].shape:
                raise RuntimeError(f"Source shape mismatch: {key}")
            converted[key] = source_value
        self.load_state_dict(converted, strict=True)
        warnings.warn(f"Initialized shared weights from {source}; {len(new_keys)} fusion tensors remain "
                      "newly initialized. Do not restore the source optimizer state.", UserWarning, stacklevel=2)
        return sorted(new_keys)
