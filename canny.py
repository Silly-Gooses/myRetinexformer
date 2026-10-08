"""Device-native, non-differentiable Canny guidance for RetinexFormer v1."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CannyEdgeDetector(nn.Module):
    def __init__(self, low_threshold=0.1, high_threshold=0.2,
                 kernel_size=5, sigma=1.0,
                 luminance_weights=(0.299, 0.587, 0.114)):
        super().__init__()
        if not (math.isfinite(low_threshold) and math.isfinite(high_threshold)
                and 0 < low_threshold < high_threshold):
            raise ValueError("Canny thresholds must be finite and 0 < low < high")
        if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("Canny kernel_size must be a positive odd integer")
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("Canny sigma must be positive and finite")
        if (len(luminance_weights) != 3
                or any(not math.isfinite(w) or w < 0 for w in luminance_weights)
                or not math.isclose(sum(luminance_weights), 1.0, abs_tol=1e-6)):
            raise ValueError("Canny luminance_weights must be three nonnegative weights summing to one")
        self.config = dict(low_threshold=low_threshold, high_threshold=high_threshold,
                           kernel_size=kernel_size, sigma=sigma,
                           luminance_weights=tuple(luminance_weights))

    @staticmethod
    def _non_maximum_suppression(magnitude, gx, gy):
        angle = torch.remainder(torch.atan2(gy, gx) * (180.0 / math.pi), 180.0)
        direction = torch.remainder(torch.floor((angle + 22.5) / 45.0), 4)
        h, w = magnitude.shape[-2:]
        padded = F.pad(magnitude, (1, 1, 1, 1))
        keep = torch.zeros_like(magnitude, dtype=torch.bool)
        # Asymmetric ties retain one side of a flat gradient ridge.
        for index, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
            ahead = padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
            behind = padded[..., 1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
            keep |= ((direction == index) & (magnitude > ahead)
                     & (magnitude >= behind))
        return magnitude * keep

    @staticmethod
    def _hysteresis(candidates, strong):
        edges = strong
        # Full eight-connected propagation. Only the convergence scalar is
        # synchronized with the host; image tensors never leave their device.
        while True:
            neighbors = F.max_pool2d(edges.float(), 3, stride=1, padding=1) > 0
            grown = edges | (candidates & neighbors)
            if torch.equal(grown, edges):
                return grown
            edges = grown

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Canny expects a floating-point [B, 3, H, W] tensor")
        if min(rgb.shape[-2:]) < 1:
            raise ValueError("Canny requires nonempty spatial dimensions")
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            image = rgb.detach().float().clamp(0, 1)
            weights = image.new_tensor(self.config["luminance_weights"]).view(1, 3, 1, 1)
            gray = (image * weights).sum(dim=1, keepdim=True)
            radius = self.config["kernel_size"] // 2
            coordinates = torch.arange(-radius, radius + 1, device=rgb.device, dtype=torch.float32)
            gaussian = torch.exp(-coordinates.square() / (2 * self.config["sigma"] ** 2))
            gaussian = gaussian / gaussian.sum()
            kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
            smooth = F.conv2d(F.pad(gray, (radius,) * 4, mode="replicate"), kernel)
            sobel_x = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel = torch.stack((sobel_x, sobel_x.t())).unsqueeze(1)
            gradients = F.conv2d(F.pad(smooth, (1,) * 4, mode="replicate"), sobel)
            gx, gy = gradients[:, :1], gradients[:, 1:]
            magnitude = torch.sqrt(gx.square() + gy.square())
            thin = self._non_maximum_suppression(magnitude, gx, gy)
            edges = self._hysteresis(thin >= self.config["low_threshold"],
                                     thin >= self.config["high_threshold"])
        return edges.to(dtype=rgb.dtype)
