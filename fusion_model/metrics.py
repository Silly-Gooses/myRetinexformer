import math

import cv2
import numpy as np


def rgb8(image):
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def float_psnr(gt, prediction):
    if gt.dtype != np.float32 or prediction.dtype != np.float32 or gt.shape != prediction.shape:
        raise ValueError("PSNR requires matching float32 RGB arrays")
    mse = float(np.mean((gt - np.clip(prediction, 0, 1)) ** 2))
    return 100.0 if mse == 0 else -10 * math.log10(mse)


_KERNEL = cv2.getGaussianKernel(11, 1.5)
_WINDOW = np.outer(_KERNEL, _KERNEL.T)


def _valid_mean(channel):
    return cv2.filter2D(channel, -1, _WINDOW)[5:-5, 5:-5]


def author_ssim(gt, prediction):
    if (gt.shape != prediction.shape or gt.ndim != 3 or gt.shape[2] != 3
            or min(gt.shape[:2]) < 11 or gt.dtype != np.uint8 or prediction.dtype != np.uint8):
        raise ValueError("SSIM requires matching RGB8 images at least 11x11")
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    scores = []
    for c in range(3):
        a, b = gt[:, :, c].astype(np.float64), prediction[:, :, c].astype(np.float64)
        ma, mb = _valid_mean(a), _valid_mean(b)
        va, vb = _valid_mean(a * a) - ma * ma, _valid_mean(b * b) - mb * mb
        cov = _valid_mean(a * b) - ma * mb
        scores.append(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) /
                              ((ma * ma + mb * mb + c1) * (va + vb + c2))))
    return float(np.mean(scores))
