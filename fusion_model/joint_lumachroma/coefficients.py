"""Unclipped affine sufficient statistics: do not substitute clipped Test scores."""
import numpy as np
from .color_fusion import dual_mse_coeffs, dual_mse

def mse_coefs(r,h,gt):
    # mean pixel MSE for alpha*r+(1-alpha)*h; GT floats in [0,1].
    d=r.astype(np.float64)-h.astype(np.float64)
    e=h.astype(np.float64)-gt.astype(np.float64)
    return float(np.mean(d*d)),float(np.mean(d*e)),float(np.mean(e*e))

def psnr_from_coeff(a,b,c,alpha):
    mse=a*alpha*alpha+2*b*alpha+c
    return float(-10*np.log10(max(mse,1e-14)))

def mean_rgb_psnr_approx(rows,alpha):
    a=float(alpha)
    return float(np.mean([-10*np.log10(max(z['rgb_a']*a*a+2*z['rgb_b']*a+z['rgb_c'],1e-14)) for z in rows]))

def mean_dual_psnr_approx(rows,ay,ac):
    return float(np.mean([-10*np.log10(dual_mse(z,ay,ac)) for z in rows]))
