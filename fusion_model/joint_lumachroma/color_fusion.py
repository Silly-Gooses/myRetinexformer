"""OpenCV float32 Y, Cr, Cb fusion; alpha values are RetinexFormer shares."""
import cv2
import numpy as np

def rgb_to_ycrcb(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb,dtype=np.float32),cv2.COLOR_RGB2YCrCb)

def ycrcb_to_rgb(ycc):
    return cv2.cvtColor(np.ascontiguousarray(ycc,dtype=np.float32),cv2.COLOR_YCrCb2RGB)

def dual_fuse(r,h,alpha_y,alpha_c):
    yr=rgb_to_ycrcb(r);yh=rgb_to_ycrcb(h)
    mixed=yh.copy()
    mixed[:,:,0]=alpha_y*yr[:,:,0]+(1.-alpha_y)*yh[:,:,0]
    mixed[:,:,1:]=alpha_c*yr[:,:,1:]+(1.-alpha_c)*yh[:,:,1:]
    return np.clip(ycrcb_to_rgb(mixed),0.,1.)

def rgb_fuse(r,h,alpha):return np.clip(alpha*r+(1.-alpha)*h,0.,1.)

def dual_components(r,h):
    # The RGB inverse color transform is (numerically) affine. Compute components
    # after conversion to avoid accidentally swapping Cr and Cb.
    yr=rgb_to_ycrcb(r);yh=rgb_to_ycrcb(h)
    b=ycrcb_to_rgb(yh)
    yonly=yh.copy();yonly[:,:,0]=yr[:,:,0]
    conly=yh.copy();conly[:,:,1:]=yr[:,:,1:]
    dy=ycrcb_to_rgb(yonly)-b
    dc=ycrcb_to_rgb(conly)-b
    return b,dy,dc

def dual_mse_coeffs(r,h,gt):
    b,dy,dc=dual_components(r,h)
    e=(b-gt).astype(np.float64)
    y=dy.astype(np.float64);c=dc.astype(np.float64)
    return {'Ayy':float(np.mean(y*y)), 'Acc':float(np.mean(c*c)),
            'Ayc':float(np.mean(y*c)), 'By':float(np.mean(y*e)),
            'Bc':float(np.mean(c*e)), 'C':float(np.mean(e*e))}

def dual_mse(z,ay,ac):
    return max(z['Ayy']*ay*ay+z['Acc']*ac*ac+2*z['Ayc']*ay*ac+
               2*z['By']*ay+2*z['Bc']*ac+z['C'],1e-14)
