"""Reference float RGB and saved-RGB8 metrics, without GT-mean correction."""
import cv2
import numpy as np

def psnr_float(pred,gt):
    mse=float(np.mean((pred.astype(np.float64)-gt.astype(np.float64))**2))
    return float(-10*np.log10(max(mse,1e-14)))

def psnr_rgb8(pred8,gt8):
    a=pred8.astype(np.float32);b=gt8.astype(np.float32)
    return float(10*np.log10(255.**2/(float(np.mean((a-b)**2))+1e-8)))

def ssim_channel(x,y):
    a=x.astype(np.float64);b=y.astype(np.float64)
    kernel=np.outer(cv2.getGaussianKernel(11,1.5),cv2.getGaussianKernel(11,1.5).T)
    ma=cv2.filter2D(a,-1,kernel)[5:-5,5:-5];mb=cv2.filter2D(b,-1,kernel)[5:-5,5:-5]
    va=cv2.filter2D(a*a,-1,kernel)[5:-5,5:-5]-ma*ma
    vb=cv2.filter2D(b*b,-1,kernel)[5:-5,5:-5]-mb*mb
    cov=cv2.filter2D(a*b,-1,kernel)[5:-5,5:-5]-ma*mb
    c1=(.01*255)**2;c2=(.03*255)**2
    return float(np.mean(((2*ma*mb+c1)*(2*cov+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2))))

def ssim_rgb8(p,g):return float(np.mean([ssim_channel(p[:,:,i],g[:,:,i]) for i in range(3)]))

def rgb8_hvi_style(x):return (np.clip(x,0,1)*255).astype(np.uint8)

def rgb8_ret_style(x):return np.round(np.clip(x,0,1)*255).astype(np.uint8)
