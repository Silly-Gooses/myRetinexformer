"""Single-pass notebook inference, preserving padding and interleaved wide-image split."""
import numpy as np
import torch
import torch.nn.functional as F
from .color_fusion import dual_fuse, rgb_fuse


def all_predictions(rr, hh, local, joint):
    """Generate all ablations without access to ground truth."""
    return {
        "ret": rr,
        "hvi": hh,
        "rgb_fit": rgb_fuse(rr, hh, local["alpha_rgb"]),
        "ycc_fit": dual_fuse(rr, hh, local["alpha_y"], local["alpha_c"]),
        "ycc_regularized": dual_fuse(rr, hh, local["alpha_y_regularized"], local["alpha_c_regularized"]),
        "ycc_joint": dual_fuse(rr, hh, joint["alpha_y"], joint["alpha_c"]),
        "ycc_joint_regularized": dual_fuse(rr, hh, joint["alpha_y_regularized"], joint["alpha_c_regularized"]),
    }

@torch.inference_mode()
def predict_ret(net,rgb,device):
    h,w=rgb.shape[:2]
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    # Matches original RetinexFormer baseline: minimum /4 pad; very wide images use 2-way interleaved split.
    factor=4;split=(h>=3000 or w>=3000)
    ph=(-h)%factor;pw=(-w)%(8 if split else 4)
    if ph or pw:x=F.pad(x,(0,pw,0,ph),'reflect')
    if not split:y=net(x)
    else:
        ya=net(x[:,:,:,1::2]);yb=net(x[:,:,:,0::2]);y=torch.empty_like(x)
        y[:,:,:,1::2]=ya;y[:,:,:,0::2]=yb
    y=y[:,:,:h,:w].clamp(0,1)
    assert torch.isfinite(y).all()
    return y[0].permute(1,2,0).cpu().numpy().astype(np.float32)

@torch.inference_mode()
def predict_hvi(net,rgb,device):
    h,w=rgb.shape[:2]
    x=torch.from_numpy(rgb.copy()).permute(2,0,1).unsqueeze(0).float().to(device)/255
    ph,pw=(-h)%8,(-w)%8
    if ph or pw:x=F.pad(x,(0,pw,0,ph),'reflect')
    y=net(x)[...,:h,:w].clamp(0,1)
    assert torch.isfinite(y).all()
    return y[0].permute(1,2,0).cpu().numpy().astype(np.float32)
