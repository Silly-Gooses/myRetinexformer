import torch
import torch.nn.functional as F

from RetinexFormer_arch_og import RetinexFormer
from .checkpoint_utils import HVI_SHA, RETINEX_SHA, load_cidnet_class, load_verified_state
from .fixed_fusion import fixed_fusion
from .self_ensemble import ensemble8


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def reflect_pad(x, multiple):
    height, width = x.shape[-2:]
    bottom, right = (-height) % multiple, (-width) % multiple
    if bottom >= height or right >= width:
        raise ValueError(f"Input {height}x{width} is too small for reflection padding to {multiple}")
    return F.pad(x, (0, right, 0, bottom), "reflect") if bottom or right else x


class Predictor:
    def __init__(self, retinex, hvi, device):
        self.device = device
        self.retinex = retinex.eval().requires_grad_(False).to(device)
        self.hvi = hvi.eval().requires_grad_(False).to(device)

    @classmethod
    def from_checkpoints(cls, retinex_checkpoint, hvi_checkpoint, hvi_repo, device):
        retinex = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1, num_blocks=[1, 2, 2])
        hvi = load_cidnet_class(hvi_repo)()
        load_verified_state(retinex, retinex_checkpoint, RETINEX_SHA)
        load_verified_state(hvi, hvi_checkpoint, HVI_SHA)
        hvi.trans.gated = False
        hvi.trans.gated2 = True
        hvi.trans.alpha = 0.8
        torch.backends.cudnn.benchmark = False
        return cls(retinex, hvi, device)

    @torch.inference_mode()
    def predict(self, low_rgb8, *, compare=False):
        """Take a single HWC uint8 RGB image. Return float32 HWC RGB arrays."""
        x = torch.from_numpy(low_rgb8.copy())
        if x.ndim != 3 or x.shape[-1] != 3 or x.dtype != torch.uint8:
            raise ValueError("Expected an HWC RGB uint8 input")
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32) / 255.0
        height, width = x.shape[-2:]
        xr = reflect_pad(x, 4)
        outputs = {}
        if compare:
            outputs["normal"] = self.retinex(xr)[..., :height, :width].clamp(0, 1)
        outputs["tta8"] = ensemble8(xr, self.retinex)[..., :height, :width].clamp(0, 1)
        xh = reflect_pad(x, 8)
        outputs["hvi"] = self.hvi(xh.pow(1.0))[..., :height, :width].clamp(0, 1)
        for name, value in outputs.items():
            if value.shape != x.shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {name} output for input {height}x{width}")
        outputs["fusion"] = fixed_fusion(outputs["tta8"], outputs["hvi"])
        if not compare:
            outputs = {"fusion": outputs["fusion"]}
        return {k: v[0].permute(1, 2, 0).cpu().contiguous().numpy() for k, v in outputs.items()}
