"""Checkpoint-compatible Model B architecture from the reference notebook."""

import torch.nn as nn
import torch.nn.functional as F

from RetinexFormer_arch import IG_MSA, RetinexFormer


ARCH = dict(in_channels=3, out_channels=3, n_feat=40, stage=1,
            num_blocks=[1, 2, 2])


class NAFInspiredMixer(nn.Module):
    """The reference attention-free mixer; IGAB supplies its residual."""

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
        x = self.dw(self.expand(x))
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = x * self.sca(F.adaptive_avg_pool2d(x, 1))
        return self.project(x).permute(0, 2, 3, 1).contiguous()


def build_model():
    """Build Model B without modifying the shared RetinexFormer classes."""
    model = RetinexFormer(**ARCH, edge_guidance=True)
    replaced = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, IG_MSA):
            parent_path, _, leaf = name.rpartition('.')
            parent = model.get_submodule(parent_path)
            if isinstance(parent, (nn.ModuleList, nn.Sequential)):
                parent[int(leaf)] = NAFInspiredMixer(mod.dim)
            else:
                setattr(parent, leaf, NAFInspiredMixer(mod.dim))
            replaced.append(name)
    if not replaced or any(isinstance(mod, IG_MSA) for mod in model.modules()):
        raise RuntimeError("The v1 IG_MSA layout differs from the Model B reference")
    return model, replaced
