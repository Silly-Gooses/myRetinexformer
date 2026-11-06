"""Original Phase 1 objective and coefficients."""
from torch.nn import functional as F


def training_loss(values, target):
    from model_d.model import rgb_to_hvi
    target_hv, target_i = rgb_to_hvi(target)
    output_hv, _ = rgb_to_hvi(values["output"])
    parts = {
        "rgb": F.l1_loss(values["output"], target),
        "intensity": F.l1_loss(values["intensity"], target_i),
        "hv": F.l1_loss(output_hv, target_hv),
        "backbone": F.l1_loss(values["backbone_rgb"], target),
        "residual": values["delta_i"].abs().mean() + values["delta_hv"].abs().mean(),
    }
    loss = (parts["rgb"] + 0.10 * parts["intensity"] + 0.05 * parts["hv"]
            + 0.02 * parts["backbone"] + 0.002 * parts["residual"])
    return loss, parts
