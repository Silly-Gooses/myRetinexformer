"""RetinexFormer with zero-initialized spatial illumination guidance modulation."""
from pathlib import Path

import torch
from torch import nn

from RetinexFormer_arch import RetinexFormer as BaselineRetinexFormer
from RetinexFormer_arch import RetinexFormer_Single_Stage as BaselineStage
from analysis_utils import extract_checkpoint_state_dict


class SpatialIlluminationModulation(nn.Module):
    """Depthwise spatial gate; alpha=0 makes guidance exactly unchanged."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, illumination):
        return illumination + self.alpha * self.gate(illumination) * illumination


class SAIMStage(BaselineStage):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2,
                 num_blocks=(1, 1, 1)):
        super().__init__(in_channels, out_channels, n_feat, level, num_blocks)
        self.saim = SpatialIlluminationModulation(n_feat)

    def forward(self, img):
        illumination, illumination_map = self.estimator(img)
        lit_up = img * illumination_map + img
        return self.denoiser(lit_up, self.saim(illumination))

    def forward_with_intermediate(self, img):
        illumination, illumination_map = self.estimator(img)
        lit_up = img * illumination_map + img
        return lit_up, self.denoiser(lit_up, self.saim(illumination))


class RetinexFormer(nn.Module):
    """Drop-in baseline signature with per-stage SAIM guidance."""

    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3,
                 num_blocks=(1, 1, 1)):
        super().__init__()
        if not isinstance(stage, int) or stage < 1:
            raise ValueError('stage must be a positive integer')
        self.stage = stage
        self.body = nn.Sequential(*[
            SAIMStage(in_channels, out_channels, n_feat, 2, num_blocks)
            for _ in range(stage)
        ])

    def forward(self, x):
        return self.body(x)

    def forward_with_intermediate(self, x):
        lightups, outputs = [], []
        for stage in self.body:
            lit_up, x = stage.forward_with_intermediate(x)
            lightups.append(lit_up)
            outputs.append(x)
        return lightups, outputs


def parameter_report(model):
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'saim_parameters': sum(p.numel() for n, p in model.named_parameters() if '.saim.' in n),
    }


def load_baseline_checkpoint(model, checkpoint, *, baseline=None, verify_input=None,
                             atol=1e-6, rtol=1e-5):
    """Require exact original keys/shapes; initialize only explicit SAIM keys.

    Accepts a path, raw state dict, or supported params/model/state_dict wrapper.
    A supplied baseline is also loaded and compared on verify_input.
    """
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source = extract_checkpoint_state_dict(checkpoint)
    target = model.state_dict()
    new_keys = {key for key in target if key.startswith('body.') and '.saim.' in key}
    baseline_keys = set(target) - new_keys
    missing = sorted(baseline_keys - set(source))
    unexpected = sorted(set(source) - baseline_keys)
    bad_shapes = sorted(key for key in baseline_keys & set(source)
                        if not torch.is_tensor(source[key]) or source[key].shape != target[key].shape)
    if missing or unexpected or bad_shapes:
        raise RuntimeError(f'Incompatible baseline checkpoint: missing={missing}, '
                           f'unexpected={unexpected}, shape_mismatch={bad_shapes}')
    model.load_state_dict({**source, **{key: target[key] for key in new_keys}}, strict=True)
    if baseline is not None:
        baseline.load_state_dict(source, strict=True)
        if verify_input is None:
            verify_input = torch.rand(1, 3, 16, 20, device=next(model.parameters()).device)
        was_baseline, was_saim = baseline.training, model.training
        baseline.eval()
        model.eval()
        try:
            with torch.inference_mode():
                actual, expected = model(verify_input), baseline(verify_input)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        finally:
            baseline.train(was_baseline)
            model.train(was_saim)
    return {'loaded_baseline_tensors': len(baseline_keys),
            'initialized_saim_keys': sorted(new_keys),
            'baseline': parameter_report(baseline) if baseline is not None else None,
            'saim': parameter_report(model),
            'output_equivalence_verified': baseline is not None}
