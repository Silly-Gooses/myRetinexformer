"""Independent, hash-pinned executed-script comparisons; never starts training."""
import ast
from pathlib import Path

import torch

from model_d.model import ModelD, build_model_b, HVIColorHead
from .checkpoint import ROOT, SOURCE_SHA256
from .runtime import runtime_info, sha_file

REFERENCE_SCRIPT = ROOT / "tests/fixtures/dpro_pilot.py"


def pilot_definitions(path=REFERENCE_SCRIPT):
    path = Path(path)
    if sha_file(path) != SOURCE_SHA256:
        raise ValueError("Reference pilot script SHA256 mismatch")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = [node for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
                or (isinstance(node, ast.FunctionDef) and node.name not in {"main", "run_pilot"})]
    namespace = {"__file__": str(path), "STOP_REQUESTED": False}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def verify_model(model, image, checkpoint_info):
    definitions = pilot_definitions()
    reference = definitions["ModelDPro"](ModelD(build_model_b(), HVIColorHead()))
    reference.load_state_dict(model.state_dict(), strict=True)
    reference = reference.to(image.device).eval()
    was_training = model.training
    model.eval()
    try:
        expected = reference(image, return_aux=True)
        actual = model(image, return_aux=True)
        if actual.keys() != expected.keys():
            raise AssertionError("Auxiliary output keys differ")
        differences = {}
        for key in expected:
            if actual[key].shape != expected[key].shape:
                raise AssertionError(f"Output shape differs: {key}")
            if not torch.isfinite(actual[key]).all() or not torch.isfinite(expected[key]).all():
                raise FloatingPointError(f"Nonfinite parity output: {key}")
            differences[key] = float((actual[key] - expected[key]).abs().max())
    finally:
        model.train(was_training)
    tolerance = 2e-5 if image.device.type == "cuda" else 1e-6
    return dict(status="passed" if max(differences.values()) <= tolerance else "failed",
                max_abs_difference=differences, absolute_tolerance=tolerance,
                script_sha256=SOURCE_SHA256, checkpoints=checkpoint_info,
                runtime=runtime_info(image.device), shape=list(image.shape))
