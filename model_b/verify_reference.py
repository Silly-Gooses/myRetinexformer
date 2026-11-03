"""Local Test A: independent notebook builder versus repository Model B."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from RetinexFormer_arch import IG_MSA, RetinexFormer
from .compare import compare_evaluation
from .data import evaluate_model, load_pairs
from .experiment import load_reference_checkpoint, sha256
from .model import ARCH


def notebook_builder(path):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][4]["source"])
    namespace = dict(ARCH=ARCH, RetinexFormer=RetinexFormer, IG_MSA=IG_MSA)
    exec(compile(source, str(path) + ":cell4", "exec"), namespace)
    return namespace["build_model"]()[0]


def forward_equivalence(repo_model, legacy_model, image_path, *, device="cuda"):
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    x = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    h, w = x.shape[-2:]
    x = torch.nn.functional.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
    repo_model, legacy_model = repo_model.to(device).eval(), legacy_model.to(device).eval()
    with torch.no_grad(), torch.autocast("cuda", enabled=device == "cuda"):
        actual = repo_model(x)
        expected = legacy_model(x)
    return dict(shape=list(actual.shape),
                max_abs_difference=float((actual.float() - expected.float()).abs().max()))


def verify(data_root, source_path, option_path, reference_notebook, output_dir,
           split_file=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Reference evaluation requires CUDA and AMP")
    _, val_pairs, train_names, val_names = load_pairs(data_root, split_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(source_sha256=sha256(source_path),
                  option_a_sha256=sha256(option_path),
                  notebook=str(reference_notebook))
    for name, path in (("source", source_path), ("best", option_path)):
        model, checkpoint, replaced = load_reference_checkpoint(
            path, train_names, val_names,
            kind="source" if name == "source" else "option_a")
        legacy = notebook_builder(reference_notebook)
        legacy.load_state_dict(checkpoint["model"], strict=True)
        equivalent = forward_equivalence(model, legacy, val_pairs[0][1])
        if equivalent["max_abs_difference"] > 1e-6:
            raise AssertionError(f"{name} notebook/repository forward mismatch: {equivalent}")
        del legacy
        model = model.cuda()
        metrics, rows = evaluate_model(model, val_pairs)
        comparison = compare_evaluation(metrics, rows, reference=name,
                                        output_dir=output_dir)
        report[name] = dict(local_step=int(checkpoint["local_step"]),
                            replaced_modules=replaced,
                            forward_equivalence=equivalent,
                            evaluation=comparison)
        print(name, json.dumps(report[name]["evaluation"], indent=2))
        del model
        torch.cuda.empty_cache()
    (output_dir / "test_a_report.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--option-a", type=Path, required=True)
    parser.add_argument("--reference-notebook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path)
    args = parser.parse_args()
    result = verify(args.data_root, args.source, args.option_a,
                    args.reference_notebook, args.output_dir, args.split_file)
    if not all(result[name]["evaluation"]["aggregate_within_tolerance"]
               for name in ("source", "best")):
        raise SystemExit("Aggregate evaluation differs from the reference tolerance")


if __name__ == "__main__":
    main()
