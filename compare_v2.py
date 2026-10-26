"""Evaluate E0/E1/E2/E3 checkpoints with one shared LOL-v2 Real protocol.

Input manifest JSON maps each label to {"checkpoint": path, "config": path}.
V2 E2 config uses gated=false; E3 uses gated=true. E0 uses version=baseline.
"""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from analysis_utils import extract_checkpoint_state_dict, infer_full_image
from dataLoader import OfficialPairedDataset
from official_workflow import inspect_dataset
from protocol_metrics import evaluate_protocol, write_csv, TEST_PROTOCOL
from RetinexFormer_arch import RetinexFormer
from train import build_model, resolve_device


def load_model(config, checkpoint, device):
    if config.get("version") == "baseline":
        model = RetinexFormer(**config["architecture"], edge_guidance=False)
    else:
        model = build_model(config, device)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and isinstance(saved.get("config"), dict):
        source = saved["config"]
        for key in ("version", "architecture", "canny", "gated"):
            if source.get(key) != config.get(key):
                raise ValueError(f"{checkpoint}: checkpoint {key} differs from comparison config")
    model.load_state_dict(extract_checkpoint_state_dict(saved), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def time_model(model, image, device, repeats=10):
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    for _ in range(2):
        infer_full_image(model, image)
    sync()
    started = time.perf_counter()
    for _ in range(repeats):
        infer_full_image(model, image)
    sync()
    return (time.perf_counter() - started) / repeats * 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    if set(manifest) != {"E0", "E1", "E2", "E3"}:
        raise ValueError("Comparison manifest requires E0, E1, E2 and E3")
    configs = {label: json.loads(Path(item["config"]).read_text()) for label, item in manifest.items()}
    expected = {"E0": ("baseline", None), "E1": ("v1", None),
                "E2": ("v2", False), "E3": ("v2", True)}
    for label, config in configs.items():
        if (config.get("version"), config.get("gated")) != expected[label]:
            raise ValueError(f"{label} has wrong version/gated setting")
    _, pairs = inspect_dataset(args.dataset_root, training=False)
    loader = DataLoader(OfficialPairedDataset(pairs["test"]), batch_size=1, shuffle=False)
    device = resolve_device(args.device, False)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = []
    for label, item in manifest.items():
        model = load_model(configs[label], item["checkpoint"], device)
        rows, scores = evaluate_protocol(model, loader, device, testing=True)
        write_csv(args.output_dir / f"{label}_per_image.csv", ("image", "psnr", "ssim"), rows)
        sample = next(iter(loader))["low"].to(device)
        runtime = time_model(model, sample, device)
        summary.append(dict(experiment=label, psnr=scores["psnr"], ssim=scores["ssim"],
                            count=scores["count"], parameters=sum(p.numel() for p in model.parameters()),
                            runtime_ms=runtime, runtime_image=next(iter(loader))["name"][0],
                            protocol=TEST_PROTOCOL, device=str(device),
                            checkpoint=str(Path(item["checkpoint"]).resolve())))
        print(f"{label}: PSNR {scores['psnr']:.4f}, SSIM {scores['ssim']:.4f}, "
              f"params {summary[-1]['parameters']}, runtime {runtime:.2f} ms", flush=True)
    write_csv(args.output_dir / "summary.csv", tuple(summary[0]), summary)


if __name__ == "__main__":
    main()
