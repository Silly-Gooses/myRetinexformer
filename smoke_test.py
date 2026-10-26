"""Audit official data, train two batches, validate, reload, and independently test.

Uses an isolated output directory and the production architecture/configuration.
The smoke weights are for testing only, never a resumable production run.
"""

import argparse
import json
from pathlib import Path

import torch

from train import build_model, resolve_device, validate_config, SCHEDULE_KEYS
from training_utils import seed_everything, train_one_epoch, EpochWarmupCosine
from official_workflow import (inspect_dataset, build_loaders, diagnostic_rng, save_weights,
                               storage_probe)
from protocol_metrics import evaluate_protocol, test_checkpoint, write_csv, protocol_metrics
from training_utils import save_epoch_visuals


def run_smoke(config, root, device, storage=None):
    validate_config(config)
    manifest, pairs = inspect_dataset(config["dataset_root"])
    print("Verified Train=689, validation=100, testing=100; no Train/Test overlap")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "checkpoints").mkdir()
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    seed_everything(config["seed"])
    train, val, generators = build_loaders(pairs, config, device)
    model = build_model(config, device)
    if config.get("version") == "v2":
        sample = next(iter(train))["low"][:1].to(device)
        prediction, details = model(sample, return_intermediates=True)
        channels = config["architecture"]["n_feat"]
        assert details["edge_map"].shape == (1, 1, *sample.shape[-2:])
        for key in ("rgb_features", "edge_features", "fused_features", "gate_map"):
            assert details[key].shape == (1, channels, *sample.shape[-2:])
        assert torch.isfinite(prediction).all() and details["gate_map"].min() >= 0
        assert details["gate_map"].max() <= 1
        prediction.abs().mean().backward()
        fusion = model.body[0].denoiser.fusion
        assert fusion.edge_encoder[0].weight.grad is not None
        assert torch.isfinite(fusion.edge_encoder[0].weight.grad).all()
        if config["gated"]:
            assert fusion.gate.weight.grad is not None
        model.zero_grad(set_to_none=True)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    settings = config["optimizer"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["warmup_start_lr"],
                                 betas=tuple(settings["betas"]), weight_decay=settings["weight_decay"])
    scheduler = EpochWarmupCosine(optimizer, **{key: config[key] for key in SCHEDULE_KEYS})
    scheduler.set_epoch(1)
    scaler = torch.amp.GradScaler("cuda", enabled=config["use_amp"])
    metrics, steps = train_one_epoch(model, train, optimizer, scaler, device,
                                     config["use_amp"], True, mixup_beta=config["mixup_beta"],
                                     grad_clip_norm=config["grad_clip_norm"], max_steps=2,
                                     metric_fn=protocol_metrics)
    assert steps == 2 and any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    after = {k: v.detach().clone() for k, v in model.state_dict().items()}
    with diagnostic_rng(generators):
        rows, summary = evaluate_protocol(model, val, device)
        if config.get("version") == "v2":
            save_epoch_visuals(model, val.dataset, [0], device, root / "visuals")
    assert model.training and all(torch.equal(after[k], v) for k, v in model.state_dict().items())
    write_csv(root / "validation_metrics.csv", ("image", "psnr", "ssim"), rows)
    snapshot = save_weights(root, model, config, 1, 2, 2, "quick_smoke")
    checkpoint = root / "checkpoints" / snapshot
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.equal(v.cpu(), restored["model"][k]) for k, v in after.items())
    test_checkpoint(checkpoint, config["dataset_root"], root / "testing", device)
    print("Storage probe passed:", storage_probe(storage or root))
    print("Smoke passed: two updates, validation without updates, checkpoint reload, independent testing")
    return metrics, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-storage", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    return run_smoke(config, args.output_dir, resolve_device(args.device, config["use_amp"]), args.check_storage)


if __name__ == "__main__":
    main()
