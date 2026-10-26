"""Fast, dataset-free V2 integration smoke; run before the LOL-v2 data smoke."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from protocol_metrics import protocol_metrics
from train import build_model, resolve_device, validate_config
from training_utils import save_epoch_visuals, seed_everything


class SyntheticPair:
    def __init__(self, low, high):
        self.low, self.high = low.cpu(), high.cpu()

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return {"low": self.low, "high": self.high, "name": "synthetic.png"}

    def __len__(self):
        return 1


def run(config, directory, device):
    validate_config(config)
    if config["version"] != "v2":
        raise ValueError("Synthetic smoke requires a V2 config")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seed_everything(config["seed"])
    model = build_model(config, device)
    low = torch.rand(1, 3, 16, 20, device=device)
    high = torch.rand_like(low)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["peak_lr"])
    for _ in range(2):
        output, details = model(low, return_intermediates=True)
        channels = config["architecture"]["n_feat"]
        assert details["edge_map"].shape == (1, 1, 16, 20)
        assert all(details[key].shape == (1, channels, 16, 20)
                   for key in ("rgb_features", "edge_features", "fused_features", "gate_map"))
        assert details["gate_map"].min() >= 0 and details["gate_map"].max() <= 1
        assert torch.isfinite(output).all()
        loss = F.l1_loss(output, high)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        fusion = model.body[0].denoiser.fusion
        assert fusion.edge_encoder[0].weight.grad is not None
        if config["gated"]:
            assert fusion.gate.weight.grad is not None
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        expected = model(low)
        psnr, ssim = protocol_metrics(expected[0], high[0])
        assert torch.isfinite(torch.tensor([psnr, ssim])).all()
        save_epoch_visuals(model, SyntheticPair(low[0], high[0]), [0], device,
                           directory / "visuals")
    path = directory / "checkpoint.pth"
    torch.save(dict(model=model.state_dict(), config=config), path)
    restored = build_model(config, device).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"], strict=True)
    with torch.inference_mode():
        assert torch.equal(expected, restored(low))
    print(f"V2 synthetic smoke passed: two steps, PSNR={psnr:.4f}, SSIM={ssim:.4f}, "
          f"checkpoint={path}, visuals={directory / 'visuals'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    run(config, args.output_dir, resolve_device(args.device, False))


if __name__ == "__main__":
    main()
