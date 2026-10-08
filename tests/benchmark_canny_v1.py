"""Warmed-up baseline/v1 forward-backward timings; suitable for Colab CUDA."""

import argparse
from pathlib import Path
import statistics
import sys
import time
import warnings

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from RetinexFormer_arch import RetinexFormer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    if args.size < 4 or args.size % 4 or min(args.batch_size, args.warmup, args.steps) < 1:
        parser.error("size must be a positive multiple of four; batch-size/warmup/steps must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.amp and args.device != "cuda":
        parser.error("--amp requires CUDA")
    torch.manual_seed(42)
    baseline = RetinexFormer(n_feat=40, stage=1, num_blocks=[1, 2, 2],
                            edge_guidance=False).to(args.device)
    v1 = RetinexFormer(n_feat=40, stage=1, num_blocks=[1, 2, 2]).to(args.device)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        v1.load_baseline_state_dict(baseline.state_dict())
    low = torch.rand(args.batch_size, 3, args.size, args.size, device=args.device) * 0.3
    target = torch.rand_like(low)

    def synchronize():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def measure(operation):
        for _ in range(args.warmup):
            operation()
        synchronize()
        times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            operation()
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
        return statistics.median(times)

    def train_step(model):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=args.device, enabled=args.amp):
            output = model(low)
            loss = torch.nn.functional.l1_loss(output, target)
        loss.backward()

    baseline_ms = measure(lambda: train_step(baseline))
    v1_ms = measure(lambda: train_step(v1))
    canny_ms = measure(lambda: v1.body[0].edge_extractor(low))
    edges = v1.body[0].edge_extractor(low)
    print(f"device={args.device}, torch={torch.__version__}, amp={args.amp}, "
          f"shape={tuple(low.shape)}, warmup={args.warmup}, steps={args.steps}")
    print(f"Baseline forward+L1+backward median: {baseline_ms:.3f} ms")
    print(f"V1 forward+L1+backward median:       {v1_ms:.3f} ms")
    print(f"Canny-only median:                  {canny_ms:.3f} ms")
    print(f"V1 relative overhead: {100 * (v1_ms / baseline_ms - 1):.2f}%")
    print(f"Edge density: {edges.float().mean().item():.4f}")
    print("Canny timings include full hysteresis and host convergence-scalar synchronization.")
    print("Synthetic low-light noise only: benchmark real training batches on the target GPU too.")


if __name__ == "__main__":
    main()
