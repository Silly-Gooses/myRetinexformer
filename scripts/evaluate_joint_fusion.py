"""Evaluate standalone backbones and all five frozen fusion ablations."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_model.joint_lumachroma.cli import evaluate_main

if __name__ == "__main__":
    evaluate_main()
