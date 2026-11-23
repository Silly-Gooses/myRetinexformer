"""Fit Train-only local and joint fusion parameters."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_model.joint_lumachroma.cli import fit_main

if __name__ == "__main__":
    fit_main()
