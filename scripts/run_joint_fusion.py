"""Run frozen joint regularized fusion on an image or folder."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_model.joint_lumachroma.cli import run_main

if __name__ == "__main__":
    run_main()
