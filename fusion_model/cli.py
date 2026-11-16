from pathlib import Path

import cv2

from .inference import Predictor, select_device
from .metrics import rgb8


def add_model_arguments(parser):
    for name in ("retinex-checkpoint", "hvi-checkpoint", "hvi-repo", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_predictor(args):
    return Predictor.from_checkpoints(args.retinex_checkpoint, args.hvi_checkpoint,
                                      args.hvi_repo, select_device(args.device))


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_prediction(path, prediction):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.png")
    if not cv2.imwrite(str(temp), cv2.cvtColor(rgb8(prediction), cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write {temp}")
    temp.replace(path)


def create_output(path):
    # Each invocation owns its run directory, including on interrupted runs.
    path.mkdir(parents=True, exist_ok=False)
