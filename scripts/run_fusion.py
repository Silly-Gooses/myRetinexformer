"""Enhance one image or all images directly inside a folder."""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion_model.cli import add_model_arguments, build_predictor, create_output, read_rgb, save_prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--compare", action="store_true", help="Save all four methods in subfolders")
    args = parser.parse_args(argv)
    files = [args.input]
    if args.input.is_dir():
        files = sorted(p for p in args.input.iterdir() if p.is_file()
                       and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not files or any(not p.is_file() for p in files):
        parser.error("No input images found")
    if len({p.stem.casefold() for p in files}) != len(files):
        parser.error("Input filenames would produce duplicate PNG output names")
    predictor = build_predictor(args)
    create_output(args.output)
    for path in files:
        for method, prediction in predictor.predict(read_rgb(path), compare=args.compare).items():
            folder = args.output / method if args.compare else args.output
            save_prediction(folder / (path.stem + ".png"), prediction)
    print(f"Saved predictions for {len(files)} image(s) to {args.output}")


if __name__ == "__main__":
    main()
