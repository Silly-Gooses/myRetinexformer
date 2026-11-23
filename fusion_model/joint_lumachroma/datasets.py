"""Exact filename pairing; Train and Test may legitimately reuse basenames."""
import random
from pathlib import Path

from .config import CFG, CV_FOLDS, SPLIT_SEED

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def file_map(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Missing folder: {folder}")
    files = {f.name: f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in EXTS}
    if not files:
        raise ValueError(f"No images in {folder}")
    return files


def exact_pairs(low, gt, expected=None):
    lm, gm = file_map(low), file_map(gt)
    if lm.keys() != gm.keys():
        raise ValueError(f"INPUT/GT mismatch: {low}; only_low={sorted(lm.keys()-gm.keys())[:8]}; "
                         f"only_gt={sorted(gm.keys()-lm.keys())[:8]}")
    if expected is not None and len(lm) != expected:
        raise ValueError(f"Expected {expected} at {low}, found {len(lm)}")
    return [(lm[k], gm[k]) for k in sorted(lm)]


def dataset_configs(data_root, overrides=None):
    overrides = overrides or {}
    return {ds: {**c, "root": Path(overrides.get(ds, Path(data_root) / c["root"].name))}
            for ds, c in CFG.items()}


def discover_splits(configs, datasets, fit_count=None):
    if fit_count is not None and fit_count < CV_FOLDS * 5:
        raise ValueError("fit_train_count must be at least CV_FOLDS * 5")
    splits = {}
    for ds in datasets:
        c = configs[ds]
        root = c["root"]
        train = exact_pairs(root / c["train_low"], root / c["train_gt"])
        test = exact_pairs(root / c["test_low"], root / c["test_gt"], c["n_test"])
        # Only paths are inspected here. Test image pixels and GT never enter fitting.
        train_paths = {p.resolve() for pair in train for p in pair}
        test_paths = {p.resolve() for pair in test for p in pair}
        if train_paths & test_paths:
            raise ValueError("Train/Test path overlap")
        if fit_count is not None and len(train) < fit_count:
            raise ValueError(f"{ds}: not enough Train pairs")
        rng = random.Random(f"{SPLIT_SEED}:{ds}")
        chosen = (set(x[0].name for x in train) if fit_count is None else
                  set(rng.sample([x[0].name for x in train], fit_count)))
        splits[ds] = {"fit": [x for x in train if x[0].name in chosen], "test": test}
    return splits
