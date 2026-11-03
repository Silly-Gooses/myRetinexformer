"""Frozen split, crop augmentation, and validation protocol from Option A."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from analysis_utils import find_paired_images, image_metrics, tensor_to_uint8


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"


def load_pairs(data_root, split_file=None, patch=128):
    data_root = Path(data_root)
    all_pairs = find_paired_images(data_root / "Train" / "Low",
                                   data_root / "Train" / "Normal")
    by_name = {pair[0]: pair for pair in all_pairs}
    frozen = json.loads((REFERENCE_DIR / "split.json").read_text(encoding="utf-8"))
    if split_file is not None:
        drive_split = json.loads(Path(split_file).read_text(encoding="utf-8"))
        if drive_split != frozen:
            raise ValueError("Drive split differs from the original frozen split")
    train_names, val_names = list(frozen["train"]), list(frozen["val"])
    if (len(train_names), len(val_names)) != (620, 69):
        raise ValueError("Expected the reference 620/69 split")
    if len(train_names) != len(set(train_names)) or len(val_names) != len(set(val_names)):
        raise ValueError("Duplicate split filenames")
    if set(train_names) & set(val_names) or set(train_names) | set(val_names) != set(by_name):
        raise ValueError("Split overlaps or does not match paired Train images")
    train_pairs = [by_name[name] for name in train_names]
    val_pairs = [by_name[name] for name in val_names]
    for name, low, gt in train_pairs + val_pairs:
        with Image.open(low) as image:
            low_size = image.size
        with Image.open(gt) as image:
            gt_size = image.size
        if low_size != gt_size:
            raise ValueError(f"Size mismatch: {name}")
        if min(low_size) < patch and name in train_names:
            raise ValueError(f"{name} is smaller than {patch}x{patch}")
    return train_pairs, val_pairs, train_names, val_names


class TrainPairs(Dataset):
    def __init__(self, pairs, patch):
        self.pairs, self.patch = pairs, patch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        name, low, gt = self.pairs[index]
        with Image.open(low) as image:
            a = image.convert("RGB")
        with Image.open(gt) as image:
            b = image.convert("RGB")
        if a.size != b.size:
            raise ValueError(f"Size mismatch: {name}")
        w, h = a.size
        p = self.patch
        if min(h, w) < p:
            raise ValueError(f"{name} is smaller than {p}x{p}")
        top = random.randrange(h - p + 1)
        left = random.randrange(w - p + 1)
        a, b = TF.crop(a, top, left, p, p), TF.crop(b, top, left, p, p)
        if random.random() < .5:
            a, b = TF.hflip(a), TF.hflip(b)
        if random.random() < .5:
            a, b = TF.vflip(a), TF.vflip(b)
        if random.random() < .5:
            a, b = TF.rotate(a, 90), TF.rotate(b, 90)
        return TF.to_tensor(a), TF.to_tensor(b)


@torch.no_grad()
def evaluate_model(model, val_pairs, *, device="cuda", use_amp=True):
    model.eval()
    rows = []
    for name, low, gt in val_pairs:
        with Image.open(low) as image:
            a = np.asarray(image.convert("RGB")).copy()
        with Image.open(gt) as image:
            b = np.asarray(image.convert("RGB")).copy()
        if a.shape != b.shape:
            raise ValueError(f"Size mismatch: {name}")
        x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 4, 0, (-h) % 4), mode="replicate")
        with torch.autocast("cuda", enabled=use_amp):
            out = model(x)[:, :, :h, :w]
        psnr, ssim = image_metrics(tensor_to_uint8(out.float()[0]), b)
        rows.append(dict(name=name, psnr=float(psnr), ssim=float(ssim)))
    model.train()
    return dict(psnr=float(np.mean([row["psnr"] for row in rows])),
                ssim=float(np.mean([row["ssim"] for row in rows])),
                count=len(rows)), rows
