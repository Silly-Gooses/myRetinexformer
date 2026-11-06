"""Original RAM staging and stateless paired crop sampling."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class PairedStore:
    """Read Train only, once into uint8 RAM (~1 GB for original 689 pairs)."""
    def __init__(self, root, train_names, val_names, patch, budget):
        from analysis_utils import find_paired_images
        root = Path(root)
        paired = find_paired_images(root / "Train/Low", root / "Train/Normal")
        indexed = {name: (low, gt) for name, low, gt in paired}
        names = train_names + val_names
        if (len(train_names) != 620 or len(val_names) != 69 or len(set(names)) != 689
                or set(names) != set(indexed)):
            raise ValueError("Dataset differs from the original 620/69 split")
        self.images, self.hashes = {}, {}
        def read_one(name):
            budget.check()
            arrays, hashes = [], []
            for path in indexed[name]:
                data = path.read_bytes()
                hashes.append(hashlib.sha256(data).hexdigest())
                with Image.open(io.BytesIO(data)) as im:
                    if im.mode != "RGB":
                        raise ValueError(f"Expected original 8-bit RGB file: {path}")
                    arrays.append(np.asarray(im).copy())
            if arrays[0].shape != arrays[1].shape or min(arrays[0].shape[:2]) < patch:
                raise ValueError(f"Bad shape/crop size: {name}")
            return name, arrays, hashes
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, (name, arrays, hashes) in enumerate(pool.map(read_one, names), 1):
                self.images[name] = arrays
                self.hashes[name] = hashes
                if i % 100 == 0 or i == len(names):
                    print(f"Data staging {i}/{len(names)} pairs (Train folders only)", flush=True)
        self.train_names, self.val_names = train_names, val_names
        self.train_sha = self.fingerprint(train_names)
        self.val_sha = self.fingerprint(val_names)
        self.bytes = sum(a.nbytes for pair in self.images.values() for a in pair)
        print(f"RGB data resident in RAM: {self.bytes / 2**30:.2f} GiB", flush=True)

    def fingerprint(self, names):
        d = hashlib.sha256()
        for name in names:
            d.update(json.dumps([name, *self.hashes[name]]).encode("utf-8"))
        return d.hexdigest()

    def batch(self, completed, micro, cfg, device):
        # Stateless sampling makes continuation independent of DataLoader workers.
        seed = cfg["seed"] * 1000003 + completed * 97 + micro
        rng = np.random.default_rng(seed)
        low_batch, gt_batch = [], []
        patch = cfg["patch"]
        for _ in range(cfg["micro_batch"]):
            name = self.train_names[int(rng.integers(len(self.train_names)))]
            low, gt = self.images[name]
            h, w = low.shape[:2]
            y, x = int(rng.integers(h - patch + 1)), int(rng.integers(w - patch + 1))
            low = low[y:y + patch, x:x + patch]
            gt = gt[y:y + patch, x:x + patch]
            k = int(rng.integers(4))
            flip = bool(rng.integers(2))
            low, gt = np.rot90(low, k), np.rot90(gt, k)
            if flip:
                low, gt = low[:, ::-1], gt[:, ::-1]
            low_batch.append(np.ascontiguousarray(low))
            gt_batch.append(np.ascontiguousarray(gt))
        def convert(batch):
            return torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device).float() / 255
        return convert(low_batch), convert(gt_batch)
