"""Small images with the real official split counts; never copied from user data."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def configuration(root):
    cfg = json.loads((REPO / 'official_lolv2_config.json').read_text())
    cfg.update(dataset_root=str(root), num_epochs=4, patch_size=16, num_workers=0,
               fixed_vis_count=2, val_freq_steps=1000)
    cfg['architecture'].update(n_feat=4, num_blocks=[0, 0, 0])
    return cfg


def dataset(root, subset='Train', count=None):
    count = (689 if subset == 'Train' else 100) if count is None else count
    for kind in ('Low', 'Normal'):
        (Path(root) / subset / kind).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4 if subset == 'Train' else 5)
    for index in range(count):
        low = rng.integers(0, 90, (13, 17, 3), dtype=np.uint8)
        Image.fromarray(low).save(Path(root) / subset / 'Low' / f'{index:04d}.png')
        Image.fromarray(low + 90).save(Path(root) / subset / 'Normal' / f'{index:04d}.png')
