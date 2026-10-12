"""Process-level training entry point, notebook launch, and legacy resume checks."""

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from canny import CannyEdgeDetector
from RetinexFormer_arch import RetinexFormer
from training_utils import (EpochWarmupCosine, build_epoch_loaders, prepare_experiment,
                            run_epoch_training, seed_everything)
from train import main, validate_config

REPO = Path(__file__).resolve().parents[1]


def dataset(root, subset='Train', count=11):
    for kind in ('Low', 'Normal'):
        (root/subset/kind).mkdir(parents=True)
    rng = np.random.default_rng(4)
    for index in range(count):
        low = rng.integers(0, 90, (16, 20, 3), dtype=np.uint8)
        Image.fromarray(low).save(root/subset/'Low'/f'{index:02d}.png')
        Image.fromarray(low + 90).save(root/subset/'Normal'/f'{index:02d}.png')


def configuration(root):
    return json.loads(json.dumps(dict(
        version='v1', training_mode='epoch', seed=42, dataset_root=str(root), val_fraction=0.1,
        num_epochs=4, warmup_epochs=3, warmup_start_lr=1e-6, peak_lr=1e-4, min_lr=1e-6,
        batch_size=8, patch_size=16, num_workers=0, use_amp=False, vis_every=10, fixed_vis_count=5,
        optimizer=dict(name='Adam', betas=[0.9, 0.999], weight_decay=0.), loss='L1',
        edge_guidance=True, architecture=dict(in_channels=3, out_channels=3, n_feat=8,
                                             stage=1, num_blocks=[0, 0, 0]),
        canny=CannyEdgeDetector().config, deterministic=True)))


class TrainCLITests(unittest.TestCase):
    def assert_nested_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def invoke(self, config_file, output_root, name, *extra, success=True):
        result = subprocess.run(
            [sys.executable, '-u', str(REPO/'train.py'), '--config', str(config_file),
             '--output-root', str(output_root), '--experiment-name', name, '--device', 'cpu', *extra],
            cwd=REPO, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            text=True, capture_output=True, timeout=120)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_notebook_launches_smoke_and_propagates_errors(self):
        notebook = json.loads((REPO/'Lightweight_RetinexFormer_Colab_Full_Training_v2.ipynb').read_text())
        sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        config_cell = next(s for s in sources if 'DATASET_ROOT =' in s)
        launcher_cell = next(s for s in sources if 'def launch_run(' in s)
        smoke_cell = next(s for s in sources if s.startswith('# @title Smoke training'))
        full_cell = next(s for s in sources if s.startswith('# @title Full training'))
        evaluation_cell = next(s for s in sources if s.startswith('# Smoke checks never'))
        with tempfile.TemporaryDirectory(prefix='retinex CLI ') as temporary:
            root = Path(temporary)
            data = root/'datasets/LOLv2/Real_captured'
            dataset(data)
            source = config_cell.replace('/content/drive/MyDrive', str(root))
            source = source.replace('PATCH_SIZE = 256', 'PATCH_SIZE = 16').replace('NUM_WORKERS = 2', 'NUM_WORKERS = 0')
            context = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, 'configuration', 'exec'), context)
                exec(compile(launcher_cell, 'launcher', 'exec'), context)
                exec(compile(smoke_cell, 'smoke', 'exec'), context)
                exec(compile(full_cell, 'full_skipped', 'exec'), context)
                exec(compile(evaluation_cell, 'test_skipped', 'exec'), context)
            experiment = context['FULL_ROOT']
            cp = torch.load(experiment/'checkpoints/last.pth', weights_only=True)
            self.assertEqual(cp['completed_epoch'], 3)
            self.assertEqual(cp['config']['num_epochs'], 200)
            self.assertEqual(len(cp['history']), 3)
            for row, expected in zip(cp['history'], [1e-6, 5.05e-5, 1e-4]):
                self.assertAlmostEqual(row['lr'], expected)
            with (experiment/'logs/training_metrics.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r['epoch'] for r in rows], ['3'])
            self.assertTrue(rows[0]['train_psnr'] and rows[0]['train_ssim'])
            for label in ('low', 'edge', 'lit_up', 'output', 'gt'):
                self.assertEqual(len(list((experiment/'visuals/epoch_0003'/label).glob('*.png'))), 2)
            self.assertFalse((experiment/'final_test_outputs').exists())
            self.assertFalse((context['OUTPUT_ROOT']/context['EXPERIMENT_NAME']).exists())
            before = (experiment/'checkpoints/last.pth').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                context['launch_run']('smoke')
            self.assertEqual(before, (experiment/'checkpoints/last.pth').read_bytes())
            context['RUN_MODE'] = 'full'
            with self.assertRaisesRegex(ValueError, 'configuration'):
                context['launch_run']('full')

    def test_cli_resumes_legacy_notebook_state_and_exports_completed_full_run(self):
        with tempfile.TemporaryDirectory(prefix='retinex CLI ') as temporary:
            root = Path(temporary)
            data = root/'data'
            dataset(data)
            cfg = configuration(data)
            config_file = root/'run config.json'
            config_file.write_text(json.dumps(cfg))
            resumed_root = root/'resumed'
            # Original notebook setup, directly using helpers (no new entry point).
            seed_everything(42)
            names = sorted(p.name for p in (data/'Train/Low').glob('*.png'))
            split = prepare_experiment(resumed_root, cfg, names)
            loaders = build_epoch_loaders(data/'Train/Low', data/'Train/Normal', split,
                                         batch_size=8, patch_size=16, num_workers=0, seed=42)
            seed_everything(42)
            model = RetinexFormer(**cfg['architecture'], canny_config=cfg['canny'])
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.)
            scheduler = EpochWarmupCosine(optimizer, **{key: cfg[key] for key in
                ('num_epochs', 'warmup_epochs', 'warmup_start_lr', 'peak_lr', 'min_lr')})
            with contextlib.redirect_stdout(io.StringIO()):
                run_epoch_training(model, loaders[0], loaders[1], optimizer, scheduler,
                    torch.amp.GradScaler('cuda', enabled=False), torch.device('cpu'), cfg,
                    split, resumed_root, loaders[2], stop_after_epoch=1, log_every=3)
            last = resumed_root/'checkpoints/last.pth'
            before = last.read_bytes()
            rejected = self.invoke(config_file, root, 'resumed', '--mode', 'full', '--evaluate-only', success=False)
            self.assertIn('Complete the configured training run', rejected.stderr)
            self.assertEqual(before, last.read_bytes())
            changed = root/'changed.json'
            changed.write_text(json.dumps(dict(cfg, batch_size=4)))
            rejected = self.invoke(changed, root, 'resumed', '--mode', 'full', '--resume', success=False)
            self.assertIn('Resume configuration differs', rejected.stderr)
            self.assertEqual(before, last.read_bytes())
            self.invoke(config_file, root, 'resumed', '--mode', 'full', '--resume')
            self.invoke(config_file, root, 'uninterrupted', '--mode', 'full')
            resumed = torch.load(last, weights_only=True)
            complete = torch.load(root/'uninterrupted/checkpoints/last.pth', weights_only=True)
            for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'split', 'best', 'global_step'):
                self.assert_nested_equal(resumed[key], complete[key])
            for a, b in zip(resumed['history'], complete['history']):
                self.assertEqual({k:v for k,v in a.items() if 'seconds' not in k},
                                 {k:v for k,v in b.items() if 'seconds' not in k})
            with (resumed_root/'logs/training_metrics.csv').open() as file:
                self.assertEqual([r['epoch'] for r in csv.DictReader(file)], ['3', '4'])
            dataset(data, 'Test', 2)
            before = last.read_bytes()
            self.invoke(config_file, root, 'resumed', '--mode', 'full', '--evaluate-only')
            self.assertEqual(before, last.read_bytes())
            with (resumed_root/'final_test_outputs/per_image_metrics.csv').open() as file:
                rows = list(csv.reader(file))
            self.assertEqual(rows[-1][0], 'AVERAGE')
            self.assertEqual(len(list((resumed_root/'final_test_outputs/output').glob('*.png'))), 2)

    def test_invalid_config_and_smoke_evaluation_are_rejected(self):
        cfg = configuration(Path('/unused'))
        for key, value in (('patch_size', 15), ('num_workers', -1), ('vis_every', 0),
                           ('edge_guidance', False), ('loss', 'MSE')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(dict(cfg, **{key: value}))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['--config', '/unused.json', '--output-root', '/unused',
                  '--mode', 'smoke', '--evaluate-only'])


if __name__ == '__main__':
    unittest.main()
