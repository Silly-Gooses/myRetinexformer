"""Process-level official trainer, independent testing, and Colab launcher."""

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
import warnings

import numpy as np
from PIL import Image
import torch

from canny import CannyEdgeDetector
from RetinexFormer_arch import RetinexFormer
from training_utils import (EpochWarmupCosine, build_epoch_loaders, prepare_experiment,
                            run_epoch_training, seed_everything)
from train import main, validate_config, training_log

REPO = Path(__file__).resolve().parents[1]


from official_fixtures import dataset, configuration


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
        with tempfile.TemporaryDirectory(prefix='retinex CLI ') as temporary:
            root = Path(temporary)
            data = root/'datasets/LOLv2/Real_captured'
            dataset(data); dataset(data, 'Test')
            source = config_cell.replace('/content/drive/MyDrive', str(root))
            source = source.replace('PATCH_SIZE = 256', 'PATCH_SIZE = 16').replace('NUM_WORKERS = 2', 'NUM_WORKERS = 0')
            source = source.replace('"n_feat": 40', '"n_feat": 4').replace('"num_blocks": [1, 2, 2]', '"num_blocks": [0, 0, 0]')
            context = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, 'configuration', 'exec'), context)
                exec(compile(launcher_cell, 'launcher', 'exec'), context)
                exec(compile(smoke_cell, 'smoke', 'exec'), context)
                exec(compile(full_cell, 'full_skipped', 'exec'), context)
            experiment = context['FULL_ROOT']
            cp = torch.load(experiment/'checkpoints/last.pth', weights_only=True)
            self.assertEqual(cp['completed_epoch'], 3)
            self.assertEqual(cp['global_step'], 261)
            self.assertEqual(cp['config']['num_epochs'], 200)
            self.assertEqual(len(cp['manifest']['splits']['train']), 689)
            for row, expected in zip(cp['history'][1:], [1e-6, 5.05e-5, 1e-4]):
                self.assertAlmostEqual(row['lr'], expected)
            with (experiment/'logs/validation_log.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([r['kind'] for r in rows], ['initial', 'smoke_final'])
            self.assertEqual([int(r['count']) for r in rows], [100, 100])
            for label in ('low', 'edge', 'lit_up', 'output', 'gt'):
                self.assertEqual(len(list((experiment/'visuals/epoch_0003'/label).glob('*.png'))), 5)
            self.assertFalse((context['OUTPUT_ROOT']/context['EXPERIMENT_NAME']).exists())
            log_before = (experiment/'logs/training.log').read_text()
            before = (experiment/'checkpoints/last.pth').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError):
                context['launch_run']('smoke')
            self.assertEqual(before, (experiment/'checkpoints/last.pth').read_bytes())
            self.assertEqual(log_before, (experiment/'logs/training.log').read_text())
            context['RUN_MODE'] = 'full'
            with self.assertRaisesRegex(ValueError, 'configuration'):
                context['launch_run']('full')

    def test_cli_full_resume_and_independent_test_without_training_directory(self):
        with tempfile.TemporaryDirectory(prefix='retinex CLI ') as temporary:
            root = Path(temporary)
            data = root/'data'
            dataset(data); dataset(data, 'Test')
            cfg = configuration(data)
            config_file = root/'run config.json'
            config_file.write_text(json.dumps(cfg))
            self.invoke(config_file, root, 'official', '--mode', 'full')
            experiment = root/'official'
            cp_path = experiment/'checkpoints/last.pth'
            before = cp_path.read_bytes()
            cp = torch.load(cp_path, weights_only=True)
            self.assertEqual(cp['global_step'], 348)
            self.assertEqual([r['kind'] for r in cp['validation_history']], ['initial','final'])
            self.invoke(config_file, root, 'official', '--mode', 'full', '--resume')
            self.assertEqual(before, cp_path.read_bytes())
            changed = root/'changed.json'
            changed.write_text(json.dumps(dict(cfg, batch_size=4)))
            rejected = self.invoke(changed, root, 'official', '--mode', 'full', '--resume', success=False)
            self.assertIn('Resume configuration differs', rejected.stderr)
            # Testing accepts a selected checkpoint without Train or the last checkpoint.
            import shutil
            shutil.rmtree(data/'Train')
            weights = experiment/'checkpoints/best.pth'
            result = subprocess.run([sys.executable, str(REPO/'test.py'),
                '--checkpoint', str(weights), '--dataset-root', str(data),
                '--output-dir', str(root/'independent'), '--device', 'cpu'],
                cwd=REPO, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertEqual(len(list((root/'independent/enhanced_images').glob('*.png'))), 100)
            self.assertEqual(before, cp_path.read_bytes())

    def test_console_log_streams_appends_and_records_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            console, errors = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(errors):
                with training_log(root, 'fresh'):
                    print('epoch message မြန်မာ')
                    with warnings.catch_warnings():
                        warnings.simplefilter('always')
                        warnings.warn('example warning')
                    # Writes are visible while the run is still open.
                    saved = (root/'logs/training.log').read_text()
                    self.assertIn('epoch message မြန်မာ', saved)
                    self.assertIn('example warning', saved)
                before = (root/'logs/training.log').read_text()
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    with training_log(root, 'resume'):
                        print('continuing')
                        raise RuntimeError('training failed')
                self.assertIs(sys.stdout, console)
                self.assertIs(sys.stderr, errors)
                with self.assertRaises(KeyboardInterrupt):
                    with training_log(root, 'resume'):
                        raise KeyboardInterrupt()
            saved = (root/'logs/training.log').read_text()
            self.assertTrue(saved.startswith(before))
            self.assertIn('RuntimeError: training failed', saved)
            self.assertIn('KeyboardInterrupt', saved)
            self.assertIn('epoch message မြန်မာ', console.getvalue())
            self.assertIn('example warning', errors.getvalue())
            self.assertEqual(saved.count('session completed'), 1)

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
