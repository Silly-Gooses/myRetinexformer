"""Epoch schedule, split isolation, checkpoint recovery and reproducible resume."""

import csv
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from RetinexFormer_arch import RetinexFormer
from analysis_utils import image_metrics, tensor_to_uint8
from training_utils import (
    EpochWarmupCosine, _atomic_write, build_epoch_loaders, epoch_learning_rate,
    load_epoch_checkpoint, make_split, prepare_experiment, run_epoch_training,
    save_epoch_visuals, seed_everything, seed_worker, train_one_epoch,
    validate_epoch, validate_split, should_log_epoch,
)


class RandomSamples(Dataset):
    def __init__(self, train=True):
        self.train = train

    def __len__(self):
        return 5 if self.train else 2

    def __getitem__(self, index):
        low = torch.full((3, 12, 12), 0.1 + index / 20)
        if self.train:
            # All three RNG sources must continue across save/resume.
            low += torch.rand_like(low) * 0.02 + random.random() * 0.02 + np.random.rand() * 0.02
        return {"low": low, "high": torch.full_like(low, 0.6), "name": f"{index}.png"}


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.dropout = torch.nn.Dropout(0.1)
        self.seen = []

    def forward(self, image):
        if self.training:
            self.seen.append(image.detach().clone())
        return self.conv(self.dropout(image))


def config(workers=0):
    return dict(seed=42, val_fraction=0.1, num_epochs=4, warmup_epochs=2,
                warmup_start_lr=1e-6, peak_lr=1e-4, min_lr=1e-6,
                batch_size=2, num_workers=workers, use_amp=False,
                fixed_vis_count=2, vis_every=10, architecture="tiny")


def components(cfg):
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['warmup_start_lr'])
    scheduler = EpochWarmupCosine(optimizer, **{k: cfg[k] for k in (
        'num_epochs', 'warmup_epochs', 'warmup_start_lr', 'peak_lr', 'min_lr')})
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    generators = {"train": torch.Generator().manual_seed(42),
                  "validation": torch.Generator().manual_seed(43)}
    loaders = [DataLoader(RandomSamples(train), batch_size=cfg['batch_size'] if train else 1,
                          shuffle=train, num_workers=cfg['num_workers'], persistent_workers=False,
                          worker_init_fn=seed_worker, generator=generators[key])
               for train, key in ((True, 'train'), (False, 'validation'))]
    return model, *loaders, optimizer, scheduler, scaler, generators


def train(parts, cfg, root, split, **kwargs):
    model, train_loader, val_loader, optimizer, scheduler, scaler, generators = parts
    return run_epoch_training(model, train_loader, val_loader, optimizer, scheduler,
                              scaler, 'cpu', cfg, split, root, generators,
                              visualize=False, **kwargs)


class EpochTrainingTests(unittest.TestCase):
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

    def test_schedule_endpoints_and_resume(self):
        settings = dict(num_epochs=200, warmup_epochs=3, warmup_start_lr=1e-6,
                        peak_lr=1e-4, min_lr=1e-6)
        rates = [epoch_learning_rate(e, **settings) for e in range(1, 201)]
        self.assertAlmostEqual(rates[0], 1e-6)
        self.assertAlmostEqual(rates[1], 5.05e-5)
        self.assertAlmostEqual(rates[2], 1e-4)
        self.assertAlmostEqual(rates[-1], 1e-6)
        self.assertTrue(all(a < b for a, b in zip(rates[:2], rates[1:3])))
        self.assertTrue(all(a > b for a, b in zip(rates[2:-1], rates[3:])))
        optimizer = torch.optim.Adam(TinyModel().parameters())
        first = EpochWarmupCosine(optimizer, **settings)
        first.set_epoch(80)
        second = EpochWarmupCosine(optimizer, **settings)
        second.load_state_dict(first.state_dict())
        self.assertEqual(second.set_epoch(81), rates[80])
        with self.assertRaises(ValueError):
            epoch_learning_rate(1, **dict(settings, num_epochs=3))
        with self.assertRaises(ValueError):
            epoch_learning_rate(201, **settings)

    def test_split_manifest_and_actual_dataset_modes(self):
        names = [f'{i:02d}.png' for i in range(11)]
        split = make_split(names)
        self.assertEqual(split, make_split(list(reversed(names))))
        self.assertEqual(len(split['validation']), 2)
        self.assertFalse(set(split['train']) & set(split['validation']))
        with self.assertRaises(ValueError):
            validate_split(split, names[:-1])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low, gt = root/'low', root/'gt'
            low.mkdir(); gt.mkdir()
            for index, name in enumerate(names):
                pixels = np.full((16, 20, 3), 40 + index, dtype=np.uint8)
                Image.fromarray(pixels).save(low/name)
                Image.fromarray(pixels + 30).save(gt/name)
            train_loader, val_loader, _ = build_epoch_loaders(low, gt, split, patch_size=12,
                                                             batch_size=8, num_workers=0)
            self.assertTrue(train_loader.dataset.dataset.augment)
            self.assertFalse(val_loader.dataset.dataset.augment)
            self.assertIsNone(val_loader.dataset.dataset.crop_size)
            self.assertEqual([len(batch['low']) for batch in train_loader], [8, 1])
            self.assertEqual(tuple(next(iter(val_loader))['low'].shape), (1, 3, 16, 20))
            run = root/'run'
            stored = prepare_experiment(run, config(), names)
            self.assertEqual(json.loads((run/'split.json').read_text()), stored)
            with self.assertRaises(FileExistsError):
                prepare_experiment(run, config(), names)

    def _resume_equivalence(self, workers):
        cfg = config(workers)
        names = [f'{i}.png' for i in range(7)]
        with tempfile.TemporaryDirectory() as temp:
            complete, resumed = Path(temp)/'complete', Path(temp)/'resumed'
            split = prepare_experiment(complete, cfg, names)
            prepare_experiment(resumed, cfg, names)
            seed_everything(42)
            all_parts = components(cfg)
            all_history = train(all_parts, cfg, complete, split)
            seed_everything(42)
            first_parts = components(cfg)
            train(first_parts, cfg, resumed, split, stop_after_epoch=1)
            # Simulate a crash leaving an uncommitted CSV row and a new process's RNG.
            with (resumed/'logs/training_metrics.csv').open('a') as file:
                file.write('duplicate,stale,row\n')
            seed_everything(999)
            remaining_parts = components(cfg)
            last = resumed/'checkpoints/last.pth'
            prepare_experiment(resumed, cfg, names, last)
            resumed_history = train(remaining_parts, cfg, resumed, split, resume_from=last)
            for a, b in zip(all_history, resumed_history):
                self.assertEqual({k: v for k, v in a.items() if 'seconds' not in k},
                                 {k: v for k, v in b.items() if 'seconds' not in k})
            expected_seen = all_parts[0].seen
            actual_seen = first_parts[0].seen + remaining_parts[0].seen
            self.assert_nested_equal(expected_seen, actual_seen)
            self.assert_nested_equal(all_parts[0].state_dict(), remaining_parts[0].state_dict())
            self.assert_nested_equal(all_parts[3].state_dict(), remaining_parts[3].state_dict())
            self.assert_nested_equal(all_parts[4].state_dict(), remaining_parts[4].state_dict())
            cp = load_epoch_checkpoint(last, cfg, split)
            self.assertEqual(cp['global_step'], 12)
            self.assertEqual(cp['completed_epoch'], 4)
            self.assertIn('rng', cp)
            with (resumed/'logs/training_metrics.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row['epoch'] for row in rows], ['1', '2', '3', '4'])

    def test_resume_equivalence_single_process(self):
        self._resume_equivalence(0)

    def test_resume_equivalence_workers(self):
        self._resume_equivalence(2)

    def test_best_last_ties_and_invalid_resume(self):
        cfg = config()
        names = [str(i) for i in range(10)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'run'
            split = prepare_experiment(root, cfg, names)
            seed_everything(42)
            parts = components(cfg)
            metrics = [dict(val_loss=0.2, val_psnr=psnr, val_ssim=0.5)
                       for psnr in (9, 11, 11, 10)]
            with patch('training_utils.validate_epoch', side_effect=metrics):
                train(parts, cfg, root, split)
            best = load_epoch_checkpoint(root/'checkpoints/best.pth', cfg, split)
            last = load_epoch_checkpoint(root/'checkpoints/last.pth', cfg, split)
            self.assertEqual(best['completed_epoch'], 2)
            self.assertEqual(last['completed_epoch'], 4)
            self.assertEqual(last['best']['epoch'], 2)
            for key in ('model', 'optimizer', 'scheduler', 'scaler', 'rng', 'history', 'split', 'config'):
                self.assertIn(key, best)
                self.assertIn(key, last)
            with self.assertRaises(ValueError):
                load_epoch_checkpoint(root/'checkpoints/last.pth', dict(cfg, batch_size=8), split)
            with self.assertRaises(ValueError):
                prepare_experiment(root, dict(cfg, seed=8), names, root/'checkpoints/last.pth')
            with self.assertRaises(ValueError):
                prepare_experiment(root, cfg, names[:-1], root/'checkpoints/last.pth')
            with self.assertRaises(FileExistsError):
                train(parts, cfg, root, split)
            old = root/'old.pth'
            torch.save({'model': parts[0].state_dict(), 'iter': 50}, old)
            with self.assertRaisesRegex(ValueError, 'iteration'):
                load_epoch_checkpoint(old, cfg, split)

    def test_atomic_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'last.pth'
            path.write_bytes(b'previous')
            def failing_writer(temporary):
                temporary.write_bytes(b'incomplete')
                raise RuntimeError('Interrupted write')
            with self.assertRaises(RuntimeError):
                _atomic_write(path, failing_writer)
            self.assertEqual(path.read_bytes(), b'previous')
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    def test_three_epoch_logging_preserves_best_and_resume(self):
        cfg = dict(config(), num_epochs=7)
        names = [str(i) for i in range(10)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'run'
            split = prepare_experiment(root, cfg, names)
            seed_everything(42)
            parts = components(cfg)
            metrics = [dict(val_loss=0.2, val_psnr=p, val_ssim=0.5)
                       for p in (10, 11, 12, 13, 12, 11, 10)]
            output = io.StringIO()
            with contextlib.redirect_stdout(output), patch('training_utils.validate_epoch', side_effect=metrics[:5]):
                train(parts, cfg, root, split, stop_after_epoch=5, log_every=3)
            last = root / 'checkpoints/last.pth'
            self.assertEqual(load_epoch_checkpoint(last, cfg, split)['completed_epoch'], 5)
            self.assertEqual(load_epoch_checkpoint(root/'checkpoints/best.pth', cfg, split)['completed_epoch'], 4)
            self.assertIn('Epoch 003/7', output.getvalue())
            self.assertNotIn('Epoch 004/7', output.getvalue())
            with (root/'logs/training_metrics.csv').open() as file:
                self.assertEqual([r['epoch'] for r in csv.DictReader(file)], ['3'])
            resumed = components(cfg)
            with contextlib.redirect_stdout(output), patch('training_utils.validate_epoch', side_effect=metrics[5:]):
                history = train(resumed, cfg, root, split, resume_from=last, log_every=3)
            self.assertEqual(len(history), 7)
            with (root/'logs/training_metrics.csv').open() as file:
                self.assertEqual([r['epoch'] for r in csv.DictReader(file)], ['3', '6', '7'])
            self.assertIn('Epoch 006/7', output.getvalue())
            self.assertIn('Epoch 007/7', output.getvalue())
            # Changing only report frequency is compatible with an existing checkpoint.
            train(components(cfg), cfg, root, split, resume_from=last, log_every=1)
            with (root/'logs/training_metrics.csv').open() as file:
                self.assertEqual(len(list(csv.DictReader(file))), 7)

    def test_log_interval_validation_and_final_epoch(self):
        epochs = [e for e in range(1, 201) if should_log_epoch(e, 3, 200)]
        self.assertEqual(len(epochs), 67)
        self.assertEqual(epochs[-2:], [198, 200])
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                should_log_epoch(1, value, 200)

    def test_three_epoch_smoke_keeps_schedule_and_saves_final_visuals(self):
        cfg = dict(config(), num_epochs=200, warmup_epochs=3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'smoke'
            split = prepare_experiment(root, cfg, [str(i) for i in range(10)])
            seed_everything(42)
            model, train_loader, val_loader, optimizer, scheduler, scaler, generators = components(cfg)
            with patch('training_utils.save_epoch_visuals') as save_visuals:
                history = run_epoch_training(
                    model, train_loader, val_loader, optimizer, scheduler, scaler,
                    'cpu', cfg, split, root, generators, stop_after_epoch=3, log_every=3)
            self.assertEqual([call.args[-1].name for call in save_visuals.call_args_list],
                             ['epoch_0001', 'epoch_0003'])
            self.assertEqual(len(history), 3)
            for row, expected in zip(history, [1e-6, 5.05e-5, 1e-4]):
                self.assertAlmostEqual(row['lr'], expected)
            last = load_epoch_checkpoint(root/'checkpoints/last.pth', cfg, split)
            self.assertEqual(last['completed_epoch'], 3)
            self.assertEqual(last['config']['num_epochs'], 200)
            self.assertEqual(last['scheduler']['schedule']['num_epochs'], 200)
            with (root/'logs/training_metrics.csv').open() as file:
                self.assertEqual([r['epoch'] for r in csv.DictReader(file)], ['3'])

    def test_sample_weighted_training_loss(self):
        model = torch.nn.Conv2d(3, 3, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        batches = [{'low': torch.zeros(2, 3, 4, 4), 'high': torch.ones(2, 3, 4, 4)},
                   {'low': torch.zeros(1, 3, 4, 4), 'high': torch.full((1, 3, 4, 4), 4.)}]
        loss, steps = train_one_epoch(model, batches, optimizer,
                                     torch.amp.GradScaler('cuda', enabled=False), 'cpu')
        self.assertEqual(loss, 2.)
        self.assertEqual(steps, 2)

    def test_training_quality_matches_per_image_metrics(self):
        model = torch.nn.Conv2d(3, 3, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        targets = [torch.full((3, 12, 12), value) for value in (0.1, 0.3, 0.8)]
        batches = [dict(low=torch.zeros(2, 3, 12, 12), high=torch.stack(targets[:2])),
                   dict(low=torch.zeros(1, 3, 12, 12), high=torch.stack(targets[2:]))]
        expected = [image_metrics(tensor_to_uint8(torch.zeros_like(gt)), tensor_to_uint8(gt))
                    for gt in targets]
        metrics, steps = train_one_epoch(model, batches, optimizer,
            torch.amp.GradScaler('cuda', enabled=False), 'cpu', return_metrics=True)
        self.assertEqual(steps, 2)
        self.assertAlmostEqual(metrics['train_loss'], 0.4, places=6)
        self.assertAlmostEqual(metrics['train_psnr'], float(np.mean([v[0] for v in expected])))
        self.assertAlmostEqual(metrics['train_ssim'], float(np.mean([v[1] for v in expected])))

    def test_quality_logging_does_not_change_training(self):
        cfg = config()
        results = []
        for collect in (False, True):
            seed_everything(42)
            model, loader, _, optimizer, _, scaler, _ = components(cfg)
            result, steps = train_one_epoch(model, loader, optimizer, scaler, 'cpu',
                                            use_amp=True, return_metrics=collect)
            results.append((model.state_dict(), optimizer.state_dict(), result, steps))
        self.assert_nested_equal(results[0][0], results[1][0])
        self.assert_nested_equal(results[0][1], results[1][1])
        self.assertEqual(results[0][2], results[1][2]['train_loss'])
        self.assertTrue(np.isfinite(results[1][2]['train_ssim']))
        self.assertEqual(results[0][3], results[1][3])

    def test_resume_legacy_history_leaves_unknown_quality_blank(self):
        cfg = config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'run'
            split = prepare_experiment(root, cfg, [str(i) for i in range(10)])
            seed_everything(42)
            train(components(cfg), cfg, root, split, stop_after_epoch=1)
            last = root/'checkpoints/last.pth'
            checkpoint = torch.load(last, weights_only=True)
            for row in checkpoint['history']:
                row.pop('train_psnr')
                row.pop('train_ssim')
            torch.save(checkpoint, last)
            train(components(cfg), cfg, root, split, resume_from=last, log_every=1)
            with (root/'logs/training_metrics.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]['train_psnr'], '')
            self.assertEqual(rows[0]['train_ssim'], '')
            self.assertTrue(all(row['train_psnr'] and row['train_ssim'] for row in rows[1:]))

    def test_v1_validation_and_five_panel_visuals(self):
        seed_everything(42)
        model = RetinexFormer(n_feat=8, stage=1, num_blocks=[0, 0, 0])
        dataset = RandomSamples(train=False)
        result = validate_epoch(model, DataLoader(dataset, batch_size=1), 'cpu')
        self.assertEqual(set(result), {'val_loss', 'val_psnr', 'val_ssim'})
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
        self.assertTrue(model.training)
        with tempfile.TemporaryDirectory() as temp:
            save_epoch_visuals(model, dataset, [0], 'cpu', temp)
            for panel in ('low', 'edge', 'lit_up', 'output', 'gt'):
                self.assertTrue((Path(temp)/panel/'0.png').is_file())
            with Image.open(Path(temp)/'0.png') as image:
                self.assertEqual(image.size, (60, 38))
            with Image.open(Path(temp)/'edge'/'0.png') as image:
                self.assertEqual(image.mode, 'L')
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
