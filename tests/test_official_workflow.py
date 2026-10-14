"""Official split, numerical protocol, event scheduling and checkpoint recovery."""
import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from dataLoader import OfficialPairedDataset
from official_workflow import (FORMAT, PROTOCOL, build_loaders, inspect_dataset, prepare_run,
                               run_training, load_checkpoint, storage_probe, save_weights)
from protocol_metrics import protocol_metrics, evaluate_protocol, test_checkpoint
from training_utils import EpochWarmupCosine, train_one_epoch, seed_everything
from train import build_model, SCHEDULE_KEYS
from official_fixtures import configuration, dataset


class RandomDataset(Dataset):
    def __init__(self, count=5): self.count = count
    def __len__(self): return self.count
    def __getitem__(self, index):
        low = torch.rand(3, 12, 12) * .3
        return dict(low=low, high=low+.4, name=f'{index}.png')


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.dropout = torch.nn.Dropout(.1)
    def forward(self, x): return self.conv(self.dropout(x))


def components(cfg, workers=0):
    seed_everything(42)
    model = TinyModel()
    generators = {'train': torch.Generator().manual_seed(42),
                  'validation': torch.Generator().manual_seed(43)}
    loaders = [DataLoader(RandomDataset(5 if key == 'train' else 2),
                          batch_size=2 if key == 'train' else 1,
                          shuffle=key == 'train', generator=generators[key],
                          num_workers=workers, persistent_workers=False)
               for key in generators]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['warmup_start_lr'])
    scheduler = EpochWarmupCosine(optimizer, **{k:cfg[k] for k in SCHEDULE_KEYS})
    return model, *loaders, optimizer, scheduler, torch.amp.GradScaler('cuda', enabled=False), generators


class OfficialWorkflowTests(unittest.TestCase):
    def test_preflight_pairing_counts_content_and_no_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset(root); dataset(root, 'Test')
            manifest, pairs = inspect_dataset(root)
            self.assertEqual([len(pairs[k]) for k in ('train', 'test')], [689, 100])
            cfg = configuration(root)
            train, val, _ = build_loaders(pairs, cfg, 'cpu')
            self.assertEqual(len(train), 87)
            self.assertEqual(len(train.dataset), 689)
            self.assertEqual(len(val.dataset), 100)
            names = [name for batch in train for name in batch['name']]
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(len(names), 689)
            self.assertEqual(sum(1 for _ in val), 100)
            # Same filenames across directories are legitimate; copied pixels are not.
            target = root/'Test/Low/0000.png'
            original = target.read_bytes()
            shutil.copyfile(root/'Train/Low/0000.png', target)
            with self.assertRaisesRegex(ValueError, 'overlap'): inspect_dataset(root)
            target.write_bytes(original)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'mismatch'): inspect_dataset(root)
            (root/'Test/Normal/0000.png').unlink()
            with self.assertRaisesRegex(ValueError, 'expected 100'): inspect_dataset(root)

    def test_official_crop_reflection_and_eight_geometric_transforms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = np.arange(12*13*3, dtype=np.uint16).reshape(12, 13, 3) % 255
            for name in ('a.png', 'b.png'): Image.fromarray(data.astype(np.uint8)).save(root/name)
            ds = OfficialPairedDataset([('x.png', root/'a.png', root/'b.png')], 16)
            padded = cv2.copyMakeBorder(data.astype(np.float32)/255,0,4,0,3,cv2.BORDER_REFLECT)
            for mode in range(8):
                with patch('dataLoader.random.randint', side_effect=[0, 0, mode]): result = ds[0]
                expected = np.rot90(padded, k=mode//2)
                if mode % 2: expected = np.flipud(expected)
                np.testing.assert_array_equal(result['low'].numpy(), expected.transpose(2,0,1))
                self.assertTrue(torch.equal(result['low'], result['high']))

    def test_metric_protocol_differs_from_quantized_psnr_and_handles_clamp(self):
        target = torch.linspace(.1, .8, 3*13*17).reshape(3,13,17)
        output = target + .001
        psnr, ssim = protocol_metrics(output, target)
        a, b = output.numpy().astype(np.float64), target.numpy().astype(np.float64)
        self.assertAlmostEqual(psnr, 10*math.log10(1/np.mean((a-b)**2)), places=10)
        test_psnr, test_ssim = protocol_metrics(output, target, testing=True)
        self.assertAlmostEqual(test_psnr, 10*math.log10(1/float(np.mean((target.numpy()-output.numpy())**2))), places=5)
        self.assertEqual(ssim, test_ssim)
        self.assertEqual(protocol_metrics(target, target)[0], float('inf'))
        self.assertEqual(protocol_metrics(target, target, testing=True)[0], 100.)
        self.assertEqual(protocol_metrics(torch.full_like(target, 2), torch.ones_like(target))[0], float('inf'))
        with self.assertRaises(FloatingPointError): protocol_metrics(target*float('nan'), target)

    def test_evaluation_preserves_parameters_buffers_gradients_and_mode(self):
        seed_everything(1)
        model = TinyModel().train()
        for p in model.parameters(): p.grad = torch.ones_like(p)
        before = copy.deepcopy(model.state_dict())
        grads = [p.grad.clone() for p in model.parameters()]
        for testing in (False, True):
            rows, summary = evaluate_protocol(model, DataLoader(RandomDataset(3)), 'cpu', testing=testing)
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(summary['psnr'], np.mean([r['psnr'] for r in rows]))
            self.assertTrue(model.training)
            for k, v in model.state_dict().items(): self.assertTrue(torch.equal(v,before[k]))
            for a, b in zip(grads, model.parameters()): self.assertTrue(torch.equal(a,b.grad))
        model.eval()
        evaluate_protocol(model, DataLoader(RandomDataset(1)), 'cpu')
        self.assertFalse(model.training)

    def test_mixup_paired_input_canny_and_clipping(self):
        cfg = configuration('/unused')
        model = build_model(cfg, 'cpu')
        low = torch.rand(2, 3, 16, 16)*.3
        high = low+.4
        seen = []
        handle = model.body[0].edge_extractor.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
        expected = .25*low + .75*low.flip(0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = copy.deepcopy(model.state_dict())
        actual_clip = torch.nn.utils.clip_grad_norm_
        with patch('training_utils.random.randint', return_value=0), \
             patch('torch.distributions.Beta.rsample', return_value=torch.tensor([[[.25]]])), \
             patch('training_utils.torch.randperm', return_value=torch.tensor([1,0])), \
             patch('torch.nn.utils.clip_grad_norm_', wraps=actual_clip) as clip:
            train_one_epoch(model, [dict(low=low, high=high)], optimizer,
                            torch.amp.GradScaler('cuda', enabled=False), 'cpu',
                            mixup_beta=1.2, grad_clip_norm=.01)
        handle.remove()
        self.assertTrue(torch.equal(seen[0], expected))
        self.assertEqual(clip.call_args.args[1], .01)
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def _run(self, cfg, manifest, root, parts, **kw):
        model, train, val, opt, sch, scaler, generators = parts
        with contextlib.redirect_stdout(io.StringIO()):
            return run_training(model, train, val, opt, sch, scaler, 'cpu', cfg,
                                manifest, root, generators, visualize=False, **kw)

    def test_exact_events_resume_replay_and_immutable_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            cfg = configuration('/unused'); cfg.update(batch_size=2, val_freq_steps=2)
            manifest = dict(protocol=PROTOCOL, splits={'train':list(range(5)), 'test':list(range(2))})
            complete, resumed = base/'complete', base/'resumed'
            for path in (complete, resumed): prepare_run(path,cfg,manifest)
            all_parts = components(cfg)
            all_history = self._run(cfg,manifest,complete,all_parts)
            first = components(cfg)
            original_train = train_one_epoch
            calls = 0
            def interrupt(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 2:
                    original = kw['on_step']
                    def stop(batch):
                        original(batch)
                        raise RuntimeError('disconnect')
                    kw['on_step'] = stop
                return original_train(*a,**kw)
            with patch('official_workflow.train_one_epoch', side_effect=interrupt), self.assertRaisesRegex(RuntimeError,'disconnect'):
                self._run(cfg,manifest,resumed,first)
            last = resumed/'checkpoints/last.pth'
            self.assertEqual(torch.load(last,weights_only=True)['completed_epoch'],1)
            saved_snapshots = {p.name:p.read_bytes() for p in (resumed/'checkpoints').glob('step_*.pth')}
            # Inject stale log rows; checkpoint history must replace them on resume.
            (resumed/'logs/validation_log.csv').write_text('stale\n')
            prepare_run(resumed,cfg,manifest,last)
            remaining = components(cfg)
            history = self._run(cfg,manifest,resumed,remaining,resume_from=last)
            for k,v in all_parts[0].state_dict().items(): self.assertTrue(torch.equal(v,remaining[0].state_dict()[k]))
            for a,b in zip(all_history,history):
                self.assertEqual({k:v for k,v in a.items() if 'seconds' not in k},
                                 {k:v for k,v in b.items() if 'seconds' not in k})
            a = load_checkpoint(complete/'checkpoints/last.pth',cfg,manifest)
            b = load_checkpoint(last,cfg,manifest)
            self.assertEqual(a['validation_history'],b['validation_history'])
            self.assertEqual([r['global_step'] for r in b['validation_history']], [0,2,4,6,8,10,12])
            self.assertEqual(len({r['event'] for r in b['validation_history']}),7)
            self.assertEqual(b['global_step'],12)
            for name, content in saved_snapshots.items(): self.assertEqual((resumed/'checkpoints'/name).read_bytes(),content)
            selected = torch.load(resumed/'checkpoints/best.pth',weights_only=True)
            self.assertEqual(selected['global_step'],b['best']['global_step'])
            with (resumed/'logs/validation_log.csv').open() as file:
                self.assertEqual(len(list(csv.DictReader(file))),7)
            # Off-cadence epochs have blank validation values, not stale scores.
            self.assertIsNone(history[1]['val_psnr'])
            with self.assertRaises(FileExistsError): prepare_run(resumed,cfg,manifest)
            changed = dict(cfg, val_freq_steps=3)
            with self.assertRaisesRegex(ValueError,'configuration'): prepare_run(resumed,changed,manifest,last)

    def test_final_validation_and_independent_testing_legacy_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg=configuration(root/'data')
            dataset(root/'data','Test')  # No Train directory: testing must still work.
            model=build_model(cfg,'cpu')
            cp=root/'legacy.pth'
            torch.save(dict(format='retinexformer-epoch-v1',model=model.state_dict(),
                            config=dict(cfg,val_fraction=.1)),cp)
            original=cp.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                test_checkpoint(cp,root/'data',root/'testing','cpu')
            self.assertEqual(cp.read_bytes(),original)
            self.assertEqual(len(list((root/'testing/enhanced_images').glob('*.png'))),100)
            with (root/'testing/per_image_metrics.csv').open() as f: rows=list(csv.DictReader(f))
            with (root/'testing/summary_metrics.csv').open() as f: summary=list(csv.DictReader(f))[0]
            self.assertEqual(len(rows),100)
            self.assertAlmostEqual(float(summary['psnr']),np.mean([float(r['psnr']) for r in rows]))
            self.assertIn('not an independent',json.loads((root/'testing/provenance.json').read_text())['selection_note'])
            # Replacing a live best/last alias after the first read must not mix
            # metadata, model weights, or the recorded checkpoint fingerprint.
            def replace_alias(*args, **kwargs):
                cp.write_bytes(b'replaced by another process')
                return inspect_dataset(*args, **kwargs)
            with patch('official_workflow.inspect_dataset', side_effect=replace_alias), contextlib.redirect_stdout(io.StringIO()):
                test_checkpoint(cp,root/'data',root/'racing_alias','cpu')
            import hashlib
            provenance=json.loads((root/'racing_alias/provenance.json').read_text())
            self.assertEqual(provenance['checkpoint_sha256'],hashlib.sha256(original).hexdigest())
            cp.write_bytes(original)
            for label in ('low','edge','lit_up','output','gt'):
                self.assertEqual(len(list((root/'testing/visuals'/label).glob('*.png'))),2)
            with Image.open(next((root/'testing/enhanced_images').glob('*.png'))) as image:
                self.assertEqual(image.size,(17,13))
            with self.assertRaises(FileExistsError): test_checkpoint(cp,root/'data',root/'testing','cpu')
            # Snapshot creation never overwrites previously saved weights.
            (root/'checkpoints').mkdir()
            one=save_weights(root,model,cfg,1,1,1,'best')
            two=save_weights(root,model,cfg,1,1,1,'best')
            self.assertNotEqual(one,two)
            self.assertEqual(storage_probe(root),str(root.resolve()))

    def test_default_interval_occurs_at_epoch_12_batch_43(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = configuration('/unused')
            cfg['num_epochs'] = 14
            manifest = dict(protocol=PROTOCOL, splits={'train':list(range(689)), 'test':list(range(100))})
            prepare_run(root, cfg, manifest)
            parts = components(cfg)
            def batches(*args, **kwargs):
                for batch in range(1, 88): kwargs['on_step'](batch)
                return dict(train_loss=.1, train_psnr=20., train_ssim=.5), 87
            with patch('official_workflow.train_one_epoch', side_effect=batches):
                self._run(cfg,manifest,root,parts)
            cp = load_checkpoint(root/'checkpoints/last.pth',cfg,manifest)
            self.assertEqual([(r['epoch'],r['batch'],r['global_step'],r['kind'])
                              for r in cp['validation_history']],
                             [(0,0,0,'initial'),(12,43,1000,'periodic'),(14,87,1218,'final')])
            periodic = list((root/'checkpoints').glob('step_00001000_periodic_*.pth'))
            self.assertEqual(len(periodic),1)

    def test_quick_smoke_and_legacy_resume_rejection(self):
        from smoke_test import run_smoke
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            dataset(base/'data'); dataset(base/'data','Test')
            cfg=configuration(base/'data')
            with contextlib.redirect_stdout(io.StringIO()):
                run_smoke(cfg,base/'quick','cpu',base/'mounted_storage')
            self.assertEqual(len(list((base/'quick/testing/enhanced_images').glob('*.png'))),100)
            self.assertFalse((base/'quick/checkpoints/last.pth').exists())
            snapshot=next((base/'quick/checkpoints').glob('*.pth'))
            self.assertEqual(torch.load(snapshot,weights_only=True)['global_step'],2)
            manifest,_=inspect_dataset(base/'data')
            with self.assertRaisesRegex(ValueError,'matching official'):
                load_checkpoint(snapshot,cfg,manifest)
            legacy=base/'legacy'
            legacy.mkdir()
            (legacy/'split.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'Legacy split runs'):
                prepare_run(legacy,cfg,manifest,legacy/'checkpoints/last.pth')


if __name__ == '__main__': unittest.main()
