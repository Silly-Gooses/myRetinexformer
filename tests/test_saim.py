"""SAIM architecture and checkpoint contract tests."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from RetinexFormer_arch import RetinexFormer as Baseline
from Proposed_SAIM_arch import RetinexFormer as SAIM, load_baseline_checkpoint
from analysis_utils import infer_full_image
from saim_experiment import run_experiment, evaluate_final_test


ARGS = dict(in_channels=3, out_channels=3, n_feat=4, stage=1, num_blocks=[1, 1, 1])


class TestSAIM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.baseline = Baseline(**ARGS)
        self.saim = SAIM(**ARGS)
        self.report = load_baseline_checkpoint(self.saim, {'params': self.baseline.state_dict()},
                                               baseline=self.baseline)

    def test_shapes_and_padding(self):
        for h, w in ((8, 12), (9, 11), (12, 9)):
            with self.subTest(h=h, w=w):
                x = torch.rand(1, 3, h, w)
                y = self.saim(x) if h % 4 == w % 4 == 0 else infer_full_image(self.saim, x)
                self.assertEqual(y.shape, x.shape)

    def test_baseline_identity(self):
        self.assertTrue(self.report['output_equivalence_verified'])
        self.assertEqual(self.report['loaded_baseline_tensors'], len(self.baseline.state_dict()))
        x = torch.rand(1, 3, 8, 12)
        with torch.no_grad():
            torch.testing.assert_close(self.saim(x), self.baseline(x), rtol=1e-5, atol=1e-6)

    def test_finite_gradients_and_gate_update(self):
        model = self.saim
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        x, target = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
        alpha_before = model.body[0].saim.alpha.detach().clone()
        for _ in range(2):
            opt.zero_grad()
            loss = F.l1_loss(model(x), target)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            opt.step()
        self.assertFalse(torch.equal(alpha_before, model.body[0].saim.alpha))
        self.assertTrue(any(p.grad is not None and torch.any(p.grad != 0)
                            for name, p in model.named_parameters() if '.saim.gate.' in name))

    def test_checkpoint_rejections(self):
        state = dict(self.baseline.state_dict())
        bad = dict(state)
        bad.pop(next(iter(bad)))
        with self.assertRaisesRegex(RuntimeError, 'missing='):
            load_baseline_checkpoint(self.saim, bad)
        bad = dict(state, surprise=torch.zeros(1))
        with self.assertRaisesRegex(RuntimeError, 'unexpected='):
            load_baseline_checkpoint(self.saim, bad)
        key = next(iter(state))
        bad = dict(state)
        bad[key] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, 'shape_mismatch='):
            load_baseline_checkpoint(self.saim, bad)

    def test_synthetic_experiment_resume_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            data = workspace / 'LOLv2/Real_captured'
            for sub in ('Train/Low', 'Train/Normal', 'Test/Low', 'Test/Normal'):
                (data / sub).mkdir(parents=True)
            for i in range(4):
                low = np.full((12, 12, 3), 40 + 20*i, dtype=np.uint8)
                high = np.clip(low.astype(int) + 25, 0, 255).astype(np.uint8)
                for split in ('Train', 'Test'):
                    if split == 'Test' and i > 0:
                        continue
                    Image.fromarray(low).save(data / split / 'Low' / f'{i}.png')
                    Image.fromarray(high).save(data / split / 'Normal' / f'{i}.png')
            cp = workspace / 'pretrained_weights/LOL_v2_real.pth'
            cp.parent.mkdir()
            torch.save({'params': self.baseline.state_dict()}, cp)
            config = dict(n_feat=4, num_blocks=[1, 1, 1], crop_size=8,
                          log_interval=100, amp=False)
            root = run_experiment(workspace, steps=100, config=config)
            self.assertTrue((root / 'psnr_comparison.png').is_file())
            self.assertTrue((root / 'last.pth').is_file())
            self.assertEqual(run_experiment(workspace, steps=100, config=config, resume=True), root)
            result = evaluate_final_test(root, data)
            self.assertIn('delta_psnr', result)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_backward(self):
        model = self.saim.cuda()
        x = torch.rand(1, 3, 8, 8, device='cuda')
        output = model(x)
        output.square().mean().backward()
        self.assertEqual(output.device.type, 'cuda')
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.isfinite(model.body[0].saim.alpha.grad))


if __name__ == '__main__':
    unittest.main()
