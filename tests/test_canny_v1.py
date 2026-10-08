"""Canny correctness, RGB restoration, and checkpoint compatibility checks."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from canny import CannyEdgeDetector
from RetinexFormer_arch import RetinexFormer
from RetinexFormer_arch_og import RetinexFormer as OriginalRetinexFormer
from analysis_utils import (evaluate, evaluate_analysis_model, export_test_set,
                            infer_full_image, infer_full_image_stages, save_comparisons)


def small_model(**kwargs):
    return RetinexFormer(n_feat=8, stage=kwargs.pop("stage", 1),
                         num_blocks=[1, 1, 1], **kwargs)


class CannyTests(unittest.TestCase):
    def test_constant_step_and_diagonal(self):
        canny = CannyEdgeDetector()
        for value in (0., 0.5, 1.):
            self.assertEqual(canny(torch.full((2, 3, 32, 40), value)).count_nonzero(), 0)
        step = torch.zeros(2, 3, 32, 40)
        step[..., 20:] = 1
        edges = canny(step)
        self.assertEqual(edges.shape, (2, 1, 32, 40))
        self.assertTrue(torch.equal(edges.sum(-1), torch.ones(2, 1, 32)))
        self.assertTrue(torch.all((edges == 0) | (edges == 1)))
        self.assertGreater(edges[..., 19:21].sum(), 0)
        diagonal = (torch.arange(32)[:, None] < torch.arange(32)[None, :]).float()
        edges = canny(diagonal.expand(1, 3, 32, 32))
        self.assertGreater(edges.sum(), 20)
        self.assertLess(edges.sum(), 100)
        coordinates = edges[0, 0].nonzero()
        self.assertTrue(torch.all((coordinates[:, 0] - coordinates[:, 1]).abs() <= 2))

    def test_hysteresis_keeps_long_connected_chains_only(self):
        candidates = torch.zeros(2, 1, 8, 100, dtype=torch.bool)
        candidates[0, 0, 1, 1:95] = True
        candidates[0, 0, 2, 95] = True  # diagonal eight-connectivity
        candidates[0, 0, 6, 1:95] = True  # disconnected weak chain
        candidates[1] = candidates[0]  # no strong seed in this batch item
        strong = torch.zeros_like(candidates)
        strong[0, 0, 1, 1] = True
        edges = CannyEdgeDetector._hysteresis(candidates, strong)
        self.assertEqual(edges.sum(), 95)
        self.assertTrue(edges[0, 0, 2, 95])
        self.assertFalse(edges[0, 0, 6].any())
        self.assertFalse(edges[1].any())

    def test_thresholds_and_nms(self):
        detector = CannyEdgeDetector(kernel_size=1)
        weak = torch.zeros(1, 3, 16, 16)
        weak[..., 8:] = 0.04  # Sobel magnitude 0.16: weak with no strong seed
        self.assertEqual(detector(weak).sum(), 0)
        self.assertGreater(detector(weak * 2).sum(), 0)
        mag = torch.zeros(1, 1, 5, 9)
        mag[..., 2:7] = torch.tensor([1., 2., 3., 2., 1.])
        thin = detector._non_maximum_suppression(mag, torch.ones_like(mag), torch.zeros_like(mag))
        self.assertEqual(thin.count_nonzero(), 5)
        self.assertTrue(torch.all(thin[..., 4] == 3))

    def test_dtype_detach_clamp_and_validation(self):
        detector = CannyEdgeDetector()
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            image = torch.rand(2, 3, 17, 19, dtype=dtype, requires_grad=True)
            result = detector(image)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, image.device)
            self.assertFalse(result.requires_grad)
            self.assertTrue(torch.isfinite(result).all())
        image = torch.randn(1, 3, 16, 20)
        saved = image.clone()
        self.assertTrue(torch.equal(detector(image), detector(image.clamp(0, 1))))
        self.assertTrue(torch.equal(saved, image))
        self.assertEqual(detector(torch.ones(1, 3, 1, 1)).sum(), 0)
        for options in ({"low_threshold": 0}, {"high_threshold": 0.05},
                        {"kernel_size": 4}, {"sigma": 0},
                        {"luminance_weights": (1, 1, 1)}):
            with self.assertRaises(ValueError):
                CannyEdgeDetector(**options)
        with self.assertRaises(ValueError):
            detector(torch.ones(1, 4, 8, 8))


class RestorationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def _training_smoke(self, device, amp=False):
        model = small_model().to(device)
        image = torch.zeros(2, 3, 16, 20, device=device)
        image[..., 10:] = 0.4
        image.requires_grad_()
        target = torch.rand_like(image)
        shapes = []
        handle = model.body[0].denoiser.embedding.register_forward_pre_hook(
            lambda module, args: shapes.append(tuple(args[0].shape)))
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        with torch.autocast(device_type=device, enabled=amp):
            result = model.forward_with_intermediate(image, return_dict=True)
            loss = nn.functional.l1_loss(result["output"], target)
        handle.remove()
        self.assertEqual(shapes, [(2, 4, 16, 20)])
        for key in ("input", "lit_up", "output"):
            self.assertEqual(result[key].shape, image.shape)
            self.assertEqual(result[key].device, image.device)
            self.assertTrue(torch.isfinite(result[key]).all())
        self.assertEqual(result["edge"].shape, (2, 1, 16, 20))
        self.assertEqual(result["edge"].dtype, image.dtype)
        self.assertEqual(result["edge"].device, image.device)
        self.assertGreater(result["edge"].sum(), 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
        embedding = model.body[0].denoiser.embedding.weight
        self.assertGreater(embedding.grad[:, 3].abs().sum(), 0)
        self.assertEqual(embedding[:, 3].count_nonzero(), 0)
        optimizer.step()
        self.assertGreater(embedding[:, 3].abs().sum(), 0)

    def test_cpu_training(self):
        self._training_smoke("cpu")

    def test_cpu_autocast_training(self):
        self._training_smoke("cpu", amp=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_training(self):
        self._training_smoke("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision(self):
        self._training_smoke("cuda", amp=True)

    def test_residual_and_intermediate_contracts(self):
        model = small_model().eval()
        nn.init.zeros_(model.body[0].denoiser.mapping.weight)
        image = torch.rand(1, 3, 16, 20)
        with torch.no_grad():
            details = model.forward_with_intermediate(image, return_dict=True)
            self.assertTrue(torch.equal(details["output"], details["lit_up"]))
            self.assertTrue(torch.equal(model(image), details["output"]))
            lightups, outputs = model.forward_with_intermediate(image)
            self.assertTrue(torch.equal(lightups[-1], details["lit_up"]))
            self.assertTrue(torch.equal(outputs[-1], details["output"]))
            stage_details = model.body[0].forward_with_intermediate(image, return_dict=True)
            stage_litup, stage_output = model.body[0].forward_with_intermediate(image)
            self.assertTrue(torch.equal(stage_litup, stage_details["lit_up"]))
            self.assertTrue(torch.equal(stage_output, stage_details["output"]))
        baseline = small_model(edge_guidance=False)
        self.assertIsNone(baseline.forward_with_intermediate(image, return_dict=True)["edge"])

    def test_multistage_uses_original_edges_once(self):
        model = small_model(stage=2).eval()
        image = torch.rand(1, 3, 16, 20)
        detector = model.body[0].edge_extractor
        edge_inputs = []
        hooks = [stage.denoiser.embedding.register_forward_pre_hook(
            lambda module, args: edge_inputs.append(args[0][:, 3:].clone()))
                 for stage in model.body]
        with torch.no_grad(), patch.object(detector, "forward", wraps=detector.forward) as extract:
            details = model.forward_with_intermediate(image, return_dict=True)
            self.assertEqual(extract.call_count, 1)
            self.assertIs(extract.call_args.args[0], image)
        for hook in hooks:
            hook.remove()
        self.assertEqual(len(details["output_stages"]), 2)
        for edge in edge_inputs:
            self.assertTrue(torch.equal(edge, details["edge"]))
        self.assertTrue(torch.allclose(model(image), details["output"]))

    def test_baseline_conversion_and_checkpoint_roundtrip(self):
        original = OriginalRetinexFormer(n_feat=8, stage=2, num_blocks=[1, 1, 1]).eval()
        baseline = small_model(stage=2, edge_guidance=False).eval()
        state = original.state_dict()
        baseline.load_state_dict(state, strict=True)
        v1 = small_model(stage=2).eval()
        with self.assertRaises(RuntimeError):
            v1.load_state_dict(state, strict=True)
        with self.assertWarnsRegex(UserWarning, "zero-initialized"):
            converted = v1.load_baseline_state_dict(state)
        self.assertEqual(len(converted), 2)
        for key, value in v1.state_dict().items():
            if key in converted:
                self.assertTrue(torch.equal(value[:, :3], state[key]))
                self.assertEqual(value[:, 3:].count_nonzero(), 0)
            else:
                self.assertTrue(torch.equal(value, state[key]))
        image = torch.rand(1, 3, 16, 20)
        with torch.no_grad():
            self.assertTrue(torch.equal(original(image), baseline(image)))
            self.assertTrue(torch.allclose(baseline(image), v1(image), atol=1e-6, rtol=1e-5))
        checkpoint = io.BytesIO()
        torch.save({"model": v1.state_dict()}, checkpoint)
        checkpoint.seek(0)
        restored = small_model(stage=2).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True)["model"], strict=True)
        self.assertTrue(torch.equal(v1(image), restored(image)))
        for bad in ({k: v for k, v in state.items() if k != converted[0]},
                    dict(state, unexpected=torch.ones(1)),
                    dict(state, **{"body.0.estimator.conv1.weight": torch.ones(1)}),
                    v1.state_dict()):
            with self.assertRaises(RuntimeError):
                v1.load_baseline_state_dict(bad)

    def test_padded_evaluation(self):
        model = small_model().eval()
        image = torch.rand(1, 3, 13, 17)
        with torch.no_grad():
            output = infer_full_image(model, image)
            lightup, captured = infer_full_image(model, image, with_intermediate=True)
            lightups, outputs = infer_full_image_stages(model, image)
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(lightup.shape, image.shape)
        self.assertTrue(torch.equal(output, captured))
        self.assertTrue(torch.equal(output, outputs[-1]))
        self.assertTrue(torch.equal(lightup, lightups[-1]))

    def test_real_model_metrics_and_exports(self):
        model = small_model()
        samples = [{"low": torch.rand(3, 13, 17), "high": torch.rand(3, 13, 17),
                    "name": "sample.png"}]
        loader = [{"low": samples[0]["low"].unsqueeze(0),
                   "high": samples[0]["high"].unsqueeze(0), "name": ["sample.png"]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            psnr, ssim = evaluate(model, loader, "cpu")
            self.assertTrue(model.training)
            self.assertTrue(torch.isfinite(torch.tensor([psnr, ssim])).all())
            saved_metrics = export_test_set(model, loader, "cpu", root / "export")
            self.assertEqual((psnr, ssim), saved_metrics)
            save_comparisons(model, samples, [0], "cpu", root / "visuals")
            rows = evaluate_analysis_model(model, samples, "cpu", root / "analysis")
            self.assertEqual(len(rows), 1)
            self.assertTrue((root / "export" / "output" / "sample.png").is_file())
            self.assertTrue((root / "visuals" / "grid" / "sample.png").is_file())
            self.assertTrue((root / "analysis" / "images" / "sample" / "lightup.png").is_file())


if __name__ == "__main__":
    unittest.main()
