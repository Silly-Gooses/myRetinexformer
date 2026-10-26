"""Focused synthetic checks for V2 fusion, Canny, gradients and restoration."""

import io
import unittest

import torch

from Proposed_V2_arch import AdaptiveRetinexFormer, OpenCVCanny
from RetinexFormer_arch import RetinexFormer


def model(gated=True, stage=1):
    return AdaptiveRetinexFormer(n_feat=8, stage=stage, num_blocks=[1, 1, 1],
                                  gated=gated)


class V2Tests(unittest.TestCase):
    def test_canny_batch_and_source(self):
        detector = OpenCVCanny(30, 60)
        input_image = torch.zeros(2, 3, 16, 20, requires_grad=True)
        input_image.data[0, :, :, 10:] = 1
        edges = detector(input_image)
        self.assertEqual(edges.shape, (2, 1, 16, 20))
        self.assertEqual(edges.dtype, torch.float32)
        self.assertFalse(edges.requires_grad)
        self.assertGreater(edges[0].sum(), 0)
        self.assertEqual(edges[1].sum(), 0)
        self.assertTrue(torch.equal(edges, detector(input_image)))

    def test_forward_gate_gradient_checkpoint_and_addition(self):
        for gated in (True, False):
            network = model(gated)
            image = torch.rand(1, 3, 16, 20)
            prediction, details = network(image, return_intermediates=True)
            self.assertEqual(prediction.shape, image.shape)
            self.assertEqual(details["edge_map"].shape, (1, 1, 16, 20))
            for name in ("rgb_features", "edge_features", "fused_features", "gate_map"):
                self.assertEqual(details[name].shape, (1, 8, 16, 20))
            self.assertTrue(torch.all((details["gate_map"] >= 0) & (details["gate_map"] <= 1)))
            self.assertTrue(torch.allclose(details["fused_features"],
                                           details["rgb_features"] +
                                           details["gate_map"] * details["edge_features"]))
            self.assertTrue(torch.isfinite(prediction).all())
            prediction.abs().mean().backward()
            fusion = network.body[0].denoiser.fusion
            self.assertIsNotNone(fusion.edge_encoder[0].weight.grad)
            self.assertGreater(fusion.edge_encoder[0].weight.grad.abs().sum(), 0)
            if gated:
                self.assertGreater(fusion.gate.weight.grad.abs().sum(), 0)
            network.eval()
            expected = network(image)
            data = io.BytesIO()
            torch.save(network.state_dict(), data)
            data.seek(0)
            restored = model(gated).eval()
            restored.load_state_dict(torch.load(data, weights_only=True), strict=True)
            self.assertTrue(torch.allclose(expected, restored(image), atol=0, rtol=0))

    def test_original_canny_shared_by_stages_and_strict_initialization(self):
        network = model(stage=2)
        image = torch.rand(1, 3, 16, 20)
        _, details = network(image, return_intermediates=True)
        self.assertTrue(torch.equal(details["edge_map"], network.edge_extractor(image)))
        self.assertEqual(len(details["feature_stages"]), 2)
        baseline = RetinexFormer(n_feat=8, stage=2, num_blocks=[1, 1, 1], edge_guidance=False)
        with self.assertWarns(UserWarning):
            new_keys = network.initialize_from(baseline.state_dict())
        self.assertTrue(new_keys)
        with self.assertRaises(RuntimeError):
            network.initialize_from({})


if __name__ == "__main__":
    unittest.main()
