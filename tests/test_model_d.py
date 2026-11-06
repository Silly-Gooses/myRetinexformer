"""Model D v1 architecture tests and opt-in original checkpoint verification."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from analysis_utils import tensor_to_uint8
from model_d.cli import (PROTOCOL, check_validation_report, infer, load_pairs,
                         notebook_definitions, prepare_output, read_rgb,
                         verify_notebook, REFERENCE_NOTEBOOK)
from model_d.model import (HVIColorHead, ModelD, REFERENCE, build_model_b,
                           hvi_to_rgb, load_backbone, load_head, load_model_d,
                           quantize_model_b, rgb_to_hvi, sha256)


def setUpModule():
    global original_threads, original_rng
    original_threads = torch.get_num_threads()
    original_rng = torch.get_rng_state()
    torch.set_num_threads(2)


def tearDownModule():
    # Avoid changing the numeric behavior of later process-level training tests.
    torch.set_num_threads(original_threads)
    torch.set_rng_state(original_rng)


class ArchitectureTests(unittest.TestCase):

    def test_hvi_round_trip_and_identity(self):
        colors = torch.tensor([[0., 0., 0.], [1., 1., 1.], [.5, .5, .5],
                               [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                               [1., 1., 0.], [0., 1., 1.], [1., 0., 1.],
                               [1e-9, 0., 1e-9]]).T[None, :, None, :]
        head = HVIColorHead().eval()
        for rgb in (colors, torch.rand(2, 3, 19, 23)):
            with torch.no_grad():
                hv, intensity = rgb_to_hvi(rgb)
                restored = hvi_to_rgb(hv, intensity)
                output, residual, corrected = head(rgb)
            torch.testing.assert_close(restored, rgb, rtol=0, atol=2e-4)
            torch.testing.assert_close(output, rgb, rtol=0, atol=2e-4)
            self.assertEqual(residual.count_nonzero(), 0)
            torch.testing.assert_close(corrected, hv, rtol=0, atol=0)

    def test_nonzero_head_bounds_intensity_and_reference(self):
        head = HVIColorHead().eval()
        torch.nn.init.normal_(head.net[-1].weight, std=.1)
        torch.nn.init.constant_(head.net[-1].bias, .2)
        legacy = notebook_definitions(REFERENCE_NOTEBOOK)["HVIColorHead"]().eval()
        legacy.load_state_dict(head.state_dict(), strict=True)
        rgb = torch.rand(2, 3, 17, 21)
        with torch.no_grad():
            actual = head(rgb)
            expected = legacy(rgb)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(a).all())
        self.assertGreater(actual[1].abs().max(), 0)
        self.assertLessEqual(actual[1].abs().max(), .12)
        torch.testing.assert_close(actual[0].max(1).values, rgb.max(1).values,
                                   rtol=0, atol=2e-7)

    def test_quantization_matches_png_round_trip(self):
        # Include ties to protect NumPy's round-to-even behavior.
        rgb = torch.cat([torch.rand(2, 3, 7, 9),
                         torch.full((1, 3, 7, 9), 10.5 / 255)], 0)
        quantized = quantize_model_b(rgb)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.png"
            for index, image in enumerate(rgb):
                Image.fromarray(tensor_to_uint8(image)).save(path)
                torch.testing.assert_close(quantized[index], read_rgb(path), rtol=0, atol=0)

    def test_wrapper_dimensions_intermediates_and_determinism(self):
        model = ModelD(build_model_b(), HVIColorHead()).eval()
        for shape in ((1, 3, 16, 20), (2, 3, 17, 19), (1, 3, 1, 1)):
            x = torch.rand(shape)
            with torch.no_grad():
                first = model(x)
                second = model(x)
                result = model.forward_with_intermediate(x)
            self.assertIsInstance(first, torch.Tensor)
            self.assertEqual(first.shape, x.shape)
            self.assertTrue(torch.isfinite(first).all())
            self.assertGreaterEqual(first.min(), 0)
            self.assertLessEqual(first.max(), 1)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first, result["output"], rtol=0, atol=0)
            torch.testing.assert_close(first, result["head_input"], rtol=0, atol=2e-4)
            self.assertEqual(set(result), {"input", "edge", "lit_up", "model_b_rgb",
                                           "head_input", "hv_residual", "output"})
            for name, value in result.items():
                self.assertEqual(value.shape[-2:], x.shape[-2:], name)
            self.assertEqual(result["edge"].shape[1], 1)
            self.assertEqual(result["hv_residual"].shape[1], 2)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

    def test_independent_notebook_forward(self):
        model = ModelD(build_model_b(), HVIColorHead()).eval()
        # Exercise actual color correction, not only zero-initialized identity.
        torch.nn.init.normal_(model.head.net[-1].weight, std=.05)
        result = verify_notebook(model, torch.rand(1, 3, 17, 23))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(max(result["max_abs_difference"].values()), 0)

    def test_invalid_inputs(self):
        model = ModelD(build_model_b(), HVIColorHead()).eval()
        for x in (torch.rand(3, 8, 8), torch.rand(1, 4, 8, 8),
                  torch.zeros(1, 3, 8, 8, dtype=torch.uint8),
                  torch.full((1, 3, 8, 8), float("nan")),
                  torch.full((1, 3, 8, 8), -1.), torch.empty(0, 3, 8, 8)):
            with self.assertRaises(ValueError):
                model(x)


class CheckpointTests(unittest.TestCase):
    def test_backbone_is_strictly_compatible_with_notebook(self):
        legacy, _ = notebook_definitions(REFERENCE_NOTEBOOK)["build_model_b"]()
        actual = build_model_b()
        actual.load_state_dict(legacy.state_dict(), strict=True)
        state = dict(legacy.state_dict())
        state.pop(next(iter(state)))
        with self.assertRaises(RuntimeError):
            actual.load_state_dict(state, strict=True)

    def test_synthetic_head_schema_and_strict_rejections(self):
        head = HVIColorHead()
        fixture = dict(head=head.state_dict(), config=copy.deepcopy(REFERENCE["config"]),
                       epoch=17, best_val_psnr=REFERENCE["metrics"]["val"]["model_d"]["psnr"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_head.pth"

            def load(value, digest=REFERENCE["config"]["model_b_sha256"]):
                torch.save(value, path)
                return load_head(path, digest)

            loaded, metadata = load(fixture)
            self.assertEqual(metadata["sha256"], sha256(path))
            for key, value in head.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "different Model B"):
                load(fixture, "wrong-pair")
            for field, value in (("head_width", 64), ("hvi_k", 2), ("max_delta", .2),
                                 ("repo_commit", "wrong"), ("val_names", [])):
                invalid = copy.deepcopy(fixture)
                invalid["config"][field] = value
                with self.assertRaisesRegex(ValueError, field):
                    load(invalid)
            for mutation in ("missing", "unexpected", "shape", "nonfinite"):
                invalid = copy.deepcopy(fixture)
                if mutation == "missing":
                    del invalid["head"]["net.0.weight"]
                elif mutation == "unexpected":
                    invalid["head"]["extra"] = torch.zeros(1)
                elif mutation == "shape":
                    invalid["head"]["net.0.weight"] = torch.zeros(1)
                else:
                    invalid["head"]["net.0.weight"].fill_(float("nan"))
                with self.assertRaises((ValueError, RuntimeError)):
                    load(invalid)
            invalid = copy.deepcopy(fixture)
            invalid["epoch"] = 30
            with self.assertRaisesRegex(ValueError, "epoch 17"):
                load(invalid)
            with self.assertRaises(ValueError):
                load({"head": fixture["head"]})

    def test_missing_or_wrong_backbone_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.pth"
            with self.assertRaises(FileNotFoundError):
                load_backbone(path)
            path.write_bytes(b"not the verified original checkpoint")
            with patch("torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_backbone(path)
                loader.assert_not_called()


class ProtocolTests(unittest.TestCase):
    def test_inference_exports_original_size_and_intermediates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.png"
            pixels = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)
            Image.fromarray(pixels).save(source)
            model = ModelD(build_model_b(), HVIColorHead()).eval()
            args = SimpleNamespace(input=source, output_dir=root / "output",
                                   model_b=root / "b.pth", head=root / "head.pth",
                                   intermediates=True)
            infer(model, {}, args, torch.device("cpu"))
            with Image.open(args.output_dir / "input.png") as image:
                self.assertEqual(image.size, (19, 17))
            with np.load(args.output_dir / "input_intermediates.npz") as tensors:
                self.assertEqual(tensors["hv_residual"].shape, (1, 2, 17, 19))
                expected = tensor_to_uint8(torch.from_numpy(tensors["output"][0]))
            with Image.open(args.output_dir / "input.png") as image:
                np.testing.assert_array_equal(np.asarray(image), expected)
            report = json.loads((args.output_dir / "inference.json").read_text())
            self.assertFalse(report["historical_protocol"])

    def test_frozen_split_and_rejection(self):
        config = REFERENCE["config"]
        pairs = [(name, Path(name), Path(name))
                 for name in config["train_names"] + config["val_names"]]
        with patch("model_d.cli.find_paired_images", return_value=pairs):
            actual = load_pairs("unused", "val")
            self.assertEqual([p[0] for p in actual], config["val_names"])
        with patch("model_d.cli.find_paired_images", return_value=pairs[:-1]):
            with self.assertRaises(ValueError):
                load_pairs("unused", "val")
            with self.assertRaises(ValueError):
                load_pairs("unused", "test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.json"
            path.write_text(json.dumps(dict(train=[], val=[])))
            with self.assertRaisesRegex(ValueError, "split.json"):
                load_pairs("unused", "val", path)

    def test_test_gate_requires_matching_validation(self):
        report = dict(protocol=PROTOCOL, split="val", count=69,
                      checkpoints={"model_b": {"sha256": "a"}, "head": {"sha256": "b"}},
                      historical_metrics_within_tolerance=True,
                      notebook_parity={"status": "passed"}, runtime={"device": "cuda"},
                      dataset_sha256="data")
        with tempfile.TemporaryDirectory() as tmp, patch(
                "model_d.cli.dataset_sha256", return_value="data"):
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            self.assertEqual(check_validation_report(path, report["checkpoints"], []), sha256(path))
            for field, value in (("count", 100), ("checkpoints", {}),
                                 ("historical_metrics_within_tolerance", False),
                                 ("notebook_parity", {}), ("runtime", {"device": "cpu"}),
                                 ("dataset_sha256", "changed")):
                invalid = copy.deepcopy(report)
                invalid[field] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    check_validation_report(path, report["checkpoints"], [])

    def test_output_never_overwrites_inputs_or_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data"
            source.mkdir()
            with self.assertRaises(ValueError):
                prepare_output(source / "outputs", [source])
            output = prepare_output(Path(tmp) / "run")
            (output / "report.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                prepare_output(output)


class OriginalCheckpointTests(unittest.TestCase):
    """Environment variables opt into real weights; never fabricate success."""

    @unittest.skipUnless(os.environ.get("MODEL_D_BACKBONE"),
                         "Unverified: set MODEL_D_BACKBONE to the original Option A best.pth")
    def test_original_backbone_and_notebook(self):
        backbone, info = load_backbone(os.environ["MODEL_D_BACKBONE"])
        self.assertEqual(info["sha256"], REFERENCE["config"]["model_b_sha256"])
        model = ModelD(backbone, HVIColorHead()).eval()
        image_path = os.environ.get("MODEL_D_IMAGE")
        image = read_rgb(image_path)[None] if image_path else torch.rand(1, 3, 17, 23)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        result = verify_notebook(model.to(device), image.to(device))
        self.assertEqual(result["status"], "passed")

    @unittest.skipUnless(os.environ.get("MODEL_D_BACKBONE") and os.environ.get("MODEL_D_HEAD"),
                         "Unverified: original Model D v1 head checkpoint unavailable")
    def test_original_trained_pair_and_notebook(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = load_model_d(os.environ["MODEL_D_BACKBONE"], os.environ["MODEL_D_HEAD"],
                               device=device)
        image_path = os.environ.get("MODEL_D_IMAGE")
        image = read_rgb(image_path)[None] if image_path else torch.rand(1, 3, 17, 23)
        result = verify_notebook(model, image.to(device))
        self.assertEqual(result["status"], "passed")


if __name__ == "__main__":
    unittest.main()
