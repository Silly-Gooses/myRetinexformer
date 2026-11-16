import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from fusion_model.checkpoint_utils import load_verified_state
from fusion_model.cli import read_rgb
from fusion_model.fixed_fusion import fixed_fusion
from fusion_model.inference import Predictor, reflect_pad, select_device
from fusion_model.metrics import author_ssim, float_psnr, rgb8
from fusion_model.self_ensemble import ensemble8
from scripts import evaluate_fusion, run_fusion


class FusionTests(unittest.TestCase):
    def test_eight_transforms_and_inverse(self):
        x = torch.arange(3 * 5 * 7).reshape(1, 3, 5, 7).float()
        views = []

        def network(view):
            views.append(view.numpy().copy())
            return view + 0.25

        torch.testing.assert_close(ensemble8(x, network), x + 0.25, rtol=0, atol=0)
        self.assertEqual(len(views), 8)
        self.assertEqual(len({v.tobytes() for v in views}), 8)
        for view, (h, w, rotate) in zip(views, product((False, True), repeat=3)):
            expected = x.numpy()[:, :, ::-1 if h else 1, ::-1 if w else 1]
            if rotate:
                expected = np.rot90(expected, axes=(-2, -1))
            np.testing.assert_array_equal(view, expected)

    def test_reflection_and_repeatability(self):
        image = np.arange(13 * 19 * 3, dtype=np.uint8).reshape(13, 19, 3)
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        for size in (4, 8):
            expected = np.pad(image, ((0, -13 % size), (0, -19 % size), (0, 0)), mode="reflect")
            np.testing.assert_array_equal(reflect_pad(x, size)[0].permute(1, 2, 0), expected)
        model = Predictor(torch.nn.Identity(), torch.nn.Identity(), torch.device("cpu"))
        outputs = model.predict(image, compare=True)
        np.testing.assert_array_equal(outputs["fusion"], model.predict(image)["fusion"])
        for result in outputs.values():
            self.assertEqual(result.shape, image.shape)
            self.assertTrue(np.isfinite(result).all() and result.min() >= 0 and result.max() <= 1)
            np.testing.assert_allclose(result, image / 255, atol=2e-7)
        with self.assertRaisesRegex(ValueError, "too small"):
            reflect_pad(torch.zeros(1, 3, 2, 2), 8)

    def test_float_fusion_and_dimensions(self):
        r, h = torch.tensor([0.12345]), torch.tensor([0.78901])
        self.assertAlmostEqual(fixed_fusion(r, h).item(), 0.25 * 0.12345 + 0.75 * 0.78901)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            fixed_fusion(r, h.repeat(2))

    def test_branch_clipping_precedes_fusion_and_nan_is_rejected(self):
        class Constant(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = value

            def forward(self, x):
                return torch.full_like(x, self.value)

        low = np.zeros((16, 24, 3), np.uint8)
        model = Predictor(Constant(3.0), Constant(-0.1), torch.device("cpu"))
        np.testing.assert_array_equal(model.predict(low)["fusion"], np.full(low.shape, 0.25))
        model.hvi = Constant(float("nan"))
        with self.assertRaisesRegex(ValueError, "Invalid hvi output"):
            model.predict(low)

    def test_hash_and_strict_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            layer = torch.nn.Linear(2, 1)
            state = {"module." + key: value for key, value in layer.state_dict().items()}
            torch.save({"params": state, "params_ema": {}}, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_verified_state(layer, path, "invalid")
            copy = torch.nn.Linear(2, 1)
            load_verified_state(copy, path, digest)
            torch.testing.assert_close(copy.weight, layer.weight)
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                load_verified_state(torch.nn.Linear(4, 1), path, digest)

    def test_metrics(self):
        black = np.zeros((13, 17, 3), np.float32)
        self.assertEqual(float_psnr(black, black), 100.0)
        self.assertAlmostEqual(float_psnr(black, black + 0.5), 6.020599913)
        self.assertEqual(float_psnr(black, black + 2.0), 0.0)
        a = rgb8(black)
        self.assertEqual(author_ssim(a, a), 1.0)
        b = a.copy()
        b[:, :, 0] = 100
        c1 = (0.01 * 255) ** 2
        self.assertAlmostEqual(author_ssim(a, b), (2 + c1 / (100**2 + c1)) / 3)
        np.testing.assert_array_equal(rgb8(np.array([0.5, 1.5, 2.5]) / 255), [0, 2, 2])

    def test_cuda_requirement(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(select_device("auto").type, "cpu")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                select_device("cuda")


class FusionCommandTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.low = self.root / "Low"
        self.low.mkdir()
        self.sample = np.zeros((12, 15, 3), np.uint8)
        self.sample[:, :, 0] = 100
        self.model = Predictor(torch.nn.Identity(), torch.nn.Identity(), torch.device("cpu"))
        self.args = ["--retinex-checkpoint", "r", "--hvi-checkpoint", "h", "--hvi-repo", "net"]

    def test_inference_single_and_folder_without_gt(self):
        path = self.low / "a.png"
        self.assertTrue(cv2.imwrite(str(path), self.sample))
        with patch.object(run_fusion, "build_predictor", return_value=self.model):
            for source, extra, name in ((path, [], "single"), (self.low, ["--compare"], "folder")):
                out = self.root / name
                run_fusion.main([*self.args, "--input", str(source), "--output", str(out), *extra])
                pngs = sorted(out.rglob("*.png"))
                self.assertEqual(len(pngs), 4 if extra else 1)
                for png in pngs:
                    np.testing.assert_array_equal(read_rgb(png), read_rgb(path))

    def test_evaluation_pairing_and_read_order(self):
        gt = self.root / "Normal"
        gt.mkdir()
        args = [*self.args, "--test-root", str(self.root)]
        for i in range(100):
            path = self.low / f"{i:03}.png"
            self.assertTrue(cv2.imwrite(str(path), self.sample))
            if i != 99:
                self.assertTrue(cv2.imwrite(str(gt / path.name), self.sample))
        with self.assertRaisesRegex(ValueError, "100 paired"):
            evaluate_fusion.paired_names(self.root)
        self.assertTrue(cv2.imwrite(str(gt / "099.png"), self.sample))
        out = self.root / "run"
        events = []

        def read(path):
            events.append(path.parent.name)
            return read_rgb(path)

        def predict(low, *, compare):
            events.append("predict")
            return self.model.predict(low, compare=compare)

        with patch.object(evaluate_fusion, "build_predictor") as build, patch.object(evaluate_fusion, "read_rgb", read):
            build.return_value.device = self.model.device
            build.return_value.predict.side_effect = predict
            evaluate_fusion.main([*args, "--output", str(out)])
            self.assertEqual(events, ["Low", "predict", "Normal"] * 100)
            self.assertTrue(cv2.imwrite(str(gt / "000.png"), self.sample[:-1]))
            with self.assertRaisesRegex(ValueError, "dimensions differ for 000"):
                evaluate_fusion.main([*args, "--output", str(self.root / "mismatch")])
        self.assertEqual(json.loads((out / "summary.json").read_text())["images"], 100)
        with (out / "per_image_official_test.csv").open() as f:
            rows = list(csv.DictReader(f))
        names = sorted(r["image"] for r in rows)
        reference = evaluate_fusion.read_reference(out / "per_image_official_test.csv", names)
        numeric = [{"image": r["image"], **{k: float(r[k]) for k in evaluate_fusion.FIELDS}} for r in reversed(rows)]
        summary = evaluate_fusion.summarize(numeric, reference)
        self.assertTrue(summary["dataset_means_within_tolerance"])
        numeric[0]["hvi_psnr_float"] += 2.0
        summary = evaluate_fusion.summarize(numeric, reference)
        self.assertFalse(summary["dataset_means_within_tolerance"])
        self.assertEqual(summary["comparison"]["hvi_psnr_float"]["worst_image"], "099.png")
