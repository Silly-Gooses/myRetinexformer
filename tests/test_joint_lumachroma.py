import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import torch

from fusion_model.joint_lumachroma import color_fusion as color
from fusion_model.joint_lumachroma import cli, evaluation, model_loader, optimize_weights as opt
from fusion_model.joint_lumachroma import train_coefficients as train
from fusion_model.joint_lumachroma.coefficients import mse_coefs, psnr_from_coeff
from fusion_model.joint_lumachroma.config import CFG, METHODS
from fusion_model.joint_lumachroma.datasets import discover_splits, exact_pairs
from fusion_model.joint_lumachroma.inference import all_predictions, predict_hvi, predict_ret
from fusion_model.joint_lumachroma.io_utils import digest, load_parameters, locked_json, read_rgb, save_png
from fusion_model.joint_lumachroma.metrics import psnr_float, psnr_rgb8, rgb8_hvi_style, rgb8_ret_style, ssim_rgb8

REFERENCE = Path(__file__).resolve().parents[1] / "reference_results/joint_lumachroma_v2"


def quadratic_table(ay=.3, ac=.7, n=6):
    return pd.DataFrame([dict(Ayy=1., Acc=1., Ayc=0., By=-ay, Bc=-ac,
                              C=ay*ay+ac*ac+.02, rgb_a=1., rgb_b=-.4, rgb_c=.18)] * n)


class NumericalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.r, self.h, self.g = [rng.uniform(.15, .85, (48, 60, 3)).astype(np.float32) for _ in range(3)]

    def test_roundtrip_and_channel_order(self):
        np.testing.assert_allclose(color.ycrcb_to_rgb(color.rgb_to_ycrcb(self.r)), self.r, atol=5e-4, rtol=0)
        red = color.rgb_to_ycrcb(np.array([[[1., 0., 0.]]], np.float32))[0, 0]
        self.assertGreater(red[1], .9)  # Red raises Cr, not Cb.
        self.assertLess(red[2], .4)
        blue = color.rgb_to_ycrcb(np.array([[[0., 0., 1.]]], np.float32))[0, 0]
        self.assertGreater(blue[2], .9)

    def test_endpoints_equal_weights_shapes_and_range(self):
        for weights, expected in [((0, 0), self.h), ((1, 1), self.r),
                                   ((.35, .35), color.rgb_fuse(self.r, self.h, .35))]:
            result = color.dual_fuse(self.r, self.h, *weights)
            self.assertEqual(result.shape, self.r.shape)
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, expected, atol=5e-4, rtol=0)
        rng = np.random.default_rng(12)
        r, h = [rng.random((32, 40, 3), dtype=np.float32) for _ in range(2)]
        for ay, ac in [(0, 1), (1, 0), (.3, .7)]:
            result = color.dual_fuse(r, h, ay, ac)
            self.assertTrue(np.isfinite(result).all())
            self.assertGreaterEqual(result.min(), 0)
            self.assertLessEqual(result.max(), 1)

    def test_coefficients_match_unclipped_affine_mse(self):
        z = color.dual_mse_coeffs(self.r, self.h, self.g)
        base, dy, dc = color.dual_components(self.r, self.h)
        for ay, ac in [(0, 0), (1, 1), (.3, .7), (.9, .1)]:
            raw = base + ay * dy + ac * dc
            self.assertLess(abs(float(np.mean((raw-self.g)**2)) - color.dual_mse(z, ay, ac)), 2e-7)
        a, b, c = mse_coefs(self.r, self.h, self.g)
        self.assertLess(abs(psnr_from_coeff(a, b, c, .35) - psnr_float(.35*self.r+.65*self.h, self.g)), 1e-5)

    def test_clipping_is_not_inserted_in_analytic_score(self):
        r = np.full((12, 12, 3), [1, 0, 0], np.float32)
        h = np.ones_like(r)
        gt = np.zeros_like(r)
        base, dy, dc = color.dual_components(r, h)
        raw = base + dc
        self.assertGreater(raw.max(), 1)
        z = color.dual_mse_coeffs(r, h, gt)
        self.assertAlmostEqual(color.dual_mse(z, 0, 1), float(np.mean(raw.astype(np.float64)**2)), places=6)
        self.assertNotAlmostEqual(color.dual_mse(z, 0, 1), float(np.mean(color.dual_fuse(r, h, 0, 1)**2)), places=3)

    def test_reference_metrics_and_rgb8_conventions(self):
        image = np.full((16, 16, 3), .5, np.float32)
        self.assertEqual(psnr_float(image, image), 140.)
        np.testing.assert_array_equal(rgb8_ret_style(image), 128)
        np.testing.assert_array_equal(rgb8_hvi_style(image), 127)
        unit = np.full((32, 32, 3), 128, np.uint8)
        self.assertGreater(ssim_rgb8(unit, unit), .999999)
        self.assertGreater(psnr_rgb8(unit, unit), 100)
        self.assertAlmostEqual(psnr_float(image, np.zeros_like(image)), 6.020599913279624)

    def test_folds_determinism_and_balance(self):
        for ds, n, counts in [("LOLv1", 485, [162, 162, 161]),
                              ("LOLv2_Real", 689, [230, 230, 229]),
                              ("LOLv2_Synthetic", 900, [300, 300, 300])]:
            a = opt.fold_labels(ds, pd.DataFrame(index=range(n)))
            np.testing.assert_array_equal(a, opt.fold_labels(ds, pd.DataFrame(index=range(n))))
            self.assertEqual(np.bincount(a).tolist(), counts)
        self.assertFalse(np.array_equal(opt.fold_labels("LOLv1", range(30)), opt.fold_labels("LOLv2_Real", range(30))))

    def test_macro_objective_and_seven_starts(self):
        tables = [quadratic_table(.2, .8, 3), quadratic_table(.7, .1, 17)]
        calls = []
        def minimize(objective, start, **kwargs):
            x = (.25, .75)
            expected = -np.mean([opt.ycc_score(t, *x) for t in tables]) + .5*((x[0]-.5)**2+(x[1]-.5)**2)
            self.assertAlmostEqual(objective(x), expected)
            self.assertNotAlmostEqual(-opt.ycc_score(pd.concat(tables), *x), -np.mean([opt.ycc_score(t, *x) for t in tables]))
            self.assertEqual(kwargs, dict(bounds=[(0, 1), (0, 1)], method="L-BFGS-B", options={"maxiter": 220, "ftol": 1e-12}))
            calls.append(tuple(start))
            return SimpleNamespace(x=start)
        with patch.object(opt, "minimize", minimize):
            opt.fit_ycc(tables, .5, .5)
        self.assertEqual(calls, [(0, 0), (1, 1), (0, 1), (1, 0), (.5, .5), (.2, .8), (.8, .2)])

    def test_optimizer_optimum_and_tie_break(self):
        np.testing.assert_allclose(opt.fit_ycc([quadratic_table()]), [.3, .7], atol=1e-6)
        self.assertAlmostEqual(opt.fit_rgb(quadratic_table()), .4, places=6)
        cv = pd.DataFrame([dict(scope="joint", dataset="LOLv1", lam=l, holdout_psnr=s)
                           for l, s in [(0., 10.004), (.5, 10.), (2., 9.999), (5., 9.998)]])
        self.assertEqual(opt.chosen_lambda(cv, "joint"), 2.)

    def test_frozen_parameters_full_precision(self):
        p = load_parameters(REFERENCE / "selected_weights_train_only.json", list(CFG))
        self.assertEqual(p["joint"]["alpha_y_regularized"], .2992029777427523)
        self.assertEqual(p["joint"]["alpha_c_regularized"], .40713060525512)
        self.assertEqual(list(all_predictions(self.r, self.h, p["local"]["LOLv1"], p["joint"])), METHODS)

    def test_complete_cv_and_frozen_json_export(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = opt.fit_tables({"LOLv1": quadratic_table()}, out)
            self.assertEqual(result["experiment"], "regjoint_v2")
            self.assertEqual(json.loads((out / "selected_weights_train_only.json").read_text()), result)
            folds = pd.read_csv(out / "train_cross_validation_folds.csv")
            self.assertEqual(len(folds), 42)  # 2 scopes * 7 lambdas * 3 folds
            self.assertTrue((folds.n_train == 4).all())
            self.assertTrue((folds.n_holdout == 2).all())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = np.full((16, 24, 3), 128, np.uint8)
        self.params = json.loads((REFERENCE / "selected_weights_train_only.json").read_text())
        self.configs = {"LOLv1": {**CFG["LOLv1"], "root": self.root, "n_test": 1}}
        self.weights = {"LOLv1": {"ret": "r", "hvi": "h", "ret_sha256": "r-sha", "hvi_sha256": "h-sha"}}
        for split, count in [("Train", 3), ("Test", 1)]:
            for folder in ["input", "target"]:
                for i in range(count):
                    save_png(self.root / split / folder / f"{i}.png", self.image)

    def test_pairing_overlap_and_reused_basenames(self):
        splits = discover_splits(self.configs, ["LOLv1"])
        self.assertEqual(splits["LOLv1"]["fit"][0][0].name, splits["LOLv1"]["test"][0][0].name)
        broken = {"LOLv1": {**self.configs["LOLv1"], "test_low": "Train/input", "test_gt": "Train/target", "n_test": 3}}
        with self.assertRaisesRegex(ValueError, "overlap"):
            discover_splits(broken, ["LOLv1"])
        save_png(self.root / "Train/input/extra.png", self.image)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            exact_pairs(self.root / "Train/input", self.root / "Train/target")

    def test_train_pipeline_does_not_read_test_pixels_or_gt(self):
        # Valid filenames but unreadable Test content: fitting must still succeed.
        for folder in ["input", "target"]:
            (self.root / "Test" / folder / "0.png").write_bytes(b"Do not read during fitting")
        reads = []
        def reader(path):
            reads.append(path)
            self.assertIn("Train", path.parts)
            return read_rgb(path)
        def fitter(tables, output):
            self.assertEqual(tables["LOLv1"].filename.tolist(), ["0.png", "1.png", "2.png"])
            return self.params
        with patch.object(train, "load_models", return_value=(torch.nn.Identity(), torch.nn.Identity())), \
                patch.object(train, "read_rgb", reader), patch.object(train, "fit_tables", fitter):
            train.fit_experiment(self.configs, ["LOLv1"], self.weights, self.root, self.root,
                                 torch.device("cpu"), self.root / "fit")
        self.assertEqual(len(reads), 6)

    def test_cache_and_protocol_rejection(self):
        lock = self.root / "lock.json"
        locked_json(lock, {"fit_count": None})
        locked_json(lock, {"fit_count": None})
        with self.assertRaisesRegex(ValueError, "changed"):
            locked_json(lock, {"fit_count": 15})
        df = quadratic_table()
        df["filename"] = [f"{i}.png" for i in range(len(df))]
        df["dataset"] = "LOLv1"
        with self.assertRaisesRegex(ValueError, "stale"):
            train.validate_cache(df, "LOLv1", list(reversed(df.filename.tolist())))

    def test_frozen_evaluation_outputs_and_no_refitting(self):
        events = []
        def reader(path):
            events.append(path.parent.name)
            return read_rgb(path)
        def predict(net, low, device):
            events.append("predict")
            return low.astype(np.float32) / 255.
        out = self.root / "evaluation"
        with patch.object(evaluation, "load_models", return_value=(None, None)), \
                patch.object(evaluation, "read_rgb", reader), \
                patch.object(evaluation, "predict_ret", predict), patch.object(evaluation, "predict_hvi", predict), \
                patch.object(opt, "fit_ycc", side_effect=AssertionError("No refitting")):
            result = evaluation.evaluate_experiment(self.configs, ["LOLv1"], self.weights, self.root, self.root,
                torch.device("cpu"), out, REFERENCE / "selected_weights_train_only.json", smoke_max_test=1)
        self.assertEqual(events, ["input", "predict", "predict", "target"])
        self.assertEqual(result.evaluation_type.tolist(), ["smoke"])
        self.assertEqual(len(list(out.rglob("*.png"))), 11)
        for name in ["selected_weights_train_only.json", "test_per_image_all.csv", "test_summary.csv",
                     "report_manifest.json", "evaluation_lock.json", "RetinexFormer_HVI_RegularizedJoint_LumaChroma_Report.zip"]:
            self.assertTrue((out / name).is_file(), name)
        manifest = json.loads((out / "report_manifest.json").read_text())
        self.assertEqual(manifest["result_status"], "new_smoke_run")

    def test_padding_crop_and_wide_image_split(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shapes = []
            def forward(self, x):
                self.shapes.append(tuple(x.shape))
                return x
        for shape, expected in [((13, 19, 3), [(1, 3, 16, 20)]),
                                ((12, 3001, 3), [(1, 3, 12, 1504), (1, 3, 12, 1504)])]:
            model = Spy()
            im = np.full(shape, 100, np.uint8)
            result = predict_ret(model, im, torch.device("cpu"))
            self.assertEqual(model.shapes, expected)
            np.testing.assert_array_equal(result, im.astype(np.float32)/255.)
        model = Spy()
        result = predict_hvi(model, np.full((13, 19, 3), 100, np.uint8), torch.device("cpu"))
        self.assertEqual(model.shapes, [(1, 3, 16, 24)])
        self.assertEqual(result.shape, (13, 19, 3))

    def test_dataset_gates_and_state_unwrap(self):
        for ds, gated, gated2 in [("LOLv1", True, False), ("LOLv2_Real", False, True),
                                 ("LOLv2_Synthetic", False, False)]:
            net = SimpleNamespace(trans=SimpleNamespace(gated=None, gated2=None, alpha=1.))
            model_loader.configure_hvi(net, ds)
            self.assertEqual((net.trans.gated, net.trans.gated2), (gated, gated2))
            self.assertEqual(net.trans.alpha, .8 if gated2 else 1.)
        p = self.root / "weights.pth"
        torch.save({"params": {"module.weight": torch.ones(2, 3)}, "params_ema": {}}, p)
        self.assertEqual(list(model_loader.unwrap_state(p)), ["weight"])
        torch.save({}, p)
        with self.assertRaisesRegex(ValueError, "Empty"):
            model_loader.unwrap_state(p)

    def test_checkpoint_hash_and_source_revision_rejection(self):
        ret_dir, hvi_dir = self.root / "ret", self.root / "hvi"
        ret_dir.mkdir()
        (hvi_dir / "LOLv1").mkdir(parents=True)
        (ret_dir / "LOL_v1.pth").write_bytes(b"wrong checkpoint")
        (hvi_dir / "LOLv1/w_perc.pth").write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            model_loader.resolve_weights(["LOLv1"], ret_dir, hvi_dir)
        with patch.object(model_loader.subprocess, "check_output", side_effect=["wrong\n", ""]):
            with self.assertRaisesRegex(ValueError, "Expected clean source"):
                model_loader.verify_source(self.root, "expected", "net")

    def test_strict_frozen_model_loading(self):
        arch = self.root / "basicsr/models/archs/RetinexFormer_arch.py"
        arch.parent.mkdir(parents=True)
        arch.write_text("import torch\nclass RetinexFormer(torch.nn.Linear):\n"
                        "    def __init__(self, **kwargs):\n        super().__init__(3, 2)\n")
        rpath, hpath = self.root / "ret.pth", self.root / "hvi.pth"
        state = torch.nn.Linear(3, 2).state_dict()
        torch.save({"params": state}, rpath)
        torch.save(state, hpath)
        def hvi_class():
            net = torch.nn.Linear(3, 2)
            net.trans = SimpleNamespace(gated=False, gated2=False, alpha=1.)
            return net
        with patch.object(model_loader, "verify_source", return_value=self.root), \
                patch.object(model_loader, "load_cidnet_class", return_value=hvi_class), \
                patch.dict(model_loader.RET_EXPECTED_HASH, {"LOL_v1.pth": digest(rpath)}), \
                patch.dict(model_loader.HVI_REFERENCE_HASH, {"LOLv1": digest(hpath)}):
            ret, hvi = model_loader.load_models({"ret": rpath, "hvi": hpath}, "LOLv1", self.root, self.root, torch.device("cpu"))
            self.assertFalse(ret.training or hvi.training)
            self.assertTrue(all(not p.requires_grad for model in [ret, hvi] for p in model.parameters()))
            torch.save({"params": {"weight": torch.ones(4, 3), "bias": torch.ones(4)}}, rpath)
            with patch.dict(model_loader.RET_EXPECTED_HASH, {"LOL_v1.pth": digest(rpath)}):
                with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                    model_loader.load_models({"ret": rpath, "hvi": hpath}, "LOLv1", self.root, self.root, torch.device("cpu"))

    def test_inference_cli_has_no_ground_truth_dependency(self):
        out = self.root / "inference"
        args = ["--dataset", "LOLv1", "--input", str(self.root / "Train/input/0.png"),
                "--weights-json", str(REFERENCE / "selected_weights_train_only.json"), "--output", str(out),
                "--ret-repo", "ret", "--hvi-repo", "hvi", "--ret-weights", "r", "--hvi-weights", "h", "--compare"]
        with patch.object(cli, "setup", return_value=(self.configs, self.weights, torch.device("cpu"))), \
                patch.object(cli, "load_models", return_value=(torch.nn.Identity(), torch.nn.Identity())):
            cli.run_main(args)
        self.assertEqual(len(list(out.rglob("*.png"))), 7)
        self.assertTrue((out / "inference_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
