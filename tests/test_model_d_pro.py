"""CPU migration parity, checkpoint, sampler, and training-state regression tests."""
import copy
import json
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from model_d.model import ModelD, build_model_b, HVIColorHead, REFERENCE, hvi_to_rgb
from model_d_pro import ModelDPro, load_model_d_pro
from model_d_pro.checkpoint import (ROOT, SCHEMA, empty_model, source_manifest,
                                   atomic_torch_save, prepare_run, rng_metadata,
                                   restore_training_state, validate_resume_payload)
from model_d_pro.cli import DEFAULT_CONFIG, compare_historical, parser, smoke
from model_d_pro.dataset import PairedStore
from model_d_pro.evaluate import evaluate
from model_d_pro.losses import training_loss
from model_d_pro.model import safe_hvi_to_rgb
from model_d_pro.runtime import Budget, BudgetStop, write_json
from model_d_pro.train import (pending_validation, protect_output, set_learning_rates,
                               validate_checkpoint, validate_config)
from model_d_pro.verify import pilot_definitions, verify_model

HISTORY = ROOT / "experiments/model_d_pro/phase1"
CFG = json.loads(DEFAULT_CONFIG.read_text())


def setUpModule():
    global threads, rng
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(2)


def tearDownModule():
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def initialized_original():
    original = ModelD(build_model_b(), HVIColorHead())
    # Learned-like nonzero NAF/head outputs, rather than identity-only fixtures.
    with torch.no_grad():
        for name, parameter in original.named_parameters():
            if "project" in name or "head.net.6" in name:
                parameter.normal_(0, .01)
    return original


class ArchitectureTests(unittest.TestCase):
    def test_seeded_initialization_matches_source(self):
        original = initialized_original()
        reference_class = pilot_definitions()["ModelDPro"]
        rng_state = torch.get_rng_state()
        actual = ModelDPro(original)
        torch.set_rng_state(rng_state)
        expected = reference_class(original)
        for key, tensor in actual.state_dict().items():
            torch.testing.assert_close(tensor, expected.state_dict()[key], rtol=0, atol=0)

    def test_independent_forward_loss_and_gradient_parity(self):
        torch.manual_seed(42)
        definitions = pilot_definitions()
        original = initialized_original()
        model = ModelDPro(original)
        reference = definitions["ModelDPro"](original)
        with torch.no_grad():
            for parameter in model.parameters():
                if torch.count_nonzero(parameter) == 0:
                    parameter.normal_(0, .001)
        self.assertEqual(list(model.state_dict()), list(reference.state_dict()))
        reference.load_state_dict(model.state_dict(), strict=True)
        counts = {g["group_name"]: sum(p.numel() for p in g["params"])
                  for g in model.parameter_groups(CFG)}
        self.assertEqual(counts, json.loads((HISTORY / "parameters.json").read_text()))
        for shape in ((1, 3, 16, 20), (2, 3, 17, 19), (1, 3, 1, 1)):
            image = torch.rand(shape)
            target = torch.rand(shape)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            actual, expected = model(image, True), reference(image, True)
            self.assertEqual(actual["output"].shape, image.shape)
            for key in expected:
                self.assertTrue(torch.isfinite(actual[key]).all())
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=1e-6)
            loss, parts = training_loss(actual, target)
            ref_loss, ref_parts = definitions["training_loss"](expected, target)
            for key in parts:
                torch.testing.assert_close(parts[key], ref_parts[key], rtol=0, atol=0)
            torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
            loss.backward()
            ref_loss.backward()
            for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
                self.assertEqual(p.grad is None, q.grad is None, name)
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=1e-6)

    def test_zero_start_float_identity_and_neutral_gradients(self):
        original = initialized_original().eval()
        model = ModelDPro(original).eval()
        image = torch.rand(1, 3, 16, 20)
        with torch.no_grad():
            expected = original.head(original.backbone(image).float().clamp(0, 1))[0]
            torch.testing.assert_close(model(image), expected, rtol=0, atol=2e-5)
        hv = torch.zeros(1, 2, 8, 8, requires_grad=True)
        intensity = torch.full((1, 1, 8, 8), .2, requires_grad=True)
        actual = safe_hvi_to_rgb(hv, intensity)
        torch.testing.assert_close(actual, hvi_to_rgb(hv, intensity), rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(hv.grad).all())
        self.assertTrue(torch.isfinite(intensity.grad).all())

    def test_warmup_and_gradient_groups(self):
        model = ModelDPro(initialized_original())
        groups = model.parameter_groups(CFG)
        parameters = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(parameters), len(set(parameters)))
        self.assertEqual(set(parameters), {id(p) for p in model.parameters()})
        self.assertEqual(model.set_phase(99, 100), "new_modules_warmup")
        for group in groups:
            self.assertTrue(all(p.requires_grad == (group["group_name"] == "new")
                                for p in group["params"]))
        self.assertEqual(model.set_phase(100, 100), "joint")
        report = smoke(model, torch.rand(1, 3, 24, 24), CFG, {})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["optimizer_updates"], 0)

    def test_loss_coefficients(self):
        image = torch.rand(1, 3, 9, 11)
        values = dict(output=image, intensity=image.amax(1, keepdim=True),
                      backbone_rgb=image, delta_i=torch.ones(1, 1, 9, 11),
                      delta_hv=torch.ones(1, 2, 9, 11) * 2)
        loss, parts = training_loss(values, image)
        self.assertEqual(set(parts), {"rgb", "intensity", "hv", "backbone", "residual"})
        torch.testing.assert_close(loss, torch.tensor(.006), rtol=0, atol=1e-8)


class DataAndProtocolTests(unittest.TestCase):
    def test_full_validation_matches_source_protocol(self):
        rng = np.random.default_rng(42)
        names = [f"{i:05d}.png" for i in range(69)]
        store = SimpleNamespace(val_names=names, val_sha="synthetic", images={})
        for name in names:
            low = rng.integers(10, 200, (16, 20, 3), dtype=np.uint8)
            store.images[name] = (low, low + 5)
        original_evaluate = pilot_definitions()["evaluate"]
        with tempfile.TemporaryDirectory() as tmp:
            actual = evaluate(torch.nn.Identity(), store, torch.device("cpu"),
                              Path(tmp) / "actual", Budget(5), "synthetic")
            expected = original_evaluate(torch.nn.Identity(), store, torch.device("cpu"),
                                         Path(tmp) / "expected", Budget(5), "synthetic")
            for key in ("count", "psnr", "ssim", "dataset_sha256", "metric_protocol", "test_evaluated"):
                self.assertEqual(actual[key], expected[key])
            self.assertEqual((Path(tmp) / "actual.csv").read_bytes(),
                             (Path(tmp) / "expected.csv").read_bytes())

    def test_sampler_matches_executed_script(self):
        reference_class = pilot_definitions()["PairedStore"]
        store, reference = PairedStore.__new__(PairedStore), reference_class.__new__(reference_class)
        rng = np.random.default_rng(42)
        images = {name: [rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)] * 2
                  for name in ["a.png", "b.png"]}
        for obj in (store, reference):
            obj.train_names, obj.images = list(images), images
        cfg = dict(CFG, patch=12)
        for step, micro in ((0, 0), (0, 3), (100, 2), (1199, 3)):
            actual, target = store.batch(step, micro, cfg, "cpu")
            expected, _ = reference.batch(step, micro, cfg, "cpu")
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual, store.batch(step, micro, cfg, "cpu")[0], rtol=0, atol=0)

    def test_frozen_split_and_malformed_datasets(self):
        frozen = json.loads((HISTORY / "split.json").read_text())
        self.assertEqual(frozen, {k: REFERENCE["config"][k + "_names"] for k in ("train", "val")})
        self.assertEqual(len(frozen["train"]), 620)
        self.assertEqual(len(frozen["val"]), 69)
        self.assertFalse(set(frozen["train"]) & set(frozen["val"]))
        with patch("analysis_utils.find_paired_images", return_value=[]):
            with self.assertRaisesRegex(ValueError, "620/69"):
                PairedStore("unused", frozen["train"], frozen["val"], 128, Budget(5))

    def test_reference_hash_and_historical_metric_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "wrong.py"
            source.write_text("raise RuntimeError('must never execute')")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                pilot_definitions(source)
        result = json.loads((HISTORY / "val_00900.json").read_text())
        report = compare_historical(result, HISTORY / "val_00900.csv")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["per_image_deltas"]), 69)
        report = compare_historical(dict(result, psnr=result["psnr"] + .003), HISTORY / "val_00900.csv")
        self.assertEqual(report["status"], "failed")


class CheckpointTests(unittest.TestCase):
    def test_historical_strict_roundtrip_and_rejections(self):
        model = empty_model()
        identity = json.loads((HISTORY / "identity.json").read_text())
        payload = dict(model=model.state_dict(), identity=identity, step=900)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pth"
            atomic_torch_save(payload, path)
            loaded, info = load_model_d_pro(checkpoint=path)
            self.assertEqual(info["step"], 900)
            self.assertTrue(info["weights_only_initialization"])
            for key, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, loaded.state_dict()[key], rtol=0, atol=0)
            self.assertEqual(verify_model(loaded, torch.rand(1, 3, 17, 19), info)["status"], "passed")
            last_payload = dict(model=payload["model"], identity=identity, state={"completed_steps": 1200})
            atomic_torch_save(last_payload, path)
            _, last_info = load_model_d_pro(checkpoint=path)
            self.assertEqual(last_info["step"], 1200)
            for mutation in ("missing", "unexpected", "shape", "nan", "identity"):
                broken = copy.deepcopy(payload)
                key = next(iter(broken["model"]))
                if mutation == "missing":
                    del broken["model"][key]
                elif mutation == "unexpected":
                    broken["model"]["extra"] = torch.zeros(1)
                elif mutation == "shape":
                    broken["model"][key] = torch.zeros(1)
                elif mutation == "nan":
                    broken["model"][key].fill_(float("nan"))
                else:
                    broken["identity"]["script_sha256"] = "wrong"
                atomic_torch_save(broken, path)
                with self.assertRaises((ValueError, RuntimeError)):
                    load_model_d_pro(checkpoint=path)
            with self.assertRaisesRegex(ValueError, "Historical last"):
                validate_resume_payload(payload, identity)

    def test_original_pair_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            wrong = Path(tmp) / "head.pth"
            wrong.write_bytes(b"wrong")
            with patch("torch.load") as loader:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    load_model_d_pro(model_b=wrong, head=wrong)
                loader.assert_not_called()

    def test_source_identity_and_foreign_run_protection(self):
        manifest = source_manifest()
        self.assertIn("model_d/model.py", manifest["protected"])
        self.assertIn("model_d_pro/model.py", manifest["package"])
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            marker = run / "identity.json"
            marker.write_text('{"version":"historical"}')
            before = marker.read_bytes()
            with self.assertRaisesRegex(ValueError, "Historical"):
                prepare_run(run, {"version": SCHEMA}, True)
            self.assertEqual(marker.read_bytes(), before)
            self.assertEqual(len(list(run.iterdir())), 1)


class TrainingStateTests(unittest.TestCase):
    def test_lr_boundaries(self):
        opt = SimpleNamespace(param_groups=[dict(group_name=k) for k in CFG["learning_rates"]])
        for step in (0, 24, 25, 99, 100, 300, 1199, 1200):
            set_learning_rates(opt, CFG, step)
            factor = .1 + .45 * (1 + math.cos(math.pi * max(0, (step-100)/1100)))
            for group in opt.param_groups:
                name = group["group_name"]
                expected = CFG["learning_rates"][name] * factor
                if name != "new" and step < 100:
                    expected = 0
                elif name == "new":
                    expected *= min(1, (step+1)/25)
                self.assertAlmostEqual(group["lr"], expected, places=14)

    def test_interrupted_validation_leaves_pending_state(self):
        state = dict(completed_steps=300, last_validated_step=0, finished=False,
                     full_validation_count=1, best={"psnr": 20.}, best_step=0,
                     patience=0, patience_reference=20.)
        self.assertTrue(pending_validation(state, CFG))
        before = copy.deepcopy(state)
        with tempfile.TemporaryDirectory() as tmp:
            with patch("model_d_pro.train.evaluate", side_effect=BudgetStop("pause")):
                with self.assertRaises(BudgetStop):
                    validate_checkpoint(None, None, None, Path(tmp), None, state,
                                        {"psnr": 20.}, CFG, {})
            self.assertEqual(state, before)
            self.assertFalse(list(Path(tmp).iterdir()))
            with patch("model_d_pro.train.evaluate", return_value={"psnr": 21., "ssim": .7}):
                validate_checkpoint(torch.nn.Linear(1, 1), None, None, Path(tmp), None,
                                    state, {"psnr": 20.}, CFG, {})
            self.assertEqual(state["best_step"], 300)
            self.assertFalse(pending_validation(state, CFG))
            state.update(completed_steps=1200, last_validated_step=900)
            self.assertTrue(pending_validation(state, CFG))
            state["finished"] = True
            self.assertFalse(pending_validation(state, CFG))

    def test_full_state_resume_matches_uninterrupted_cpu_updates(self):
        identity = {"version": SCHEMA, "recipe": CFG}
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        def update(m, opt, scale):
            opt.zero_grad(set_to_none=True)
            loss = m(torch.rand(3, 2)).square().sum()
            scale.scale(loss).backward()
            scale.step(opt)
            scale.update()
        update(model, optimizer, scaler)
        payload = dict(identity=identity, model=model.state_dict(), optimizer=optimizer.state_dict(),
                       scaler=scaler.state_dict(), state={"completed_steps": 1},
                       torch_rng=torch.get_rng_state(), cuda_rng=[], **rng_metadata())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pth"
            atomic_torch_save(payload, path)
            update(model, optimizer, scaler)
            expected_random = (random.random(), float(np.random.rand()))
            fresh = torch.nn.Linear(2, 1)
            opt = torch.optim.AdamW(fresh.parameters(), lr=.01, foreach=False)
            scale = torch.amp.GradScaler("cpu", enabled=False)
            saved = torch.load(path, weights_only=True)
            state = restore_training_state(saved, identity, fresh, opt, scale, "cpu")
            self.assertEqual(state["completed_steps"], 1)
            update(fresh, opt, scale)
            self.assertEqual((random.random(), float(np.random.rand())), expected_random)
            for actual, expected in zip(fresh.parameters(), model.parameters()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "identity"):
                restore_training_state(saved, dict(identity, recipe={}), fresh, opt, scale, "cpu")

    def test_config_and_output_guards(self):
        validate_config(CFG, 45)
        for changes in ({"save_every": 0}, {"micro_batch": 3}, {"run_name": "../pilot_01"}):
            with self.assertRaises(ValueError):
                validate_config(dict(CFG, **changes), 45)
        with self.assertRaises(ValueError):
            protect_output(ROOT / "datasets/run", [ROOT / "datasets"])
        args = parser().parse_args(["train"])
        self.assertFalse(args.worker)
        self.assertFalse(CFG["resume"])

    def test_no_default_notebook_training(self):
        import ast
        notebook = json.loads((ROOT / "notebooks/Model_DPro_Phase1_Colab.ipynb").read_text())
        settings = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                tree = ast.parse("".join(cell["source"]))
                for node in tree.body:
                    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                        for name in node.targets:
                            if isinstance(name, ast.Name):
                                settings[name.id] = node.value.value
        self.assertIs(settings["RUN_TRAINING"], False)
        self.assertIs(settings["RUN_VALIDATION"], False)
        self.assertNotEqual(settings["RUN_NAME"], "pilot_01")


class RealCheckpointTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MODEL_D_PRO_CHECKPOINT"),
                         "Unverified: set MODEL_D_PRO_CHECKPOINT to historical Phase 1 best.pth")
    def test_real_best_strict_load_and_parity(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, info = load_model_d_pro(checkpoint=os.environ["MODEL_D_PRO_CHECKPOINT"], device=device)
        self.assertEqual(verify_model(model, torch.rand(1, 3, 32, 36, device=device), info)["status"], "passed")

    @unittest.skipUnless(os.environ.get("MODEL_D_BACKBONE") and os.environ.get("MODEL_D_HEAD"),
                         "Unverified: original trained Model B/HVI pair unavailable")
    def test_original_pair(self):
        model, info = load_model_d_pro(model_b=os.environ["MODEL_D_BACKBONE"], head=os.environ["MODEL_D_HEAD"])
        self.assertEqual(verify_model(model, torch.rand(1, 3, 32, 36), info)["status"], "passed")


if __name__ == "__main__":
    unittest.main()
