"""Fast synthetic checks for the isolated fine-tuning control flow."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

import finetune_v2_1 as ft


class Pairs(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        low = torch.full((3, 12, 12), 0.2 + index * 0.01)
        return dict(low=low, high=low + 0.4, name=f"{index}.png")


class FinetuneTests(unittest.TestCase):
    def fixtures(self, root, baseline=100.):
        (root / "original").mkdir()
        source = root / "original/original_best.pth"
        source_model = torch.nn.Conv2d(3, 3, 1)
        torch.save(dict(model=source_model.state_dict()), source)
        config = dict(seed=42, peak_lr=1e-4, optimizer=dict(betas=[.9, .999], weight_decay=0.),
                      use_amp=False, mixup_beta=1.2, grad_clip_norm=.01,
                      dataset_root="unused")
        info = dict(source_checkpoint=str(source), source_sha256=ft.sha256(source),
                    source_format="synthetic", source_epoch=50, source_batch=3,
                    source_global_step=1234, source_effective_lr=5e-5,
                    baseline_psnr=baseline, baseline_ssim=.7,
                    baseline_origin="checkpoint metadata", reference_lr=1e-4,
                    validation_protocol=ft.VALIDATION_PROTOCOL)
        manifest = dict(protocol=ft.PROTOCOL, splits=dict(train=["0", "1"], test=["0", "1"]))
        def source_info(*_):
            model = torch.nn.Conv2d(3, 3, 1)
            model.load_state_dict(torch.load(source, weights_only=True)["model"])
            return info, config, manifest, model
        def loaders(*_):
            generators = {key: torch.Generator().manual_seed(seed)
                          for key, seed in (("train", 42), ("validation", 43))}
            return DataLoader(Pairs(), batch_size=1), DataLoader(Pairs(), batch_size=1), generators
        return source, info, manifest, source_info, loaders

    def test_schedule_endpoints(self):
        self.assertAlmostEqual(ft.cosine_lr(1, 30, 2e-5), 2e-5)
        self.assertAlmostEqual(ft.cosine_lr(30, 30, 2e-5), 2e-6)
        self.assertAlmostEqual(ft.cosine_lr(1, 30, 1e-5), 1e-5)

    def test_source_preflight_loads_strictly_and_reports_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = torch.nn.Conv2d(3, 3, 1)
            config = dict(version="v2", gated=True, num_epochs=200, peak_lr=1e-4)
            path = root / "best.pth"
            torch.save(dict(format="snapshot", config=config, epoch=40, batch=12,
                            global_step=400, model=model.state_dict()), path)
            final = dict(config=config, manifest={"splits": {}},
                         best=dict(epoch=40, global_step=400, psnr=20.73, ssim=.7833),
                         history=[dict(lr=2e-5) for _ in range(201)])
            with patch.object(ft, "select_checkpoints", return_value=({"best": path}, final)), \
                 patch.object(ft, "validate_config"), \
                 patch.object(ft, "build_model", side_effect=lambda *_: torch.nn.Conv2d(3, 3, 1)):
                info, _, _, _ = ft.source_info(root)
                self.assertEqual((info["source_epoch"], info["source_batch"],
                                  info["source_global_step"]), (40, 12, 400))
                self.assertEqual(info["source_effective_lr"], 2e-5)
                self.assertEqual(info["source_sha256"], ft.sha256(path))
                torch.save(dict(format="snapshot", config=config, epoch=40, batch=12,
                                global_step=400, model={"wrong": torch.zeros(1)}), path)
                with self.assertRaises(RuntimeError):
                    ft.source_info(root)

    def test_smoke_isolation_logging_and_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, info, manifest, source_info, loaders = self.fixtures(root)
            original = source.read_bytes()
            with patch.object(ft, "source_info", side_effect=source_info), \
                 patch.object(ft, "inspect_dataset", return_value=(manifest, {})), \
                 patch.object(ft, "build_loaders", side_effect=loaders), \
                 patch.object(ft, "build_model", side_effect=lambda *_: torch.nn.Conv2d(3, 3, 1)):
                output = ft.run(root / "original", root / "outputs", "finetune_lr5", "cpu", smoke=True)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual((output / "checkpoints/best.pth").read_bytes(), original)
            with (output / "logs/finetune.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[1]["additional_epoch"]), 1)
            self.assertEqual(int(rows[1]["global_step"]), 1236)
            self.assertAlmostEqual(float(rows[1]["learning_rate"]), 2e-5)
            with (output / "logs/validation_per_image.csv").open(newline="") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 2)
            saved = torch.load(output / "checkpoints/latest.pth", weights_only=True)
            self.assertEqual(saved["source"]["source_sha256"], info["source_sha256"])
            self.assertEqual(json.loads((output / "selection.json").read_text())["best"]["additional_epoch"], 0)
            for name in ("training_loss", "validation_psnr", "validation_ssim", "learning_rate"):
                self.assertTrue((output / "visuals" / f"{name}.png").is_file())

    def test_resume_replays_only_uncommitted_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _, manifest, source_info, loaders = self.fixtures(root, baseline=-100.)
            original = source.read_bytes()
            original_train = ft.train_one_epoch
            calls = 0
            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("disconnect")
                return original_train(*args, **kwargs)
            with patch.object(ft, "source_info", side_effect=source_info), \
                 patch.object(ft, "inspect_dataset", return_value=(manifest, {})), \
                 patch.object(ft, "build_loaders", side_effect=loaders):
                with patch.object(ft, "train_one_epoch", side_effect=interrupted):
                    with self.assertRaisesRegex(RuntimeError, "disconnect"):
                        ft.run(root / "original", root / "outputs", "finetune_lr10", "cpu",
                               max_epochs=3, validation_interval=1)
                best_alias = root / "outputs/v2.1/finetune_lr10/checkpoints/best.pth"
                best_alias.unlink()  # Simulate interruption after latest.pth but before best.pth.
                output = ft.run(root / "original", root / "outputs", "finetune_lr10", "cpu",
                                resume=True, max_epochs=3, validation_interval=1)
            saved = torch.load(output / "checkpoints/latest.pth", weights_only=True)
            self.assertEqual(saved["additional_epoch"], 3)
            self.assertEqual([r["additional_epoch"] for r in saved["history"]], [0, 1, 2, 3])
            self.assertEqual(saved["global_step"], 1240)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(saved["best"]["origin"], "measured fine-tuning validation")
            self.assertTrue(best_alias.is_file())

    def test_resume_after_disconnect_in_first_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, manifest, source_info, loaders = self.fixtures(root)
            with patch.object(ft, "source_info", side_effect=source_info), \
                 patch.object(ft, "inspect_dataset", return_value=(manifest, {})), \
                 patch.object(ft, "build_loaders", side_effect=loaders):
                with patch.object(ft, "train_one_epoch", side_effect=RuntimeError("disconnect")):
                    with self.assertRaisesRegex(RuntimeError, "disconnect"):
                        ft.run(root / "original", root / "outputs", "finetune_lr5", "cpu",
                               max_epochs=2, validation_interval=1)
                output = ft.run(root / "original", root / "outputs", "finetune_lr5", "cpu",
                                resume=True, max_epochs=2, validation_interval=1)
            saved = torch.load(output / "checkpoints/latest.pth", weights_only=True)
            self.assertEqual(saved["additional_epoch"], 2)

    def test_comparison_keeps_original_when_neither_run_improves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, manifest, source_info, loaders = self.fixtures(root)
            with patch.object(ft, "source_info", side_effect=source_info), \
                 patch.object(ft, "inspect_dataset", return_value=(manifest, {})), \
                 patch.object(ft, "build_loaders", side_effect=loaders):
                for experiment in ("finetune_lr5", "finetune_lr10"):
                    ft.run(root / "original", root / "outputs", experiment, "cpu",
                           max_epochs=2, validation_interval=1, patience=0)
                rows = ft.compare(root / "outputs", root / "original")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["additional_epochs"] == 0 for row in rows))
            summary = json.loads((root / "outputs/v2.1/comparison/comparison.json").read_text())
            self.assertFalse(summary["psnr_improved"])
            self.assertEqual(summary["selected"]["model"], "Proposed V2 original best")
            self.assertTrue((root / "outputs/v2.1/comparison/comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
