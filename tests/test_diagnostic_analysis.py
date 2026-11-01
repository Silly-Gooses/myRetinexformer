"""Synthetic, read-only checks for the V2 notebook diagnostic workflow."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

import diagnostic_analysis as diagnostic
from official_workflow import FORMAT, SNAPSHOT_FORMAT
from train import build_model


class DiagnosticAnalysisTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        config = json.loads((Path(__file__).resolve().parents[1] /
                             "official_lolv2_v2_config.json").read_text())
        config.update(dataset_root=str(root / "data"), batch_size=2)
        config["architecture"].update(n_feat=4, num_blocks=[0, 0, 0])
        pairs = []
        for index in range(3):
            low_path = root / "data/Test/Low" / f"{index}.png"
            gt_path = root / "data/Test/Normal" / f"{index}.png"
            low_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            low = np.full((16, 20, 3), 20 + index*10, dtype=np.uint8)
            low[:, 10:] += 45
            Image.fromarray(low).save(low_path)
            Image.fromarray(low + 80).save(gt_path)
            pairs.append((f"{index}.png", low_path, gt_path))
        manifest = {"splits": {"train": list(range(3)),
                               "test": [{"name": item[0]} for item in pairs]}}
        experiment = root / "run"
        (experiment / "checkpoints").mkdir(parents=True)
        (experiment / "logs").mkdir()
        model = build_model(config, "cpu")
        snapshot = "selected_best_snapshot.pth"
        best_info = dict(epoch=58, global_step=116, psnr=20., ssim=.7,
                         snapshot=snapshot)
        best_path = experiment / "checkpoints" / snapshot
        torch.save(dict(format=SNAPSHOT_FORMAT, config=config, model=model.state_dict(),
                        epoch=58, global_step=116), best_path)
        final_path = experiment / "checkpoints/last.pth"
        torch.save(dict(format=FORMAT, config=config, manifest=manifest,
                        model=model.state_dict(), completed_epoch=200, global_step=400,
                        best=best_info, history=[dict(epoch=0, global_step=0, train_loss=None, lr=None),
                                                 dict(epoch=200, global_step=400, train_loss=.1, lr=1e-6)],
                        validation_history=[dict(epoch=58, global_step=116, psnr=20., ssim=.7)]),
                   final_path)
        with (experiment / "logs/train_log.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("epoch", "global_step", "train_loss", "lr"))
            writer.writeheader()
            writer.writerow(dict(epoch=200, global_step=400, train_loss=.1, lr=1e-6))
        with (experiment / "logs/validation_log.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=("epoch", "global_step", "psnr", "ssim"))
            writer.writeheader()
            writer.writerow(dict(epoch=58, global_step=116, psnr=20., ssim=.7))
        return experiment, config, manifest, pairs, best_path, final_path

    def test_complete_analysis_outputs_and_original_files_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment, config, manifest, pairs, best_path, final_path = self.fixture(temporary)
            original = {path: path.read_bytes() for path in (best_path, final_path,
                experiment / "logs/train_log.csv", experiment / "logs/validation_log.csv")}
            paths, final = diagnostic.select_checkpoints(experiment)
            self.assertEqual(paths, {"best": best_path, "final": final_path})
            training, validation, notes = diagnostic.read_logs(experiment, final)
            self.assertTrue(any("Gradient norm" in note for note in notes))
            output = diagnostic.create_run_directory(experiment)
            self.assertNotEqual(output, diagnostic.create_run_directory(experiment))
            plots = diagnostic.save_stability_plots(training, validation, 116, output)
            self.assertEqual(len(plots), 4)
            with patch.object(diagnostic, "inspect_dataset", return_value=(
                    {"splits": {"test": manifest["splits"]["test"]}}, {"test": pairs})):
                metrics, edges, visuals = diagnostic.analyze_images(
                    paths, final, config["dataset_root"], 3, "cpu", visual_count=2)
            self.assertEqual(len(metrics), 6)
            self.assertEqual(len(edges), 6)
            self.assertEqual(len(visuals), 4)
            self.assertTrue(all("edge_psnr_delta" in row for row in edges))
            self.assertTrue(all(0 <= row["edge_pixel_density"] <= 1 for row in edges))
            comparison = diagnostic.compare_checkpoints(metrics)
            self.assertEqual(len(comparison), 3)
            diagnostic.save_visuals(visuals, output)
            self.assertEqual(len(list((output / "visuals").glob("*.png"))), 4)
            diagnostic.export_results(output, metrics, edges, comparison, final,
                                      notes, paths)
            for name in ("illumination_denoising_analysis.csv", "edge_analysis.csv",
                         "checkpoint_comparison.csv", "aggregate_summary.json",
                         "diagnostic_report.md"):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual({path: path.read_bytes() for path in original}, original)

    def test_epoch_200_checkpoint_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment, _, _, _, _, final_path = self.fixture(temporary)
            state = torch.load(final_path, weights_only=True)
            archive = experiment / "checkpoints/epoch_0200.pth"
            archive.write_bytes(final_path.read_bytes())
            state["completed_epoch"] = 300
            torch.save(state, final_path)
            paths, _ = diagnostic.select_checkpoints(experiment)
            self.assertEqual(paths["final"], archive)
            archive.unlink()
            with self.assertRaisesRegex(ValueError, "epoch-200"):
                diagnostic.select_checkpoints(experiment)

    def test_zero_edge_hook_is_removed_after_inference_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment, config, manifest, pairs, _, _ = self.fixture(temporary)
            paths, final = diagnostic.select_checkpoints(experiment)
            model = build_model(config, "cpu").eval()
            calls = 0
            def inference(network, low):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("synthetic inference failure")
                return network(low)
            with patch.object(diagnostic, "inspect_dataset", return_value=(
                    {"splits": {"test": manifest["splits"]["test"]}}, {"test": pairs})), \
                 patch.object(diagnostic, "_load_model", return_value=model), \
                 patch.object(diagnostic, "infer_full_image", side_effect=inference), \
                 self.assertRaisesRegex(RuntimeError, "synthetic inference failure"):
                diagnostic.analyze_images(paths, final, config["dataset_root"], 1, "cpu")
            self.assertEqual(len(model.edge_extractor._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
