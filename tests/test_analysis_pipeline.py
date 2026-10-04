"""Small synthetic checks for stage capture, metrics, and comparison exports."""

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from RetinexFormer_arch import RetinexFormer
from analysis_utils import (
    classify_litup_diagnostics,
    classify_output_delta,
    compare_analysis_results,
    evaluate_analysis_model,
    extract_checkpoint_state_dict,
    find_paired_images,
    image_metrics,
    litup_diagnostics,
    load_checkpoint_strict,
    write_analysis_report,
    write_analysis_metrics,
)
from analysis import compare_author_vs_retrained as comparison_script


class TinyDataset:
    def __len__(self):
        return 3

    def __getitem__(self, index):
        low = torch.full((3, 13, 17), 0.2 + index * 0.1)
        high = torch.full((3, 13, 17), 0.5)
        return {"low": low, "high": high, "name": f"{index:05d}.png"}


class TinyModel(torch.nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset

    def forward_with_intermediate(self, x):
        lightup = x + self.offset / 2
        output = x + self.offset
        return [lightup], [output]


class TinyAnalysisModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.offset

    def forward_with_intermediate(self, x):
        return [x + self.offset / 2], [self.forward(x)]


class AnalysisPipelineTests(unittest.TestCase):
    def test_checkpoint_formats_and_litup_grouping(self):
        model = TinyAnalysisModel()
        direct = model.state_dict()
        for wrapped in (
            direct,
            {"params": direct},
            {"model": direct},
            {"state_dict": direct},
            {"checkpoint": {"state_dict": direct}},
        ):
            self.assertEqual(set(extract_checkpoint_state_dict(wrapped)), set(direct))
        prefixed = {f"module.{key}": value for key, value in direct.items()}
        self.assertEqual(set(extract_checkpoint_state_dict(prefixed)), set(direct))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrapped.pth"
            torch.save({"checkpoint": {"params": prefixed}}, path)
            loaded, message = load_checkpoint_strict(TinyAnalysisModel(), path, "cpu", "Tiny")
            self.assertIn("Loaded Tiny checkpoint", message)
            self.assertTrue(torch.allclose(loaded.offset, model.offset))
            bad_path = Path(temp) / "incompatible.pth"
            torch.save({"state_dict": {"wrong.weight": torch.ones(1)}}, bad_path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_checkpoint_strict(TinyAnalysisModel(), bad_path, "cpu", "Tiny")

        author = np.full((13, 17, 3), 64, dtype=np.uint8)
        retrained = np.full((13, 17, 3), 128, dtype=np.uint8)
        metrics = litup_diagnostics(author, retrained)
        self.assertGreater(metrics["litup_mean_luminance_delta"], 0.02)
        self.assertEqual(classify_litup_diagnostics(metrics), "Brighter")
        self.assertEqual(classify_output_delta(0.5), "Improved")
        self.assertEqual(classify_output_delta(-0.5), "Degraded")
        self.assertEqual(classify_output_delta(0.1), "Similar")

    def test_dedicated_smoke_analysis_exports_required_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            low_dir, gt_dir, output_dir = root / "low", root / "gt", root / "results"
            low_dir.mkdir()
            gt_dir.mkdir()
            for index in range(2):
                Image.fromarray(np.full((13, 17, 3), 40 + index, dtype=np.uint8)).save(
                    low_dir / f"{index:05d}.png"
                )
                Image.fromarray(np.full((13, 17, 3), 120, dtype=np.uint8)).save(
                    gt_dir / f"{index:05d}.png"
                )
            author_ckpt, retrained_ckpt = root / "author.pth", root / "retrained.pth"
            torch.save({"params": TinyAnalysisModel().state_dict()}, author_ckpt)
            retrained = TinyAnalysisModel()
            retrained.offset.data.fill_(0.2)
            torch.save({"checkpoint": {"state_dict": retrained.state_dict()}}, retrained_ckpt)
            args = Namespace(
                author_ckpt=author_ckpt, retrained_ckpt=retrained_ckpt,
                low_dir=low_dir, gt_dir=gt_dir, output_dir=output_dir,
                max_images=1, device="cpu", output_threshold_db=0.5,
                brightness_delta=0.02, ratio_delta=0.02, dark_threshold=0.10,
                highlight_threshold=0.98,
            )
            with patch.object(comparison_script, "_model", lambda device: TinyAnalysisModel()):
                rows = comparison_script.analyze(args)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(find_paired_images(low_dir, gt_dir)), 2)
            for folder in (
                "low", "author_litup", "retrained_litup", "author_output",
                "retrained_output", "gt", "grids",
            ):
                self.assertTrue((output_dir / folder / "00000.png").is_file())
            with (output_dir / "per_image_analysis.csv").open(newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))
            self.assertEqual(len(csv_rows), 1)
            self.assertIn("author_vs_retrained_litup_mae", csv_rows[0])
            self.assertIn("author_litup_psnr", csv_rows[0])
            self.assertIn("litup_ssim_delta", csv_rows[0])
            self.assertIn("combined_group", csv_rows[0])
            self.assertAlmostEqual(
                float(csv_rows[0]["litup_psnr_delta"]),
                float(csv_rows[0]["retrained_litup_psnr"]) -
                float(csv_rows[0]["author_litup_psnr"]),
            )
            self.assertAlmostEqual(
                float(csv_rows[0]["litup_ssim_delta"]),
                float(csv_rows[0]["retrained_litup_ssim"]) -
                float(csv_rows[0]["author_litup_ssim"]),
            )
            self.assertTrue((output_dir / "summary.csv").is_file())

    def test_stage_path_matches_forward(self):
        torch.manual_seed(7)
        model = RetinexFormer(n_feat=8, stage=2, num_blocks=[0, 0, 0]).eval()
        x = torch.rand(1, 3, 16, 20)
        with torch.inference_mode():
            expected = model(x)
            lightups, outputs = model.forward_with_intermediate(x)
        self.assertEqual((len(lightups), len(outputs)), (2, 2))
        self.assertTrue(torch.allclose(expected, outputs[-1], atol=1e-6, rtol=1e-5))

    def test_exports_and_metrics_match_saved_pixels(self):
        dataset = TinyDataset()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            author_rows = evaluate_analysis_model(TinyModel(0.1), dataset, "cpu", root / "author")
            ours_rows = evaluate_analysis_model(TinyModel(0.2), dataset, "cpu", root / "ours")
            self.assertEqual(len(author_rows), 3)
            for row in author_rows:
                images = root / "author" / "images" / row["filename"][:-4]
                with Image.open(images / "lightup.png") as lightup, Image.open(images / "gt.png") as gt:
                    lightup_psnr, lightup_ssim = image_metrics(np.asarray(lightup), np.asarray(gt))
                with Image.open(images / "output.png") as output, Image.open(images / "gt.png") as gt:
                    output_psnr, output_ssim = image_metrics(np.asarray(output), np.asarray(gt))
                self.assertAlmostEqual(row["lightup_psnr"], lightup_psnr)
                self.assertAlmostEqual(row["lightup_ssim"], lightup_ssim)
                self.assertAlmostEqual(row["output_psnr"], output_psnr)
                self.assertAlmostEqual(row["output_ssim"], output_ssim)
                self.assertTrue((images / "lightup_stage1.png").is_file())
                self.assertTrue((images / "output_stage1.png").is_file())
            author_avg = write_analysis_metrics(author_rows, root / "author")
            self.assertAlmostEqual(
                author_avg["lightup"]["psnr"],
                np.mean([r["lightup_psnr"] for r in author_rows]),
            )
            self.assertAlmostEqual(
                author_avg["output"]["ssim"],
                np.mean([r["output_ssim"] for r in author_rows]),
            )
            ours_avg = write_analysis_metrics(ours_rows, root / "ours")
            comparisons, selected = compare_analysis_results(
                author_rows, ours_rows, root / "author", root / "ours", root,
                threshold=0.5, representative_count=2
            )
            write_analysis_report(comparisons, author_avg, ours_avg, root)
            self.assertEqual(len(comparisons), 3)
            self.assertEqual(sum(len(rows) for rows in selected.values()), 3)
            with (root / "comparison.csv").open(newline="", encoding="utf-8") as file:
                csv_rows = list(csv.DictReader(file))
            self.assertEqual(len(csv_rows), 3)
            self.assertIn("delta_lightup_psnr", csv_rows[0])
            self.assertIn("delta_output_ssim", csv_rows[0])
            report = (root / "results.md").read_text(encoding="utf-8")
            self.assertIn("## Aggregate metrics", report)
            self.assertIn("## Per-image metrics", report)
            for row in comparisons:
                grid = root / row["group"] / row["filename"][:-4] / "comparison.png"
                self.assertTrue(grid.is_file())
                self.assertAlmostEqual(
                    row["delta_lightup_psnr"],
                    row["ours_lightup_psnr"] - row["author_lightup_psnr"],
                )
                self.assertAlmostEqual(
                    row["delta_lightup_ssim"],
                    row["ours_lightup_ssim"] - row["author_lightup_ssim"],
                )
                self.assertAlmostEqual(
                    row["delta_output_psnr"],
                    row["ours_output_psnr"] - row["author_output_psnr"],
                )
                self.assertAlmostEqual(
                    row["delta_output_ssim"],
                    row["ours_output_ssim"] - row["author_output_ssim"],
                )


if __name__ == "__main__":
    unittest.main()
