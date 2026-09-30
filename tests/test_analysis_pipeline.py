"""Small synthetic checks for stage capture, metrics, and comparison exports."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from RetinexFormer_arch import RetinexFormer
from analysis_utils import (
    compare_analysis_results,
    evaluate_analysis_model,
    image_metrics,
    write_analysis_metrics,
)


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


class AnalysisPipelineTests(unittest.TestCase):
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
                with Image.open(images / "output.png") as output, Image.open(images / "gt.png") as gt:
                    psnr, ssim = image_metrics(np.asarray(output), np.asarray(gt))
                self.assertAlmostEqual(row["psnr"], psnr)
                self.assertAlmostEqual(row["ssim"], ssim)
                self.assertTrue((images / "lightup_stage1.png").is_file())
                self.assertTrue((images / "output_stage1.png").is_file())
            author_avg = write_analysis_metrics(author_rows, root / "author")
            self.assertAlmostEqual(author_avg["psnr"], np.mean([r["psnr"] for r in author_rows]))
            comparisons, selected = compare_analysis_results(
                author_rows, ours_rows, root / "author", root / "ours", root,
                threshold=0.5, representative_count=2
            )
            self.assertEqual(len(comparisons), 3)
            self.assertEqual(sum(len(rows) for rows in selected.values()), 3)
            with (root / "comparison.csv").open(newline="", encoding="utf-8") as file:
                self.assertEqual(len(list(csv.DictReader(file))), 3)
            for row in comparisons:
                grid = root / row["group"] / row["filename"][:-4] / "comparison.png"
                self.assertTrue(grid.is_file())
                self.assertAlmostEqual(row["delta_psnr"], row["ours_psnr"] - row["author_psnr"])


if __name__ == "__main__":
    unittest.main()
