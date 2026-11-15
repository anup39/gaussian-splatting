"""Run with: python tools/test_boundary_debug.py (requires NumPy/OpenCV)."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from evaluate_gis_semantics import image_metrics, load_pairs, save_boundary_debug


class BoundaryDebugTests(unittest.TestCase):
    def test_export_matches_scoring(self):
        reference = np.zeros((100, 100), dtype=bool)
        reference[20:80, 20:80] = True
        for kind in ("identical", "expanded", "empty"):
            probability = reference.astype(np.float32)
            if kind == "expanded":
                probability[10:90, 10:90] = 1
            if kind == "empty":
                probability[:] = 0
            original = probability.copy()
            metrics = image_metrics(probability, reference, 0.4, 3)
            with tempfile.TemporaryDirectory() as root:
                save_boundary_debug(Path(root), kind, probability, reference, 0.4, 3,
                                    metrics, {}, {})
                folder = Path(root) / "boundary_debug" / kind
                info = json.loads((folder / "debug_info.json").read_text())
                for name in ("prediction", "reference"):
                    boundary = cv2.imread(str(folder / f"{name}_boundary.png"), 0)
                    self.assertEqual(int((boundary > 0).sum()), info[f"{name}_boundary_pixels"])
                if info["prediction_boundary_pixels"]:
                    self.assertAlmostEqual(
                        metrics["boundary_precision"],
                        info["matched_prediction_boundary_pixels"] / info["prediction_boundary_pixels"])
                self.assertEqual(cv2.imread(str(folder / "probability_eval_u16.png"), -1).dtype, np.uint16)
                self.assertEqual(len(list(folder.glob("*.png"))), 9)
                self.assertTrue(np.array_equal(probability, original))
                self.assertEqual(metrics, image_metrics(probability, reference, 0.4, 3))
            self.assertEqual(metrics["boundary_f1"], 1.0 if kind == "identical" else 0.0)

    def test_resize_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pred, ref = root / "pred", root / "ref"
            pred.mkdir(); ref.mkdir()
            cv2.imwrite(str(pred / "sample.png"), np.full((20, 30), 65535, np.uint16))
            cv2.imwrite(str(ref / "sample.png"), np.full((80, 120), 255, np.uint8))
            metadata = {}
            pairs = load_pairs(pred, ref, None, metadata)
            self.assertEqual(pairs[0][1].shape, (80, 120))
            self.assertEqual(metadata["sample"]["resize_scale_xy"], [4.0, 4.0])


if __name__ == "__main__":
    unittest.main()