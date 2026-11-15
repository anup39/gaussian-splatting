#!/usr/bin/env python3
"""Evaluate rendered GIS semantics against independent reference masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def read_probability(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Could not read prediction: {path}")
    if image.ndim == 3:
        image = image[..., 0]
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def read_reference(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Could not read reference mask: {path}")
    return image > 127


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    data = mask.astype(np.uint8)
    eroded = cv2.erode(data, np.ones((3, 3), np.uint8), iterations=1)
    return (data > eroded).astype(np.uint8)


def boundary_fscore(prediction: np.ndarray, reference: np.ndarray,
                    tolerance_px: int) -> Tuple[float, float, float]:
    pred_boundary = mask_boundary(prediction)
    ref_boundary = mask_boundary(reference)
    pred_count, ref_count = pred_boundary.sum(), ref_boundary.sum()
    if pred_count == 0 and ref_count == 0:
        return 1.0, 1.0, 1.0
    if pred_count == 0 or ref_count == 0:
        return 0.0, 0.0, 0.0
    distance_to_ref = cv2.distanceTransform(1 - ref_boundary, cv2.DIST_L2, 3)
    distance_to_pred = cv2.distanceTransform(1 - pred_boundary, cv2.DIST_L2, 3)
    precision = float((distance_to_ref[pred_boundary.astype(bool)] <= tolerance_px).mean())
    recall = float((distance_to_pred[ref_boundary.astype(bool)] <= tolerance_px).mean())
    fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, fscore


def image_metrics(probability: np.ndarray, reference: np.ndarray,
                  threshold: float, boundary_tolerance_px: int) -> Dict[str, float]:
    prediction = probability >= threshold
    tp = int(np.logical_and(prediction, reference).sum())
    fp = int(np.logical_and(prediction, ~reference).sum())
    fn = int(np.logical_and(~prediction, reference).sum())
    tn = int(np.logical_and(~prediction, ~reference).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    boundary_precision, boundary_recall, boundary_f1 = boundary_fscore(
        prediction, reference, boundary_tolerance_px
    )
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "dice_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "boundary_precision": boundary_precision,
        "boundary_recall": boundary_recall,
        "boundary_f1": boundary_f1,
        "brier": float(np.mean((probability - reference.astype(np.float32)) ** 2)),
        "predicted_fraction": float(prediction.mean()),
        "reference_fraction": float(reference.mean()),
    }


def load_pairs(prediction_dir: Path, reference_dir: Path,
               include_file: Optional[Path]) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    included = None
    if include_file:
        included = {
            Path(line.strip()).stem
            for line in include_file.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    predictions = {path.stem: path for path in prediction_dir.rglob("*.png")}
    references = {path.stem: path for path in reference_dir.rglob("*.png")}
    names = sorted(set(predictions) & set(references))
    if included is not None:
        names = [name for name in names if name in included]
    pairs = []
    for name in names:
        probability = read_probability(predictions[name])
        reference = read_reference(references[name])
        if probability.shape != reference.shape:
            probability = cv2.resize(
                probability, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        pairs.append((name, probability, reference))
    if not pairs:
        raise RuntimeError("No matching prediction/reference PNG stems were found")
    return pairs


def macro_iou(pairs, threshold: float) -> float:
    values = []
    for _, probability, reference in pairs:
        prediction = probability >= threshold
        intersection = np.logical_and(prediction, reference).sum()
        union = np.logical_or(prediction, reference).sum()
        values.append(intersection / max(union, 1))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", required=True, type=Path,
                        help="probability_u16 directory from render_gis_semantics.py")
    parser.add_argument("--reference-dir", required=True, type=Path,
                        help="independently labelled reference-mask directory")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-from", type=Path,
                        help="read the selected threshold from a previous semantic_metrics.json")
    parser.add_argument("--select-threshold", action="store_true",
                        help="select threshold on this validation set by macro IoU")
    parser.add_argument("--threshold-min", type=float, default=0.30)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--boundary-tolerance-px", type=int, default=3)
    parser.add_argument("--include-file", type=Path,
                        help="optional list of image names defining validation or test subset")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.threshold_from is not None and args.select_threshold:
        parser.error("use either --threshold-from or --select-threshold, not both")
    if args.threshold_from is not None:
        args.threshold = float(json.loads(args.threshold_from.read_text())["threshold"])

    pairs = load_pairs(args.prediction_dir, args.reference_dir, args.include_file)
    threshold_curve = []
    if args.select_threshold:
        thresholds = np.arange(
            args.threshold_min, args.threshold_max + args.threshold_step * 0.5, args.threshold_step
        )
        threshold_curve = [
            {"threshold": float(value), "macro_iou": macro_iou(pairs, float(value))}
            for value in thresholds
        ]
        args.threshold = max(threshold_curve, key=lambda row: row["macro_iou"])["threshold"]

    rows = []
    for name, probability, reference in pairs:
        row = {"image_name": name}
        row.update(image_metrics(probability, reference, args.threshold, args.boundary_tolerance_px))
        rows.append(row)

    metric_names = [key for key in rows[0] if key != "image_name"]
    summary = {
        "images": len(rows),
        "threshold": args.threshold,
        "boundary_tolerance_px": args.boundary_tolerance_px,
        "macro_mean": {key: float(np.mean([row[key] for row in rows])) for key in metric_names},
        "macro_std": {key: float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0
                      for key in metric_names},
        "threshold_curve": threshold_curve,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image_semantic_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "semantic_metrics.json").open("w") as stream:
        json.dump(summary, stream, indent=2)

    print(f"Images: {summary['images']}")
    print(f"Threshold: {summary['threshold']:.4f}")
    for key, value in summary["macro_mean"].items():
        print(f"mean_{key}: {value:.6f}")
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
