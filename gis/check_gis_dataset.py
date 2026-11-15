#!/usr/bin/env python3
"""Audit images, GIS masks, hold-out split, and inverse-depth inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_files(folder):
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_split(path):
    if path is None:
        return set()
    return {
        Path(line.strip()).stem
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--images", default="images")
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--depth-dir", type=Path)
    parser.add_argument("--eval-split-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reject-empty-masks", action="store_true",
                        help="treat valid all-background masks as an error")
    args = parser.parse_args()

    images_dir = args.source / args.images
    if not images_dir.is_dir():
        parser.error("image directory does not exist: {}".format(images_dir))
    if not args.mask_dir.is_dir():
        parser.error("mask directory does not exist: {}".format(args.mask_dir))

    images = image_files(images_dir)
    if not images:
        parser.error("no images found in {}".format(images_dir))
    holdout = read_split(args.eval_split_file)
    image_stems = {path.stem for path in images}
    records = []
    problems = []

    for image_path in images:
        with Image.open(str(image_path)) as image:
            width, height = image.size
        mask_path = args.mask_dir / (image_path.stem + ".png")
        record = {
            "image": image_path.name,
            "width": width,
            "height": height,
            "split": "test" if image_path.stem in holdout else "train",
            "mask": str(mask_path),
            "mask_exists": mask_path.exists(),
        }
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                problems.append("unreadable mask: {}".format(mask_path))
                record["mask_readable"] = False
            else:
                record["mask_readable"] = True
                record["mask_width"] = int(mask.shape[1])
                record["mask_height"] = int(mask.shape[0])
                record["mask_positive_fraction"] = float((mask > 127).mean())
                unique_values = sorted(np.unique(mask).astype(int).tolist())
                record["mask_binary_values"] = unique_values[:20]
                if not set(unique_values).issubset({0, 255}):
                    problems.append("non-binary mask values: {}".format(image_path.name))
                if mask.shape != (height, width):
                    problems.append("mask size mismatch: {}".format(image_path.name))
                if record["mask_positive_fraction"] == 0.0 and args.reject_empty_masks:
                    problems.append("empty mask: {}".format(image_path.name))
        else:
            problems.append("missing mask: {}".format(image_path.name))

        if args.depth_dir is not None:
            depth_path = args.depth_dir / (image_path.stem + ".png")
            record["depth"] = str(depth_path)
            record["depth_exists"] = depth_path.exists()
            if depth_path.exists():
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth is None:
                    problems.append("unreadable depth: {}".format(depth_path))
                else:
                    if depth.ndim == 3:
                        depth = depth[..., 0]
                    record["depth_dtype"] = str(depth.dtype)
                    record["depth_width"] = int(depth.shape[1])
                    record["depth_height"] = int(depth.shape[0])
                    record["depth_valid_fraction"] = float((depth > 0).mean())
                    if depth.shape != (height, width):
                        problems.append("depth size mismatch: {}".format(image_path.name))
                    if record["depth_valid_fraction"] == 0.0:
                        problems.append("empty depth: {}".format(image_path.name))
            else:
                problems.append("missing depth: {}".format(image_path.name))
        records.append(record)

    unknown_holdout = sorted(holdout - image_stems)
    if unknown_holdout:
        problems.append("hold-out names absent from images: {}".format(", ".join(unknown_holdout)))
    if args.eval_split_file is not None and not holdout:
        problems.append("hold-out split file is empty")

    depth_parameters = args.source / "sparse" / "0" / "depth_params.json"
    if args.depth_dir is not None and not depth_parameters.exists():
        problems.append("missing depth alignment file: {}".format(depth_parameters))

    positive_fractions = [
        row.get("mask_positive_fraction", 0.0) for row in records if row.get("mask_readable")
    ]
    report = {
        "source": str(args.source),
        "image_count": len(images),
        "train_count": sum(row["split"] == "train" for row in records),
        "test_count": sum(row["split"] == "test" for row in records),
        "mask_count": sum(row["mask_exists"] for row in records),
        "empty_mask_count": sum(
            row.get("mask_positive_fraction") == 0.0 for row in records
        ),
        "mean_mask_positive_fraction": (
            float(np.mean(positive_fractions)) if positive_fractions else None
        ),
        "depth_params_exists": depth_parameters.exists(),
        "problems": problems,
        "records": records,
    }
    output = args.output or (args.source / "gis_dataset_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print("Images: {} (train {}, test {})".format(
        report["image_count"], report["train_count"], report["test_count"]
    ))
    print("Masks: {}/{} ({} all-background)".format(
        report["mask_count"], report["image_count"], report["empty_mask_count"]
    ))
    print("Problems: {}".format(len(problems)))
    for problem in problems[:30]:
        print("  - {}".format(problem))
    print("Report: {}".format(output))
    if problems:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
