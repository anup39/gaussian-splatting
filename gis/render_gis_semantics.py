#!/usr/bin/env python3
"""Render per-Gaussian GIS membership for train and held-out cameras."""
from __future__ import annotations

import csv
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arguments import ModelParams, PipelineParams, get_combined_args  # noqa: E402
from gaussian_renderer import GaussianModel, render_semantics  # noqa: E402
from scene import Scene  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402


def save_u16(path: Path, value: torch.Tensor) -> None:
    array = value.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.round(array * 65535.0).astype(np.uint16)):
        raise IOError(f"Could not write {path}")


def save_u8(path: Path, value: torch.Tensor) -> None:
    array = value.detach().clamp(0.0, 1.0).squeeze().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.round(array * 255.0).astype(np.uint8)):
        raise IOError(f"Could not write {path}")


def render_split(name, cameras, gaussians, pipeline, model_path, iteration,
                 threshold, minimum_coverage):
    root = Path(model_path) / name / f"ours_{iteration}" / "gis_semantics"
    probability_dir = root / "probability_u16"
    mask_dir = root / "masks"
    coverage_dir = root / "coverage_u16"
    rows = []

    for camera in tqdm(cameras, desc=f"Rendering GIS semantics ({name})"):
        package = render_semantics(
            camera, gaussians, pipeline, geometry_gradient_scale=0.0,
            minimum_coverage=minimum_coverage,
        )
        probability = package["semantic_probability"]
        coverage = package["coverage"]
        valid = coverage >= minimum_coverage
        binary = (probability >= threshold) & valid
        filename = f"{Path(camera.image_name).stem}.png"
        save_u16(probability_dir / filename, probability)
        save_u8(mask_dir / filename, binary.float())
        save_u16(coverage_dir / filename, coverage)
        rows.append({
            "image_name": camera.image_name,
            "split": name,
            "threshold": threshold,
            "mean_probability": float(probability[valid].mean()) if valid.any() else 0.0,
            "coverage_fraction": float(valid.float().mean()),
            "positive_fraction": float(binary.float().mean()),
        })

    root.mkdir(parents=True, exist_ok=True)
    with (root / "render_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()) if rows else ["image_name"])
        writer.writeheader()
        writer.writerows(rows)
    with (root / "render_config.json").open("w") as stream:
        json.dump({
            "iteration": iteration,
            "threshold": threshold,
            "minimum_coverage": minimum_coverage,
            "images": len(rows),
        }, stream, indent=2)
    return root


def main() -> None:
    parser = ArgumentParser(description="Render GIS semantic probability, masks, and coverage")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--minimum_coverage", default=0.01, type=float)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    safe_state(args.quiet)
    dataset = model.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        outputs = []
        if not args.skip_train:
            outputs.append(render_split(
                "train", scene.getTrainCameras(), gaussians, pipeline.extract(args),
                dataset.model_path, scene.loaded_iter, args.threshold, args.minimum_coverage,
            ))
        if not args.skip_test:
            outputs.append(render_split(
                "test", scene.getTestCameras(), gaussians, pipeline.extract(args),
                dataset.model_path, scene.loaded_iter, args.threshold, args.minimum_coverage,
            ))
    for output in outputs:
        print(f"Semantic render output: {output}")


if __name__ == "__main__":
    main()
