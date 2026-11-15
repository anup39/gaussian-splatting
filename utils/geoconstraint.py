"""Differentiable losses for GIS-supervised Gaussian splatting.

GIS vector conversion remains in ``tools/mask.py``; this module handles
uncertainty weighting, loss scheduling, and depth-aware cross-view reprojection.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Return the same forward value while multiplying its backward gradient."""
    scale = float(scale)
    if scale == 1.0:
        return tensor
    return tensor.detach() + scale * (tensor - tensor.detach())


def ramped_weight(final_weight: float, iteration: int, start_iteration: int = 0,
                  ramp_iterations: int = 0) -> float:
    """Linearly introduce a loss after geometry has had time to stabilize."""
    if final_weight <= 0 or iteration < start_iteration:
        return 0.0
    if ramp_iterations <= 0:
        return float(final_weight)
    progress = (iteration - start_iteration) / float(ramp_iterations)
    return float(final_weight) * min(max(progress, 0.0), 1.0)


def boundary_confidence(mask: torch.Tensor, band_px: int = 0,
                        boundary_weight: float = 0.0) -> torch.Tensor:
    """Downweight a band around a binary mask boundary.

    ``mask`` has shape ``(1, H, W)``. Pixels farther than ``band_px`` from the
    boundary receive weight one; pixels in the uncertainty band receive
    ``boundary_weight``.
    """
    if band_px <= 0:
        return torch.ones_like(mask)
    radius = int(band_px)
    kernel = 2 * radius + 1
    batched = mask.unsqueeze(0)
    dilated = F.max_pool2d(batched, kernel, stride=1, padding=radius)
    eroded = 1.0 - F.max_pool2d(1.0 - batched, kernel, stride=1, padding=radius)
    band = (dilated - eroded).clamp(0.0, 1.0).squeeze(0)
    return 1.0 - band * (1.0 - float(boundary_weight))


def weighted_dice_bce_loss(prediction: torch.Tensor, target: torch.Tensor,
                           weights: Optional[torch.Tensor] = None,
                           eps: float = 1e-6) -> torch.Tensor:
    """Weighted BCE plus soft Dice for tensors of shape ``(1, H, W)``."""
    prediction = prediction.clamp(eps, 1.0 - eps)
    if weights is None:
        weights = torch.ones_like(target)
    weights = weights.to(dtype=prediction.dtype)
    weight_sum = weights.sum().clamp_min(eps)
    bce = (F.binary_cross_entropy(prediction, target, reduction="none") * weights).sum() / weight_sum
    intersection = (prediction * target * weights).sum()
    dice = 1.0 - (2.0 * intersection + eps) / (
        (prediction * weights).sum() + (target * weights).sum() + eps
    )
    return bce + dice


def dice_bce_loss(prediction: torch.Tensor, target: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """Backward-compatible unweighted binary mask loss."""
    return weighted_dice_bce_loss(prediction, target, None, eps)


def boundary_weighted_l1(prediction: torch.Tensor, target: torch.Tensor,
                         signed_distance: torch.Tensor,
                         band_px: float = 8.0) -> torch.Tensor:
    """Weight mask disagreement near a supplied signed-distance boundary."""
    weights = torch.exp(-signed_distance.abs() / max(band_px, 1e-6))
    return (weights * (prediction - target).abs()).sum() / (weights.sum() + 1e-6)


def _resize_map(value: torch.Tensor, size: Tuple[int, int], mode: str) -> torch.Tensor:
    kwargs: Dict[str, Any] = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(value.unsqueeze(0), **kwargs).squeeze(0)


def _camera_tensors(camera: Any, device: torch.device,
                    dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.as_tensor(np.asarray(camera.R), device=device, dtype=dtype)
    translation = torch.as_tensor(np.asarray(camera.T), device=device, dtype=dtype)
    return rotation, translation


def depth_aware_cross_view_loss(
    source_semantic: torch.Tensor,
    source_invdepth: torch.Tensor,
    source_coverage: torch.Tensor,
    target_semantic: torch.Tensor,
    target_invdepth: torch.Tensor,
    target_coverage: torch.Tensor,
    source_camera: Any,
    target_camera: Any,
    source_region: Optional[torch.Tensor] = None,
    source_confidence: Optional[torch.Tensor] = None,
    downsample: float = 0.25,
    minimum_coverage: float = 0.05,
    relative_depth_tolerance: float = 0.05,
    minimum_valid_pixels: int = 128,
    charbonnier_epsilon: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compare semantic predictions after depth-aware reprojection.

    Source pixels are back-projected using opacity-normalized rendered inverse
    depth and projected into the target camera. Only mutually visible,
    depth-consistent pixels contribute. Visibility tests are detached so the
    optimizer cannot satisfy the loss by invalidating correspondences.
    """
    if not (0 < downsample <= 1.0):
        raise ValueError("cross-view downsample must be in (0, 1]")

    source_h, source_w = source_semantic.shape[-2:]
    target_h, target_w = target_semantic.shape[-2:]
    sh = max(2, int(round(source_h * downsample)))
    sw = max(2, int(round(source_w * downsample)))
    th = max(2, int(round(target_h * downsample)))
    tw = max(2, int(round(target_w * downsample)))

    source_semantic = _resize_map(source_semantic, (sh, sw), "bilinear")
    source_invdepth = _resize_map(source_invdepth, (sh, sw), "bilinear")
    source_coverage = _resize_map(source_coverage, (sh, sw), "bilinear")
    target_semantic = _resize_map(target_semantic, (th, tw), "bilinear")
    target_invdepth = _resize_map(target_invdepth, (th, tw), "bilinear")
    target_coverage = _resize_map(target_coverage, (th, tw), "bilinear")

    if source_region is not None:
        source_region = _resize_map(source_region, (sh, sw), "nearest")
    if source_confidence is not None:
        source_confidence = _resize_map(source_confidence, (sh, sw), "bilinear")

    device, dtype = source_semantic.device, source_semantic.dtype
    y, x = torch.meshgrid(
        torch.arange(sh, device=device, dtype=dtype),
        torch.arange(sw, device=device, dtype=dtype),
        indexing="ij",
    )

    fx_s = sw / (2.0 * math.tan(float(source_camera.FoVx) * 0.5))
    fy_s = sh / (2.0 * math.tan(float(source_camera.FoVy) * 0.5))
    cx_s, cy_s = (sw - 1.0) * 0.5, (sh - 1.0) * 0.5
    safe_source_invdepth = source_invdepth[0].clamp_min(1e-8)
    source_z = 1.0 / safe_source_invdepth
    source_camera_points = torch.stack(
        ((x - cx_s) * source_z / fx_s, (y - cy_s) * source_z / fy_s, source_z), dim=-1
    )

    source_rotation, source_translation = _camera_tensors(source_camera, device, dtype)
    target_rotation, target_translation = _camera_tensors(target_camera, device, dtype)
    world_points = torch.matmul(
        source_camera_points - source_translation.view(1, 1, 3),
        source_rotation.transpose(0, 1),
    )
    target_camera_points = torch.matmul(world_points, target_rotation) + target_translation.view(1, 1, 3)
    target_z = target_camera_points[..., 2]

    fx_t = tw / (2.0 * math.tan(float(target_camera.FoVx) * 0.5))
    fy_t = th / (2.0 * math.tan(float(target_camera.FoVy) * 0.5))
    cx_t, cy_t = (tw - 1.0) * 0.5, (th - 1.0) * 0.5
    target_x = fx_t * target_camera_points[..., 0] / target_z.clamp_min(1e-8) + cx_t
    target_y = fy_t * target_camera_points[..., 1] / target_z.clamp_min(1e-8) + cy_t
    grid = torch.stack(
        (2.0 * target_x / max(tw - 1, 1) - 1.0, 2.0 * target_y / max(th - 1, 1) - 1.0), dim=-1
    ).unsqueeze(0)

    def sample(value: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(
            value.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).squeeze(0)

    sampled_target_semantic = sample(target_semantic)
    sampled_target_invdepth = sample(target_invdepth)
    sampled_target_coverage = sample(target_coverage)
    expected_target_invdepth = (1.0 / target_z.clamp_min(1e-8)).unsqueeze(0)
    relative_depth_error = (
        (sampled_target_invdepth - expected_target_invdepth).abs()
        / torch.maximum(sampled_target_invdepth.abs(), expected_target_invdepth.abs()).clamp_min(1e-8)
    )

    grid_x = grid[..., 0].squeeze(0).unsqueeze(0)
    grid_y = grid[..., 1].squeeze(0).unsqueeze(0)
    valid = (
        (source_coverage >= minimum_coverage)
        & (sampled_target_coverage >= minimum_coverage)
        & (source_invdepth > 0)
        & (sampled_target_invdepth > 0)
        & (target_z.unsqueeze(0) > 0.2)
        & (grid_x >= -1.0) & (grid_x <= 1.0)
        & (grid_y >= -1.0) & (grid_y <= 1.0)
        & (relative_depth_error <= relative_depth_tolerance)
    )
    if source_region is not None:
        valid = valid & (source_region > 0.5)
    valid = valid.detach()

    weights = valid.to(dtype)
    if source_confidence is not None:
        weights = weights * source_confidence.detach().clamp(0.0, 1.0)
    valid_count = weights.sum()
    difference = source_semantic - sampled_target_semantic
    zero = source_semantic.sum() * 0.0 + target_semantic.sum() * 0.0
    robust_difference = torch.sqrt(difference.square() + charbonnier_epsilon**2)
    candidate_loss = (weights * robust_difference).sum() / valid_count.clamp_min(1.0)
    loss = torch.where(valid_count >= minimum_valid_pixels, candidate_loss, zero)

    binary_agreement = (((source_semantic >= 0.5) == (sampled_target_semantic >= 0.5)).to(dtype) * weights).sum()
    stats = {
        "valid_pixels": valid_count.detach(),
        "valid_fraction": (valid_count / float(sh * sw)).detach(),
        "mean_absolute_difference": ((difference.abs() * weights).sum() / valid_count.clamp_min(1.0)).detach(),
        "binary_agreement": (binary_agreement / valid_count.clamp_min(1.0)).detach(),
    }
    return loss, stats
