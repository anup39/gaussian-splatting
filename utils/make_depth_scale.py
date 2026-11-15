import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
from joblib import Parallel, delayed

from read_write_model import qvec2rotmat, read_model


def bilinear_sample(image, coordinates):
    """Sample a 2-D image at Nx2 pixel coordinates without cv2.remap limits."""
    if image.ndim != 2:
        raise ValueError("bilinear_sample expects a two-dimensional image")

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2)")
    if coordinates.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    height, width = image.shape
    x = np.clip(coordinates[:, 0], 0.0, width - 1.0)
    y = np.clip(coordinates[:, 1], 0.0, height - 1.0)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    wx = x - x0
    wy = y - y0
    image = image.astype(np.float64, copy=False)

    return (
        image[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + image[y0, x1] * wx * (1.0 - wy)
        + image[y1, x0] * (1.0 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def make_result(image_name, status, scale=0.0, offset=0.0,
                correspondences=0, colmap_spread=0.0, mono_spread=0.0):
    """Return JSON-safe scale parameters and useful per-image diagnostics."""
    return {
        "image_name": image_name,
        "scale": float(scale),
        "offset": float(offset),
        "status": status,
        "correspondences": int(correspondences),
        "colmap_spread": float(colmap_spread),
        "mono_spread": float(mono_spread),
    }


def get_scales(key, cameras, images, points3d_ordered, args):
    image_meta = images[key]
    image_name = os.path.splitext(image_meta.name)[0]
    cam_intrinsic = cameras[image_meta.camera_id]

    depth_path = os.path.join(args.depths_dir, image_name + ".png")
    invmonodepthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if invmonodepthmap is None:
        return make_result(image_name, "missing_depth")
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]
    if invmonodepthmap.size == 0:
        return make_result(image_name, "empty_depth")

    # Depth Anything output is stored as unsigned 16-bit inverse depth. The
    # global factor is immaterial to the affine alignment, but this keeps the
    # same convention as the training loader.
    invmonodepthmap = invmonodepthmap.astype(np.float64) / float(2 ** 16)

    pts_idx = np.asarray(image_meta.point3D_ids)
    valid_ids = (pts_idx >= 0) & (pts_idx < len(points3d_ordered))
    pts_idx = pts_idx[valid_ids].astype(np.int64, copy=False)
    valid_xys = np.asarray(image_meta.xys, dtype=np.float64)[valid_ids]

    if len(pts_idx) < args.minimum_correspondences:
        return make_result(
            image_name, "insufficient_sparse_points", correspondences=len(pts_idx)
        )

    points_camera = np.dot(points3d_ordered[pts_idx], qvec2rotmat(image_meta.qvec).T)
    points_camera += image_meta.tvec
    z = points_camera[:, 2]

    depth_height, depth_width = invmonodepthmap.shape
    scale_x = depth_width / float(cam_intrinsic.width)
    scale_y = depth_height / float(cam_intrinsic.height)
    sample_xy = valid_xys.copy()
    sample_xy[:, 0] *= scale_x
    sample_xy[:, 1] *= scale_y

    valid = (
        np.isfinite(sample_xy).all(axis=1)
        & np.isfinite(z)
        & (z > 1e-8)
        & (sample_xy[:, 0] >= 0.0)
        & (sample_xy[:, 1] >= 0.0)
        & (sample_xy[:, 0] < depth_width)
        & (sample_xy[:, 1] < depth_height)
    )
    sample_xy = sample_xy[valid]
    z = z[valid]

    if len(z) < args.minimum_correspondences:
        return make_result(
            image_name, "insufficient_valid_points", correspondences=len(z)
        )

    invcolmapdepth = 1.0 / z
    # NumPy sampling supports any number of correspondences. cv2.remap treats
    # the N samples as an N-pixel output dimension and fails once N >= 32767.
    invmonodepth = bilinear_sample(invmonodepthmap, sample_xy)

    finite = np.isfinite(invcolmapdepth) & np.isfinite(invmonodepth)
    invcolmapdepth = invcolmapdepth[finite]
    invmonodepth = invmonodepth[finite]
    if len(invcolmapdepth) < args.minimum_correspondences:
        return make_result(
            image_name, "insufficient_finite_samples",
            correspondences=len(invcolmapdepth),
        )

    t_colmap = np.median(invcolmapdepth)
    s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))
    t_mono = np.median(invmonodepth)
    s_mono = np.mean(np.abs(invmonodepth - t_mono))

    if not np.isfinite(s_colmap) or s_colmap <= args.minimum_colmap_spread:
        return make_result(
            image_name, "degenerate_sparse_depth",
            correspondences=len(invcolmapdepth), colmap_spread=s_colmap,
            mono_spread=s_mono,
        )
    if not np.isfinite(s_mono) or s_mono <= args.minimum_mono_spread:
        return make_result(
            image_name, "degenerate_mono_depth",
            correspondences=len(invcolmapdepth), colmap_spread=s_colmap,
            mono_spread=s_mono,
        )

    scale = s_colmap / s_mono
    offset = t_colmap - t_mono * scale
    if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0.0:
        return make_result(
            image_name, "invalid_alignment",
            correspondences=len(invcolmapdepth), colmap_spread=s_colmap,
            mono_spread=s_mono,
        )

    return make_result(
        image_name, "ok", scale=scale, offset=offset,
        correspondences=len(invcolmapdepth), colmap_spread=s_colmap,
        mono_spread=s_mono,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Align monocular inverse-depth maps to sparse SfM depths."
    )
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--depths_dir", required=True)
    parser.add_argument("--model_type", default="bin", choices=("bin", "txt"))
    parser.add_argument("--jobs", type=int, default=-1,
                        help="Parallel workers; use 1 for sequential diagnostics.")
    parser.add_argument("--minimum_correspondences", type=int, default=20)
    parser.add_argument("--minimum_mono_spread", type=float, default=1e-8)
    parser.add_argument("--minimum_colmap_spread", type=float, default=1e-8)
    args = parser.parse_args()

    sparse_dir = os.path.join(args.base_dir, "sparse", "0")
    cam_intrinsics, images_metas, points3d = read_model(
        sparse_dir, ext="." + args.model_type
    )
    if not points3d:
        raise RuntimeError("No sparse 3-D points were found in " + sparse_dir)

    pts_indices = np.array([point.id for point in points3d.values()], dtype=np.int64)
    pts_xyzs = np.array([point.xyz for point in points3d.values()], dtype=np.float64)
    points3d_ordered = np.zeros((int(pts_indices.max()) + 1, 3), dtype=np.float64)
    points3d_ordered[pts_indices] = pts_xyzs

    depth_param_list = Parallel(n_jobs=args.jobs, backend="threading")(
        delayed(get_scales)(
            key, cam_intrinsics, images_metas, points3d_ordered, args
        )
        for key in images_metas
    )

    depth_params = {
        result["image_name"]: {
            field: value for field, value in result.items() if field != "image_name"
        }
        for result in depth_param_list
    }
    output_path = os.path.join(sparse_dir, "depth_params.json")
    with open(output_path, "w") as output_file:
        json.dump(depth_params, output_file, indent=2)

    status_counts = Counter(result["status"] for result in depth_param_list)
    print("Wrote {} depth alignments to {}".format(len(depth_params), output_path))
    print("Status summary: " + ", ".join(
        "{}={}".format(status, count)
        for status, count in sorted(status_counts.items())
    ))
    if status_counts.get("ok", 0) == 0:
        raise RuntimeError(
            "No reliable depth alignments were produced. Check that the depth "
            "PNGs are non-constant inverse-depth maps and match the image stems."
        )


if __name__ == "__main__":
    main()
