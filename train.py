#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import numpy as np
import torch
from random import choice, randint
from utils.loss_utils import l1_loss, ssim
from utils.geoconstraint import (
    boundary_confidence,
    depth_aware_cross_view_loss,
    ramped_weight,
    weighted_dice_bce_loss,
)
from gaussian_renderer import render, render_semantics, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def build_camera_neighbors(cameras, neighbor_count):
    """Return nearest camera candidates for overlapping-view sampling."""
    if len(cameras) < 2:
        return {}
    centers = np.stack([camera.camera_center.detach().cpu().numpy() for camera in cameras])
    squared_distances = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(squared_distances, np.inf)
    count = min(max(int(neighbor_count), 1), len(cameras) - 1)
    return {
        id(camera): [cameras[index] for index in np.argsort(squared_distances[row])[:count]]
        for row, camera in enumerate(cameras)
    }


def validate_geospatial_options(opt):
    nonnegative = {
        "geo_mask_weight": opt.geo_mask_weight,
        "geo_boundary_weight": opt.geo_boundary_weight,
        "geo_densify_weight": opt.geo_densify_weight,
        "geo_cross_view_weight": opt.geo_cross_view_weight,
        "geo_start_iteration": opt.geo_start_iteration,
        "geo_ramp_iterations": opt.geo_ramp_iterations,
        "geo_cross_view_min_pixels": opt.geo_cross_view_min_pixels,
        "geo_boundary_ignore_px": opt.geo_boundary_ignore_px,
        "geo_boundary_band_px": opt.geo_boundary_band_px,
        "geo_cross_view_interval": opt.geo_cross_view_interval,
    }
    for name, value in nonnegative.items():
        if value < 0:
            raise ValueError("--{} must be non-negative".format(name))
    for name, value in {
        "geo_geometry_gradient_scale": opt.geo_geometry_gradient_scale,
        "geo_boundary_min_weight": opt.geo_boundary_min_weight,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError("--{} must be in [0, 1]".format(name))
    if not 0.0 < opt.geo_cross_view_downsample <= 1.0:
        raise ValueError("--geo_cross_view_downsample must be in (0, 1]")
    if not 0.0 < opt.geo_min_coverage <= 1.0:
        raise ValueError("--geo_min_coverage must be in (0, 1]")
    if opt.geo_cross_view_depth_tolerance <= 0:
        raise ValueError("--geo_cross_view_depth_tolerance must be positive")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    validate_geospatial_options(opt)
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, opt, pipe)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    train_cameras = scene.getTrainCameras()
    masked_train_cameras = [camera for camera in train_cameras if camera.gis_mask is not None]
    gis_requested = (
        opt.geo_mask_weight > 0
        or opt.geo_boundary_weight > 0
        or opt.geo_cross_view_weight > 0
        or opt.geo_depth_inside_mask
    )
    if gis_requested:
        if not masked_train_cameras:
            raise RuntimeError(
                "GIS supervision was requested, but no training masks were loaded. "
                "Check --gis_mask_dir and image-stem filenames."
            )
        missing_count = len(train_cameras) - len(masked_train_cameras)
        if missing_count and not opt.geo_allow_missing_masks:
            raise RuntimeError(
                f"GIS supervision requested but {missing_count}/{len(train_cameras)} training masks are missing. "
                "Fix the dataset or pass --geo_allow_missing_masks to continue intentionally."
            )
        print(f"GIS masks loaded for {len(masked_train_cameras)}/{len(train_cameras)} training cameras")

    positive_mask_cameras = [
        camera for camera in masked_train_cameras if float(camera.gis_mask.sum()) > 0
    ]
    camera_neighbors = build_camera_neighbors(positive_mask_cameras, opt.geo_cross_view_neighbors)
    if opt.geo_cross_view_weight > 0 and not camera_neighbors:
        raise RuntimeError(
            "Cross-view supervision requires at least two training cameras with non-empty GIS masks"
        )

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    ema_geo_loss_for_log = 0.0
    ema_cross_view_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # GeoConstraintGS.  The existing rasterizer is reused: a second pass
        # splats each Gaussian's learnable feature probability as a grey value.
        # This keeps the method Python/PyTorch-only while retaining gradients.
        geo_mask_loss = torch.zeros((), device="cuda")
        geo_boundary_loss = torch.zeros((), device="cuda")
        cross_view_loss = torch.zeros((), device="cuda")
        cross_view_stats = None
        effective_geo_weight = ramped_weight(
            opt.geo_mask_weight, iteration, opt.geo_start_iteration, opt.geo_ramp_iterations
        )
        effective_cross_view_weight = ramped_weight(
            opt.geo_cross_view_weight, iteration, opt.geo_start_iteration, opt.geo_ramp_iterations
        )
        semantic_pkg = None
        semantic_requested = effective_geo_weight > 0 or opt.geo_boundary_weight > 0 or effective_cross_view_weight > 0
        if viewpoint_cam.gis_mask is not None and semantic_requested:
            semantic_pkg = render_semantics(
                viewpoint_cam,
                gaussians,
                pipe,
                geometry_gradient_scale=opt.geo_geometry_gradient_scale,
                minimum_coverage=opt.geo_min_coverage,
            )
            rendered_mask = (
                semantic_pkg["semantic_probability"]
                if opt.geo_normalize_semantic else semantic_pkg["semantic_numerator"]
            )
            target_mask = viewpoint_cam.gis_mask
            confidence = boundary_confidence(
                target_mask, opt.geo_boundary_ignore_px, opt.geo_boundary_min_weight
            )
            if opt.geo_normalize_semantic:
                confidence = confidence * (semantic_pkg["coverage"].detach() >= opt.geo_min_coverage)
            if effective_geo_weight > 0:
                geo_mask_loss = weighted_dice_bce_loss(rendered_mask, target_mask, confidence)
                loss += effective_geo_weight * geo_mask_loss
            if opt.geo_boundary_weight > 0:
                boundary_only = 1.0 - boundary_confidence(
                    target_mask, int(opt.geo_boundary_band_px), 0.0
                )
                geo_boundary_loss = (
                    boundary_only * (rendered_mask - target_mask).abs()
                ).sum() / boundary_only.sum().clamp_min(1.0)
                loss += opt.geo_boundary_weight * geo_boundary_loss

            if (
                effective_cross_view_weight > 0
                and iteration % max(opt.geo_cross_view_interval, 1) == 0
                and id(viewpoint_cam) in camera_neighbors
            ):
                paired_camera = choice(camera_neighbors[id(viewpoint_cam)])
                paired_pkg = render_semantics(
                    paired_camera,
                    gaussians,
                    pipe,
                    geometry_gradient_scale=opt.geo_geometry_gradient_scale,
                    minimum_coverage=opt.geo_min_coverage,
                )
                cross_view_loss, cross_view_stats = depth_aware_cross_view_loss(
                    semantic_pkg["semantic_probability"],
                    semantic_pkg["normalized_invdepth"],
                    semantic_pkg["coverage"],
                    paired_pkg["semantic_probability"],
                    paired_pkg["normalized_invdepth"],
                    paired_pkg["coverage"],
                    viewpoint_cam,
                    paired_camera,
                    source_region=target_mask,
                    source_confidence=confidence,
                    downsample=opt.geo_cross_view_downsample,
                    minimum_coverage=opt.geo_min_coverage,
                    relative_depth_tolerance=opt.geo_cross_view_depth_tolerance,
                    minimum_valid_pixels=opt.geo_cross_view_min_pixels,
                )
                loss += effective_cross_view_weight * cross_view_loss

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda() * (mono_invdepth > 0)
            depth_residual = torch.abs(invDepth - mono_invdepth)
            if opt.geo_depth_charbonnier_epsilon > 0:
                depth_residual = torch.sqrt(
                    depth_residual.square() + opt.geo_depth_charbonnier_epsilon**2
                )
            if opt.geo_depth_inside_mask:
                if viewpoint_cam.gis_mask is None:
                    depth_mask = torch.zeros_like(depth_mask)
                else:
                    depth_confidence = boundary_confidence(
                        viewpoint_cam.gis_mask,
                        opt.geo_boundary_ignore_px,
                        opt.geo_boundary_min_weight,
                    )
                    depth_mask = depth_mask * viewpoint_cam.gis_mask * depth_confidence
                Ll1depth_pure = (depth_residual * depth_mask).sum() / depth_mask.sum().clamp_min(1.0)
            else:
                Ll1depth_pure = (depth_residual * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log
            ema_geo_loss_for_log = 0.4 * float(geo_mask_loss.detach()) + 0.6 * ema_geo_loss_for_log
            ema_cross_view_loss_for_log = 0.4 * float(cross_view_loss.detach()) + 0.6 * ema_cross_view_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.7f}",
                    "Depth": f"{ema_Ll1depth_for_log:.7f}",
                    "GIS": f"{ema_geo_loss_for_log:.7f}",
                    "CrossView": f"{ema_cross_view_loss_for_log:.7f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            loss_terms = {
                "gis_unweighted": float(geo_mask_loss.detach()),
                "gis_effective_weight": effective_geo_weight,
                "boundary_unweighted": float(geo_boundary_loss.detach()),
                "cross_view_unweighted": float(cross_view_loss.detach()),
                "cross_view_effective_weight": effective_cross_view_weight,
            }
            if cross_view_stats is not None:
                loss_terms.update({
                    f"cross_view_{key}": float(value)
                    for key, value in cross_view_stats.items()
                })
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, loss_terms)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii,
                                                geo_densify_weight=opt.geo_densify_weight)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args, optimization_args=None, pipeline_args=None):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    experiment_config = {"model": vars(args)}
    if optimization_args is not None:
        experiment_config["optimization"] = vars(optimization_args)
    if pipeline_args is not None:
        experiment_config["pipeline"] = vars(pipeline_args)
    with open(os.path.join(args.model_path, "experiment_config.json"), "w") as config_file:
        json.dump(experiment_config, config_file, indent=2, default=str)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, loss_terms=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        for name, value in (loss_terms or {}).items():
            tb_writer.add_scalar(f"train_loss_patches/{name}", value, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
