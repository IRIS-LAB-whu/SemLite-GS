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
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _module_dir in (
    "submodules/diff-gaussian-rasterization_fastgs",
    "submodules/fused-ssim",
    "submodules/simple-knn",
):
    sys.path.insert(0, os.path.join(_ROOT, _module_dir))

import torch
import numpy as np
import random, time
import json
from random import randint
from utils.loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fastgs, network_gui_ws
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_fastgs, sampling_cameras
from utils.region_balanced_sampler import RegionBalancedCameraSampler
from utils.training_schedule import should_reset_opacity


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    sampling_mode = str(opt.camera_sampling_mode)
    if sampling_mode not in ("image_uniform", "region_balanced"):
        raise ValueError(
            "--camera_sampling_mode must be image_uniform or region_balanced, "
            f"got {sampling_mode}"
        )
    region_sampler = None
    sampling_log = None
    if sampling_mode == "region_balanced":
        if not opt.region_camera_map:
            raise ValueError(
                "region_balanced sampling requires --region_camera_map"
            )
        region_sampler = RegionBalancedCameraSampler(
            viewpoint_stack,
            opt.region_camera_map,
            seed=opt.camera_sampling_seed,
            camera_min_count=opt.region_camera_min_count,
            camera_max_count=opt.region_camera_max_count,
            planned_iterations=opt.iterations,
            camera_priority_path=opt.region_camera_priority_path or None,
            priority_fraction=opt.region_camera_priority_fraction,
            priority_quantile=opt.region_camera_priority_quantile,
            priority_start_iteration=opt.region_camera_priority_start_iteration,
        )
        print(
            f"Region-balanced camera sampling: {len(region_sampler.region_ids)} "
            f"regions from {opt.region_camera_map}; "
            f"camera bounds=[{opt.region_camera_min_count}, "
            f"{opt.region_camera_max_count or 'unbounded'}]; "
            f"priority={opt.region_camera_priority_fraction:.2f} from "
            f"iteration {opt.region_camera_priority_start_iteration}"
        )
        if opt.camera_sampling_log_every > 0:
            sampling_log = open(
                os.path.join(dataset.model_path, "camera_sampling.jsonl"),
                "w",
                encoding="utf-8",
                buffering=1,
            )


    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    roi_loss_reported = False
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fastgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()

        gaussians.update_learning_rate(iteration)


        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()


        selected_region_id = None
        if region_sampler is not None:
            viewpoint_cam, selected_region_id = region_sampler.sample(iteration)
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            _ = viewpoint_indices.pop(rand_idx)
        if (
            sampling_log is not None
            and iteration % int(opt.camera_sampling_log_every) == 0
        ):
            sampling_log.write(
                json.dumps(
                    {
                        "iteration": int(iteration),
                        "region_id": selected_region_id,
                        "camera": viewpoint_cam.image_name,
                    }
                )
                + "\n"
            )


        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_fastgs(viewpoint_cam, gaussians, pipe, bg, opt.mult)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


        gt_image = viewpoint_cam.original_image.cuda()
        loss_roi = None
        if region_sampler is not None and opt.region_loss_outside_weight < 1.0:
            loss_roi = region_sampler.loss_roi(
                selected_region_id,
                viewpoint_cam.image_name,
                image.shape[2],
                image.shape[1],
            )
        if loss_roi is None:
            Ll1 = l1_loss(image, gt_image)
            ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            roi_mask_cpu, (x0, y0, x1, y1) = loss_roi
            roi_mask = torch.from_numpy(roi_mask_cpu).to(
                device=image.device, dtype=image.dtype
            )
            outside_weight = float(opt.region_loss_outside_weight)
            pixel_weights = roi_mask + outside_weight * (1.0 - roi_mask)
            if not roi_loss_reported:
                roi_fraction = float(roi_mask.mean().item())
                weighted_roi_share = roi_fraction / max(
                    roi_fraction + outside_weight * (1.0 - roi_fraction), 1e-12
                )
                print(
                    f"Projected ROI loss enabled: outside_weight={outside_weight:.4f}, "
                    f"first_camera={viewpoint_cam.image_name}, roi_fraction={roi_fraction:.4f}, "
                    f"weighted_roi_share={weighted_roi_share:.4f}, bbox={(x0, y0, x1, y1)}"
                )
                roi_loss_reported = True
            Ll1 = (
                torch.abs(image - gt_image) * pixel_weights.unsqueeze(0)
            ).sum() / (pixel_weights.sum() * image.shape[0]).clamp_min(1.0)
            ssim_value = fast_ssim(
                image[:, y0:y1, x0:x1].unsqueeze(0),
                gt_image[:, y0:y1, x0:x1].unsqueeze(0),
            )
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        iter_end.record()

        with torch.no_grad():

            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)


            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            optim_start.record()


            if iteration < opt.densify_until_iter:

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None


                    my_viewpoint_stack = (
                        region_sampler.eligible_cameras.copy()
                        if region_sampler is not None
                        else scene.getTrainCameras().copy()
                    )
                    camlist = sampling_cameras(my_viewpoint_stack)


                    importance_score, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt, DENSIFY=True)
                    gaussians.densify_and_prune_fastgs(max_screen_size = size_threshold,
                                                min_opacity = 0.005,
                                                extent = scene.cameras_extent,
                                                radii=radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score)


                if should_reset_opacity(
                    iteration=iteration,
                    opacity_reset_interval=opt.opacity_reset_interval,
                    densify_until_iter=opt.densify_until_iter,
                    white_background=dataset.white_background,
                    densify_from_iter=opt.densify_from_iter,
                ):
                    gaussians.reset_opacity()


            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = (
                    region_sampler.eligible_cameras.copy()
                    if region_sampler is not None
                    else scene.getTrainCameras().copy()
                )
                camlist = sampling_cameras(my_viewpoint_stack)

                _, pruning_score = compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, opt)
                gaussians.final_prune_fastgs(min_opacity = 0.1, pruning_score = pruning_score)


            if iteration < opt.iterations:
                if opt.optimizer_type == "default":
                    gaussians.optimizer_step(iteration)
                elif opt.optimizer_type == "sparse_adam":
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)


            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3


    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")
    if sampling_log is not None:
        sampling_log.close()
    if region_sampler is not None:
        sampling_summary = region_sampler.summary(opt.iterations - first_iter + 1)
        summary_path = os.path.join(dataset.model_path, "camera_sampling_summary.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump(sampling_summary, file, ensure_ascii=False, indent=2)
        print(f"Camera sampling summary: {summary_path}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)


    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":

    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)


    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets
    )


    print("\nTraining complete.")
