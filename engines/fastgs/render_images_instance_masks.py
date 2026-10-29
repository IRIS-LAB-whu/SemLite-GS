import os
import sys
from os import makedirs
from argparse import ArgumentParser
from typing import Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
for module_dir in (
    "submodules/diff-gaussian-rasterization-semantic",
    "submodules/simple-knn",
):
    sys.path.insert(0, os.path.join(ROOT, module_dir))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from scene import Scene
from gaussian_renderer.semantic_renderer import render_semantic
from utils.instance_encoding import instance_colors
from utils.instance_pipeline_config import (
    ALPHA_THRESHOLD,
    BUILDING_ABS_THRESHOLD,
    BUILDING_RATIO_THRESHOLD,
    SEMANTIC_ALPHA_MIN,
    WINNER_RATIO_THRESHOLD,
)
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_model import GaussianModel

try:
    import diff_gaussian_rasterization

    SPARSE_ADAM_AVAILABLE = hasattr(
        diff_gaussian_rasterization,
        "SparseGaussianAdam",
    )
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def instance_ids_to_rgb(instance_map: np.ndarray) -> np.ndarray:
    ids = np.asarray(instance_map)
    h, w = ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    valid = ids >= 0
    if not np.any(valid):
        return rgb

    max_id = int(ids[valid].max())
    lut = instance_colors(np.arange(max_id + 1))
    rgb[valid] = lut[ids[valid]]
    return rgb


def depth_to_vis(depth_np: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_np) & (depth_np > 0)
    vis = np.zeros(depth_np.shape, dtype=np.uint8)
    if not np.any(valid):
        return vis
    values = depth_np[valid]
    lo = float(np.percentile(values, 2.0))
    hi = float(np.percentile(values, 98.0))
    if hi <= lo:
        hi = lo + 1e-6
    vis[valid] = np.clip((depth_np[valid] - lo) / (hi - lo) * 255.0, 0, 255).astype(
        np.uint8
    )
    return vis


def save_instance_render(
    output_root,
    image_name,
    instance_map: torch.Tensor,
    depth_map: Optional[torch.Tensor] = None,
    save_raw: bool = False,
    save_depth: bool = False,
):
    vis_dir = os.path.join(output_root, "vis")
    fg_dir = os.path.join(output_root, "fg")
    labels_dir = os.path.join(output_root, "labels")
    makedirs(vis_dir, exist_ok=True)
    makedirs(fg_dir, exist_ok=True)
    makedirs(labels_dir, exist_ok=True)

    instance_np = instance_map.detach().to(dtype=torch.int16).cpu().numpy()
    if save_raw:
        raw_dir = os.path.join(output_root, "raw")
        makedirs(raw_dir, exist_ok=True)
        np.save(os.path.join(raw_dir, f"{image_name}.npy"), instance_np)
    Image.fromarray(instance_ids_to_rgb(instance_np), mode="RGB").save(
        os.path.join(vis_dir, f"{image_name}.png")
    )


    labels = np.zeros_like(instance_np, dtype=np.uint8)
    valid = instance_np >= 0
    if np.any(valid):
        maximum = int(instance_np[valid].max()) + 1
        if maximum > 255:
            raise ValueError(
                f"{image_name}: {maximum} labels cannot be stored in uint8 PNG"
            )
        labels[valid] = instance_np[valid].astype(np.int32) + 1
    Image.fromarray(labels, mode="L").save(
        os.path.join(labels_dir, f"{image_name}.png")
    )

    fg = np.zeros_like(instance_np, dtype=np.uint8)
    fg[instance_np >= 0] = 255
    Image.fromarray(fg, mode="L").save(os.path.join(fg_dir, f"{image_name}.png"))

    if save_depth and depth_map is not None:
        depth_np = depth_map.detach().to(dtype=torch.float32).cpu().numpy()
        depth_np = np.squeeze(depth_np)
        depth_root = os.path.dirname(output_root)
        depth_raw_dir = os.path.join(depth_root, "depth", "raw")
        depth_vis_dir = os.path.join(depth_root, "depth", "vis")
        makedirs(depth_raw_dir, exist_ok=True)
        makedirs(depth_vis_dir, exist_ok=True)
        np.save(os.path.join(depth_raw_dir, f"{image_name}.npy"), depth_np)
        Image.fromarray(depth_to_vis(depth_np), mode="L").save(
            os.path.join(depth_vis_dir, f"{image_name}.png")
        )


def render_train_views(
    model_path,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    separate_sh,
    output_dir=None,
    max_views=0,
    save_raw=False,
    save_depth=False,
):
    output_dir = output_dir or os.path.join(model_path, "train", f"ours_{iteration}")
    instance_root = os.path.join(output_dir, "instance_masks")
    makedirs(instance_root, exist_ok=True)

    semantic_only_color = None
    if BUILDING_RATIO_THRESHOLD < 0:


        semantic_only_color = torch.zeros_like(gaussians.get_xyz)

    selected_views = views
    if max_views and max_views > 0:
        selected_views = selected_views[:max_views]
    for view in tqdm(selected_views, desc="Rendering train"):
        render_pkg = render_semantic(
            view,
            gaussians,
            pipeline,
            background,
            use_trained_exp=train_test_exp,
            separate_sh=separate_sh,
            override_color=semantic_only_color,
            return_instance=True,
            instance_none=-1,
            alpha_thresh=ALPHA_THRESHOLD,
            building_abs_thresh=BUILDING_ABS_THRESHOLD,
            building_ratio_thresh=BUILDING_RATIO_THRESHOLD,
            winner_ratio_thresh=WINNER_RATIO_THRESHOLD,
            semantic_alpha_min=SEMANTIC_ALPHA_MIN,
        )
        instance_map = render_pkg["instance"]
        depth_map = render_pkg["depth"]
        if train_test_exp:
            instance_map = instance_map[..., instance_map.shape[-1] // 2 :]
            depth_map = depth_map[..., depth_map.shape[-1] // 2 :]
        save_instance_render(
            instance_root,
            view.image_name,
            instance_map,
            depth_map=depth_map,
            save_raw=save_raw,
            save_depth=save_depth,
        )


def render_train_set(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    separate_sh: bool,
    output_dir: str = "",
    max_views: int = 0,
    save_raw: bool = False,
    save_depth: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            load_images=False,
            load_test=False,
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        has_instance_ids = (
            hasattr(gaussians, "get_instance_id")
            and gaussians.get_instance_id.numel() > 0
            and bool(torch.any(gaussians.get_instance_id >= 0).item())
        )
        if not has_instance_ids:
            raise RuntimeError(
                "No valid instance_id field was found in the loaded PLY."
            )

        render_train_views(
            dataset.model_path,
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
            getattr(dataset, "train_test_exp", False),
            separate_sh,
            output_dir=output_dir or None,
            max_views=max_views,
            save_raw=save_raw,
            save_depth=save_depth,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--save-depth", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_train_set(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        SPARSE_ADAM_AVAILABLE,
        args.output_dir,
        args.max_views,
        args.save_raw,
        args.save_depth,
    )
