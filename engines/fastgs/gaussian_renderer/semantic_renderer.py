import math

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render_semantic(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
    return_instance=False,
    instance_none=-1,
    alpha_thresh=0.005,
    building_abs_thresh=0.0005,
    building_ratio_thresh=0.30,
    winner_ratio_thresh=0.20,
    semantic_alpha_min=0.0,
):
    del separate_sh
    screenspace_points = (
        torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda")
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
        tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=getattr(pipe, "antialiasing", False),
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    cov3d = None
    scales = None
    rotations = None
    if pipe.compute_cov3D_python:
        cov3d = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors = None
    if override_color is not None:
        colors = override_color
    elif pipe.convert_SHs_python:
        shs_view = pc.get_features.transpose(1, 2).view(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        directions = pc.get_xyz - viewpoint_camera.camera_center.repeat(
            pc.get_features.shape[0], 1
        )
        directions = directions / directions.norm(dim=1, keepdim=True)
        colors = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs_view, directions) + 0.5, 0.0
        )
    else:
        shs = pc.get_features

    outputs = rasterizer(
        means3D=pc.get_xyz,
        means2D=screenspace_points,
        shs=shs,
        colors_precomp=colors,
        opacities=pc.get_opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3d,
        instance_ids=pc.get_instance_id if return_instance else None,
        instance_confidences=pc.get_instance_confidence if return_instance else None,
        return_instance=return_instance,
        instance_none=instance_none,
        alpha_thresh=alpha_thresh,
        building_abs_thresh=building_abs_thresh,
        building_ratio_thresh=building_ratio_thresh,
        winner_ratio_thresh=winner_ratio_thresh,
        semantic_alpha_min=semantic_alpha_min,
    )

    if return_instance:
        image, radii, depth, instance, winner, stats = outputs
    else:
        image, radii, depth = outputs
        instance = winner = stats = None

    if use_trained_exp and hasattr(pc, "get_exposure_from_name"):
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        image = (
            torch.matmul(image.permute(1, 2, 0), exposure[:3, :3])
            .permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    result = {
        "render": image.clamp(0, 1),
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth,
    }
    if return_instance:
        result["instance"] = instance
        result["instance_winner"] = winner
        result["instance_stats"] = stats
    return result
