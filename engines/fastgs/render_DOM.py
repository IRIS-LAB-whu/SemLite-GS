from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0,
    os.path.join(
        ROOT,
        "submodules",
        "diff-gaussian-rasterization_fastgs_ortho",
    ),
)

from utils.instance_pipeline_config import (
    KEEP_RATIO,
    MARGIN_RATIO,
    VIEW_DIR,
    YAW_DEG,
)
from utils.ortho_projection import (
    build_overlapping_grid,
    build_world_z_rotation,
    camera_position_from_view,
    choose_virtual_camera_position,
    compute_robust_camera_box,
    load_training_camera_orientation,
    transform_points_to_camera,
    view_matrix,
)

DEFAULT_RASTER_MULT = 0.5


def load_gaussians_from_ply_with_sh(
    ply_path: str,
    device: str = "cuda",
):
    ply = PlyData.read(ply_path)
    v = ply["vertex"].data
    names = v.dtype.names

    def req(name):
        if name not in names:
            raise KeyError(f"PLY 中缺少字段: {name}")
        return np.asarray(v[name], dtype=np.float32)

    xyz = np.stack([req("x"), req("y"), req("z")], axis=1)
    opacity_raw = req("opacity")
    scale_raw = np.stack([req("scale_0"), req("scale_1"), req("scale_2")], axis=1)
    rot = np.stack([req("rot_0"), req("rot_1"), req("rot_2"), req("rot_3")], axis=1)

    f_dc = np.stack([req("f_dc_0"), req("f_dc_1"), req("f_dc_2")], axis=1)

    f_rest_names = sorted(
        [n for n in names if n.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    f_rest = np.stack([req(n) for n in f_rest_names], axis=1)

    if f_rest.shape[1] != 45:
        raise ValueError(f"当前 f_rest 维数为 {f_rest.shape[1]}，不是期望的 45。")

    sh_degree = 3
    num_bases = (sh_degree + 1) ** 2


    f_rest = f_rest.reshape(-1, 3, num_bases - 1).transpose(0, 2, 1)
    f_dc = f_dc[:, None, :]
    sh_coeffs = np.concatenate([f_dc, f_rest], axis=1)

    means = torch.from_numpy(xyz).to(device=device, dtype=torch.float32)
    opacities = torch.sigmoid(
        torch.from_numpy(opacity_raw).to(device=device, dtype=torch.float32)
    )
    scales = torch.exp(
        torch.from_numpy(scale_raw).to(device=device, dtype=torch.float32)
    )
    quats = F.normalize(
        torch.from_numpy(rot).to(device=device, dtype=torch.float32), dim=-1
    )
    sh_coeffs = torch.from_numpy(sh_coeffs).to(device=device, dtype=torch.float32)

    return means, quats, scales, opacities, sh_coeffs, sh_degree


def build_fastgs_ortho_matrices(
    viewmat_np: np.ndarray,
    ortho_width_world: float,
    ortho_height_world: float,
    near_plane: float,
    far_plane: float,
    device: str,
):
    projection = np.zeros((4, 4), dtype=np.float32)
    projection[0, 0] = 2.0 / max(float(ortho_width_world), 1e-6)
    projection[1, 1] = 2.0 / max(float(ortho_height_world), 1e-6)
    projection[2, 2] = 1.0 / max(float(far_plane - near_plane), 1e-6)
    projection[2, 3] = -float(near_plane) * projection[2, 2]
    projection[3, 3] = 1.0

    world_view = (
        torch.from_numpy(viewmat_np)
        .to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    projection_t = (
        torch.from_numpy(projection)
        .to(device=device, dtype=torch.float32)
        .transpose(0, 1)
        .contiguous()
    )
    full_projection = (world_view.unsqueeze(0).bmm(projection_t.unsqueeze(0))).squeeze(
        0
    )
    return world_view, full_projection


def render_fastgs_ortho(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    sh_coeffs: torch.Tensor,
    sh_degree: int,
    world_view: torch.Tensor,
    full_projection: torch.Tensor,
    camera_position,
    background: torch.Tensor,
    width: int,
    height: int,
    mult: float = 0.5,
):
    from diff_gaussian_rasterization_fastgs_ortho import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    device = means.device
    camera_position = torch.as_tensor(
        camera_position,
        dtype=torch.float32,
        device=device,
    )
    screenspace_points = torch.zeros(
        (means.shape[0], 4),
        dtype=means.dtype,
        device=device,
    )
    metric_map = torch.zeros(
        int(width) * int(height),
        dtype=torch.int32,
        device=device,
    )
    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=1.0,
        tanfovy=1.0,
        bg=background,
        scale_modifier=1.0,
        viewmatrix=world_view,
        projmatrix=full_projection,
        sh_degree=int(sh_degree),
        campos=camera_position,
        mult=float(mult),
        prefiltered=False,
        debug=False,
        get_flag=False,
        metric_map=metric_map,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer(
        means3D=means,
        means2D=screenspace_points,
        dc=sh_coeffs[:, :1, :].contiguous(),
        shs=sh_coeffs[:, 1:, :].contiguous(),
        colors_precomp=None,
        opacities=opacities[:, None].contiguous(),
        scales=scales.contiguous(),
        rotations=quats.contiguous(),
        cov3D_precomp=None,
    )


def save_render_image(render_colors: torch.Tensor, out_path: str) -> tuple[int, ...]:
    if render_colors.ndim == 4:
        img = render_colors[0].detach().cpu().numpy()
    elif render_colors.ndim == 3:
        img = render_colors.detach().cpu().numpy()
    else:
        raise RuntimeError(f"不支持的 render_colors 维度: {tuple(render_colors.shape)}")

    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))

    img = np.clip(img, 0.0, 1.0)
    img_u8 = (img * 255.0).astype(np.uint8)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(img_u8).save(out_path)
    return img_u8.shape


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ply", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="单图模式输出路径；分区模式使用 --regions-dir",
    )
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--height", type=int, default=2000)
    parser.add_argument(
        "--sh-degree",
        type=int,
        choices=range(4),
        default=3,
        help="渲染使用的 SH 阶数；0 仅使用稳定的 DC 颜色，默认 3",
    )
    parser.add_argument(
        "--raster-mult",
        type=float,
        default=DEFAULT_RASTER_MULT,
        help="FastGS compact-box beta；1.0 保留更完整的高斯尾部，0.5 更快",
    )
    parser.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
        help="无覆盖区域背景色",
    )
    parser.add_argument(
        "--camera-height-margin",
        type=float,
        default=1.0,
        help="world_z 模式下虚拟相机高于场景稳健最高点的距离；可降至 0.25~0.5",
    )
    parser.add_argument("--grid-rows", type=int, default=1)
    parser.add_argument("--grid-cols", type=int, default=1)
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.1,
        help="相邻区域的重叠比例，范围 [0, 1)",
    )
    parser.add_argument(
        "--regions-dir",
        type=str,
        default=None,
        help="分区图输出目录；grid 大于 1 时必填",
    )
    parser.add_argument(
        "--region-config",
        type=str,
        default=None,
        help="自定义 camera-space 区域 JSON；用于只渲染选定 ROI",
    )
    parser.add_argument(
        "--orientation-mode",
        "--orientation_mode",
        dest="orientation_mode",
        choices=["world_z", "train_camera"],
        default="world_z",
        help="world_z：按世界 Z 轴俯视；train_camera：使用指定训练相机朝向",
    )
    parser.add_argument(
        "--cameras-json",
        "--cameras_json",
        dest="cameras_json",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--camera-idx",
        "--camera_idx",
        dest="camera_idx",
        type=int,
        default=None,
    )

    args = parser.parse_args()
    if args.raster_mult <= 0.0:
        raise ValueError("--raster-mult 必须 > 0")
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("DOM 渲染需要 CUDA，但 torch.cuda.is_available() = False")


    means, quats, scales, opacities, sh_coeffs, loaded_sh_degree = (
        load_gaussians_from_ply_with_sh(args.ply, device=device)
    )
    sh_degree = min(int(args.sh_degree), int(loaded_sh_degree))
    xyz_world = means.detach().cpu().numpy()

    if args.orientation_mode == "train_camera":
        if args.cameras_json is None or args.camera_idx is None:
            raise ValueError(
                "train_camera 模式需要同时提供 --cameras-json 和 --camera-idx"
            )
        cam_pos, R_w2c, convention, positive_ratio = (
            load_training_camera_orientation(
                args.cameras_json,
                args.camera_idx,
                xyz_world,
            )
        )
        print(
            f"[相机] 训练相机 id={args.camera_idx}，旋转约定={convention}，"
            f"正深度比例={positive_ratio:.4f}"
        )
    else:
        cam_pos = choose_virtual_camera_position(
            world_points=xyz_world,
            view_dir=VIEW_DIR,
            keep_ratio=KEEP_RATIO,
            z_margin=args.camera_height_margin,
        )
        R_w2c = build_world_z_rotation(yaw_deg=YAW_DEG, view_dir=VIEW_DIR)
    pts_cam = transform_points_to_camera(xyz_world, R_w2c, cam_pos)


    box = compute_robust_camera_box(
        pts_cam,
        margin_ratio=MARGIN_RATIO,
        keep_ratio=KEEP_RATIO,
    )

    z_min = box["z_min"]
    z_max = box["z_max"]
    if args.grid_rows < 1 or args.grid_cols < 1:
        raise ValueError("--grid-rows 和 --grid-cols 必须 >= 1")
    custom_regions = None
    if args.region_config is not None:
        with open(args.region_config, "r", encoding="utf-8") as file:
            custom_regions = json.load(file).get("regions", [])
        if not custom_regions:
            raise ValueError("--region-config 中没有 regions")
    multi_region = (
        custom_regions is not None
        or args.grid_rows > 1
        or args.grid_cols > 1
    )
    if multi_region and args.regions_dir is None:
        raise ValueError("分区渲染时必须提供 --regions-dir")
    if not multi_region and args.out is None:
        raise ValueError("单图渲染时必须提供 --out")

    full_x_min = box["x_mid"] - 0.5 * box["ortho_w"]
    full_x_max = box["x_mid"] + 0.5 * box["ortho_w"]
    full_y_min = box["y_mid"] - 0.5 * box["ortho_h"]
    full_y_max = box["y_mid"] + 0.5 * box["ortho_h"]
    regions = custom_regions or build_overlapping_grid(
        full_x_min,
        full_x_max,
        full_y_min,
        full_y_max,
        args.grid_rows,
        args.grid_cols,
        args.tile_overlap,
    )
    background_value = 1.0 if args.background == "white" else 0.0
    background = torch.full(
        (3,), background_value, dtype=torch.float32, device=device
    )


    near_plane = max(1e-3, float(z_min) - 1.0)
    far_plane = max(near_plane + 1e-3, float(z_max) + 1.0)
    metadata = {
        "source_ply": os.path.abspath(args.ply),
        "orientation_mode": args.orientation_mode,
        "camera_position": [float(value) for value in cam_pos],
        "rotation_world_to_camera": R_w2c.tolist(),
        "camera_height_margin": float(args.camera_height_margin),
        "near_plane": float(near_plane),
        "far_plane": float(far_plane),
        "grid_rows": int(args.grid_rows),
        "grid_cols": int(args.grid_cols),
        "tile_overlap": float(args.tile_overlap),
        "image_width": int(args.width),
        "image_height": int(args.height),
        "sh_degree": int(sh_degree),
        "loaded_sh_degree": int(loaded_sh_degree),
        "raster_mult": float(args.raster_mult),
        "background": args.background,
        "regions": [],
    }
    if multi_region:
        os.makedirs(args.regions_dir, exist_ok=True)

    for region in regions:
        x_mid = float(region["x_mid"])
        y_mid = float(region["y_mid"])
        ortho_w = float(region["ortho_w"])
        ortho_h = float(region["ortho_h"])
        viewmat_np = view_matrix(R_w2c, cam_pos, x_mid, y_mid)
        render_camera_position = camera_position_from_view(viewmat_np)
        world_view, full_projection = build_fastgs_ortho_matrices(
            viewmat_np=viewmat_np,
            ortho_width_world=ortho_w,
            ortho_height_world=ortho_h,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device,
        )
        if multi_region:
            filename = region.get("filename")
            if filename is None:
                filename = f'region_r{int(region["row"]):02d}_c{int(region["col"]):02d}.png'
            out_path = os.path.join(args.regions_dir, filename)
        else:
            filename = os.path.basename(args.out)
            out_path = args.out
        print(
            f"Rendering region {region.get('region_id', filename)} with "
            f"{len(means)} Gaussians at {args.width}x{args.height}; "
            f"extent={ortho_w:.3f}x{ortho_h:.3f}, "
            f"center=({x_mid:.3f}, {y_mid:.3f})"
        )
        render_colors, _, _ = render_fastgs_ortho(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            sh_degree=sh_degree,
            world_view=world_view,
            full_projection=full_projection,
            camera_position=render_camera_position,
            background=background,
            width=args.width,
            height=args.height,
            mult=args.raster_mult,
        )
        image_shape = save_render_image(render_colors, out_path)
        print(f"Saved {out_path} ({image_shape})")
        metadata["regions"].append(
            {
                **region,
                "filename": filename,
                "camera_position": [float(value) for value in render_camera_position],
            }
        )

    if multi_region:
        metadata_path = os.path.join(args.regions_dir, "regions.json")
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
