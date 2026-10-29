from __future__ import annotations

import os
import json
import argparse
import numpy as np
from PIL import Image, ImageFile
from plyfile import PlyData, PlyElement
from scipy import ndimage
from utils.instance_encoding import instance_colors
from utils.instance_pipeline_config import (
    COMPONENT_MIN_POINTS,
    COMPONENT_VOXEL_SIZE,
    FLOATING_HEIGHT_MARGIN,
    FLOATING_HIGH_PERCENTILE,
    FLOATING_LOW_PERCENTILE,
    FLOATING_MIN_POINTS,
    KEEP_RATIO,
    MARGIN_RATIO,
    MASK_THRESHOLD,
    MIN_INSTANCE_AREA,
    OPEN_RADIUS,
    SAMPLE_RADIUS,
    VIEW_DIR,
    YAW_DEG,
)
from utils.ortho_projection import (
    build_world_z_rotation,
    choose_virtual_camera_position,
    compute_robust_camera_box,
    load_training_camera_orientation,
    project_points_to_ortho_pixels,
    transform_points_to_camera,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def make_disk(radius: int):
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def make_disk_offsets(radius: int):
    radius = int(radius)
    st = make_disk(radius)
    ys, xs = np.where(st)
    ys = ys.astype(np.int32) - radius
    xs = xs.astype(np.int32) - radius
    return np.stack([ys, xs], axis=1)


def filter_floating_points_by_instance_height(
    xyz_world: np.ndarray,
    instance_ids_3d: np.ndarray,
    high_percentile: float,
    low_percentile: float,
    margin: float,
    min_points: int,
) -> np.ndarray:
    remove = np.zeros(len(instance_ids_3d), dtype=bool)
    valid_ids = sorted(int(x) for x in np.unique(instance_ids_3d) if int(x) >= 0)
    if not valid_ids:
        return remove

    high_percentile = float(high_percentile)
    low_percentile = float(low_percentile)
    margin = float(margin)
    min_points = max(1, int(min_points))
    z = xyz_world[:, 2]
    for inst_id in valid_ids:
        keep = instance_ids_3d == inst_id
        if int(keep.sum()) < min_points:
            continue
        z_inst = z[keep]
        high_limit = float(np.percentile(z_inst, high_percentile)) + margin
        low_limit = float(np.percentile(z_inst, low_percentile)) - margin
        remove |= keep & ((z > high_limit) | (z < low_limit))

    return remove


def filter_small_3d_components_by_instance(
    xyz_world: np.ndarray,
    instance_ids_3d: np.ndarray,
    voxel_size: float,
    min_instance_points: int,
) -> np.ndarray:

    remove = np.zeros(len(instance_ids_3d), dtype=bool)
    valid_ids = sorted(int(x) for x in np.unique(instance_ids_3d) if int(x) >= 0)
    if not valid_ids:
        return remove

    voxel_size = max(float(voxel_size), 1e-6)
    min_instance_points = max(1, int(min_instance_points))
    neighbor_offsets = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=np.int64,
    )

    for inst_id in valid_ids:
        inst_mask = instance_ids_3d == inst_id
        inst_indices = np.nonzero(inst_mask)[0]
        if int(inst_indices.size) < min_instance_points:
            continue

        pts = xyz_world[inst_indices]
        origin = pts.min(axis=0)
        vox = np.floor((pts - origin[None, :]) / voxel_size).astype(np.int64)
        uniq, inverse, counts = np.unique(
            vox, axis=0, return_inverse=True, return_counts=True
        )
        num_voxels = int(uniq.shape[0])
        if num_voxels <= 1:
            continue

        voxel_to_idx = {tuple(v.tolist()): i for i, v in enumerate(uniq)}
        labels = np.full(num_voxels, -1, dtype=np.int32)
        component_point_counts = []
        component_id = 0

        for start in range(num_voxels):
            if labels[start] >= 0:
                continue
            labels[start] = component_id
            stack = [start]
            point_count = 0
            while stack:
                cur = stack.pop()
                point_count += int(counts[cur])
                base = uniq[cur]
                for off in neighbor_offsets:
                    nxt = tuple((base + off).tolist())
                    nxt_idx = voxel_to_idx.get(nxt)
                    if nxt_idx is not None and labels[nxt_idx] < 0:
                        labels[nxt_idx] = component_id
                        stack.append(nxt_idx)
            component_point_counts.append(point_count)
            component_id += 1

        component_point_counts = np.asarray(component_point_counts, dtype=np.int64)
        if component_point_counts.size <= 1:
            continue
        largest = int(np.argmax(component_point_counts))
        keep_components = np.zeros(component_point_counts.size, dtype=bool)
        keep_components[largest] = True
        point_components = labels[inverse]
        remove_inst = ~keep_components[point_components]
        remove[inst_indices[remove_inst]] = True

    return remove


def load_ply_vertices(ply_path: str):
    ply = PlyData.read(ply_path)
    return ply["vertex"].data


def extract_xyz_from_vertex(vertex_data):
    x = np.asarray(vertex_data["x"], dtype=np.float32)
    y = np.asarray(vertex_data["y"], dtype=np.float32)
    z = np.asarray(vertex_data["z"], dtype=np.float32)
    xyz = np.stack([x, y, z], axis=1)
    return xyz


def save_vertices(vertices, out_path: str):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    el = PlyElement.describe(vertices, "vertex")
    PlyData([el], text=False).write(out_path)


def structured_add_field(data, field_name, field_dtype, values):
    if field_name in data.dtype.names:
        new_data = data.copy()
        new_data[field_name] = values
        return new_data

    new_dtype = data.dtype.descr + [(field_name, field_dtype)]
    new_data = np.empty(data.shape, dtype=new_dtype)
    for name in data.dtype.names:
        new_data[name] = data[name]
    new_data[field_name] = values
    return new_data


def ensure_rgb_fields(data):
    color_fields = ("red", "green", "blue")
    missing_fields = [name for name in color_fields if name not in data.dtype.names]
    if not missing_fields:
        return data.copy()

    new_dtype = list(data.dtype.descr) + [(name, "u1") for name in missing_fields]
    new_data = np.empty(data.shape, dtype=new_dtype)
    for name in data.dtype.names:
        new_data[name] = data[name]
    for name in missing_fields:
        new_data[name] = 255

    return new_data


def sample_instance_id_with_radius(
    label_map: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:


    height, width = label_map.shape[:2]
    result = np.zeros(len(u), dtype=np.int32)

    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    idxs = np.where(valid)[0]

    if radius <= 0:
        result[idxs] = label_map[v[idxs], u[idxs]]
        return result, valid

    offsets = make_disk_offsets(radius)
    padded = np.pad(label_map, radius, mode="constant", constant_values=0)
    uu = u[idxs] + radius
    vv = v[idxs] + radius
    batch_size = 200000

    for start in range(0, len(idxs), batch_size):
        end = min(start + batch_size, len(idxs))
        uu_batch = uu[start:end]
        vv_batch = vv[start:end]
        vals = np.stack(
            [padded[vv_batch + dy, uu_batch + dx] for dy, dx in offsets],
            axis=1,
        ).astype(np.int32, copy=False)


        candidates = np.sort(vals, axis=1)
        counts = np.stack(
            [
                (
                    (candidates == candidates[:, column, None])
                    & (candidates[:, column, None] > 0)
                ).sum(axis=1)
                for column in range(candidates.shape[1])
            ],
            axis=1,
        )
        best_column = counts.argmax(axis=1)
        result[idxs[start:end]] = candidates[
            np.arange(candidates.shape[0]),
            best_column,
        ]

    return result, valid


def instance_segment_mask(
    mask: np.ndarray,
    min_instance_area: int = 50,
    mask_format: str = "binary",
) -> tuple[np.ndarray, dict[int, dict]]:
    if mask_format == "labels":
        source = np.asarray(mask)
        if source.ndim != 2:
            raise ValueError(f"标签 mask 必须是二维图，当前 shape={source.shape}")
        filtered = np.zeros(source.shape, dtype=np.int32)
        instance_info: dict[int, dict] = {}
        next_id = 1
        for source_id in sorted(int(value) for value in np.unique(source) if int(value) > 0):
            pixels = source == source_id
            area = int(pixels.sum())
            if area < max(1, int(min_instance_area)):
                continue
            ys, xs = np.nonzero(pixels)
            filtered[pixels] = next_id
            instance_info[next_id] = {
                "source_label_id": source_id,
                "pixel_area": area,
                "bbox_xy": {
                    "minx": int(xs.min()),
                    "maxx": int(xs.max()),
                    "miny": int(ys.min()),
                    "maxy": int(ys.max()),
                },
            }
            next_id += 1
        return filtered, instance_info
    if mask_format != "binary":
        raise ValueError(f"未知 mask_format={mask_format!r}")

    binary_mask = mask > MASK_THRESHOLD
    work_mask = ndimage.binary_opening(
        binary_mask,
        structure=make_disk(OPEN_RADIUS),
    )

    structure = np.ones((3, 3), dtype=np.uint8)
    raw_labels, num = ndimage.label(work_mask, structure=structure)

    if num == 0:
        return np.zeros_like(mask, dtype=np.int32), {}

    areas = np.bincount(raw_labels.ravel())
    kept_labels = np.flatnonzero(areas >= min_instance_area)
    kept_labels = kept_labels[kept_labels > 0]
    old_to_new = np.zeros(len(areas), dtype=np.int32)
    old_to_new[kept_labels] = np.arange(1, len(kept_labels) + 1)
    filtered = old_to_new[raw_labels]

    filtered_areas = np.bincount(filtered.ravel())
    instance_info = {}
    for label, region in enumerate(ndimage.find_objects(filtered), start=1):
        if region is None:
            continue
        y_slice, x_slice = region
        instance_info[label] = {
            "pixel_area": int(filtered_areas[label]),
            "bbox_xy": {
                "minx": int(x_slice.start),
                "maxx": int(x_slice.stop - 1),
                "miny": int(y_slice.start),
                "maxy": int(y_slice.stop - 1),
            },
        }

    return filtered, instance_info


def colorize_vertices_by_instance(vertices, instance_ids: np.ndarray):


    data = ensure_rgb_fields(vertices)


    data["red"] = 80
    data["green"] = 80
    data["blue"] = 80

    valid = instance_ids >= 0
    colors = instance_colors(instance_ids[valid])
    data["red"][valid] = colors[:, 0]
    data["green"][valid] = colors[:, 1]
    data["blue"][valid] = colors[:, 2]

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="整场景 point_cloud.ply")
    parser.add_argument(
        "--mask", type=str, required=True, help="建筑物 DOM 二值图或层级标签图"
    )
    parser.add_argument(
        "--mask-format",
        choices=["binary", "labels"],
        default="binary",
        help="binary 保持原连通域流程；labels 保留已有非零区域 ID 和组内空隙",
    )
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument(
        "--dom-out-dir",
        type=str,
        default=None,
        help="二维 DOM 实例图输出目录；默认与 --out-dir 相同",
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default=None,
        help="统计 JSON 输出路径；默认写入 --out-dir/summary.json",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=["world_z", "train_camera"],
        default="world_z",
    )
    parser.add_argument("--cameras-json", type=str, default=None)
    parser.add_argument("--camera-idx", type=int, default=None)
    parser.add_argument(
        "--floating-high-percentile",
        type=float,
        default=FLOATING_HIGH_PERCENTILE,
    )
    parser.add_argument(
        "--floating-low-percentile",
        type=float,
        default=FLOATING_LOW_PERCENTILE,
    )
    parser.add_argument(
        "--floating-height-margin",
        type=float,
        default=FLOATING_HEIGHT_MARGIN,
    )
    parser.add_argument(
        "--floating-min-points",
        type=int,
        default=FLOATING_MIN_POINTS,
    )
    parser.add_argument(
        "--component-voxel-size",
        type=float,
        default=COMPONENT_VOXEL_SIZE,
    )
    parser.add_argument(
        "--component-min-points",
        type=int,
        default=COMPONENT_MIN_POINTS,
    )

    args = parser.parse_args()

    ensure_dir(args.out_dir)
    dom_out_dir = args.dom_out_dir or args.out_dir
    summary_out = args.summary_out or os.path.join(args.out_dir, "summary.json")
    ensure_dir(dom_out_dir)
    ensure_dir(os.path.dirname(os.path.abspath(summary_out)))


    vertex_data = load_ply_vertices(args.ply)
    xyz_world = extract_xyz_from_vertex(vertex_data)


    if args.orientation_mode == "train_camera":
        if args.cameras_json is None or args.camera_idx is None:
            raise ValueError(
                "train_camera 模式需要同时提供 --cameras-json 和 --camera-idx"
            )
        cam_pos, R_w2c, _, _ = load_training_camera_orientation(
            args.cameras_json,
            args.camera_idx,
            xyz_world,
        )
    else:
        cam_pos = choose_virtual_camera_position(
            world_points=xyz_world,
            view_dir=VIEW_DIR,
            keep_ratio=KEEP_RATIO,
        )
        R_w2c = build_world_z_rotation(yaw_deg=YAW_DEG, view_dir=VIEW_DIR)
    pts_cam_raw = transform_points_to_camera(xyz_world, R_w2c, cam_pos)


    box = compute_robust_camera_box(
        pts_cam_raw,
        margin_ratio=MARGIN_RATIO,
        keep_ratio=KEEP_RATIO,
    )

    x_mid = box["x_mid"]
    y_mid = box["y_mid"]
    z_min = box["z_min"]
    z_max = box["z_max"]

    ortho_w = box["ortho_w"]
    ortho_h = box["ortho_h"]


    mask_img = Image.open(args.mask).convert("L")
    mask = np.array(mask_img)
    height, width = mask.shape[:2]


    instance_map_2d, instance_info_2d = instance_segment_mask(
        mask=mask,
        min_instance_area=MIN_INSTANCE_AREA,
        mask_format=args.mask_format,
    )
    instance_mask_vis = np.zeros((*instance_map_2d.shape, 3), dtype=np.uint8)
    instance_pixels = instance_map_2d > 0
    instance_mask_vis[instance_pixels] = instance_colors(
        instance_map_2d[instance_pixels] - 1
    )
    Image.fromarray(instance_mask_vis, mode="RGB").save(
        os.path.join(dom_out_dir, "instance_mask_vis.png")
    )


    u_float, v_float, _ = project_points_to_ortho_pixels(
        world_points=xyz_world,
        camera_position=cam_pos,
        rotation_world_to_camera=R_w2c,
        width=width,
        height=height,
        ortho_width_world=ortho_w,
        ortho_height_world=ortho_h,
        x_mid=x_mid,
        y_mid=y_mid,
    )

    u = np.round(u_float).astype(np.int32)
    v = np.round(v_float).astype(np.int32)


    sampled_ids_1based, in_image_mask = sample_instance_id_with_radius(
        label_map=instance_map_2d,
        u=u,
        v=v,
        radius=SAMPLE_RADIUS,
    )
    final_building_mask = (sampled_ids_1based > 0) & in_image_mask


    instance_ids_3d = np.full(len(xyz_world), -1, dtype=np.int32)
    instance_ids_3d[final_building_mask] = sampled_ids_1based[final_building_mask] - 1

    removed_floating_mask = filter_floating_points_by_instance_height(
        xyz_world=xyz_world,
        instance_ids_3d=instance_ids_3d,
        high_percentile=args.floating_high_percentile,
        low_percentile=args.floating_low_percentile,
        margin=args.floating_height_margin,
        min_points=args.floating_min_points,
    )
    instance_ids_3d[removed_floating_mask] = -1

    removed_component_mask = filter_small_3d_components_by_instance(
        xyz_world=xyz_world,
        instance_ids_3d=instance_ids_3d,
        voxel_size=args.component_voxel_size,
        min_instance_points=args.component_min_points,
    )
    instance_ids_3d[removed_component_mask] = -1
    final_building_mask = instance_ids_3d >= 0

    labeled_vertices = structured_add_field(
        vertex_data, "instance_id", "i4", instance_ids_3d.astype(np.int32)
    )
    color_all_vertices = colorize_vertices_by_instance(
        labeled_vertices, instance_ids_3d
    )
    save_vertices(
        color_all_vertices, os.path.join(args.out_dir, "labeled_instances_color.ply")
    )
    save_vertices(
        color_all_vertices[final_building_mask],
        os.path.join(args.out_dir, "building_only_instances_color.ply"),
    )

    uniq_inst = sorted([x for x in np.unique(instance_ids_3d) if x >= 0])

    summary = {
        "camera_info": {
            "camera_position": cam_pos.tolist(),
            "R_w2c": R_w2c.tolist(),
        },
        "dom_projection_params": {
            "width": int(width),
            "height": int(height),
            "keep_ratio": KEEP_RATIO,
            "margin_ratio": MARGIN_RATIO,
            "yaw_deg": YAW_DEG,
            "view_dir": VIEW_DIR,
            "ortho_width_world": float(ortho_w),
            "ortho_height_world": float(ortho_h),
            "x_mid": float(x_mid),
            "y_mid": float(y_mid),
            "z_range": [float(z_min), float(z_max)],
            "sample_radius": SAMPLE_RADIUS,
        },
        "mask_instance_params": {
            "mask_format": args.mask_format,
            "mask_thresh": MASK_THRESHOLD,
            "open_radius": OPEN_RADIUS,
            "min_instance_area": MIN_INSTANCE_AREA,
            "floating_high_percentile": args.floating_high_percentile,
            "floating_low_percentile": args.floating_low_percentile,
            "floating_height_margin": args.floating_height_margin,
            "floating_min_points": args.floating_min_points,
            "component_voxel_size": args.component_voxel_size,
            "component_min_points": args.component_min_points,
        },
        "stats": {
            "num_total_points": int(len(xyz_world)),
            "num_in_image": int(in_image_mask.sum()),
            "num_removed_floating_points": int(removed_floating_mask.sum()),
            "num_removed_component_points": int(removed_component_mask.sum()),
            "num_building_points": int(final_building_mask.sum()),
            "num_non_building_points": int((~final_building_mask).sum()),
            "num_instance_points": int((instance_ids_3d >= 0).sum()),
            "num_instances_2d": len(instance_info_2d),
            "num_instances_3d": int(len(uniq_inst)),
        },
        "instances_2d": instance_info_2d,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"instances={len(uniq_inst)} "
        f"building_points={int(final_building_mask.sum())} "
        f"removed_height={int(removed_floating_mask.sum())} "
        f"removed_components={int(removed_component_mask.sum())}"
    )
    print(f"saved: {args.out_dir}")


if __name__ == "__main__":
    main()
