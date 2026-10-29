from __future__ import annotations

import json

import numpy as np


def build_overlapping_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    rows: int,
    cols: int,
    overlap_ratio: float = 0.1,
) -> list[dict[str, float | int]]:


    rows = int(rows)
    cols = int(cols)
    overlap_ratio = float(overlap_ratio)
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be >= 1, got {rows}x{cols}")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError(
            f"overlap_ratio must be in [0, 1), got {overlap_ratio}"
        )
    if not float(x_max) > float(x_min) or not float(y_max) > float(y_min):
        raise ValueError("grid bounds must have positive width and height")

    tile_width = (float(x_max) - float(x_min)) / (
        cols - (cols - 1) * overlap_ratio
    )
    tile_height = (float(y_max) - float(y_min)) / (
        rows - (rows - 1) * overlap_ratio
    )
    x_stride = tile_width * (1.0 - overlap_ratio)
    y_stride = tile_height * (1.0 - overlap_ratio)

    tiles: list[dict[str, float | int]] = []
    for row in range(rows):
        for col in range(cols):
            tile_x_min = float(x_min) + col * x_stride
            tile_y_min = float(y_min) + row * y_stride
            tiles.append(
                {
                    "row": row,
                    "col": col,
                    "x_min": tile_x_min,
                    "x_max": tile_x_min + tile_width,
                    "y_min": tile_y_min,
                    "y_max": tile_y_min + tile_height,
                    "x_mid": tile_x_min + 0.5 * tile_width,
                    "y_mid": tile_y_min + 0.5 * tile_height,
                    "ortho_w": tile_width,
                    "ortho_h": tile_height,
                }
            )
    return tiles


def robust_percentile_range(
    values: np.ndarray, keep_ratio: float
) -> tuple[float, float]:
    keep_ratio = float(keep_ratio)
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if keep_ratio == 1.0:
        return float(values.min()), float(values.max())
    tail = (1.0 - keep_ratio) / 2.0
    return (
        float(np.quantile(values, tail)),
        float(np.quantile(values, 1.0 - tail)),
    )


def compute_robust_camera_box(
    camera_points: np.ndarray,
    margin_ratio: float = 0.05,
    keep_ratio: float = 0.98,
) -> dict[str, float]:
    x_min, x_max = robust_percentile_range(camera_points[:, 0], keep_ratio)
    y_min, y_max = robust_percentile_range(camera_points[:, 1], keep_ratio)
    z_min, z_max = robust_percentile_range(camera_points[:, 2], keep_ratio)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "x_mid": 0.5 * (x_min + x_max),
        "y_mid": 0.5 * (y_min + y_max),
        "ortho_w": (x_max - x_min) * (1.0 + 2.0 * margin_ratio),
        "ortho_h": (y_max - y_min) * (1.0 + 2.0 * margin_ratio),
    }


def choose_virtual_camera_position(
    world_points: np.ndarray,
    view_dir: str,
    keep_ratio: float,
    z_margin: float = 1.0,
) -> np.ndarray:
    x_min, x_max = robust_percentile_range(world_points[:, 0], keep_ratio)
    y_min, y_max = robust_percentile_range(world_points[:, 1], keep_ratio)
    z_min, z_max = robust_percentile_range(world_points[:, 2], keep_ratio)
    z = z_max + z_margin if view_dir == "minus_z" else z_min - z_margin
    return np.asarray(
        [0.5 * (x_min + x_max), 0.5 * (y_min + y_max), z],
        dtype=np.float32,
    )


def load_training_camera_orientation(
    cameras_json_path: str,
    camera_id: int,
    world_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    with open(cameras_json_path, "r", encoding="utf-8") as file:
        cameras = json.load(file)
    if not isinstance(cameras, list):
        raise ValueError("cameras.json 顶层必须是相机列表")

    camera = next(
        (
            item
            for item in cameras
            if "id" in item and int(item["id"]) == int(camera_id)
        ),
        None,
    )
    if camera is None:
        raise ValueError(
            f"cameras.json 中没有 id={camera_id} 的相机，共 {len(cameras)} 个相机"
        )
    for key in ("position", "rotation"):
        if key not in camera:
            raise KeyError(f"相机 id={camera_id} 缺少字段：{key}")

    camera_position = np.asarray(camera["position"], dtype=np.float32).reshape(3)
    raw_rotation = np.asarray(camera["rotation"], dtype=np.float32).reshape(3, 3)

    candidates = (("raw", raw_rotation), ("transpose", raw_rotation.T))
    scored = []
    for name, rotation in candidates:
        camera_points = transform_points_to_camera(
            world_points,
            rotation,
            camera_position,
        )
        positive_ratio = float((camera_points[:, 2] > 0).mean())
        scored.append((positive_ratio, name, rotation))

    positive_ratio, convention, rotation_world_to_camera = max(
        scored,
        key=lambda item: item[0],
    )
    return (
        camera_position,
        rotation_world_to_camera.astype(np.float32),
        convention,
        positive_ratio,
    )


def build_world_z_rotation(
    yaw_deg: float = 0.0, view_dir: str = "minus_z"
) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    z_sign = -1.0 if view_dir == "minus_z" else 1.0
    return np.asarray(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, z_sign],
        ],
        dtype=np.float32,
    )


def transform_points_to_camera(
    world_points: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    camera_position: np.ndarray,
) -> np.ndarray:
    return (rotation_world_to_camera @ (world_points - camera_position[None, :]).T).T


def view_matrix(
    rotation_world_to_camera: np.ndarray,
    camera_position: np.ndarray,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> np.ndarray:
    translation = -rotation_world_to_camera @ camera_position
    translation[0] -= x_offset
    translation[1] -= y_offset
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation_world_to_camera
    matrix[:3, 3] = translation
    return matrix


def camera_position_from_view(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    return (-rotation.T @ translation).astype(np.float32)


def project_points_to_ortho_pixels(
    world_points: np.ndarray,
    camera_position: np.ndarray,
    rotation_world_to_camera: np.ndarray,
    width: int,
    height: int,
    ortho_width_world: float,
    ortho_height_world: float,
    x_mid: float,
    y_mid: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = transform_points_to_camera(
        world_points, rotation_world_to_camera, camera_position
    )
    camera_points[:, 0] -= x_mid
    camera_points[:, 1] -= y_mid
    u = width / max(ortho_width_world, 1e-6) * camera_points[:, 0] + width / 2.0
    v = height / max(ortho_height_world, 1e-6) * camera_points[:, 1] + height / 2.0
    return u, v, camera_points
