#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from plyfile import PlyData
from utils.instance_encoding import instance_colors
from hierarchical_grouping.partition import PARTITION_MODES, resolve_mask_source
from utils.instance_pipeline_config import (
    ALPHA_THRESHOLD,
    BUILDING_ABS_THRESHOLD,
    BUILDING_RATIO_THRESHOLD,
    COMPONENT_MIN_POINTS,
    COMPONENT_VOXEL_SIZE,
    FLOATING_HEIGHT_MARGIN,
    FLOATING_HIGH_PERCENTILE,
    FLOATING_LOW_PERCENTILE,
    FLOATING_MIN_POINTS,
    KEEP_RATIO,
    MIN_INSTANCE_AREA,
    RENDER_RESOLUTION,
    RENDER_SPLIT,
    SEMANTIC_ALPHA_MIN,
    WINNER_RATIO_THRESHOLD,
)

PROJECT_DIR = Path(__file__).resolve().parent
SEGMENT_SCRIPT = PROJECT_DIR / "segment_ply_by_dom_mask_instances.py"
RENDER_SCRIPT = PROJECT_DIR / "render_images_instance_masks.py"


def log(message: str) -> None:
    print(f"[render_from_DOM_mask] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_command(
    cmd: Sequence[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    pretty = " ".join(str(x) for x in cmd)
    if cwd is not None:
        log(f"cwd: {cwd}")
    log(f"cmd: {pretty}")
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, env=env, check=True)


def infer_source_dir(model_dir: Path) -> Path:
    cfg_path = model_dir / "cfg_args"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Model config not found: {cfg_path}. Pass --source-dir explicitly."
        )
    expression = ast.parse(
        cfg_path.read_text(encoding="utf-8"),
        mode="eval",
    ).body
    if not (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "Namespace"
    ):
        raise ValueError(f"Unsupported model config format: {cfg_path}")
    values = {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in expression.keywords
        if keyword.arg is not None
    }
    source_path = values.get("source_path")
    if not source_path:
        raise ValueError(
            f"source_path is missing from {cfg_path}. Pass --source-dir explicitly."
        )
    return Path(source_path).expanduser().resolve()


def read_instance_ids_from_ply(ply_path: Path) -> list[int]:
    ply = PlyData.read(str(ply_path))
    names = ply["vertex"].data.dtype.names
    if names is None or "instance_id" not in names:
        raise RuntimeError(f"PLY has no instance_id field: {ply_path}")
    ids = np.asarray(ply["vertex"].data["instance_id"], dtype=np.int64)
    valid = sorted(int(x) for x in np.unique(ids) if int(x) >= 0)
    return valid


def write_instance_metadata(out_dir: Path, instance_ids: list[int]) -> None:
    colors = instance_colors(np.asarray(instance_ids, dtype=np.int64))
    id_map = {str(instance_id): instance_id for instance_id in instance_ids}
    color_map = {"-1": [0, 0, 0]}
    color_map.update(
        {
            str(instance_id): color.tolist()
            for instance_id, color in zip(instance_ids, colors)
        }
    )
    (out_dir / "instance_mask.json").write_text(
        json.dumps(id_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "instance_mask_colors.json").write_text(
        json.dumps(color_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "training_label_ids.json").write_text(
        json.dumps(
            {
                "background_id": 0,
                "label_ids": [int(instance_id) + 1 for instance_id in instance_ids],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_segment_command(
    args: argparse.Namespace,
    point_cloud_dir: Path,
    dom_instances_dir: Path,
    summary_path: Path,
) -> list[str]:

    cmd = [
        sys.executable,
        str(SEGMENT_SCRIPT),
        "--ply",
        str(args.ply),
        "--mask",
        str(args.dom_mask),
        "--mask-format",
        args.dom_mask_format,
        "--out-dir",
        str(point_cloud_dir),
        "--dom-out-dir",
        str(dom_instances_dir),
        "--summary-out",
        str(summary_path),
        "--orientation-mode",
        args.orientation_mode,
        "--floating-high-percentile",
        str(args.floating_high_percentile),
        "--floating-low-percentile",
        str(args.floating_low_percentile),
        "--floating-height-margin",
        str(args.floating_height_margin),
        "--floating-min-points",
        str(args.floating_min_points),
        "--component-voxel-size",
        str(args.component_voxel_size),
        "--component-min-points",
        str(args.component_min_points),
    ]
    if args.orientation_mode == "train_camera":
        cmd.extend(
            [
                "--cameras-json",
                str(args.cameras_json),
                "--camera-idx",
                str(args.camera_idx),
            ]
        )
    return cmd


def build_render_command(
    args: argparse.Namespace,
    render_output_dir: Path,
) -> list[str]:

    cmd = [
        sys.executable,
        str(RENDER_SCRIPT),
        "-s",
        str(args.source_dir),
        "-m",
        str(args.model_dir),
        "--iteration",
        str(args.iteration),
        "-r",
        str(RENDER_RESOLUTION),
        "--data_device",
        "cpu",
        "--output-dir",
        str(render_output_dir),
    ]
    if args.save_render_raw:
        cmd.append("--save-raw")
    if args.save_render_depth:
        cmd.append("--save-depth")
    if args.max_render_views > 0:
        cmd.extend(["--max-views", str(args.max_render_views)])
    return cmd


def write_pipeline_summary(
    args: argparse.Namespace,
    out_dir: Path,
    instance_ids: list[int],
    paths: Mapping[str, Path],
) -> None:

    summary = {
        "inputs": {
            "model_dir": str(args.model_dir),
            "ply": str(args.ply),
            "dom_mask": str(args.dom_mask),
            "partition_mode": args.partition_mode,
            "iteration": int(args.iteration),
        },
        "dom_segmentation_params": {
            "keep_ratio": KEEP_RATIO,
            "min_instance_area": MIN_INSTANCE_AREA,
            "floating_high_percentile": args.floating_high_percentile,
            "floating_low_percentile": args.floating_low_percentile,
            "floating_height_margin": args.floating_height_margin,
            "floating_min_points": args.floating_min_points,
            "component_voxel_size": args.component_voxel_size,
            "component_min_points": args.component_min_points,
        },
        "render_params": {
            "resolution": RENDER_RESOLUTION,
            "alpha_thresh": ALPHA_THRESHOLD,
            "building_abs_thresh": BUILDING_ABS_THRESHOLD,
            "building_ratio_thresh": BUILDING_RATIO_THRESHOLD,
            "winner_ratio_thresh": WINNER_RATIO_THRESHOLD,
            "semantic_alpha_min": SEMANTIC_ALPHA_MIN,
            "max_render_views": int(args.max_render_views),
            "cuda_visible_devices": args.cuda_visible_devices,
            "render_split": RENDER_SPLIT,
            "save_render_raw": bool(args.save_render_raw),
            "save_render_depth": bool(args.save_render_depth),
        },
        "instance_ids": instance_ids,
        "num_instances": len(instance_ids),
        "outputs": {k: str(v) for k, v in paths.items()},
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DOM building mask -> point-cloud instance_id -> training-view instance masks"
    )

    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="COLMAP dataset directory. Default: source_path from model-dir/cfg_args",
    )
    parser.add_argument("--dom-mask", type=Path, required=True)
    parser.add_argument(
        "--partition_mode",
        choices=PARTITION_MODES,
        default="none",
        help="none 使用 --dom-mask；level0/1/2 从 --hierarchical-masks-dir 选择标签图。",
    )
    parser.add_argument(
        "--hierarchical-masks-dir",
        type=Path,
        default=None,
        help="包含 level0/level1/level2 的层级 mask 根目录。",
    )
    parser.add_argument("--dom-image", type=Path, default=None)
    parser.add_argument("--dom-overlay", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--render-output-dir", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--cuda-visible-devices", type=str, default="")
    parser.add_argument("--max-render-views", type=int, default=0)
    parser.add_argument(
        "--save-render-raw",
        action="store_true",
        help="保存 instance_masks/raw/*.npy，供后处理读取实例 ID。",
    )
    parser.add_argument(
        "--save-render-depth",
        action="store_true",
        help="保存 depth/raw/*.npy 和 depth/vis/*.png。",
    )
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="只重建 DOM 实例、点云和 JSON，复用已有多视角渲染结果",
    )
    parser.add_argument(
        "--orientation-mode",
        choices=["world_z", "train_camera"],
        default="world_z",
    )
    parser.add_argument("--cameras-json", type=Path, default=None)
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

    args.model_dir = args.model_dir.resolve()
    if args.source_dir is None:
        args.source_dir = infer_source_dir(args.model_dir)
    else:
        args.source_dir = args.source_dir.resolve()
    original_dom_mask = args.dom_mask.resolve()
    if args.hierarchical_masks_dir is not None:
        args.hierarchical_masks_dir = args.hierarchical_masks_dir.resolve()
    if args.partition_mode == "none":
        args.dom_mask = original_dom_mask
        args.dom_mask_format = "binary"
    else:
        if args.hierarchical_masks_dir is None:
            parser.error("--partition_mode 非 none 时必须提供 --hierarchical-masks-dir")
        args.dom_mask = resolve_mask_source(
            args.partition_mode,
            original_dom_mask,
            args.hierarchical_masks_dir,
        ).resolve()
        args.dom_mask_format = "labels"
    if not args.dom_mask.is_file():
        parser.error(f"找不到所选 DOM mask: {args.dom_mask}")
    if args.dom_image is not None:
        args.dom_image = args.dom_image.resolve()
    if args.dom_overlay is not None:
        args.dom_overlay = args.dom_overlay.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.cameras_json is None:
        args.cameras_json = args.model_dir / "cameras.json"
    else:
        args.cameras_json = args.cameras_json.resolve()
    if args.orientation_mode == "train_camera" and args.camera_idx is None:
        parser.error("--orientation-mode train_camera 需要提供 --camera-idx")
    if args.summary_dir is None:
        args.summary_dir = args.out_dir
    else:
        args.summary_dir = args.summary_dir.resolve()

    args.ply = (
        args.model_dir
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )

    if args.render_output_dir is None:
        args.render_output_dir = (
            args.out_dir / "04_render_mask" / f"ours_{args.iteration}"
        )
    else:
        args.render_output_dir = args.render_output_dir.resolve()

    return args


def main() -> None:
    args = parse_args()

    stage_dom_dir = args.out_dir / "01_dom_instances"
    stage_pc_dir = args.out_dir / "02_point_cloud"
    stage_json_dir = args.out_dir / "03_json"
    render_output_dir = args.render_output_dir

    for d in [
        args.out_dir,
        args.summary_dir,
        stage_dom_dir,
        stage_pc_dir,
        stage_json_dir,
        render_output_dir,
    ]:
        ensure_dir(d)

    for stale_name in (
        "building_only_instances_color.ply",
        "labeled_instances_color.ply",
        "summary.json",
    ):
        stale_path = stage_dom_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    stale_point_cloud_summary = stage_pc_dir / "summary.json"
    if stale_point_cloud_summary.exists():
        stale_point_cloud_summary.unlink()

    dom_sources = [args.dom_image, args.dom_mask, args.dom_overlay]
    if args.dom_image is None:
        dom_sources[0] = next(
            (
                path
                for path in sorted(args.dom_mask.parent.glob("DOM*.png"))
                if path != args.dom_mask
            ),
            None,
        )
    if args.dom_overlay is None:
        inferred_overlay = args.dom_mask.parent / "building_overlay.png"
        dom_sources[2] = inferred_overlay if inferred_overlay.exists() else None
    for source in dom_sources:
        if source is not None and source.exists():
            shutil.copy2(source, stage_dom_dir / source.name)
    raw_dom_mask = args.dom_mask.parent / "TOM_mask_sam3_raw.png"
    if raw_dom_mask.exists():
        shutil.copy2(raw_dom_mask, stage_dom_dir / raw_dom_mask.name)

    log("DOM mask -> Gaussian instance IDs")
    segment_cmd = build_segment_command(
        args,
        stage_pc_dir,
        stage_dom_dir,
        stage_json_dir / "dom_instance_summary.json",
    )
    run_command(segment_cmd, cwd=PROJECT_DIR)

    render_ply = stage_pc_dir / "labeled_instances_color.ply"
    building_only_ply = stage_pc_dir / "building_only_instances_color.ply"
    if not render_ply.exists():
        raise FileNotFoundError(f"Segmented PLY not found: {render_ply}")
    if not building_only_ply.exists():
        raise FileNotFoundError(f"Building-only PLY not found: {building_only_ply}")
    instance_ids = read_instance_ids_from_ply(render_ply)
    write_instance_metadata(stage_json_dir, instance_ids)

    model_ply = (
        args.model_dir
        / "point_cloud"
        / f"iteration_{args.iteration}"
        / "point_cloud.ply"
    )

    if args.skip_render:
        log("Reuse existing multi-view masks; skip rendering")
    else:
        log("Render training-view instance masks with the labeled PLY")
        render_model_dir = args.out_dir / "render_model"
        render_cloud_dir = render_model_dir / "point_cloud" / f"iteration_{args.iteration}"
        ensure_dir(render_cloud_dir)
        shutil.copy2(render_ply, render_cloud_dir / "point_cloud.ply")
        shutil.copy2(args.model_dir / "cfg_args", render_model_dir / "cfg_args")
        args.model_dir = render_model_dir

        env = os.environ.copy()
        env["GS_RENDER_NO_GT"] = "1"
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

        render_cmd = build_render_command(args, render_output_dir)
        run_command(render_cmd, cwd=PROJECT_DIR, env=env)

    log("Write pipeline_summary.json")
    paths = {
        "dom_instances_dir": stage_dom_dir,
        "point_cloud_dir": stage_pc_dir,
        "json_dir": stage_json_dir,
        "render_instance_masks_dir": render_output_dir / "instance_masks",
        "render_training_labels_dir": render_output_dir / "instance_masks" / "labels",
        "render_ply": render_ply,
        "building_only_ply": building_only_ply,
        "model_ply": model_ply,
    }
    write_pipeline_summary(args, args.summary_dir, instance_ids, paths)

    log("Pipeline file organization is ready.")
    log(f"DOM instance outputs: {stage_dom_dir}")
    log(f"Point-cloud outputs: {stage_pc_dir}")
    log(f"Instance JSON outputs: {stage_json_dir}")
    log(f"2DGS uint8 labels: {render_output_dir / 'instance_masks' / 'labels'}")
    log(f"Render mask outputs: {render_output_dir / 'instance_masks'}")


if __name__ == "__main__":
    main()
