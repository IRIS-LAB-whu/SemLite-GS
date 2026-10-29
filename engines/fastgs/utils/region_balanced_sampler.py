from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw


class RegionBalancedCameraSampler:
    def __init__(
        self,
        cameras,
        region_map_path: str | Path,
        seed: int = 0,
        camera_min_count: int = 0,
        camera_max_count: int = 0,
        planned_iterations: Optional[int] = None,
        camera_priority_path: str | Path | None = None,
        priority_fraction: float = 0.0,
        priority_quantile: float = 0.4,
        priority_start_iteration: int = 1,
    ):
        self.cameras = list(cameras)
        self.camera_min_count = int(camera_min_count)
        self.camera_max_count = int(camera_max_count)
        self.planned_iterations = (
            int(planned_iterations) if planned_iterations is not None else None
        )
        self.priority_fraction = float(priority_fraction)
        self.priority_quantile = float(priority_quantile)
        self.priority_start_iteration = int(priority_start_iteration)
        if not 0.0 <= self.priority_fraction <= 1.0:
            raise ValueError("priority fraction must be in [0, 1]")
        if not 0.0 < self.priority_quantile <= 1.0:
            raise ValueError("priority quantile must be in (0, 1]")
        if self.priority_start_iteration < 1:
            raise ValueError("priority start iteration must be >= 1")
        if self.camera_min_count < 0 or self.camera_max_count < 0:
            raise ValueError("camera min/max counts must be >= 0")
        if (
            self.camera_max_count > 0
            and self.camera_min_count > self.camera_max_count
        ):
            raise ValueError("camera minimum count cannot exceed maximum count")
        camera_by_name = {camera.image_name: camera for camera in self.cameras}
        data = json.loads(Path(region_map_path).read_text(encoding="utf-8"))
        priority_data = {}
        if camera_priority_path:
            raw_priority = json.loads(
                Path(camera_priority_path).read_text(encoding="utf-8")
            )
            priority_data = raw_priority.get("regions", raw_priority)
        if self.priority_fraction > 0 and not priority_data:
            raise ValueError("priority sampling requires a camera priority file")
        self.region_ids = []
        self.region_cameras = {}
        self.region_priority_scores = {}
        self.region_priority_names = {}
        self.region_camera_loss_polygons = {}
        self.region_queues = {}
        self.rngs = {}
        self._loss_roi_cache = {}
        for index, region in enumerate(data.get("regions", [])):
            if region.get("active", True) is False:
                continue
            region_id = str(region["region_id"])
            members = [
                camera_by_name[name]
                for name in region.get("camera_names", [])
                if name in camera_by_name
            ]
            if not members:
                continue
            self.region_ids.append(region_id)
            self.region_cameras[region_id] = members
            raw_polygons = region.get("camera_loss_polygons_normalized", {})
            self.region_camera_loss_polygons[region_id] = {
                name: polygon
                for name, polygon in raw_polygons.items()
                if name in camera_by_name and len(polygon) >= 3
            }
            raw_scores = priority_data.get(region_id, {})
            if isinstance(raw_scores, list):
                raw_scores = {
                    item["img_name"]: item.get("score", 0.0) for item in raw_scores
                }
            scores = {
                camera.image_name: max(1e-6, float(raw_scores.get(camera.image_name, 0.0)))
                for camera in members
            }
            ranked = sorted(
                members, key=lambda camera: scores[camera.image_name], reverse=True
            )
            priority_count = max(1, int(round(len(ranked) * self.priority_quantile)))
            self.region_priority_scores[region_id] = scores
            self.region_priority_names[region_id] = {
                camera.image_name for camera in ranked[:priority_count]
            }
            self.region_queues[region_id] = []
            self.rngs[region_id] = random.Random(int(seed) + 1009 * (index + 1))
        if not self.region_ids:
            raise ValueError(
                f"region map has no active regions with loaded cameras: {region_map_path}"
            )
        self.eligible_camera_names = sorted(
            {
                camera.image_name
                for members in self.region_cameras.values()
                for camera in members
            }
        )
        self.eligible_cameras = [
            camera_by_name[name] for name in self.eligible_camera_names
        ]
        if self.planned_iterations is not None:
            minimum_required = self.camera_min_count * len(self.eligible_camera_names)
            maximum_capacity = (
                self.camera_max_count * len(self.eligible_camera_names)
                if self.camera_max_count > 0
                else None
            )
            if minimum_required > self.planned_iterations:
                raise ValueError(
                    f"camera minimum requires {minimum_required} samples, but only "
                    f"{self.planned_iterations} iterations are planned"
                )
            if maximum_capacity is not None and maximum_capacity < self.planned_iterations:
                raise ValueError(
                    f"camera maximum permits {maximum_capacity} samples, but "
                    f"{self.planned_iterations} iterations are planned"
                )
        self.region_counts = Counter()
        self.camera_counts = Counter()
        self.missing_camera_names = sorted(
            {
                name
                for region in data.get("regions", [])
                for name in region.get("camera_names", [])
                if name not in camera_by_name
            }
        )

    def sample(self, iteration: int):
        if int(iteration) < 1:
            raise ValueError(f"iteration must be >= 1, got {iteration}")
        region_id = self.region_ids[(int(iteration) - 1) % len(self.region_ids)]
        priority_active = (
            self.priority_fraction > 0
            and int(iteration) >= self.priority_start_iteration
        )
        if self.camera_min_count > 0 or self.camera_max_count > 0 or priority_active:
            members = self.region_cameras[region_id]
            eligible = [
                camera
                for camera in members
                if self.camera_max_count <= 0
                or self.camera_counts[camera.image_name] < self.camera_max_count
            ]
            if not eligible:
                raise RuntimeError(
                    f"camera maximum count {self.camera_max_count} is infeasible for "
                    f"region {region_id} at iteration {iteration}"
                )
            below_minimum = [
                camera
                for camera in eligible
                if self.camera_counts[camera.image_name] < self.camera_min_count
            ]
            if below_minimum:
                pool = below_minimum
                camera = self._least_used_choice(region_id, pool)
            elif priority_active and self.rngs[region_id].random() < self.priority_fraction:
                priority_names = self.region_priority_names[region_id]
                pool = [camera for camera in eligible if camera.image_name in priority_names]
                if not pool:
                    pool = eligible
                scores = self.region_priority_scores[region_id]
                camera = self.rngs[region_id].choices(
                    pool,
                    weights=[scores[item.image_name] for item in pool],
                    k=1,
                )[0]
            else:
                camera = self._least_used_choice(region_id, eligible)
        else:
            queue = self.region_queues[region_id]
            if not queue:
                queue.extend(self.region_cameras[region_id])
                self.rngs[region_id].shuffle(queue)
            camera = queue.pop()
        self.region_counts[region_id] += 1
        self.camera_counts[camera.image_name] += 1
        return camera, region_id

    def loss_roi(self, region_id: str, camera_name: str, width: int, height: int):

        key = (str(region_id), str(camera_name), int(width), int(height))
        if key in self._loss_roi_cache:
            return self._loss_roi_cache[key]
        polygon = self.region_camera_loss_polygons.get(str(region_id), {}).get(
            str(camera_name)
        )
        if not polygon:
            self._loss_roi_cache[key] = None
            return None
        points = [
            (
                round(min(max(float(point[0]), 0.0), 1.0) * max(int(width) - 1, 0)),
                round(min(max(float(point[1]), 0.0), 1.0) * max(int(height) - 1, 0)),
            )
            for point in polygon
        ]
        image = Image.new("L", (int(width), int(height)), 0)
        ImageDraw.Draw(image).polygon(points, fill=1)
        mask = np.asarray(image, dtype=np.float32)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            result = None
        else:
            result = (
                mask,
                (
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ),
            )
        self._loss_roi_cache[key] = result
        return result

    def _least_used_choice(self, region_id, pool):
        lowest_count = min(self.camera_counts[camera.image_name] for camera in pool)
        least_used = [
            camera
            for camera in pool
            if self.camera_counts[camera.image_name] == lowest_count
        ]
        return self.rngs[region_id].choice(least_used)

    def summary(self, iterations_requested: int) -> dict:
        counts = {region_id: int(self.region_counts[region_id]) for region_id in self.region_ids}
        values = list(counts.values())
        eligible_counts = {
            name: int(self.camera_counts[name]) for name in self.eligible_camera_names
        }
        camera_values = list(eligible_counts.values())
        minimum_satisfied = (
            min(camera_values) >= self.camera_min_count if camera_values else False
        )
        maximum_satisfied = (
            self.camera_max_count <= 0
            or (max(camera_values) <= self.camera_max_count if camera_values else False)
        )
        return {
            "sampling_mode": "region_balanced",
            "iterations_requested": int(iterations_requested),
            "region_count": len(self.region_ids),
            "region_counts": counts,
            "region_count_min": min(values) if values else 0,
            "region_count_max": max(values) if values else 0,
            "region_count_difference": max(values) - min(values) if values else 0,
            "camera_counts": eligible_counts,
            "eligible_camera_count": len(self.eligible_camera_names),
            "configured_camera_min_count": self.camera_min_count,
            "configured_camera_max_count": self.camera_max_count,
            "priority_fraction": self.priority_fraction,
            "priority_quantile": self.priority_quantile,
            "priority_start_iteration": self.priority_start_iteration,
            "camera_count_min": min(camera_values) if camera_values else 0,
            "camera_count_max": max(camera_values) if camera_values else 0,
            "camera_minimum_satisfied": minimum_satisfied,
            "camera_maximum_satisfied": maximum_satisfied,
            "missing_camera_names": self.missing_camera_names,
            "region_guarantee_satisfied": (max(values) - min(values) <= 1) if values else False,
            "guarantee_satisfied": (
                (max(values) - min(values) <= 1)
                and minimum_satisfied
                and maximum_satisfied
            ) if values else False,
        }
