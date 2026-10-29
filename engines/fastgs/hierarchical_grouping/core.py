from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GroupingConfig:
    road_close_radius: int = 2
    road_dilate_radius: int = 1
    vegetation_close_radius: int = 2
    vegetation_dilate_radius: int = 1
    min_block_area: int = 32
    min_building_area: int = 16


    adjacency_radius: float = 12.0
    relative_adjacency_threshold: float = 0.75
    barrier_fraction_threshold: float = 0.20
    level1_region_mode: str = "solid_hull"
    reliable_fill_ratio: float = 0.55
    reliable_compactness: float = 0.08
    visibility_similarity: float = 0.5

    def __post_init__(self) -> None:
        integer_nonnegative = {
            "road_close_radius": self.road_close_radius,
            "road_dilate_radius": self.road_dilate_radius,
            "vegetation_close_radius": self.vegetation_close_radius,
            "vegetation_dilate_radius": self.vegetation_dilate_radius,
            "min_block_area": self.min_block_area,
            "min_building_area": self.min_building_area,
        }
        for name, value in integer_nonnegative.items():
            if int(value) != value or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        if self.min_block_area < 1 or self.min_building_area < 1:
            raise ValueError("min_block_area and min_building_area must be at least 1")
        if self.adjacency_radius < 0:
            raise ValueError("adjacency_radius must be non-negative")
        if self.relative_adjacency_threshold < 0:
            raise ValueError("relative_adjacency_threshold must be non-negative")
        if self.level1_region_mode not in {"solid_hull", "building_pixels"}:
            raise ValueError(
                "level1_region_mode must be 'solid_hull' or 'building_pixels'"
            )
        for name, value in {
            "barrier_fraction_threshold": self.barrier_fraction_threshold,
            "reliable_fill_ratio": self.reliable_fill_ratio,
            "reliable_compactness": self.reliable_compactness,
            "visibility_similarity": self.visibility_similarity,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value!r}")


@dataclass(frozen=True)
class HierarchicalMasks:
    level0: np.ndarray
    level1: np.ndarray
    level2: np.ndarray
    groups: dict[int, dict[str, Any]]
    components: dict[int, dict[str, Any]]


def _disk(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return xx * xx + yy * yy <= radius * radius


def _sequential_labels(labels: np.ndarray, foreground: Optional[np.ndarray] = None) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2-D label map, got shape {labels.shape}")
    result = np.zeros(labels.shape, dtype=np.int32)
    allowed = np.ones(labels.shape, dtype=bool) if foreground is None else foreground.astype(bool)
    next_id = 1
    for source_id in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        source = (labels == source_id) & allowed
        split, count = ndimage.label(source, structure=np.ones((3, 3), dtype=bool))
        for local_id in range(1, count + 1):
            result[split == local_id] = next_id
            next_id += 1
    return result


def _filter_and_relabel(binary: np.ndarray, min_area: int) -> np.ndarray:
    labels, count = ndimage.label(binary.astype(bool), structure=np.ones((3, 3), dtype=bool))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    kept = np.flatnonzero(areas >= max(1, int(min_area)))
    kept = kept[kept > 0]
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    return lookup[labels]


def refine_road_mask(road_mask: np.ndarray, config: GroupingConfig) -> np.ndarray:

    road = np.asarray(road_mask, dtype=bool)
    if road.ndim != 2:
        raise ValueError(f"Expected a 2-D road mask, got shape {road.shape}")
    if config.road_close_radius > 0:
        road = ndimage.binary_closing(road, structure=_disk(config.road_close_radius))
    if config.road_dilate_radius > 0:
        road = ndimage.binary_dilation(road, structure=_disk(config.road_dilate_radius))
    return road


def refine_vegetation_mask(
    vegetation_mask: np.ndarray,
    config: GroupingConfig,
) -> np.ndarray:

    vegetation = np.asarray(vegetation_mask, dtype=bool)
    if vegetation.ndim != 2:
        raise ValueError(f"Expected a 2-D vegetation mask, got shape {vegetation.shape}")
    if config.vegetation_close_radius > 0:
        vegetation = ndimage.binary_closing(
            vegetation,
            structure=_disk(config.vegetation_close_radius),
        )
    if config.vegetation_dilate_radius > 0:
        vegetation = ndimage.binary_dilation(
            vegetation,
            structure=_disk(config.vegetation_dilate_radius),
        )
    return vegetation


def build_block_map(road_mask: np.ndarray, config: GroupingConfig) -> np.ndarray:

    return _filter_and_relabel(~refine_road_mask(road_mask, config), config.min_block_area)


def build_component_map(
    building_mask: np.ndarray,
    config: GroupingConfig,
    instance_mask: Optional[np.ndarray] = None,
) -> np.ndarray:

    building = np.asarray(building_mask, dtype=bool)
    if building.ndim != 2:
        raise ValueError(f"Expected a 2-D building mask, got shape {building.shape}")
    if instance_mask is None:
        return _filter_and_relabel(building, config.min_building_area)
    instances = np.asarray(instance_mask)
    if instances.shape != building.shape:
        raise ValueError("instance_mask and semantic masks must have identical dimensions")


    if len([value for value in np.unique(instances) if int(value) > 0]) <= 1:
        return _filter_and_relabel(
            (instances > 0) & building,
            config.min_building_area,
        )
    relabelled = _sequential_labels(instances, foreground=building)
    count = int(relabelled.max())
    areas = np.bincount(relabelled.ravel(), minlength=count + 1)
    kept = np.flatnonzero(areas >= max(1, config.min_building_area))
    kept = kept[kept > 0]
    lookup = np.zeros(count + 1, dtype=np.int32)
    lookup[kept] = np.arange(1, len(kept) + 1, dtype=np.int32)
    return lookup[relabelled]


def _component_metadata(component_map: np.ndarray, block_map: np.ndarray) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    objects = ndimage.find_objects(component_map)
    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        local_labels = component_map[y_slice, x_slice]
        mask = local_labels == component_id
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        block_values = block_map[y_slice, x_slice][mask]
        positive = block_values[block_values > 0]
        block_id = int(np.bincount(positive).argmax()) if positive.size else 0
        y0, y1 = int(y_slice.start), int(y_slice.stop)
        x0, x1 = int(x_slice.start), int(x_slice.stop)
        area = int(mask.sum())
        fill_ratio = area / float((y1 - y0) * (x1 - x0))
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        perimeter = int(np.count_nonzero(mask & ~eroded))
        compactness = 4.0 * np.pi * area / max(float(perimeter * perimeter), 1.0)
        equivalent_diameter = 2.0 * np.sqrt(area / np.pi)
        metadata[component_id] = {
            "component_id": component_id,
            "block_id": block_id,
            "area": area,
            "bbox": [x0, y0, x1, y1],
            "fill_ratio": float(fill_ratio),
            "compactness": float(compactness),
            "equivalent_diameter": float(equivalent_diameter),
        }
    return metadata


def _bbox_gap(a: Sequence[int], b: Sequence[int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(ax0 - bx1, bx0 - ax1, 0)
    dy = max(ay0 - by1, by0 - ay1, 0)
    return float(np.hypot(dx, dy))


def _component_boundary_trees(component_map: np.ndarray) -> dict[int, tuple[cKDTree, np.ndarray]]:

    result: dict[int, tuple[cKDTree, np.ndarray]] = {}
    objects = ndimage.find_objects(component_map)
    for component_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        y_slice, x_slice = slices
        mask = component_map[y_slice, x_slice] == component_id
        if not mask.any():
            continue
        eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        coords = np.column_stack(np.nonzero(mask & ~eroded)).astype(np.float64, copy=False)
        coords[:, 0] += int(y_slice.start)
        coords[:, 1] += int(x_slice.start)
        result[component_id] = (cKDTree(coords), coords)
    return result


def _nearest_gap(
    boundaries: Mapping[int, tuple[cKDTree, np.ndarray]],
    first: int,
    second: int,
    upper_bound: float,
) -> tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    first_tree, first_coords = boundaries[first]
    second_tree, second_coords = boundaries[second]
    if len(first_coords) <= len(second_coords):
        query_coords, target_tree = first_coords, second_tree
    else:
        query_coords, target_tree = second_coords, first_tree
    distances, indices = target_tree.query(
        query_coords,
        k=1,
        distance_upper_bound=float(upper_bound),
        workers=1,
    )
    best = int(np.argmin(distances)) if len(distances) else 0
    distance = float(distances[best]) if len(distances) else float("inf")
    if not np.isfinite(distance):
        return distance, None, None
    query_point = query_coords[best]
    target_point = target_tree.data[int(indices[best])]
    if query_coords is first_coords:
        return distance, query_point, target_point
    return distance, target_point, query_point


def _corridor_barrier_fraction(
    barrier_mask: Optional[np.ndarray],
    first_point: Optional[np.ndarray],
    second_point: Optional[np.ndarray],
) -> float:
    if barrier_mask is None or first_point is None or second_point is None:
        return 0.0
    distance = float(np.linalg.norm(second_point - first_point))
    samples = max(2, int(np.ceil(distance)) + 1)
    coordinates = np.linspace(first_point, second_point, samples)
    yy = np.clip(
        np.rint(coordinates[:, 0]).astype(np.intp),
        0,
        barrier_mask.shape[0] - 1,
    )
    xx = np.clip(
        np.rint(coordinates[:, 1]).astype(np.intp),
        0,
        barrier_mask.shape[1] - 1,
    )
    return float(np.count_nonzero(barrier_mask[yy, xx]) / samples)


def _visibility_similar(
    first: int,
    second: int,
    visibility: Optional[Mapping[int, Sequence[str]]],
    threshold: float,
) -> bool:
    if visibility is None:
        return True
    a, b = set(visibility.get(first, ())), set(visibility.get(second, ()))
    if not a or not b:
        return True
    return len(a & b) / float(len(a | b)) >= threshold


class _DisjointSet:
    def __init__(self, values: Sequence[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def group_components(
    component_map: np.ndarray,
    block_map: np.ndarray,
    config: GroupingConfig,
    visibility: Optional[Mapping[int, Sequence[str]]] = None,
    member_building_ids: Optional[Mapping[int, int]] = None,
    barrier_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:

    metadata = _component_metadata(component_map, block_map)
    component_ids = sorted(metadata)
    graph = _DisjointSet(component_ids)
    boundaries = _component_boundary_trees(component_map)

    for index, first in enumerate(component_ids):
        a = metadata[first]
        unreliable_a = (
            a["fill_ratio"] < config.reliable_fill_ratio
            or a["compactness"] < config.reliable_compactness
        )
        for second in component_ids[index + 1 :]:
            b = metadata[second]
            if a["block_id"] == 0 or a["block_id"] != b["block_id"]:
                continue
            unreliable_b = (
                b["fill_ratio"] < config.reliable_fill_ratio
                or b["compactness"] < config.reliable_compactness
            )
            if not (unreliable_a or unreliable_b):
                continue
            if _bbox_gap(a["bbox"], b["bbox"]) > config.adjacency_radius:
                continue
            gap, first_point, second_point = _nearest_gap(
                boundaries,
                first,
                second,
                config.adjacency_radius,
            )
            mean_diameter = 0.5 * (
                a["equivalent_diameter"] + b["equivalent_diameter"]
            )
            relative_gap = gap / max(mean_diameter, 1.0)
            if gap > config.adjacency_radius:
                continue
            if relative_gap > config.relative_adjacency_threshold:
                continue
            if (
                _corridor_barrier_fraction(
                    barrier_mask,
                    first_point,
                    second_point,
                )
                > config.barrier_fraction_threshold
            ):
                continue
            if _visibility_similar(first, second, visibility, config.visibility_similarity):
                graph.union(first, second)

    roots = sorted({graph.find(component_id) for component_id in component_ids})
    root_to_group = {root: index + 1 for index, root in enumerate(roots)}
    component_to_group = np.zeros(int(component_map.max()) + 1, dtype=np.int32)
    groups: dict[int, dict[str, Any]] = {}
    for component_id in component_ids:
        group_id = root_to_group[graph.find(component_id)]
        component_to_group[component_id] = group_id
        group = groups.setdefault(
            group_id,
            {"group_id": group_id, "component_ids": [], "block_id": metadata[component_id]["block_id"], "member_buildings": []},
        )
        group["component_ids"].append(component_id)
        member_id = component_id if member_building_ids is None else member_building_ids.get(component_id, component_id)
        group["member_buildings"].append(int(member_id))
    group_map = component_to_group[component_map]
    return group_map, groups, metadata


def build_solid_group_map(
    building_group_map: np.ndarray,
    block_map: np.ndarray,
    groups: dict[int, dict[str, Any]],
) -> np.ndarray:

    result = np.zeros(building_group_map.shape, dtype=np.int32)
    candidates = _solid_group_candidates(building_group_map, block_map, groups)
    conflict_pixels: dict[int, int] = {}
    for group_id in sorted(candidates):
        y_slice, x_slice, candidate = candidates[group_id]
        local_result = result[y_slice, x_slice]
        conflicts = candidate & (local_result > 0)
        conflict_pixels[group_id] = int(np.count_nonzero(conflicts))
        local_result[candidate & ~conflicts] = group_id

    total_conflicts = sum(conflict_pixels.values())
    if total_conflicts:
        raise RuntimeError(
            f"Solid Level-1 envelopes still overlap by {total_conflicts} pixels; "
            "call merge_overlapping_group_envelopes first"
        )

    if not np.all(result[building_group_map > 0] == building_group_map[building_group_map > 0]):
        raise RuntimeError("Solid Level-1 ownership changed a source building group ID")

    building_counts = np.bincount(
        building_group_map.ravel(),
        minlength=int(building_group_map.max()) + 1,
    )
    region_counts = np.bincount(result.ravel(), minlength=int(result.max()) + 1)
    for group_id, group in groups.items():
        building_pixels = int(building_counts[group_id])
        region_pixels = int(region_counts[group_id])
        group["building_pixels"] = building_pixels
        group["level1_region_pixels"] = region_pixels
        group["filled_between_pixels"] = region_pixels - building_pixels
        group["raw_hull_overlap_pixels"] = conflict_pixels.get(group_id, 0)
        group["output_overlap_pixels"] = 0
    return result


def _block_owner_map(block_map: np.ndarray) -> np.ndarray:
    if np.any(block_map > 0):
        nearest = ndimage.distance_transform_edt(
            block_map == 0,
            return_distances=False,
            return_indices=True,
        )
        return block_map[nearest[0], nearest[1]]
    return block_map


def _solid_group_candidates(
    building_group_map: np.ndarray,
    block_map: np.ndarray,
    groups: Mapping[int, Mapping[str, Any]],
) -> dict[int, tuple[slice, slice, np.ndarray]]:

    block_owner = _block_owner_map(block_map)
    group_objects = ndimage.find_objects(building_group_map)
    candidates: dict[int, tuple[slice, slice, np.ndarray]] = {}
    for group_id, slices in enumerate(group_objects, start=1):
        if slices is None or group_id not in groups:
            continue
        y_slice, x_slice = slices
        local_group = building_group_map[y_slice, x_slice] == group_id
        yy, xx = np.nonzero(local_group)
        if yy.size < 3:
            candidate = local_group
        else:
            points = np.column_stack((xx, yy)).astype(np.int32).reshape(-1, 1, 2)
            hull = cv2.convexHull(points)
            candidate_u8 = np.zeros(local_group.shape, dtype=np.uint8)
            cv2.fillConvexPoly(candidate_u8, hull, 1)
            candidate = candidate_u8.astype(bool)
        block_id = int(groups[group_id]["block_id"])
        if block_id > 0:
            candidate &= block_owner[y_slice, x_slice] == block_id
        else:
            candidate &= local_group
        candidates[group_id] = (y_slice, x_slice, candidate)
    return candidates


def _candidate_overlap(
    first: tuple[slice, slice, np.ndarray],
    second: tuple[slice, slice, np.ndarray],
) -> bool:
    ay, ax, a = first
    by, bx, b = second
    y0, y1 = max(ay.start, by.start), min(ay.stop, by.stop)
    x0, x1 = max(ax.start, bx.start), min(ax.stop, bx.stop)
    if y0 >= y1 or x0 >= x1:
        return False
    a_view = a[y0 - ay.start : y1 - ay.start, x0 - ax.start : x1 - ax.start]
    b_view = b[y0 - by.start : y1 - by.start, x0 - bx.start : x1 - bx.start]
    return bool(np.any(a_view & b_view))


def merge_overlapping_group_envelopes(
    building_group_map: np.ndarray,
    block_map: np.ndarray,
    groups: Mapping[int, Mapping[str, Any]],
) -> tuple[np.ndarray, dict[int, dict[str, Any]]]:


    current_map = np.asarray(building_group_map, dtype=np.int32).copy()
    current_groups: dict[int, dict[str, Any]] = {}
    for group_id, source in groups.items():
        group = dict(source)
        group["component_ids"] = list(source.get("component_ids", ()))
        group["member_buildings"] = list(source.get("member_buildings", ()))
        group["initial_group_ids"] = list(source.get("initial_group_ids", (group_id,)))
        group["envelope_merge_rounds"] = int(source.get("envelope_merge_rounds", 0))
        current_groups[int(group_id)] = group

    rounds = 0
    while True:
        candidates = _solid_group_candidates(current_map, block_map, current_groups)
        group_ids = sorted(candidates)
        unions = _DisjointSet(group_ids)
        for index, first in enumerate(group_ids):
            first_block = int(current_groups[first].get("block_id", 0))
            for second in group_ids[index + 1 :]:
                if first_block != int(current_groups[second].get("block_id", 0)):
                    continue
                if _candidate_overlap(candidates[first], candidates[second]):
                    unions.union(first, second)

        roots = {group_id: unions.find(group_id) for group_id in group_ids}
        if all(root == group_id for group_id, root in roots.items()):
            break

        rounds += 1
        unique_roots = sorted(set(roots.values()))
        root_to_group = {root: index + 1 for index, root in enumerate(unique_roots)}
        old_to_new = np.zeros(max(group_ids) + 1, dtype=np.int32)
        merged_groups: dict[int, dict[str, Any]] = {}
        for old_id in group_ids:
            new_id = root_to_group[roots[old_id]]
            old_to_new[old_id] = new_id
            source = current_groups[old_id]
            target = merged_groups.setdefault(
                new_id,
                {
                    "group_id": new_id,
                    "component_ids": [],
                    "block_id": int(source.get("block_id", 0)),
                    "member_buildings": [],
                    "initial_group_ids": [],
                    "envelope_merge_rounds": rounds,
                },
            )
            target["component_ids"].extend(source.get("component_ids", ()))
            target["member_buildings"].extend(source.get("member_buildings", ()))
            target["initial_group_ids"].extend(source.get("initial_group_ids", (old_id,)))
            target["envelope_merge_rounds"] = max(
                int(target["envelope_merge_rounds"]),
                int(source.get("envelope_merge_rounds", 0)),
                rounds,
            )
        current_map = old_to_new[current_map]
        for group in merged_groups.values():
            group["component_ids"] = sorted(int(v) for v in group["component_ids"])
            group["member_buildings"] = sorted(int(v) for v in group["member_buildings"])
            group["initial_group_ids"] = sorted(set(int(v) for v in group["initial_group_ids"]))
            group["envelope_merged_group_count"] = len(group["initial_group_ids"]) - 1
        current_groups = merged_groups

    for group in current_groups.values():
        group.setdefault("envelope_merged_group_count", len(group["initial_group_ids"]) - 1)
    return current_map, current_groups


def _validate_uint8(name: str, labels: np.ndarray) -> np.ndarray:
    maximum = int(labels.max()) if labels.size else 0
    if maximum > 255:
        raise ValueError(f"{name} contains {maximum} regions; uint8 PNG supports at most 255")
    return labels.astype(np.uint8, copy=False)


def build_hierarchy(
    building_mask: np.ndarray,
    road_mask: np.ndarray,
    instance_mask: Optional[np.ndarray] = None,
    config: Optional[GroupingConfig] = None,
    visibility: Optional[Mapping[int, Sequence[str]]] = None,
    vegetation_mask: Optional[np.ndarray] = None,
) -> HierarchicalMasks:
    config = config or GroupingConfig()
    if np.asarray(building_mask).shape != np.asarray(road_mask).shape:
        raise ValueError("building_mask and road_mask must have identical dimensions")
    if (
        vegetation_mask is not None
        and np.asarray(vegetation_mask).shape != np.asarray(building_mask).shape
    ):
        raise ValueError("vegetation_mask and building_mask must have identical dimensions")
    refined_road = refine_road_mask(road_mask, config)
    refined_vegetation = (
        refine_vegetation_mask(vegetation_mask, config)
        if vegetation_mask is not None
        else np.zeros(np.asarray(building_mask).shape, dtype=bool)
    )
    building = np.asarray(building_mask, dtype=bool) & ~refined_road
    level0 = _filter_and_relabel(~refined_road, config.min_block_area)
    level2 = build_component_map(building, config, instance_mask)
    member_building_ids: dict[int, int] = {}
    if instance_mask is not None:
        source = np.asarray(instance_mask)
        positive_ids = [int(value) for value in np.unique(source) if int(value) > 0]
        objects = ndimage.find_objects(level2)
        for component_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            if len(positive_ids) <= 1:
                member_building_ids[component_id] = component_id
                continue
            values = source[slices][level2[slices] == component_id]
            values = values[values > 0].astype(np.int64, copy=False)
            member_building_ids[component_id] = (
                int(np.bincount(values).argmax()) if values.size else component_id
            )
    level1_buildings, groups, components = group_components(
        level2,
        level0,
        config,
        visibility,
        member_building_ids or None,
        barrier_mask=(refined_road | refined_vegetation) & (level2 == 0),
    )
    if config.level1_region_mode == "solid_hull":
        level1_buildings, groups = merge_overlapping_group_envelopes(
            level1_buildings,
            level0,
            groups,
        )
        level1 = build_solid_group_map(level1_buildings, level0, groups)
    else:
        level1 = level1_buildings
    if config.level1_region_mode == "building_pixels":
        counts = np.bincount(level1.ravel(), minlength=int(level1.max()) + 1)
        for group_id, group in groups.items():
            pixels = int(counts[group_id])
            group["building_pixels"] = pixels
            group["level1_region_pixels"] = pixels
            group["filled_between_pixels"] = 0
            group["raw_hull_overlap_pixels"] = 0
            group["output_overlap_pixels"] = 0
    for component_id, source_id in member_building_ids.items():
        components[component_id]["member_building_id"] = source_id
    return HierarchicalMasks(
        level0=_validate_uint8("level0", level0),
        level1=_validate_uint8("level1", level1),
        level2=_validate_uint8("level2", level2),
        groups=groups,
        components=components,
    )
