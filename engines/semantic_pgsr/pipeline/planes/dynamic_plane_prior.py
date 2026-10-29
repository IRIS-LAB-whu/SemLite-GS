import json
import math
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, cKDTree

def _arg(args, name, default):
    return getattr(args, name, default)

def _env_flag(name):
    return os.environ.get(name, '').strip().lower() in {'1', 'true', 'yes', 'on'}

def _bulk_component_neighbors_enabled(args):
    return bool(_arg(args, 'plane_bulk_component_neighbors', False)) or _env_flag('GS_PLANE_BULK_COMPONENT_NEIGHBORS')

def _parse_up(value):
    values = np.asarray([float(item) for item in str(value).split(',')], np.float64)
    if values.shape != (3,) or np.linalg.norm(values) < 1e-12:
        raise ValueError(f'--plane_up_vector must contain three non-zero values, got {value}')
    return values / np.linalg.norm(values)

def _deterministic_stratified_sample(weights, limit, seed):

    weights = np.asarray(weights, np.float64).reshape(-1)
    count = len(weights)
    if limit <= 0 or count <= int(limit):
        return np.arange(count, dtype=np.int64)
    finite = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(finite, kind='mergesort')
    strata = [part for part in np.array_split(order, min(8, count)) if len(part)]
    base, remainder = divmod(int(limit), len(strata))
    rng = np.random.default_rng(int(seed))
    selected = []
    for stratum_index, indices in enumerate(strata):
        quota = min(base + (1 if stratum_index < remainder else 0), len(indices))
        if quota <= 0:
            continue
        local_weights = np.maximum(finite[indices], 0.0)
        if float(local_weights.sum()) <= 1e-12:
            chosen = rng.choice(indices, size=quota, replace=False)
        else:
            chosen = rng.choice(indices, size=quota, replace=False, p=local_weights / local_weights.sum())
        selected.extend((int(value) for value in chosen))
    if len(selected) < int(limit):
        used = np.zeros(count, dtype=bool)
        used[np.asarray(selected, np.int64)] = True
        remaining = np.flatnonzero(~used)
        ranked = remaining[np.argsort(-finite[remaining], kind='mergesort')]
        selected.extend(ranked[:int(limit) - len(selected)].tolist())
    return np.asarray(sorted(selected[:int(limit)]), dtype=np.int64)
_VIEW_SPACING_CACHE_KEY = '_dynamic_plane_view_spacing'

def estimate_spacing(points):
    if len(points) < 2:
        return 0.001
    sample = points
    if len(sample) > 100000:
        sample = sample[np.linspace(0, len(sample) - 1, 100000, dtype=np.int64)]
    distances, _ = cKDTree(sample).query(sample, k=2)
    positive = distances[:, 1][np.isfinite(distances[:, 1]) & (distances[:, 1] > 0)]
    return float(np.median(positive)) if len(positive) else 0.001

def _visibility_rays_from_records(records, args):
    fields = ('points', 'normals', 'camera_origins', 'ray_directions', 'weights', 'camera_ids', 'alpha', 'normal_consistency')
    output = {field: [] for field in fields}
    instance_chunks = []
    for instance_id in sorted((int(value) for value in records)):
        for chunk in records[instance_id]:
            if not all((field in chunk for field in fields)):
                continue
            length = len(chunk['points'])
            for field in fields:
                output[field].append(np.asarray(chunk[field]))
            instance_chunks.append(np.full(length, instance_id, np.int32))
    if not instance_chunks:
        return None
    concatenate = {field: np.concatenate(chunks) for field, chunks in output.items()}
    concatenate['instance_ids'] = np.concatenate(instance_chunks)
    limit = int(_arg(args, 'plane_visibility_rays_max_points', 200000))
    seed = int(_arg(args, 'plane_visibility_rays_seed', 0))
    chosen_parts = []
    unique_instances = np.unique(concatenate['instance_ids'])
    if limit > 0:
        base, remainder = divmod(limit, max(1, len(unique_instances)))
    for order, instance_id in enumerate(unique_instances):
        candidates = np.flatnonzero(concatenate['instance_ids'] == instance_id)
        quota = 0 if limit <= 0 else base + (1 if order < remainder else 0)
        if limit > 0 and quota <= 0:
            local = np.empty(0, np.int64)
        else:
            local = _deterministic_stratified_sample(concatenate['weights'][candidates], quota, seed + int(instance_id) * 1009)
        chosen_parts.append(candidates[local])
    chosen = np.sort(np.concatenate(chosen_parts)) if chosen_parts else np.empty(0, np.int64)
    return {'surface_points': concatenate['points'][chosen].astype(np.float32, copy=False), 'surface_normals': concatenate['normals'][chosen].astype(np.float32, copy=False), 'camera_origins': concatenate['camera_origins'][chosen].astype(np.float32, copy=False), 'ray_directions': concatenate['ray_directions'][chosen].astype(np.float32, copy=False), 'weights': concatenate['weights'][chosen].astype(np.float32, copy=False), 'instance_ids': concatenate['instance_ids'][chosen].astype(np.int32, copy=False), 'camera_ids': concatenate['camera_ids'][chosen].astype(np.int32, copy=False), 'alpha': concatenate['alpha'][chosen].astype(np.float32, copy=False), 'normal_consistency': concatenate['normal_consistency'][chosen].astype(np.float32, copy=False)}

def _empty_fused_voxels():
    return (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32), np.empty((0,), np.float32), np.empty((0,), np.int32))

def _fuse_instance_voxels_legacy(points, normals, weights, camera_ids, voxel_size, min_views, min_concentration, prefilter_min_views=False):
    if len(points) == 0:
        return _empty_fused_voxels()
    keys = np.floor(points / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)), len(keys)]
    group_count = len(boundaries) - 1
    camera_counts_by_group = None
    if bool(prefilter_min_views) and int(min_views) > 1 and (group_count > 0):
        group_ids = np.repeat(np.arange(group_count, dtype=np.int64), np.diff(boundaries))
        sorted_camera_ids = camera_ids[order].astype(np.int64, copy=False)
        camera_min = int(sorted_camera_ids.min())
        camera_span = int(sorted_camera_ids.max()) - camera_min + 1
        pair_keys = group_ids * camera_span + (sorted_camera_ids - camera_min)
        unique_pair_groups = np.unique(pair_keys) // camera_span
        camera_counts_by_group = np.bincount(unique_pair_groups, minlength=group_count).astype(np.int32, copy=False)
        selected_groups = np.flatnonzero(camera_counts_by_group >= int(min_views))
        del group_ids, pair_keys, unique_pair_groups
    else:
        selected_groups = range(group_count)
    output = []
    for group_index in selected_groups:
        start = int(boundaries[group_index])
        end = int(boundaries[group_index + 1])
        indices = order[start:end]
        voxel_weights = weights[indices]
        reference = normals[indices[np.argmax(voxel_weights)]]
        aligned = normals[indices].copy()
        aligned[aligned @ reference < 0] *= -1
        normal_sum = (aligned * voxel_weights[:, None]).sum(0)
        total_weight = float(voxel_weights.sum())
        concentration = float(np.linalg.norm(normal_sum) / max(total_weight, 1e-12))
        camera_count = int(camera_counts_by_group[group_index]) if camera_counts_by_group is not None else int(len(np.unique(camera_ids[indices])))
        if camera_count < min_views or concentration < min_concentration:
            continue
        output.append(((points[indices] * voxel_weights[:, None]).sum(0) / total_weight, normal_sum / max(np.linalg.norm(normal_sum), 1e-12), total_weight, camera_count))
    if not output:
        return _empty_fused_voxels()
    fused_points, fused_normals, fused_weights, camera_counts = zip(*output)
    return (np.asarray(fused_points, np.float32), np.asarray(fused_normals, np.float32), np.asarray(fused_weights, np.float32), np.asarray(camera_counts, np.int32))

def _fuse_instance_voxels_vectorized(points, normals, weights, camera_ids, voxel_size, min_views, min_concentration):


    if len(points) == 0:
        return _empty_fused_voxels()
    keys = np.floor(points / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sorted_keys = keys[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)), len(sorted_keys)]
    starts = boundaries[:-1]
    sizes = np.diff(boundaries)
    group_count = len(starts)
    group_ids = np.repeat(np.arange(group_count, dtype=np.int64), sizes)
    sorted_camera_ids = camera_ids[order].astype(np.int64, copy=False)
    camera_min = int(sorted_camera_ids.min())
    camera_span = int(sorted_camera_ids.max()) - camera_min + 1
    pair_keys = group_ids * camera_span + (sorted_camera_ids - camera_min)
    unique_pair_groups = np.unique(pair_keys) // camera_span
    camera_counts = np.bincount(unique_pair_groups, minlength=group_count).astype(np.int32, copy=False)
    sorted_points = points[order].astype(np.float32, copy=False)
    sorted_normals = normals[order].astype(np.float32, copy=False)
    sorted_weights = weights[order].astype(np.float32, copy=False)
    max_weights = np.maximum.reduceat(sorted_weights, starts)
    positions = np.arange(len(order), dtype=np.int64)
    reference_positions = np.minimum.reduceat(np.where(sorted_weights == max_weights[group_ids], positions, len(order)), starts)
    references = sorted_normals[reference_positions]
    aligned_normals = sorted_normals.copy()
    flip = np.einsum('ij,ij->i', aligned_normals, references[group_ids], optimize=False) < 0
    aligned_normals[flip] *= -1
    total_weights = np.add.reduceat(sorted_weights, starts)
    normal_sums = np.add.reduceat(aligned_normals * sorted_weights[:, None], starts, axis=0)
    point_sums = np.add.reduceat(sorted_points * sorted_weights[:, None], starts, axis=0)
    normal_lengths = np.linalg.norm(normal_sums, axis=1)
    concentrations = normal_lengths / np.maximum(total_weights, 1e-12)
    selected = (camera_counts >= int(min_views)) & (concentrations >= float(min_concentration))
    if not np.any(selected):
        return _empty_fused_voxels()
    fused_points = point_sums[selected] / total_weights[selected, None]
    fused_normals = normal_sums[selected] / np.maximum(normal_lengths[selected, None], 1e-12)
    return (fused_points.astype(np.float32, copy=False), fused_normals.astype(np.float32, copy=False), total_weights[selected].astype(np.float32, copy=False), camera_counts[selected].astype(np.int32, copy=False))

def fuse_instance_voxels(points, normals, weights, camera_ids, voxel_size, min_views, min_concentration, prefilter_min_views=False, backend='vectorized'):
    backend = str(backend).lower()
    if backend == 'legacy':
        return _fuse_instance_voxels_legacy(points, normals, weights, camera_ids, voxel_size, min_views, min_concentration, prefilter_min_views=prefilter_min_views)
    if backend != 'vectorized':
        raise ValueError(f"Unknown plane voxel fusion backend: {backend!r}; expected 'legacy' or 'vectorized'")
    return _fuse_instance_voxels_vectorized(points, normals, weights, camera_ids, voxel_size, min_views, min_concentration)

def _normal_clusters(normals, weights, angle_deg):
    cosine = math.cos(math.radians(angle_deg))
    remaining = np.ones(len(normals), dtype=bool)
    clusters = []
    for seed_index in np.argsort(-weights):
        if not remaining[seed_index]:
            continue
        direction = normals[seed_index].copy()
        members = np.empty(0, np.int64)
        for _ in range(3):
            candidates = np.flatnonzero(remaining & (np.abs(normals @ direction) >= cosine))
            if not len(candidates):
                break
            aligned = normals[candidates].copy()
            aligned[aligned @ direction < 0] *= -1
            direction = (aligned * weights[candidates, None]).sum(0)
            direction /= max(np.linalg.norm(direction), 1e-12)
            members = candidates
        if len(members):
            remaining[members] = False
            clusters.append((members, direction))
    return clusters

def _extract_plane_hypotheses(points, normals, weights, indices, normal_angle_deg, distance_epsilon, min_seed_points):

    remaining = np.asarray(indices, dtype=np.int64)
    hypotheses = []
    cosine = math.cos(math.radians(normal_angle_deg))
    while len(remaining) >= min_seed_points:
        seed_local = int(np.argmax(weights[remaining]))
        seed_index = int(remaining[seed_local])
        normal = normals[seed_index].astype(np.float64).copy()
        normal /= max(np.linalg.norm(normal), 1e-12)
        offset = -float(normal @ points[seed_index])
        members = np.empty(0, dtype=np.int64)
        for _ in range(4):
            normal_similarity = np.abs(normals[remaining] @ normal)
            plane_distance = np.abs(points[remaining] @ normal + offset)
            selected = (normal_similarity >= cosine) & (plane_distance <= distance_epsilon)
            candidate = remaining[selected]
            if len(candidate) < min_seed_points:
                break
            selected_points = points[candidate]
            selected_weights = weights[candidate]
            center = (selected_points * selected_weights[:, None]).sum(0) / selected_weights.sum()
            centered = selected_points - center
            covariance = (centered * selected_weights[:, None]).T @ centered
            _, vectors = np.linalg.eigh(covariance)
            fitted_normal = vectors[:, 0]
            if fitted_normal @ normal < 0:
                fitted_normal *= -1
            normal = fitted_normal
            offset = -float(normal @ center)
            members = candidate
        if len(members) >= min_seed_points:
            hypotheses.append((members, normal))
            member_mask = np.isin(remaining, members, assume_unique=False)
            remaining = remaining[~member_mask]
        else:
            remaining = np.delete(remaining, seed_local)
    return hypotheses

def _plane_basis(normal):
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u /= max(np.linalg.norm(axis_u), 1e-12)
    return (axis_u, np.cross(normal, axis_u))

def _components(points, indices, normal, radius, *, bulk_neighbors=False):
    axis_u, axis_v = _plane_basis(normal)
    uv = np.column_stack((points[indices] @ axis_u, points[indices] @ axis_v))
    tree = cKDTree(uv)
    neighborhoods = None
    if bulk_neighbors:
        neighborhoods = tree.query_ball_point(uv, radius, return_sorted=False)
    visited = np.zeros(len(indices), dtype=bool)
    components = []
    for seed in range(len(indices)):
        if visited[seed]:
            continue
        queue = deque([seed])
        visited[seed] = True
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            neighbors = neighborhoods[current] if neighborhoods is not None else tree.query_ball_point(uv[current], radius)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(indices[np.asarray(component, np.int64)])
    return components

def _fit_plane(points, normals, weights, indices, reference_normal):
    selected_points = points[indices]
    selected_weights = weights[indices]
    center = (selected_points * selected_weights[:, None]).sum(0) / selected_weights.sum()
    centered = selected_points - center
    covariance = (centered * selected_weights[:, None]).T @ centered
    _, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, 0]
    if normal @ reference_normal < 0:
        normal *= -1
    offset = -float(normal @ center)
    errors = np.abs(selected_points @ normal + offset)
    axis_u, axis_v = _plane_basis(normal)
    uv = np.column_stack((selected_points @ axis_u, selected_points @ axis_v))
    area = float(ConvexHull(uv).volume) if len(uv) >= 3 and np.linalg.matrix_rank(uv - uv.mean(0)) == 2 else 0.0
    aligned_normals = normals[indices].copy()
    aligned_normals[aligned_normals @ normal < 0] *= -1
    concentration = np.linalg.norm((aligned_normals * selected_weights[:, None]).sum(0)) / selected_weights.sum()
    return (normal, offset, center, errors, area, float(concentration))

def _classify_plane(normal, centroid, instance_points, up, wall_deg, roof_deg):
    heights = instance_points @ up
    low, high = np.quantile(heights, [0.05, 0.95])
    ratio = float((centroid @ up - low) / max(high - low, 1e-08))
    up_cosine = abs(float(normal @ up))
    if up_cosine <= math.sin(math.radians(wall_deg)):
        return 'wall'
    if up_cosine >= math.cos(math.radians(roof_deg)) and ratio >= 0.5:
        return 'horizontal_roof'
    if up_cosine < math.cos(math.radians(roof_deg)) and ratio >= 0.5:
        return 'sloped_roof'
    if up_cosine >= math.cos(math.radians(roof_deg)):
        return 'horizontal_low'
    return 'other_planar'

def detect_instance_planes(points, normals, weights, camera_counts, instance_id, voxel_size, args, diagnostics=None):
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.update(normal_clusters=0, offset_groups=0, components=0, rejected_too_few=0, rejected_mean_error=0, rejected_p95_error=0, rejected_area=0, rejected_view_support=0, accepted_planes=0)
    min_points = int(_arg(args, 'plane_min_points', 150))
    offset_eps = float(_arg(args, 'plane_offset_cluster_eps', -1))
    offset_eps = 2.5 * voxel_size if offset_eps < 0 else offset_eps
    component_eps = float(_arg(args, 'plane_component_eps', -1))
    component_eps = 5.0 * voxel_size if component_eps < 0 else component_eps
    max_mean = float(_arg(args, 'plane_max_mean_error', -1))
    max_mean = 1.5 * voxel_size if max_mean < 0 else max_mean
    max_p95 = float(_arg(args, 'plane_max_p95_error', -1))
    max_p95 = 3.0 * voxel_size if max_p95 < 0 else max_p95
    min_area = float(_arg(args, 'plane_min_area', -1))
    min_area = (5.0 * voxel_size) ** 2 if min_area < 0 else min_area
    up = _parse_up(_arg(args, 'plane_up_vector', '0,0,1'))
    bulk_component_neighbors = _bulk_component_neighbors_enabled(args)
    planes = []
    normal_clusters = _normal_clusters(normals, weights, float(_arg(args, 'plane_normal_cluster_angle_deg', 12)))
    diagnostics['normal_clusters'] = len(normal_clusters)
    for normal_indices, cluster_normal in normal_clusters:
        hypotheses = _extract_plane_hypotheses(points, normals, weights, normal_indices, normal_angle_deg=float(_arg(args, 'plane_normal_cluster_angle_deg', 12)), distance_epsilon=offset_eps, min_seed_points=max(12, min_points // 4))
        diagnostics['offset_groups'] += len(hypotheses)
        for hypothesis_indices, hypothesis_normal in hypotheses:
            components = _components(points, hypothesis_indices, hypothesis_normal, component_eps, bulk_neighbors=bulk_component_neighbors)
            diagnostics['components'] += len(components)
            for component in components:
                if len(component) < min_points:
                    diagnostics['rejected_too_few'] += 1
                    continue
                normal, offset, centroid, errors, area, concentration = _fit_plane(points, normals, weights, component, hypothesis_normal)
                if errors.mean() > max_mean:
                    diagnostics['rejected_mean_error'] += 1
                    continue
                if np.percentile(errors, 95) > max_p95:
                    diagnostics['rejected_p95_error'] += 1
                    continue
                if area < min_area:
                    diagnostics['rejected_area'] += 1
                    continue
                if int(camera_counts[component].max()) < int(_arg(args, 'plane_min_view_support', 2)):
                    diagnostics['rejected_view_support'] += 1
                    continue
                planes.append({'instance_id': int(instance_id), 'local_plane_id': len(planes), 'normal': normal.astype(np.float32), 'offset': float(offset), 'centroid': centroid.astype(np.float32), 'area': area, 'spacing': float(voxel_size), 'point_count': int(len(component)), 'camera_count': int(camera_counts[component].max()), 'mean_error': float(errors.mean()), 'median_error': float(np.median(errors)), 'p95_error': float(np.percentile(errors, 95)), 'max_error': float(errors.max()), 'normal_concentration': concentration, 'class': _classify_plane(normal, centroid, points, up, float(_arg(args, 'plane_wall_angle_deg', 20)), float(_arg(args, 'plane_roof_angle_deg', 30))), 'support_indices': component})
                diagnostics['accepted_planes'] += 1
    return planes

def _build_instance_prior_component(instance_id, chunks, args):
    profile_start = time.perf_counter()
    instance_id = int(instance_id)
    if not chunks:
        return {'instance_id': instance_id, 'has_samples': False, 'fused_points': np.empty((0, 3), np.float32), 'fused_normals': np.empty((0, 3), np.float32), 'fused_weights': np.empty((0,), np.float32), 'camera_counts': np.empty((0,), np.int32), 'planes': [], 'plane_support_points': [], 'stats': {'sample_points': 0, 'fused_points': 0, 'planes': 0}, 'log': None}
    points = np.concatenate([chunk['points'] for chunk in chunks])
    normals = np.concatenate([chunk['normals'] for chunk in chunks])
    weights = np.concatenate([chunk['weights'] for chunk in chunks])
    camera_ids = np.concatenate([chunk['camera_ids'] for chunk in chunks])
    concatenate_end = time.perf_counter()
    view_spacings = [float(chunk[_VIEW_SPACING_CACHE_KEY]) if _VIEW_SPACING_CACHE_KEY in chunk else estimate_spacing(chunk['points']) for chunk in chunks if len(chunk['points']) >= 2]
    raw_spacing = float(np.median(view_spacings)) if view_spacings else estimate_spacing(points)
    spacing_end = time.perf_counter()
    configured_voxel = float(_arg(args, 'plane_voxel_size', -1))
    voxel_size = 2.0 * raw_spacing if configured_voxel < 0 else configured_voxel
    fused = fuse_instance_voxels(points, normals, weights, camera_ids, voxel_size, int(_arg(args, 'plane_min_view_support', 2)), float(_arg(args, 'plane_min_normal_concentration', 0.8)), prefilter_min_views=bool(_arg(args, 'plane_voxel_prefilter_min_views', False)), backend=str(_arg(args, 'plane_voxel_fusion_backend', 'vectorized')))
    fused_points, fused_normals, fused_weights, camera_counts = fused
    fusion_end = time.perf_counter()
    detection_diagnostics = {}
    instance_planes = detect_instance_planes(fused_points, fused_normals, fused_weights, camera_counts, instance_id, voxel_size, args, diagnostics=detection_diagnostics)
    detection_end = time.perf_counter()
    plane_support_points = []
    plane_support_indices = []
    for plane in instance_planes:
        indices = np.asarray(plane['support_indices'], np.int64)
        plane_support_indices.append(indices.copy())
        plane_support_points.append(fused_points[indices])
    stats = {'views': int(len(chunks)), 'sample_points': int(len(points)), 'fused_points': int(len(fused_points)), 'planes': int(len(instance_planes)), 'raw_spacing': float(raw_spacing), 'voxel_size': float(voxel_size), 'voxel_fusion_backend': str(_arg(args, 'plane_voxel_fusion_backend', 'vectorized')), 'detection': detection_diagnostics, 'profile_ms': {'concatenate': float((concatenate_end - profile_start) * 1000.0), 'spacing': float((spacing_end - concatenate_end) * 1000.0), 'voxel_fusion': float((fusion_end - spacing_end) * 1000.0), 'plane_detection': float((detection_end - fusion_end) * 1000.0), 'total': float((detection_end - profile_start) * 1000.0)}}
    log = f'[DYNAMIC-PLANE] instance={instance_id}, views={len(chunks)}, samples={len(points)}, raw_spacing={raw_spacing:.6g}, voxel={voxel_size:.6g}, fused={len(fused_points)}, planes={len(instance_planes)}, detection={detection_diagnostics}'
    return {'instance_id': instance_id, 'has_samples': True, 'fused_points': fused_points, 'fused_normals': fused_normals, 'fused_weights': fused_weights, 'camera_counts': camera_counts, 'planes': instance_planes, 'plane_support_points': plane_support_points, 'plane_support_indices': plane_support_indices, 'stats': stats, 'log': log}

def build_prior_from_samples(records, instance_ids, args):
    all_points, all_normals, all_weights, all_instances, all_camera_counts = ([], [], [], [], [])
    planes, plane_support_points = ([], [])
    plane_support_normals, plane_support_weights = ([], [])
    plane_support_camera_counts, plane_support_indices = ([], [])
    plane_support_instance_ids, plane_support_ids = ([], [])
    per_instance = {}
    instance_ids = [int(instance_id) for instance_id in instance_ids]
    workers = int(_arg(args, 'plane_build_instance_workers', 0) or 0)
    if workers <= 0:
        workers = min(len(instance_ids), max(1, min(4, os.cpu_count() or 1)))
    workers = max(1, min(workers, len(instance_ids)))
    jobs = [(instance_id, records.get(instance_id, []), args) for instance_id in instance_ids]
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda job: _build_instance_prior_component(*job), jobs))
    else:
        results = [_build_instance_prior_component(*job) for job in jobs]
    for result in results:
        instance_id = int(result['instance_id'])
        per_instance[str(instance_id)] = result['stats']
        if result['log'] is not None:
            print(result['log'])
            profile_ms = result['stats'].get('profile_ms', {})
            print(f"[DYNAMIC-PROFILE] instance={instance_id}, concatenate_ms={profile_ms.get('concatenate', 0.0):.2f}, spacing_ms={profile_ms.get('spacing', 0.0):.2f}, voxel_fusion_ms={profile_ms.get('voxel_fusion', 0.0):.2f}, voxel_backend={result['stats'].get('voxel_fusion_backend', 'legacy')}, component_neighbors={('bulk' if _bulk_component_neighbors_enabled(args) else 'scalar')}, plane_detection_ms={profile_ms.get('plane_detection', 0.0):.2f}, instance_total_ms={profile_ms.get('total', 0.0):.2f}")
        if not result['has_samples']:
            continue
        fused_points = result['fused_points']
        fused_normals = result['fused_normals']
        fused_weights = result['fused_weights']
        camera_counts = result['camera_counts']
        for plane, support_points, support_indices in zip(result['planes'], result['plane_support_points'], result['plane_support_indices']):
            plane['global_plane_id'] = len(planes)
            plane_id = len(planes)
            plane_support_points.append(support_points)
            plane_support_normals.append(fused_normals[support_indices])
            plane_support_weights.append(fused_weights[support_indices])
            plane_support_camera_counts.append(camera_counts[support_indices])
            plane_support_indices.append(np.asarray(support_indices, np.int64))
            plane_support_instance_ids.append(np.full(len(support_indices), instance_id, np.int64))
            plane_support_ids.append(np.full(len(support_indices), plane_id, np.int64))
            planes.append(plane)
        all_points.append(fused_points)
        all_normals.append(fused_normals)
        all_weights.append(fused_weights)
        all_instances.append(np.full(len(fused_points), instance_id, np.int64))
        all_camera_counts.append(camera_counts)
    if not planes:
        raise RuntimeError(f'Dynamic plane construction found no planes after geometric filtering. Per-instance diagnostics: {json.dumps(per_instance)}')
    concatenate = lambda chunks, shape, dtype: np.concatenate(chunks) if chunks else np.empty(shape, dtype)
    class_counts = {}
    for plane in planes:
        class_counts[plane['class']] = class_counts.get(plane['class'], 0) + 1
    return {'support_points': concatenate(all_points, (0, 3), np.float32), 'support_normals': concatenate(all_normals, (0, 3), np.float32), 'support_weights': concatenate(all_weights, (0,), np.float32), 'support_instance_ids': concatenate(all_instances, (0,), np.int64), 'support_camera_counts': concatenate(all_camera_counts, (0,), np.int32), 'plane_normals': np.asarray([plane['normal'] for plane in planes], np.float32), 'plane_offsets': np.asarray([plane['offset'] for plane in planes], np.float32), 'plane_instance_ids': np.asarray([plane['instance_id'] for plane in planes], np.int64), 'plane_local_ids': np.asarray([plane['local_plane_id'] for plane in planes], np.int64), 'plane_classes': [plane['class'] for plane in planes], 'plane_centroids': np.asarray([plane['centroid'] for plane in planes], np.float32), 'plane_areas': np.asarray([plane['area'] for plane in planes], np.float32), 'plane_spacings': np.asarray([plane['spacing'] for plane in planes], np.float32), 'plane_support_points': plane_support_points, 'plane_support_normals': plane_support_normals, 'plane_support_weights': plane_support_weights, 'plane_support_camera_counts': plane_support_camera_counts, 'plane_support_indices': plane_support_indices, 'plane_support_ids': plane_support_ids, 'plane_support_instance_ids': plane_support_instance_ids, 'planes': planes, 'statistics': {'instances': per_instance, 'plane_count': len(planes), 'class_counts': class_counts}}

def _write_cloud(path, points, normals, instance_ids, weights=None, camera_counts=None, plane_ids=None, plane_confidences=None):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('instance_id', 'i4'), ('weight', 'f4'), ('camera_count', 'i4')]
    if plane_ids is not None:
        dtype.append(('plane_id', 'i4'))
    if plane_confidences is not None:
        dtype.append(('plane_confidence', 'f4'))
    data = np.empty(len(points), dtype=dtype)
    data['x'], data['y'], data['z'] = points.T
    data['nx'], data['ny'], data['nz'] = normals.T
    data['instance_id'] = instance_ids
    data['weight'] = np.ones(len(points)) if weights is None else weights
    data['camera_count'] = np.ones(len(points), np.int32) if camera_counts is None else camera_counts
    if plane_ids is not None:
        data['plane_id'] = plane_ids
    if plane_confidences is not None:
        data['plane_confidence'] = np.clip(plane_confidences, 0.0, 1.0)
    PlyData([PlyElement.describe(data, 'vertex')]).write(str(path))

def _materialize_plane_support_evidence(prior):
    points_by_plane = list(prior['plane_support_points'])
    normals_by_plane = prior.get('plane_support_normals')
    weights_by_plane = prior.get('plane_support_weights')
    cameras_by_plane = prior.get('plane_support_camera_counts')
    indices_by_plane = prior.get('plane_support_indices')
    if all((values is not None and len(values) == len(points_by_plane) for values in (normals_by_plane, weights_by_plane, cameras_by_plane, indices_by_plane))):
        return (points_by_plane, list(normals_by_plane), list(weights_by_plane), list(cameras_by_plane), list(indices_by_plane))
    support_points = np.asarray(prior['support_points'])
    support_instances = np.asarray(prior['support_instance_ids'])
    normals_by_plane, weights_by_plane, cameras_by_plane, indices_by_plane = ([], [], [], [])
    for plane_id, points in enumerate(points_by_plane):
        instance_id = int(prior['plane_instance_ids'][plane_id])
        eligible = np.flatnonzero(support_instances == instance_id)
        if not len(eligible):
            matched = np.zeros(len(points), np.int64)
        else:
            _, local = cKDTree(support_points[eligible]).query(points, k=1)
            matched = eligible[np.asarray(local, np.int64)]
        indices_by_plane.append(matched)
        normals_by_plane.append(np.asarray(prior['support_normals'])[matched])
        weights_by_plane.append(np.asarray(prior['support_weights'])[matched])
        cameras_by_plane.append(np.asarray(prior['support_camera_counts'])[matched])
    return (points_by_plane, normals_by_plane, weights_by_plane, cameras_by_plane, indices_by_plane)

def _plane_support_confidence(points, normals, weights, camera_counts, plane):
    weights = np.maximum(np.asarray(weights, np.float64), 0.0)
    camera_counts = np.maximum(np.asarray(camera_counts, np.float64), 0.0)
    weight_scale = max(float(np.percentile(weights, 95)) if len(weights) else 0.0, 1e-08)
    camera_scale = max(float(np.percentile(camera_counts, 95)) if len(camera_counts) else 0.0, 1.0)
    evidence = np.sqrt(np.clip(weights / weight_scale, 0.0, 1.0) * np.clip(camera_counts / camera_scale, 0.0, 1.0))
    normal = np.asarray(plane['normal'], np.float64)
    residual = np.abs(np.asarray(points) @ normal + float(plane['offset']))
    spacing = max(float(plane.get('spacing', 0.001)), 1e-08)
    residual_score = np.exp(-0.5 * (residual / (2.0 * spacing)) ** 2)
    normal_alignment = np.clip(np.abs(np.asarray(normals) @ normal), 0.0, 1.0)
    concentration = np.clip(float(plane.get('normal_concentration', 0.0)), 0.0, 1.0)
    confidence = (0.25 + 0.75 * evidence) * residual_score
    confidence *= np.sqrt(np.maximum(normal_alignment * concentration, 0.0))
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)

def save_dynamic_plane_prior(prior, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_cloud(output_dir / 'fused_points_all.ply', prior['support_points'], prior['support_normals'], prior['support_instance_ids'], prior['support_weights'], prior['support_camera_counts'])
    for instance_id in np.unique(prior['support_instance_ids']):
        selected = prior['support_instance_ids'] == instance_id
        _write_cloud(output_dir / f'fused_points_instance_{int(instance_id)}.ply', prior['support_points'][selected], prior['support_normals'][selected], prior['support_instance_ids'][selected], prior['support_weights'][selected], prior['support_camera_counts'][selected])
    points_by_plane, normals_by_plane, weights_by_plane, cameras_by_plane, indices_by_plane = _materialize_plane_support_evidence(prior)
    support_points = np.concatenate(points_by_plane)
    support_normals = np.concatenate(normals_by_plane).astype(np.float32, copy=False)
    support_weights = np.concatenate(weights_by_plane).astype(np.float32, copy=False)
    support_cameras = np.concatenate(cameras_by_plane).astype(np.int32, copy=False)
    support_plane_ids = np.concatenate([np.full(len(points), index, np.int64) for index, points in enumerate(points_by_plane)])
    support_instances = prior['plane_instance_ids'][support_plane_ids]
    confidence_by_plane = [_plane_support_confidence(points, normals, weights, cameras, plane) for points, normals, weights, cameras, plane in zip(points_by_plane, normals_by_plane, weights_by_plane, cameras_by_plane, prior['planes'])]
    support_confidence = np.concatenate(confidence_by_plane)
    _write_cloud(output_dir / 'dynamic_plane_support.ply', support_points, support_normals, support_instances, support_weights, support_cameras, plane_ids=support_plane_ids, plane_confidences=support_confidence)
    serializable_planes = []
    support_start = 0
    for plane_id, (plane, points, confidence, cameras, indices) in enumerate(zip(prior['planes'], points_by_plane, confidence_by_plane, cameras_by_plane, indices_by_plane)):
        serializable = dict(plane)
        serializable['normal'] = np.asarray(serializable['normal']).tolist()
        serializable['centroid'] = np.asarray(serializable['centroid']).tolist()
        serializable.pop('support_indices', None)
        support_end = support_start + len(points)
        serializable.update({'support_count': int(len(points)), 'confidence_mean': float(np.mean(confidence)) if len(confidence) else 0.0, 'confidence_median': float(np.median(confidence)) if len(confidence) else 0.0, 'camera_count_mean': float(np.mean(cameras)) if len(cameras) else 0.0, 'camera_count_max': int(np.max(cameras)) if len(cameras) else 0, 'support_range': [int(support_start), int(support_end)], 'support_source_indices': np.asarray(indices, np.int64).tolist(), 'support_bounds': [np.asarray(points).min(0).astype(float).tolist(), np.asarray(points).max(0).astype(float).tolist()] if len(points) else None, 'global_plane_id': int(plane_id)})
        serializable_planes.append(serializable)
        support_start = support_end
    with open(output_dir / 'dynamic_plane_priors.json', 'w', encoding='utf-8') as handle:
        json.dump({'planes': serializable_planes}, handle, indent=2)
    visibility_rays = prior.get('visibility_rays')
    if visibility_rays is not None:
        np.savez_compressed(output_dir / 'multiview_surface_rays.npz', **visibility_rays)
    with open(output_dir / 'plane_detection_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(prior['statistics'], handle, indent=2)


