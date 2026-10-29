import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from plyfile import PlyData


BUILDING_PRISM_CACHE_SCHEMA = "building_prism_model_cache_v1"
BUILDING_PRISM_ALGORITHM_VERSION = "legacy_building_prism_v1"
BUILDING_PRISM_SOURCE_MANIFEST_SCHEMA = "building_prism_source_sha256_v1"


class _BuildingPrismCacheMiss(RuntimeError):
    pass


def _canonical_json(payload):
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_fingerprint(path):
    real_path = os.path.realpath(os.fspath(path))
    stat_result = os.stat(real_path)
    return {
        "realpath": real_path,
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _load_source_digest_manifest(path):
    path = Path(path)
    if not path.is_file():
        return {"schema": BUILDING_PRISM_SOURCE_MANIFEST_SCHEMA, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"schema": BUILDING_PRISM_SOURCE_MANIFEST_SCHEMA, "entries": {}}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != BUILDING_PRISM_SOURCE_MANIFEST_SCHEMA
        or not isinstance(payload.get("entries"), dict)
    ):
        return {"schema": BUILDING_PRISM_SOURCE_MANIFEST_SCHEMA, "entries": {}}
    entries = {
        key: value
        for key, value in payload["entries"].items()
        if isinstance(key, str) and _valid_sha256(value)
    }
    return {"schema": BUILDING_PRISM_SOURCE_MANIFEST_SCHEMA, "entries": entries}


def _building_source_sha256(path, cache_root, force_rehash=False):

    fingerprint = _source_fingerprint(path)
    fingerprint_key = _canonical_json(fingerprint)
    manifest_path = Path(cache_root) / "source_sha256_manifest.json"
    manifest = _load_source_digest_manifest(manifest_path)
    if not force_rehash:
        cached_digest = manifest["entries"].get(fingerprint_key)
        if _valid_sha256(cached_digest):
            return cached_digest, "manifest_hit"

    digest = _sha256_file(fingerprint["realpath"])


    latest_manifest = _load_source_digest_manifest(manifest_path)
    latest_manifest["entries"][fingerprint_key] = digest
    _atomic_write_json(manifest_path, latest_manifest)
    return digest, "content_hashed"


def _optional_file_sha256(path):
    if not path or not Path(path).is_file():
        return "missing"
    return _sha256_file(path)


def _building_prism_cache_key_payload(
    instance_ids,
    building_ply_path,
    transform_path,
    value_mapping_path,
    color_mapping_path,
    xy_padding,
    z_padding,
    trim_quantile,
    cache_root,
    force_source_rehash=False,
):
    building_digest, digest_reason = _building_source_sha256(
        building_ply_path,
        cache_root,
        force_rehash=force_source_rehash,
    )
    payload = {
        "schema": BUILDING_PRISM_CACHE_SCHEMA,
        "algorithm": BUILDING_PRISM_ALGORITHM_VERSION,
        "instance_ids": sorted({int(value) for value in instance_ids}),
        "inputs": {
            "building_ply_sha256": building_digest,
            "transform_sha256": _sha256_file(transform_path),
            "value_mapping_sha256": _optional_file_sha256(value_mapping_path),
            "color_mapping_sha256": _optional_file_sha256(color_mapping_path),
        },
        "parameters": {
            "xy_padding": float(xy_padding).hex(),
            "z_padding": float(z_padding).hex(),
            "trim_quantile": float(trim_quantile).hex(),
        },
    }
    cache_key = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return cache_key, payload, digest_reason


def _building_prism_cache_path(cache_root, cache_key):
    return (
        Path(cache_root)
        / BUILDING_PRISM_CACHE_SCHEMA
        / "objects"
        / cache_key[:2]
        / f"{cache_key}.npz"
    )


def _scalar_string(array, field_name):
    array = np.asarray(array)
    if array.shape != ():
        raise _BuildingPrismCacheMiss(f"invalid_{field_name}_shape")
    value = array.item()
    if not isinstance(value, str):
        raise _BuildingPrismCacheMiss(f"invalid_{field_name}_type")
    return value


def _load_building_prism_cache(path, expected_key, expected_payload_json):
    path = Path(path)
    if not path.is_file():
        raise _BuildingPrismCacheMiss("object_not_found")
    try:
        with np.load(path, allow_pickle=False) as payload:
            required = {
                "schema",
                "cache_key",
                "key_payload_json",
                "transform",
                "hull_values",
                "hull_offsets",
                "z_min",
                "z_max",
                "metadata_json",
            }
            if not required.issubset(payload.files):
                raise _BuildingPrismCacheMiss("missing_fields")
            if _scalar_string(payload["schema"], "schema") != BUILDING_PRISM_CACHE_SCHEMA:
                raise _BuildingPrismCacheMiss("schema_mismatch")
            if _scalar_string(payload["cache_key"], "cache_key") != expected_key:
                raise _BuildingPrismCacheMiss("key_mismatch")
            if (
                _scalar_string(payload["key_payload_json"], "key_payload_json")
                != expected_payload_json
            ):
                raise _BuildingPrismCacheMiss("key_payload_mismatch")

            transform = np.asarray(payload["transform"])
            hull_values = np.asarray(payload["hull_values"])
            hull_offsets = np.asarray(payload["hull_offsets"])
            z_min = np.asarray(payload["z_min"])
            z_max = np.asarray(payload["z_max"])
            if transform.dtype != np.float64 or transform.shape != (3, 3):
                raise _BuildingPrismCacheMiss("invalid_transform")
            if (
                hull_values.dtype != np.float64
                or hull_values.ndim != 2
                or hull_values.shape[1] != 2
            ):
                raise _BuildingPrismCacheMiss("invalid_hulls")
            if hull_offsets.dtype != np.int64 or hull_offsets.ndim != 1:
                raise _BuildingPrismCacheMiss("invalid_hull_offsets")
            if z_min.dtype != np.float64 or z_min.ndim != 1:
                raise _BuildingPrismCacheMiss("invalid_z_min")
            if z_max.dtype != np.float64 or z_max.shape != z_min.shape:
                raise _BuildingPrismCacheMiss("invalid_z_max")
            if (
                len(hull_offsets) != len(z_min) + 1
                or len(hull_offsets) == 0
                or int(hull_offsets[0]) != 0
                or int(hull_offsets[-1]) != len(hull_values)
                or np.any(hull_offsets[1:] < hull_offsets[:-1])
            ):
                raise _BuildingPrismCacheMiss("invalid_hull_offsets")

            metadata_json = _scalar_string(payload["metadata_json"], "metadata_json")
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError as error:
                raise _BuildingPrismCacheMiss("invalid_metadata_json") from error
            if not isinstance(metadata, dict):
                raise _BuildingPrismCacheMiss("invalid_metadata")
            if int(metadata.get("prisms", -1)) != len(z_min):
                raise _BuildingPrismCacheMiss("metadata_prism_count_mismatch")
            if not isinstance(metadata.get("selected_colors"), int):
                raise _BuildingPrismCacheMiss("invalid_metadata_selected_colors")
            if not isinstance(metadata.get("match_mode"), str):
                raise _BuildingPrismCacheMiss("invalid_metadata_match_mode")

            prisms = []
            for index in range(len(z_min)):
                start = int(hull_offsets[index])
                stop = int(hull_offsets[index + 1])
                if stop - start < 3:
                    raise _BuildingPrismCacheMiss("invalid_hull_vertex_count")
                prisms.append(
                    {
                        "hull": hull_values[start:stop].copy(),
                        "z_min": float(z_min[index]),
                        "z_max": float(z_max[index]),
                    }
                )
            return {
                "transform": transform.copy(),
                "prisms": prisms,
                "metadata": metadata,
            }
    except _BuildingPrismCacheMiss:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        EOFError,
        KeyError,
        zipfile.BadZipFile,
    ) as error:
        raise _BuildingPrismCacheMiss(
            f"load_error_{type(error).__name__}"
        ) from error


def _write_building_prism_cache(
    path,
    cache_key,
    key_payload_json,
    prism_model,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prisms = list(prism_model["prisms"])
    hull_offsets = [0]
    hull_parts = []
    for prism in prisms:
        hull = np.asarray(prism["hull"], dtype=np.float64)
        hull_parts.append(hull)
        hull_offsets.append(hull_offsets[-1] + len(hull))
    hull_values = (
        np.concatenate(hull_parts, axis=0)
        if hull_parts
        else np.empty((0, 2), dtype=np.float64)
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez(
                handle,
                schema=np.asarray(BUILDING_PRISM_CACHE_SCHEMA),
                cache_key=np.asarray(cache_key),
                key_payload_json=np.asarray(key_payload_json),
                transform=np.asarray(prism_model["transform"], dtype=np.float64),
                hull_values=hull_values,
                hull_offsets=np.asarray(hull_offsets, dtype=np.int64),
                z_min=np.asarray(
                    [prism["z_min"] for prism in prisms], dtype=np.float64
                ),
                z_max=np.asarray(
                    [prism["z_max"] for prism in prisms], dtype=np.float64
                ),
                metadata_json=np.asarray(
                    json.dumps(
                        dict(prism_model["metadata"]),
                        separators=(",", ":"),
                        ensure_ascii=True,
                    )
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def read_transform(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            values = [float(value) for value in raw.replace("[", " ").replace("]", " ").split()]
            if values:
                rows.append(values)
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (4, 4):
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0])):
            raise ValueError(f"Unsupported non-affine transform in {path}")
        return matrix[:3, :3]
    raise ValueError(f"Expected a 3x3 or 4x4 transform in {path}, got {matrix.shape}")


def read_colored_ply_points(path):
    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(np.float64)
    rgb = np.column_stack([vertex[name] for name in ("red", "green", "blue")]).astype(np.uint8)
    return xyz, rgb


def load_json(path):
    if not path or not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def convex_hull_2d(points):
    points = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(points) <= 2:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def cross(origin, first, second):
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if len(hull) >= 3 else points


def expand_degenerate_hull(hull, padding):
    if len(hull) >= 3:
        return hull
    center = hull.mean(axis=0) if len(hull) else np.zeros(2, dtype=np.float64)
    padding = max(float(padding), 1e-3)
    return np.asarray(
        [
            [center[0] - padding, center[1] - padding],
            [center[0] + padding, center[1] - padding],
            [center[0] + padding, center[1] + padding],
            [center[0] - padding, center[1] + padding],
        ],
        dtype=np.float64,
    )


def pad_polygon_from_center(polygon, padding):
    padding = float(padding)
    if padding <= 0:
        return polygon
    center = polygon.mean(axis=0)
    direction = polygon - center
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return polygon + padding * direction / norm


def select_building_color_groups(points_rgb, instance_ids, value_mapping_path, color_mapping_path):
    unique_colors = sorted(
        tuple(int(channel) for channel in color)
        for color in np.unique(points_rgb, axis=0).tolist()
    )
    selected_values = {int(instance_id) for instance_id in instance_ids}
    value_mapping = load_json(value_mapping_path)
    color_mapping = load_json(color_mapping_path)

    if value_mapping and color_mapping:
        value_to_instance = {
            int(value): int(instance_id)
            for instance_id, value in value_mapping.items()
        }
        requested_instance_ids = {
            value_to_instance[value]
            for value in selected_values
            if value in value_to_instance
        }
        instance_to_color = {
            int(instance_id): tuple(int(channel) for channel in color)
            for instance_id, color in color_mapping.items()
        }
        selected_colors = [
            instance_to_color[instance_id]
            for instance_id in sorted(requested_instance_ids)
            if instance_id in instance_to_color
            and instance_to_color[instance_id] in unique_colors
        ]
        if selected_colors:
            return selected_colors, "mask_value_to_color_mapping"

        sorted_instance_ids = sorted(int(instance_id) for instance_id in value_mapping.keys())
        if len(unique_colors) == len(sorted_instance_ids):
            color_by_instance = {
                instance_id: color
                for instance_id, color in zip(sorted_instance_ids, unique_colors)
            }
            selected_colors = [
                color_by_instance[instance_id]
                for instance_id in sorted(requested_instance_ids)
                if instance_id in color_by_instance
            ]
            if selected_colors:
                return selected_colors, "sorted_color_order_mapping"

    if len(unique_colors) == len(selected_values):
        return unique_colors, "all_ply_color_groups_match_requested_count"

    raise ValueError(
        "Could not map requested instance mask values to building PLY colors. "
        f"requested_values={sorted(selected_values)}, ply_color_groups={len(unique_colors)}. "
        "Set the value/color mapping paths, or provide a PLY containing exactly "
        "the requested building color groups."
    )


def _build_building_prism_model_uncached(
    instance_ids,
    building_ply_path,
    transform_path,
    value_mapping_path=None,
    color_mapping_path=None,
    xy_padding=0.0,
    z_padding=0.0,
    trim_quantile=0.02,
):
    if not os.path.exists(building_ply_path):
        raise FileNotFoundError(f"Building PLY was not found: {building_ply_path}")
    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Z-align transform was not found: {transform_path}")

    building_points, building_rgb = read_colored_ply_points(building_ply_path)
    transform = read_transform(transform_path)
    selected_colors, match_mode = select_building_color_groups(
        building_rgb,
        instance_ids,
        value_mapping_path,
        color_mapping_path,
    )
    aligned_points = building_points @ transform.T

    trim_quantile = float(np.clip(trim_quantile, 0.0, 0.49))
    prisms = []
    for color in selected_colors:
        color_mask = np.all(building_rgb == np.asarray(color, dtype=np.uint8), axis=1)
        points = aligned_points[color_mask]
        if len(points) < 3:
            continue
        fit_points = points
        if trim_quantile > 0:
            low = np.quantile(points, trim_quantile, axis=0)
            high = np.quantile(points, 1.0 - trim_quantile, axis=0)
            keep = np.all((points >= low) & (points <= high), axis=1)
            if int(keep.sum()) >= 3:
                fit_points = points[keep]
        hull = expand_degenerate_hull(convex_hull_2d(fit_points[:, :2]), xy_padding)
        hull = pad_polygon_from_center(hull, xy_padding)
        z_min = float(fit_points[:, 2].min() - z_padding)
        z_max = float(fit_points[:, 2].max() + z_padding)
        if z_max <= z_min:
            z_max = z_min + max(float(z_padding), 1e-3)
        prisms.append({"hull": hull, "z_min": z_min, "z_max": z_max})

    return {
        "transform": transform,
        "prisms": prisms,
        "metadata": {
            "prisms": len(prisms),
            "selected_colors": len(set(selected_colors)),
            "match_mode": match_mode,
        },
    }


def build_building_prism_model(
    instance_ids,
    building_ply_path,
    transform_path,
    value_mapping_path=None,
    color_mapping_path=None,
    xy_padding=0.0,
    z_padding=0.0,
    trim_quantile=0.02,
    cache_dir=None,
    rebuild_cache=False,
):


    if cache_dir is None or not os.fspath(cache_dir).strip():
        return _build_building_prism_model_uncached(
            instance_ids,
            building_ply_path,
            transform_path,
            value_mapping_path,
            color_mapping_path,
            xy_padding,
            z_padding,
            trim_quantile,
        )


    if not os.path.exists(building_ply_path):
        raise FileNotFoundError(f"Building PLY was not found: {building_ply_path}")
    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Z-align transform was not found: {transform_path}")

    cache_root = Path(cache_dir).expanduser().resolve()
    try:
        cache_key, key_payload, source_digest_reason = (
            _building_prism_cache_key_payload(
                instance_ids,
                building_ply_path,
                transform_path,
                value_mapping_path,
                color_mapping_path,
                xy_padding,
                z_padding,
                trim_quantile,
                cache_root,
                force_source_rehash=bool(rebuild_cache),
            )
        )
    except (OSError, ValueError, TypeError) as error:
        print(
            "[BUILDING_PRISM_CACHE] "
            f"status=bypass key=unavailable reason=key_error_{type(error).__name__} "
            f"cache_dir={cache_root}"
        )
        return _build_building_prism_model_uncached(
            instance_ids,
            building_ply_path,
            transform_path,
            value_mapping_path,
            color_mapping_path,
            xy_padding,
            z_padding,
            trim_quantile,
        )

    key_payload_json = _canonical_json(key_payload)
    cache_path = _building_prism_cache_path(cache_root, cache_key)
    miss_reason = "rebuild_requested" if rebuild_cache else "object_not_found"
    if not rebuild_cache:
        try:
            model = _load_building_prism_cache(
                cache_path,
                cache_key,
                key_payload_json,
            )
        except _BuildingPrismCacheMiss as error:
            miss_reason = str(error)
        else:
            print(
                "[BUILDING_PRISM_CACHE] "
                f"status=hit key={cache_key} reason=validated "
                f"source_digest={source_digest_reason} path={cache_path}"
            )
            return model

    print(
        "[BUILDING_PRISM_CACHE] "
        f"status=miss key={cache_key} reason={miss_reason} "
        f"source_digest={source_digest_reason} path={cache_path}"
    )
    model = _build_building_prism_model_uncached(
        instance_ids,
        building_ply_path,
        transform_path,
        value_mapping_path,
        color_mapping_path,
        xy_padding,
        z_padding,
        trim_quantile,
    )
    try:
        _write_building_prism_cache(
            cache_path,
            cache_key,
            key_payload_json,
            model,
        )
    except (OSError, ValueError, TypeError) as error:
        print(
            "[BUILDING_PRISM_CACHE] "
            f"status=write_failed key={cache_key} "
            f"reason={type(error).__name__} path={cache_path}"
        )
    else:
        print(
            "[BUILDING_PRISM_CACHE] "
            f"status=stored key={cache_key} reason={miss_reason} path={cache_path}"
        )
    return model


def point_in_polygon_numpy(points_xy, polygon_xy):
    points_xy = np.asarray(points_xy, dtype=np.float64)
    polygon_xy = np.asarray(polygon_xy, dtype=np.float64)
    inside = np.zeros(len(points_xy), dtype=bool)
    if len(polygon_xy) < 3:
        return inside
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    for index in range(len(polygon_xy)):
        first = polygon_xy[index]
        second = polygon_xy[(index + 1) % len(polygon_xy)]
        crosses = (first[1] > y) != (second[1] > y)
        x_at_y = (
            (second[0] - first[0])
            * (y - first[1])
            / (second[1] - first[1] + 1e-12)
            + first[0]
        )
        inside ^= crosses & (x < x_at_y)
    return inside


def points_in_prisms_numpy(points, prism_model, chunk_size=1_000_000):
    points = np.asarray(points, dtype=np.float64)
    result = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        aligned = points[start : start + chunk_size] @ prism_model["transform"].T
        inside = np.zeros(len(aligned), dtype=bool)
        for prism in prism_model["prisms"]:
            inside |= (
                (aligned[:, 2] >= prism["z_min"])
                & (aligned[:, 2] <= prism["z_max"])
                & point_in_polygon_numpy(aligned[:, :2], prism["hull"])
            )
        result[start : start + len(aligned)] = inside
    return result


