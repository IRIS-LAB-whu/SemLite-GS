import math
import time
import torch
import torch.nn.functional as F


def _target_mask(camera, instance_ids, device):
    labels = getattr(camera, 'instance_mask', None)
    if labels is None:
        return None
    labels = labels.to(device=device)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    if labels.shape != (camera.image_height, camera.image_width):
        raise ValueError('Instance mask shape must match camera resolution')
    target = torch.zeros_like(labels, dtype=torch.bool)
    for instance_id in instance_ids:
        target |= labels == int(instance_id)
    return target


def _report(keep, started, **counts):
    if keep.is_cuda:
        torch.cuda.synchronize(keep.device)
    return dict(total=int(keep.numel()), kept=int(keep.sum().item()),
                pruned=int((~keep).sum().item()), elapsed_s=time.perf_counter()-started, **counts)


@torch.no_grad()
def initial_mask_keep(points, cameras, instance_ids, dilation_pixels=0):


    started = time.perf_counter()
    keep = torch.zeros(len(points), device=points.device, dtype=torch.bool)
    dilation_pixels = max(0, int(dilation_pixels))
    mask_cameras = 0
    for camera in cameras:
        target = _target_mask(camera, instance_ids, points.device)
        if target is None or not target.any():
            continue
        mask_cameras += 1
        if dilation_pixels:
            target = F.max_pool2d(target[None, None].float(), 2*dilation_pixels+1,
                                  stride=1, padding=dilation_pixels)[0, 0].bool()
        x, y, _, visible = project_points_to_camera(points, camera)
        indices = visible.nonzero().flatten()
        keep[indices[target[y[indices].long(), x[indices].long()]]] = True
    if not mask_cameras:
        raise RuntimeError('Initial Gaussian pruning found no camera containing a requested instance mask')
    if not keep.any():
        raise RuntimeError('Initial Gaussian pruning rejected every point; check camera poses, mask labels and init_ply')
    return keep, _report(keep, started, mask_cameras=mask_cameras)


def project_points_to_camera(points, camera):
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    view = torch.matmul(points_hom, camera.world_view_transform)
    projected = torch.matmul(points_hom, camera.full_proj_transform)
    denom = projected[:, 3]
    valid_depth = denom > 1e-7
    denom_safe = torch.where(valid_depth, denom, torch.ones_like(denom))
    ndc = projected[:, :3] / denom_safe.unsqueeze(1)
    pix_x = ((ndc[:, 0] + 1.0) * 0.5) * float(camera.image_width)
    pix_y = ((ndc[:, 1] + 1.0) * 0.5) * float(camera.image_height)
    in_image = (
        valid_depth
        & (pix_x >= 0.0)
        & (pix_x < float(camera.image_width))
        & (pix_y >= 0.0)
        & (pix_y < float(camera.image_height))
    )
    return pix_x, pix_y, view[:, 2], in_image


def accumulate_occluded_frozen_view_votes(
    point_indices,
    pix_x,
    pix_y,
    frozen_depth,
    radii,
    target_mask,
    trainable_alpha,
    trainable_depth,
    relevant_views,
    exposed_views,
    depth_margin,
    alpha_threshold,
    occluded_ratio_threshold,
    skip_exposed=True,
):
    if skip_exposed:


        pending = exposed_views[point_indices] == 0
        point_indices = point_indices[pending]
        pix_x = pix_x[pending]
        pix_y = pix_y[pending]
        frozen_depth = frozen_depth[pending]
        radii = radii[pending]
    if point_indices.numel() == 0:
        return

    height, width = target_mask.shape
    pix_x = pix_x.round().to(dtype=torch.long)
    pix_y = pix_y.round().to(dtype=torch.long)
    radii = radii.to(dtype=torch.long).clamp_min(1)
    unique_radii = torch.unique(radii)
    max_samples_per_chunk = 262144
    max_vectorized_footprint_pixels = 65536

    for radius_value in unique_radii.tolist():
        radius = int(radius_value)
        group_mask = radii == radius
        group_positions = torch.nonzero(group_mask, as_tuple=False).squeeze(1)
        if group_positions.numel() == 0:
            continue

        square_pixels = (2 * radius + 1) * (2 * radius + 1)
        if square_pixels > max_vectorized_footprint_pixels:
            for local_position in group_positions.tolist():
                cx = int(pix_x[local_position].item())
                cy = int(pix_y[local_position].item())
                x0 = max(0, cx - radius)
                x1 = min(width, cx + radius + 1)
                y0 = max(0, cy - radius)
                y1 = min(height, cy + radius + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = torch.meshgrid(
                    torch.arange(y0, y1, device=target_mask.device),
                    torch.arange(x0, x1, device=target_mask.device),
                    indexing="ij",
                )
                disk = ((xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)) <= radius * radius
                target_pixels = disk & target_mask[y0:y1, x0:x1]
                relevant_pixels = int(target_pixels.sum().item())
                if relevant_pixels <= 0:
                    continue
                occluded_pixels = (
                    target_pixels
                    & (trainable_alpha[y0:y1, x0:x1] > alpha_threshold)
                    & (trainable_depth[y0:y1, x0:x1] < (frozen_depth[local_position] - depth_margin))
                )
                full_index = point_indices[local_position]
                relevant_views[full_index] += 1
                if float(occluded_pixels.sum().item()) / float(relevant_pixels) < occluded_ratio_threshold:
                    exposed_views[full_index] += 1
            continue

        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=target_mask.device),
            torch.arange(-radius, radius + 1, device=target_mask.device),
            indexing="ij",
        )
        disk = (offsets_x * offsets_x + offsets_y * offsets_y) <= radius * radius
        offsets_x = offsets_x[disk].reshape(1, -1)
        offsets_y = offsets_y[disk].reshape(1, -1)
        footprint_pixels = int(offsets_x.shape[1])
        if footprint_pixels == 0:
            continue

        chunk_size = max(1, max_samples_per_chunk // footprint_pixels)
        for start in range(0, int(group_positions.numel()), chunk_size):
            local = group_positions[start : start + chunk_size]
            xs = pix_x[local].reshape(-1, 1) + offsets_x
            ys = pix_y[local].reshape(-1, 1) + offsets_y
            in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            xs_safe = xs.clamp(0, width - 1)
            ys_safe = ys.clamp(0, height - 1)

            target_pixels = in_bounds & target_mask[ys_safe, xs_safe]
            relevant_pixel_counts = target_pixels.sum(dim=1)
            relevant = relevant_pixel_counts > 0
            if not relevant.any():
                continue

            alpha_pixels = trainable_alpha[ys_safe, xs_safe]
            depth_pixels = trainable_depth[ys_safe, xs_safe]
            occluded_pixels = (
                target_pixels
                & (alpha_pixels > alpha_threshold)
                & (depth_pixels < (frozen_depth[local].reshape(-1, 1) - depth_margin))
            )
            occluded_pixel_counts = occluded_pixels.sum(dim=1)
            occluded_ratio = occluded_pixel_counts.to(dtype=torch.float32) / relevant_pixel_counts.clamp_min(1).to(dtype=torch.float32)
            occluded = relevant & (occluded_ratio >= occluded_ratio_threshold)
            exposed = relevant & ~occluded

            full_indices = point_indices[local]
            relevant_views[full_indices[relevant]] += 1
            exposed_views[full_indices[exposed]] += 1

@torch.no_grad()
def occluded_frozen_keep(model, cameras, instance_ids, pipe, background, *,
                         depth_margin=1., alpha_threshold=.5, occluded_ratio=.8,
                         min_relevant_views=1, pixel_radius=1, render_batch_size=2,
                         check_all_views=False, render_fn=None):


    if not math.isfinite(depth_margin) or depth_margin < 0:
        raise ValueError('depth_margin must be finite and nonnegative')
    if not 0 <= alpha_threshold <= 1 or not 0 < occluded_ratio <= 1:
        raise ValueError('Invalid alpha or occluded ratio threshold')
    if min_relevant_views < 1 or int(min_relevant_views) != min_relevant_views:
        raise ValueError('min_relevant_views must be a positive integer')
    if render_batch_size < 1 or int(render_batch_size) != render_batch_size:
        raise ValueError('render_batch_size must be a positive integer')
    started = time.perf_counter()
    points = model.get_xyz
    trainable = model.trainable_mask
    frozen = ~trainable
    keep = torch.ones(len(points), device=points.device, dtype=torch.bool)
    frozen_indices = frozen.nonzero().flatten()
    cameras = list(cameras)
    relevant = torch.zeros(len(points), device=points.device, dtype=torch.int32)
    exposed = torch.zeros_like(relevant)
    render_batches = render_views = 0
    candidate_view_pairs = 0
    counts = dict(frozen_total=int(frozen.sum().item()), trainable_total=int(trainable.sum().item()),
                  selected_views=len(cameras))
    if not len(points):
        raise RuntimeError('Cannot filter an empty Gaussian population')
    if not frozen.any() or not cameras or not trainable.any():
        return keep, _report(keep, started, **counts, kept_frozen=counts['frozen_total'],
                             relevant_frozen=0, exposed_frozen=0, render_batches=0,
                             render_views=0, candidate_view_pairs=0)
    if render_fn is None:
        from pipeline.rendering import render_many
        render_fn = render_many
    backup = model._opacity.detach()[frozen].clone()
    hidden = torch.full_like(backup, 1e-8).logit()
    fallback_radius = max(1, int(pixel_radius))
    skip_exposed = not check_all_views

    def process(payloads):
        nonlocal render_batches, render_views, candidate_view_pairs
        if not payloads:
            return
        packages = render_fn([p[0] for p in payloads], model, pipe, background,
                             return_plane=True, return_depth_normal=False)
        if len(packages) != len(payloads):
            raise RuntimeError('Native PGSR render batch returned the wrong view count')
        render_batches += 1
        render_views += len(payloads)
        for payload, package in zip(payloads, packages):
            camera, target, indices, x, y, z = payload
            alpha = torch.nan_to_num(package['rendered_alpha'][0], nan=0., posinf=0., neginf=0.)
            depth = package['plane_depth'][0]

            depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.inf)
            radii = package['radii'][indices].float().ceil().long().clamp_min(fallback_radius)
            candidate_view_pairs += int((exposed[indices] == 0).sum().item()) if skip_exposed else len(indices)
            accumulate_occluded_frozen_view_votes(indices, x, y, z, radii, target,
                alpha, depth, relevant, exposed, depth_margin, alpha_threshold,
                occluded_ratio, skip_exposed=skip_exposed)

    try:
        model._opacity[frozen] = hidden
        pending_by_resolution = {}
        for camera in cameras:
            candidates = frozen_indices[exposed[frozen_indices] == 0] if skip_exposed else frozen_indices
            if not len(candidates):
                break
            target = _target_mask(camera, instance_ids, points.device)
            if target is None or not target.any():
                continue
            x, y, z, visible = project_points_to_camera(points[candidates], camera)
            if not visible.any():
                continue
            key = (int(camera.image_height), int(camera.image_width))
            pending = pending_by_resolution.setdefault(key, [])
            pending.append((camera, target, candidates[visible], x[visible], y[visible], z[visible]))
            if len(pending) >= render_batch_size:
                process(pending)
                pending.clear()
        for pending in pending_by_resolution.values():
            process(pending)
            pending.clear()
    finally:
        model._opacity[frozen] = backup
    remove = frozen & (relevant >= min_relevant_views) & (exposed == 0)
    keep = ~remove
    if not keep.any():
        raise RuntimeError('Occlusion filtering rejected every point')
    return keep, _report(keep, started, **counts,
        kept_frozen=int((frozen & keep).sum().item()),
        relevant_frozen=int((frozen & (relevant > 0)).sum().item()),
        exposed_frozen=int((frozen & (exposed > 0)).sum().item()),
        render_batches=render_batches, render_views=render_views,
        candidate_view_pairs=candidate_view_pairs)
