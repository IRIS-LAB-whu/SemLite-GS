from pathlib import Path
import numpy as np
from pipeline.planes.dynamic_plane_prior import (
    build_prior_from_samples, save_dynamic_plane_prior, _visibility_rays_from_records,
)


def export_dynamic_evidence(gaussians, cameras, instance_ids, pipe, background, args, output_dir):


    import torch
    import torch.nn.functional as F
    from pipeline.rendering import render_many
    ids = [int(i) for i in instance_ids]
    records = {i: [] for i in ids}
    limit = int(getattr(args, 'plane_max_points_per_view', 20000))
    independent = getattr(gaussians, 'context_mode', 'joint') == 'independent'
    eligible = []
    rois = []
    owners = []
    for camera in cameras:
        mask = getattr(camera, 'instance_mask', None)
        if mask is None:
            continue
        labels = torch.as_tensor(mask).reshape(int(camera.image_height), int(camera.image_width))
        if independent:
            for owner in ids:
                coordinates = torch.nonzero(labels == owner)
                if not len(coordinates):
                    continue
                lower, upper = coordinates.min(0).values, coordinates.max(0).values + 1
                rois.append((int(lower[1]), int(lower[0]), int(upper[1]-lower[1]), int(upper[0]-lower[0])))
                eligible.append(camera)
                owners.append(owner)
            continue
        selected = torch.zeros_like(labels, dtype=torch.bool)
        for instance_id in ids:
            selected |= labels == instance_id
        coordinates = torch.nonzero(selected)
        if not len(coordinates):
            continue
        lower, upper = coordinates.min(0).values, coordinates.max(0).values + 1
        rois.append((int(lower[1]), int(lower[0]), int(upper[1]-lower[1]), int(upper[0]-lower[0])))
        eligible.append(camera)
    batch_size = int(getattr(args, 'plane_render_batch_size', getattr(args, 'camera_batch_size', 1)))
    if batch_size < 1:
        raise ValueError('plane_render_batch_size must be positive')
    with torch.no_grad():
        if independent:
            from pipeline.active import ActiveSetCache, CompactModel
            cache = ActiveSetCache()
            compacts = {owner: CompactModel(gaussians, cache.select(gaussians, owner), owner) for owner in ids}
        for start in range(0, len(eligible), batch_size):
            batch = eligible[start:start+batch_size]
            packages = render_many(batch, gaussians, pipe, background,
                                   models=[compacts[o] for o in owners[start:start+batch_size]] if independent else None,
                                   rois=rois[start:start+batch_size],
                                   return_plane=True, return_depth_normal=True)
            for offset, (camera, pkg) in enumerate(zip(batch, packages)):
                _append_camera_records(camera, pkg, [owners[start+offset]] if independent else ids, records, args, limit)
    prior = build_prior_from_samples(records, ids, args)
    prior['visibility_rays'] = _visibility_rays_from_records(records, args)
    save_dynamic_plane_prior(prior, Path(output_dir))
    return prior


def _append_camera_records(camera, pkg, ids, records, args, limit):
    import torch
    import torch.nn.functional as F
    depth = pkg['plane_depth'][0]
    device, dtype = depth.device, depth.dtype
    rx, ry, rw, rh = pkg['roi']
    labels = torch.as_tensor(camera.instance_mask, device=device).reshape(int(camera.image_height), int(camera.image_width))[ry:ry+rh, rx:rx+rw]
    if labels.shape != depth.shape:
        raise ValueError(f'{camera.image_name}: mask/depth shape mismatch')
    y, x = torch.meshgrid(torch.arange(depth.shape[0], device=device, dtype=dtype),
                          torch.arange(depth.shape[1], device=device, dtype=dtype), indexing='ij')
    local = torch.stack(((x+rx-camera.Cx)/camera.Fx, (y+ry-camera.Cy)/camera.Fy,
                         torch.ones_like(x)), dim=-1) * depth[..., None]
    transform = camera.world_view_transform.to(device=device, dtype=dtype)

    points = (local - transform[3, :3]) @ transform[:3, :3].T
    normal_cam = pkg['rendered_normal'].permute(1, 2, 0)
    normal = F.normalize(normal_cam @ transform[:3, :3].T, dim=-1)
    depth_normal = pkg['depth_normal'].permute(1, 2, 0) @ transform[:3, :3].T
    alpha = pkg['rendered_alpha'].squeeze(0).clamp(0, 1)
    surface_normal = F.normalize(depth_normal, dim=-1)
    dot = (normal * surface_normal).sum(-1)
    normal = torch.where(dot[..., None] < 0, -normal, normal)
    consistency = dot.abs()
    normal = F.normalize(normal + surface_normal, dim=-1)
    origins = camera.camera_center.to(device=device, dtype=dtype)
    rays = F.normalize(points - origins, dim=-1)
    valid = (torch.isfinite(depth) & (depth > 0) & torch.isfinite(points).all(-1)
             & (alpha >= float(getattr(args, 'plane_min_alpha', .5)))
             & (consistency >= float(getattr(args, 'plane_min_normal_consistency', .8))))
    for instance_id in ids:
        indices = torch.nonzero((valid & (labels == instance_id)).reshape(-1)).flatten()
        if not len(indices):
            continue
        if limit > 0 and len(indices) > limit:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(camera.uid)*1009+instance_id)
            indices = indices[torch.randperm(len(indices), generator=generator, device=device)[:limit]]
        def sample(value):
            trailing = value.shape[2:]
            return value.reshape((-1, *trailing))[indices].cpu().numpy()
        count = len(indices)
        records[instance_id].append({
            'points': sample(points), 'normals': sample(normal),
            'weights': sample(alpha * consistency),
            'camera_ids': np.full(count, int(camera.uid), np.int64),
            'camera_origins': np.broadcast_to(origins.cpu().numpy(), (count, 3)).copy(),
            'ray_directions': sample(rays), 'alpha': sample(alpha),
            'normal_consistency': sample(consistency),
        })
