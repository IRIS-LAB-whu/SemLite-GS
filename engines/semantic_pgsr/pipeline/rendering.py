# Derived from PGSR gaussian_renderer; see LICENSE.md.
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


_native = Path(__file__).resolve().parents[1] / 'submodules/diff-plane-rasterization'
sys.path.insert(0, str(_native))
from diff_plane_rasterization import GaussianRasterizationSettings, GaussianRasterizer, rasterize_gaussians_batch
from utils.graphics_utils import depth_pcd2normal
from utils.sh_utils import eval_sh


def validate_roi(camera, roi):
    width, height = int(camera.image_width), int(camera.image_height)
    if roi is None:
        return (0, 0, width, height)
    if len(roi) != 4 or any(int(v) != v for v in roi):
        raise ValueError('ROI must contain four integers (x,y,width,height)')
    x, y, w, h = map(int, roi)
    if min(x, y) < 0 or min(w, h) <= 0 or x+w > width or y+h > height:
        raise ValueError('ROI is empty or outside the camera image')
    return x, y, w, h


def pixel_rays(camera, roi, device, dtype=torch.float32):
    x, y, w, h = roi
    yy, xx = torch.meshgrid(torch.arange(y, y+h, device=device, dtype=dtype),
                            torch.arange(x, x+w, device=device, dtype=dtype), indexing='ij')
    return torch.stack(((xx-camera.Cx)/camera.Fx, (yy-camera.Cy)/camera.Fy,
                        torch.ones_like(xx)), dim=-1)


def _prepare_render(camera, model, pipe, background, roi=None, app_model=None,
           return_plane=True, return_depth_normal=True, scaling_modifier=1.0,
           override_color=None, halo=5, defer_depth_normal=False, defer_crop=False, maps_override=None):
    if int(halo) != halo or halo < 0:
        raise ValueError('halo must be a nonnegative integer')
    halo = int(halo)
    roi = validate_roi(camera, roi)
    x, y, w, h = roi
    rx, ry = max(0, x-halo), max(0, y-halo)
    rw = min(camera.image_width, x+w+halo)-rx
    rh = min(camera.image_height, y+h+halo)-ry
    raw_roi = (rx, ry, rw, rh)
    xyz = model.get_xyz
    means2d = torch.zeros_like(xyz, requires_grad=True) + 0
    means2d_abs = torch.zeros_like(xyz, requires_grad=True) + 0
    for value in (means2d, means2d_abs):
        if value.requires_grad:
            value.retain_grad()
    if 'roi_x' not in GaussianRasterizationSettings._fields:
        raise RuntimeError('Build the project-local PGSR rasterizer with scripts/build_pipeline_extensions.sh')
    settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height), image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx*.5), tanfovy=math.tan(camera.FoVy*.5),
        bg=background, scale_modifier=scaling_modifier,
        viewmatrix=camera.world_view_transform, projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree, campos=camera.camera_center,
        prefiltered=False, render_geo=return_plane, debug=pipe.debug,
        roi_x=rx, roi_y=ry, roi_width=rw, roi_height=rh,
        principal_x=float(camera.Cx), principal_y=float(camera.Cy))
    covariance = model.get_covariance(scaling_modifier) if pipe.compute_cov3D_python else None
    scales = None if covariance is not None else model.get_scaling
    rotations = None if covariance is not None else model.get_rotation
    shs, colors = None, override_color
    if colors is None:
        if pipe.convert_SHs_python:
            features = model.get_features.transpose(1, 2).reshape(-1, 3, (model.max_sh_degree+1)**2)
            direction = F.normalize(xyz-camera.camera_center, dim=-1)
            colors = (eval_sh(model.active_sh_degree, features, direction)+.5).clamp_min(0)
        else:
            shs = model.get_features
    maps = None
    if return_plane:
        from pipeline import render_inputs
        if maps_override is not None:
            maps = maps_override
        elif render_inputs.USE_FUSED_MAPS and hasattr(model, 'local_trainable'):
            maps = render_inputs.camera_maps(xyz,model.get_smallest_axis(),
                camera.world_view_transform,camera.camera_center,model.raster_gradient_mask)
        else:
            normal = model.get_normal(camera) @ camera.world_view_transform[:3, :3]
            points = xyz @ camera.world_view_transform[:3, :3] + camera.world_view_transform[3, :3]
            distance = (normal*points).sum(-1, keepdim=True).abs()
            maps = torch.cat((normal, torch.ones_like(distance), distance), dim=-1)
    inputs = dict(means3D=xyz, means2D=means2d, means2D_abs=means2d_abs, shs=shs,
                  colors_precomp=colors, opacities=model.get_opacity, scales=scales,
                  rotations=rotations, cov3D_precomp=covariance, all_map=maps)

    def finish(outputs):
        color, radii, observe, all_map, depth = outputs
        raw = {'render': color}
        if return_plane:
            raw.update(rendered_normal=all_map[:3], rendered_alpha=all_map[3:4],
                       rendered_distance=all_map[4:5], plane_depth=depth)
            if return_depth_normal and not defer_depth_normal:
                points = pixel_rays(camera, raw_roi, xyz.device, xyz.dtype)*depth.permute(1, 2, 0)
                if rh < 3 or rw < 3:
                    depth_normal = points.permute(2, 0, 1)*0
                else:
                    depth_normal = depth_pcd2normal(points).permute(2, 0, 1)
                raw['depth_normal'] = depth_normal*all_map[3:4].detach()
        if app_model is not None and model.use_app and not getattr(camera, 'is_virtual', False):
            ab = app_model.appear_ab[camera.uid]
            raw['app_image'] = ab[0].exp()*color+ab[1]
        result = dict(raw) if defer_crop else {key: value[..., y-ry:y-ry+h, x-rx:x-rx+w] for key, value in raw.items()}
        result.update(roi=roi, raw_roi=raw_roi, raw=raw,
                      viewspace_points=means2d, viewspace_points_abs=means2d_abs,
                      visibility_filter=radii > 0, radii=radii, out_observe=observe)
        return result
    return settings, inputs, finish


def render(camera, model, pipe, background, roi=None, app_model=None,
           return_plane=True, return_depth_normal=True, scaling_modifier=1.0,
           override_color=None, halo=5):
    settings, inputs, finish = _prepare_render(
        camera, model, pipe, background, roi=roi, app_model=app_model,
        return_plane=return_plane, return_depth_normal=return_depth_normal,
        scaling_modifier=scaling_modifier, override_color=override_color, halo=halo)
    return finish(GaussianRasterizer(settings)(**inputs))


USE_SELECTIVE_NORMALS = True
USE_ACTIVE_BACKWARD = True


def render_many(cameras, model, pipe, background, rois=None, models=None, depth_normal_mask=None, **kwargs):

    cameras = list(cameras)
    rois = [None] * len(cameras) if rois is None else list(rois)
    if len(rois) != len(cameras):
        raise ValueError('one ROI is required per camera')
    models = [model] * len(cameras) if models is None else list(models)
    if len(models) != len(cameras):
        raise ValueError('one compact model is required per camera')
    normal_mask = [True] * len(cameras) if depth_normal_mask is None or not USE_SELECTIVE_NORMALS else list(depth_normal_mask)
    if len(normal_mask) != len(cameras):
        raise ValueError('one depth-normal selection is required per camera')
    batch_normals = kwargs.get('return_plane',True) and kwargs.get('return_depth_normal',True)
    fused_crop = os.environ.get('PGSR_BATCH_CROPS','0') == '1'
    from pipeline import render_inputs
    fused_maps = (os.environ.get('PGSR_BATCH_MAPS','0') == '1' and render_inputs.USE_FUSED_MAPS
                  and kwargs.get('return_plane',True) and 0 < len(models) <= 8
                  and all(hasattr(m,'local_trainable') and m.get_xyz.is_cuda and m.get_xyz.dtype == torch.float32 for m in models)
                  and all(not (c.world_view_transform.requires_grad or c.camera_center.requires_grad) for c in cameras))
    maps = [None]*len(models)
    if fused_maps:
        from pipeline.view_batch import maps_many
        maps = maps_many(cameras,models)
    prepared = [_prepare_render(camera, local_model, pipe, background, roi=roi,
                defer_depth_normal=batch_normals, defer_crop=fused_crop, maps_override=mp, **kwargs)
                for camera, local_model, roi, mp in zip(cameras, models, rois, maps)]
    if USE_ACTIVE_BACKWARD:
        for local_model, item in zip(models, prepared):
            if hasattr(local_model, 'raster_gradient_mask'):
                item[1]['gradient_mask'] = local_model.raster_gradient_mask
    outputs = rasterize_gaussians_batch([p[0] for p in prepared], [p[1] for p in prepared])
    packages = [p[2](output) for p, output in zip(prepared, outputs)]
    normal_items = [(c,p) for c,p,needed in zip(cameras,packages,normal_mask) if needed]
    if batch_normals and normal_items:
        heights=[p['raw_roi'][3] for c,p in normal_items]
        widths=[p['raw_roi'][2] for c,p in normal_items]
        h,w=max(heights),max(widths)
        points=[]
        for camera,p in normal_items:
            depth=p['raw']['plane_depth']
            xyz=pixel_rays(camera,p['raw_roi'],depth.device,depth.dtype)*depth.permute(1,2,0)
            points.append(F.pad(xyz.permute(2,0,1),(0,w-xyz.shape[1],0,h-xyz.shape[0])))
        xyz=torch.stack(points).permute(0,2,3,1)
        if h>=3 and w>=3:
            normal=F.normalize(torch.cross(xyz[:,1:-1,2:]-xyz[:,1:-1,:-2],
                                          xyz[:,:-2,1:-1]-xyz[:,2:,1:-1],dim=-1),dim=-1).permute(0,3,1,2)
        for i,(camera,p) in enumerate(normal_items):
            rh,rw=heights[i],widths[i]
            raw_normal=(F.pad(normal[i,:,:rh-2,:rw-2],(1,1,1,1)) if rh>=3 and rw>=3 else points[i][:,:rh,:rw]*0)
            raw_normal=raw_normal*p['raw']['rendered_alpha'].detach()
            p['raw']['depth_normal']=raw_normal
            x,y,lw,lh=p['roi'];rx,ry,_,_=p['raw_roi']
            p['depth_normal']=raw_normal if fused_crop else raw_normal[:,y-ry:y-ry+lh,x-rx:x-rx+lw]
    if fused_crop:
        from pipeline.view_batch import crop_packages
        crop_packages(packages)
    return packages
