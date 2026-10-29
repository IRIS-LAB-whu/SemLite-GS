#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple, Optional
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    means2D_abs,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    all_map,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_map,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        means2D_abs,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        all_maps,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            all_maps,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.render_geo,
            raster_settings.debug,
            *_roi_args(raster_settings),
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(out_all_map, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, out_observe, out_all_map, out_plane_depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_out_observe, grad_out_all_map, grad_out_plane_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        all_map_pixels, colors_precomp, all_maps, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                all_map_pixels,
                means3D, 
                radii, 
                colors_precomp, 
                all_maps,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_all_map,
                grad_out_plane_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.render_geo,
                raster_settings.debug,
                raster_settings.image_width, raster_settings.image_height,
                *_roi_args(raster_settings))

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_means2D_abs, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, gard_all_map = _C.rasterize_gaussians_backward(*args)
        # print(f"grad_means2D {grad_means2D.sum()}, grad_means2D_abs {grad_means2D_abs.sum()}")

        grads = (
            grad_means3D,
            grad_means2D,
            grad_means2D_abs,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            gard_all_map,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    """Full-camera intrinsics with optional native output ROI.

    ROI dimensions of zero extend to the image edge. Principal coordinates
    affect plane-depth rays; projmatrix must already encode the same camera.
    Rasterization retains global 16px tile alignment, including partial edge
    tiles, so nonaligned ROI pixels match the official full-frame result.
    """
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    render_geo : bool
    debug : bool
    roi_x: int = 0
    roi_y: int = 0
    roi_width: int = 0
    roi_height: int = 0
    principal_x: Optional[float] = None
    principal_y: Optional[float] = None

def _roi_args(settings):
    x, y = settings.roi_x, settings.roi_y
    w = settings.roi_width or (settings.image_width - x)
    h = settings.roi_height or (settings.image_height - y)
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > settings.image_width or y + h > settings.image_height:
        raise ValueError("ROI must be a nonempty rectangle inside the full image")
    cx = settings.image_width * 0.5 if settings.principal_x is None else settings.principal_x
    cy = settings.image_height * 0.5 if settings.principal_y is None else settings.principal_y
    return x, y, w, h, cx, cy


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, means2D_abs, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, all_map=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if all_map is None:
            all_map = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            means2D_abs,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            all_map,
            raster_settings, 
        )



USE_FUSED_FORWARD = True


def rasterize_gaussians_batch(settings, inputs):
    """One native camera batch over shared or per-camera differentiable inputs.

    Camera outputs keep their native ROI shapes. Tile namespaces are concatenated
    before a single CUB sort; no full-image padding or per-camera count readbacks.
    """
    settings, inputs = tuple(settings), tuple(inputs)
    if len(settings) != len(inputs):
        raise ValueError('one input dictionary is required per camera')
    if not settings:
        return []
    flattened = []
    gradient_masks = []
    keys = ('means3D', 'means2D', 'means2D_abs', 'shs', 'colors_precomp',
            'opacities', 'scales', 'rotations', 'cov3D_precomp', 'all_map')
    for s, values in zip(settings, inputs):
        _roi_args(s)
        mask = values.get('gradient_mask')
        if mask is None:
            mask = values['means3D'].new_empty(0, dtype=torch.bool)
        if mask.dtype != torch.bool or mask.device != values['means3D'].device or mask.ndim != 1 or mask.numel() not in (0, len(values['means3D'])):
            raise ValueError('gradient_mask must be a device bool vector with one entry per Gaussian')
        gradient_masks.append(mask.contiguous())
        if (values.get('shs') is None) == (values.get('colors_precomp') is None):
            raise ValueError('provide exactly one of SHs or precomputed colors')
        scale, rotation, cov = (values.get(k) for k in ('scales', 'rotations', 'cov3D_precomp'))
        if ((scale is None or rotation is None) and cov is None) or ((scale is not None or rotation is not None) and cov is not None):
            raise ValueError('provide scale/rotation or precomputed covariance')
        for key in keys:
            value = values.get(key)
            flattened.append(value.contiguous() if value is not None else values['means3D'].new_empty(0))
    outputs = _RasterizeGaussiansBatch.apply(settings, tuple(gradient_masks), *flattened)
    return [tuple(outputs[i:i+5]) for i in range(0, len(outputs), 5)]


class _RasterizeGaussiansBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, settings, gradient_masks, *inputs):
        args = []
        for i, s in enumerate(settings):
            means, screen, absolute, sh, colors, opacity, scales, rotations, cov, maps = inputs[10*i:10*i+10]
            args.append((s.bg.contiguous(), means, colors, opacity, scales, rotations,
                         s.scale_modifier, cov, maps, s.viewmatrix.contiguous(),
                         s.projmatrix.contiguous(), s.tanfovx, s.tanfovy,
                         s.image_height, s.image_width, sh, s.sh_degree,
                         s.campos.contiguous(), s.prefiltered, s.render_geo, s.debug, *_roi_args(s)))
        native = _C.rasterize_gaussians_batch(args, USE_FUSED_FORWARD)
        ctx.settings = settings
        ctx.num_rendered = [value[0] for value in native]
        # args owns any contiguous camera copies until all deferred CUDA launches.
        ctx.save_for_backward(*inputs, *(t for value in native for t in (value[4], value[2], *value[6:])), *gradient_masks)
        outputs = tuple(t for value in native for t in value[1:6])
        ctx.mark_non_differentiable(*(value[j] for value in native for j in (2, 3)))
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        count = len(ctx.settings)
        saved = ctx.saved_tensors
        args = []
        for i, s in enumerate(ctx.settings):
            means, screen, absolute, sh, colors, opacity, scales, rotations, cov, maps = saved[10*i:10*i+10]
            pixels, radii, geom, binning, image = saved[10*count+5*i:10*count+5*i+5]
            grads = grad_outputs[5*i:5*i+5]
            args.append((s.bg.contiguous(), pixels, means, radii, colors, maps, scales, rotations,
                         s.scale_modifier, cov, s.viewmatrix.contiguous(), s.projmatrix.contiguous(), s.tanfovx,
                         s.tanfovy, grads[0].contiguous(), grads[3].contiguous(),
                         grads[4].contiguous(), sh, s.sh_degree, s.campos.contiguous(), geom,
                         ctx.num_rendered[i], binning, image, s.render_geo, s.debug,
                         s.image_width, s.image_height, *_roi_args(s)))
        native = _C.rasterize_gaussians_batch_backward(args, list(saved[15*count:]))
        ordered = []
        for i, value in enumerate(native):
            for original, g in zip(saved[10*i:10*i+10], (value[4], value[0], value[1], value[6], value[2], value[3], value[7], value[8], value[5], value[9])):
                ordered.append(g.reshape(original.shape) if original.numel() == 0 and g.numel() == 0 else (None if original.numel() == 0 else g))
        return (None, None, *ordered)
