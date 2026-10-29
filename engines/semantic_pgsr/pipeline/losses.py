# Derived from PGSR train.py and utils/loss_utils.py; see LICENSE.md.
from collections import OrderedDict

import torch
import torch.nn.functional as F
from utils.loss_utils import ssim2, lncc
from pipeline.rendering import pixel_rays


def crop(image, roi):
    x, y, w, h = roi
    return image[..., y:y+h, x:x+w]


def local_mask(mask, package):
    shape = package['render'].shape[-2:]
    if mask is None:
        return torch.ones(shape, device=package['render'].device, dtype=torch.bool)
    from pipeline.supervision import ROIMask
    if isinstance(mask, ROIMask):
        x,y,w,h = package['roi']
        mx,my,mw,mh = mask.roi
        if x < mx or y < my or x+w > mx+mw or y+h > my+mh:
            raise ValueError('Requested loss rectangle lies outside cached mask')
        return crop(mask.tensor,(x-mx,y-my,w,h)).to(package['render'].device)
    return crop(mask.reshape(mask.shape[-2:]), package['roi']).to(package['render'].device).bool()


def _mean(values, mask):
    return torch.where(mask, values, 0).sum()/mask.sum().clamp_min(1)


def _image_weight(image):


    if min(image.shape[-2:]) < 3:
        return torch.ones_like(image[0])
    dx = (image[:, 1:-1, 2:]-image[:, 1:-1, :-2]).abs().mean(0)
    dy = (image[:, :-2, 1:-1]-image[:, 2:, 1:-1]).abs().mean(0)
    grad = torch.maximum(dx, dy)
    grad = (grad-grad.min())/(grad.max()-grad.min()).clamp_min(1e-8)
    return (1-F.pad(grad, (1, 1, 1, 1), value=1)).clamp(0, 1).square().detach()


_IMAGE_WEIGHTS = OrderedDict()


def cached_image_weight(camera, image):
    key = (id(camera), image.data_ptr(), image._version, tuple(image.shape))
    if key in _IMAGE_WEIGHTS:
        value = _IMAGE_WEIGHTS.pop(key)
    else:
        value = _image_weight(image)
    _IMAGE_WEIGHTS[key] = value
    while len(_IMAGE_WEIGHTS) > 2:
        _IMAGE_WEIGHTS.popitem(last=False)
    return value


def per_view_loss(camera, package, model, opt, iteration, mask=None):
    from pipeline.supervision import supervision_for
    entry = (None if package.get('_disable_supervision_cache') else
             supervision_for(camera, package.get('instance_id')))
    gt = None if entry is not None else camera.get_image()[0]
    selected = local_mask(mask, package)
    rgb = package['render']
    raw = package.get('raw', package)
    raw_roi = package.get('raw_roi', package['roi'])


    if entry is None:
        gt_raw = crop(gt, raw_roi).to(rgb.device)
    else:
        rx,ry,rw,rh = raw_roi
        ex,ey,ew,eh = entry.raw_roi
        if rx < ex or ry < ey or rx+rw > ex+ew or ry+rh > ey+eh:
            raise ValueError('Requested SSIM halo exceeds cached supervision')
        gt_raw = crop(entry.rgb,(rx-ex,ry-ey,rw,rh)).to(rgb.device)
    similarity = ssim2(raw['render'], gt_raw)
    x, y, w, h = package['roi']
    rx, ry, _, _ = raw_roi
    gt_local = gt_raw[..., y-ry:y-ry+h, x-rx:x-rx+w]
    similarity = similarity[y-ry:y-ry+h, x-rx:x-rx+w]
    ssim_loss = _mean(1-similarity, selected)
    exposed = (torch.where(ssim_loss.detach() < .5, package['app_image'], rgb)
               if 'app_image' in package else rgb)
    photo = (1-opt.lambda_dssim)*_mean((exposed-gt_local).abs().mean(0), selected)
    loss = photo+opt.lambda_dssim*ssim_loss
    visibility = package['visibility_filter'].clone()
    trainable = getattr(model, 'trainable_mask', None)
    if trainable is not None:
        visibility &= trainable
    if 'instance_id' in package and hasattr(model, 'instance_ids'):
        visibility &= model.instance_ids == package['instance_id']
    scale = _mean(model.get_scaling.min(dim=-1).values, visibility)
    loss = loss+opt.scale_loss_weight*scale*selected.any().to(scale.dtype)
    if iteration > opt.single_view_weight_from_iter:
        delta = (package['depth_normal']-package['rendered_normal']).abs().sum(0)
        if not opt.wo_image_weight:
            if entry is None:
                weight = crop(cached_image_weight(camera, gt), package['roi'])
            else:
                ex,ey,_,_ = entry.roi
                weight = crop(entry.weight,(x-ex,y-ey,w,h))
            delta = delta*weight.to(delta.device)
        loss = loss+opt.single_view_weight*_mean(delta, selected)
    return loss


def _grid(pixels, width, height):
    return torch.stack((2*pixels[..., 0]/max(width-1, 1)-1,
                        2*pixels[..., 1]/max(height-1, 1)-1), dim=-1)


def _sample(image, pixels):
    height, width = image.shape[-2:]
    return F.grid_sample(image.reshape(1, -1, height, width),
                         _grid(pixels, width, height).reshape(1, -1, 1, 2),
                         align_corners=True).reshape(image.numel()//(height*width), -1).T


def _intrinsic(camera, device, dtype, scale=1):
    return torch.tensor([[camera.Fx/scale, 0, camera.Cx/scale],
                         [0, camera.Fy/scale, camera.Cy/scale],
                         [0, 0, 1]], device=device, dtype=dtype)


def pair_loss(camera, near, package, near_package, model, opt, mask=None):


    depth = package['plane_depth'][0]
    device, dtype = depth.device, depth.dtype
    zero = depth.sum()*0
    x, y, w, h = package['roi']
    rays = pixel_rays(camera, package['roi'], device, dtype)
    points = (rays*depth[..., None]).reshape(-1, 3)
    view = camera.world_view_transform
    nview = near.world_view_transform
    world = (points-view[3, :3]) @ view[:3, :3].T
    nearest = world @ nview[:3, :3]+nview[3, :3]
    projected = torch.stack((nearest[:, 0]*near.Fx/nearest[:, 2].clamp_min(1e-8)+near.Cx,
                             nearest[:, 1]*near.Fy/nearest[:, 2].clamp_min(1e-8)+near.Cy), -1)
    nx, ny, nw, nh = near_package['roi']
    local = projected-projected.new_tensor([nx, ny])
    mapped = _sample(near_package['plane_depth'], local)[:, 0]
    valid = (local[:, 0] >= 0) & (local[:, 0] <= nw-1) & (local[:, 1] >= 0) & (local[:, 1] <= nh-1)
    valid &= (nearest[:, 2] > 0) & (mapped > 0) & (depth.reshape(-1) > 0)
    valid &= local_mask(mask, package).reshape(-1)
    if 'loss_mask' in near_package:
        valid &= _sample(near_package['loss_mask'].float(), local)[:, 0] > .999

    nearest_back = nearest/nearest[:, 2:3].clamp_min(1e-8)*mapped[:, None]
    world_back = (nearest_back-nview[3, :3]) @ nview[:3, :3].T
    reference = world_back @ view[:3, :3]+view[3, :3]
    roundtrip = torch.stack((reference[:, 0]*camera.Fx/reference[:, 2].clamp_min(1e-8)+camera.Cx,
                             reference[:, 1]*camera.Fy/reference[:, 2].clamp_min(1e-8)+camera.Cy), -1)
    yy, xx = torch.meshgrid(torch.arange(y, y+h, device=device, dtype=dtype),
                            torch.arange(x, x+w, device=device, dtype=dtype), indexing='ij')
    pixels = torch.stack((xx, yy), -1).reshape(-1, 2)
    noise = torch.linalg.vector_norm(roundtrip-pixels, dim=-1)
    valid &= (reference[:, 2] > 0) & torch.isfinite(noise)
    if not opt.wo_use_geo_occ_aware:
        valid &= noise < opt.multi_view_pixel_noise_th
        weights = torch.exp(-noise).detach()
    else:
        weights = torch.ones_like(noise)
    if not valid.any():
        return zero
    loss = opt.multi_view_geo_weight*(weights[valid]*noise[valid]).mean()
    ids = valid.nonzero(as_tuple=True)[0]
    if ids.numel() > opt.multi_view_sample_num:
        ids = ids[torch.randperm(ids.numel(), device=device)[:opt.multi_view_sample_num]]
    if ids.numel() == 0 or opt.multi_view_ncc_weight == 0:
        return loss
    radius = opt.multi_view_patch_size
    offsets = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    offsets = torch.stack((ox, oy), -1).reshape(1, -1, 2)
    ref_scale, near_scale = float(camera.ncc_scale), float(near.ncc_scale)
    ref_pixels = pixels[ids, None]/ref_scale+offsets
    normals = package['rendered_normal'].permute(1, 2, 0).reshape(-1, 3)[ids]
    distance = package['rendered_distance'].reshape(-1)[ids]
    rotation = nview[:3, :3].T @ view[:3, :3]
    translation = -rotation @ view[3, :3]+nview[3, :3]
    homography = rotation[None]-translation[None, :, None]*normals[:, None, :]/distance.clamp_min(1e-8)[:, None, None]
    homography = _intrinsic(near, device, dtype, near_scale)[None] @ homography @ torch.linalg.inv(_intrinsic(camera, device, dtype, ref_scale))
    homogeneous = torch.cat((ref_pixels, torch.ones_like(ref_pixels[..., :1])), -1)
    warped = homogeneous @ homography.transpose(-1, -2)
    denom = warped[..., 2:3]
    safe_denom = torch.where(denom.abs() > 1e-8, denom, torch.full_like(denom, 1e-8))
    near_pixels = warped[..., :2]/safe_denom
    from pipeline.supervision import supervision_gray
    reference_gray = supervision_gray(camera)
    nearest_gray = supervision_gray(near)
    reference_gray = reference_gray.to(device)
    nearest_gray = nearest_gray.to(device)
    rh, rw = reference_gray.shape[-2:]
    nh, nw = nearest_gray.shape[-2:]
    inside = ((ref_pixels[..., 0] >= 0) & (ref_pixels[..., 0] <= rw-1) &
              (ref_pixels[..., 1] >= 0) & (ref_pixels[..., 1] <= rh-1) &
              (near_pixels[..., 0] >= 0) & (near_pixels[..., 0] <= nw-1) &
              (near_pixels[..., 1] >= 0) & (near_pixels[..., 1] <= nh-1) &
              torch.isfinite(near_pixels).all(-1)).all(-1) & (distance > 1e-8)
    if inside.any():
        ref_values = _sample(reference_gray, ref_pixels[inside]).reshape(int(inside.sum()), -1)
        near_values = _sample(nearest_gray, near_pixels[inside]).reshape(int(inside.sum()), -1)
        ncc, ncc_valid = lncc(ref_values, near_values)
        loss = loss+opt.multi_view_ncc_weight*_mean(ncc[:, 0]*weights[ids][inside], ncc_valid[:, 0])
    return loss
