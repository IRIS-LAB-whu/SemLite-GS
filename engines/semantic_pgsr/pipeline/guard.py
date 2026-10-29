from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class GuardTarget:
    roi: tuple
    mask: torch.Tensor
    rgb: torch.Tensor


def union_roi(first, second):
    if second is None:
        return first
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x, y = min(ax, bx), min(ay, by)
    return x, y, max(ax+aw, bx+bw)-x, max(ay+ah, by+bh)-y


def load_guard_targets(cameras, directory, workers=0):
    root = Path(directory)
    if not (root/'guard').is_dir() or not (root/'frozen_rgb').is_dir():
        raise FileNotFoundError(f'Expected guard/ and frozen_rgb/ under {root}')
    from pipeline.setup_io import ordered_prefetch
    def decode(camera):
        name = Path(camera.mask_path).name if getattr(camera, 'mask_path', None) else camera.image_name+'.png'
        guard_path, rgb_path = root/'guard'/name, root/'frozen_rgb'/name
        if not guard_path.is_file() or not rgb_path.is_file():
            raise FileNotFoundError(f'Missing frozen-context guard target for {camera.image_name}')
        resolution = (int(camera.image_width), int(camera.image_height))
        with Image.open(guard_path) as image:
            guard = np.asarray(image.convert('L').resize(resolution, Image.Resampling.NEAREST)) > 0
        yy, xx = np.nonzero(guard)
        if not len(xx):
            return camera, None, 0
        x, y = int(xx.min()), int(yy.min())
        w, h = int(xx.max())-x+1, int(yy.max())-y+1
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert('RGB').resize(resolution, Image.Resampling.BILINEAR))
        target = GuardTarget(
            (x, y, w, h), torch.from_numpy(guard[y:y+h, x:x+w].copy()),
            torch.from_numpy(rgb[y:y+h, x:x+w].copy()).permute(2, 0, 1).float().div_(255))
        return camera, target, len(xx)
    total = 0
    for camera, target, count in ordered_prefetch(decode, cameras, workers):
        camera.frozen_guard = target
        total += count
    print(f'Frozen-context guard: cameras={len(cameras)}, pixels={total}', flush=True)


def guard_for(camera, instance=None):
    return getattr(camera, '_instance_guards', {}).get(instance, getattr(camera, 'frozen_guard', None))


def guard_rgb_loss(package, camera):
    target = guard_for(camera, package.get('instance_id'))
    prediction = package['render']
    if target is None:
        return prediction.sum()*0
    x, y, w, h = package['roi']
    gx, gy, gw, gh = target.roi
    if not (x <= gx and y <= gy and x+w >= gx+gw and y+h >= gy+gh):
        raise ValueError(f'Render ROI excludes guard pixels for {camera.image_name}')
    crop = prediction[:, gy-y:gy-y+gh, gx-x:gx-x+gw]
    mask = target.mask.to(device=prediction.device)
    rgb = target.rgb.to(device=prediction.device)
    return ((crop-rgb).abs().mean(0)*mask).sum()/mask.sum().clamp_min(1)
