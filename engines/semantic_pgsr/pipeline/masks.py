from pathlib import Path
import numpy as np
import torch
from PIL import Image


def find_mask_path(root, camera):
    names = [camera.image_name, Path(camera.image_name).stem, Path(camera.image_path).name]
    candidates = [Path(root) / (name + ext) for name in names for ext in ('', '.png', '.tif', '.tiff')]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f'No instance mask for {camera.image_name} in {root}')
    return path


def load_mask_path(path, resolution):
    with Image.open(path) as im:
        if im.mode not in ('L', 'I', 'I;16', 'P'):
            raise ValueError(f'Expected integer label mask, got {im.mode}: {path}')
        labels = np.asarray(im.resize(resolution, Image.Resampling.NEAREST))
        if labels.dtype == np.uint16:
            labels = labels.astype(np.int32)
    return torch.from_numpy(labels.copy())


def load_instance_mask(root, camera):
    return load_mask_path(find_mask_path(root,camera),(camera.image_width,camera.image_height))


def mask_roi(mask):
    y, x = torch.where(mask)
    if not len(x):
        raise ValueError('Cannot render an empty instance mask')
    x0, y0 = int(x.min()), int(y.min())
    return x0, y0, int(x.max())-x0+1, int(y.max())-y0+1


def infer_instance_ids(cameras, background_id=255):
    labels=set()
    for camera in cameras:
        available=getattr(camera,'available_instance_ids',None)
        if available is not None:
            labels.update(available)
        elif camera.instance_mask is not None:
            labels.update(int(v) for v in camera.instance_mask.unique())
    return sorted(value for value in labels if value != background_id)
