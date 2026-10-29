# Derived from PGSR camera perturbation; see LICENSE.md.
import numpy as np
from scene.cameras import Camera

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                        preload_img=False, data_device = "cuda")

    virtul_cam.Fx, virtul_cam.Fy = cam.Fx, cam.Fy
    virtul_cam.Cx, virtul_cam.Cy = cam.Cx, cam.Cy
    virtul_cam.projection_matrix = cam.projection_matrix.clone()
    virtul_cam.full_proj_transform = virtul_cam.world_view_transform @ virtul_cam.projection_matrix
    virtul_cam.ncc_scale = cam.ncc_scale
    virtul_cam.is_virtual = True
    return virtul_cam


def restore_colmap_principal_points(cameras, source_path):

    from pathlib import Path
    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
    from utils.graphics_utils import getProjectionMatrixCenterShift
    root = Path(source_path)/'sparse'
    if (root/'0').is_dir() and not (root/'cameras.bin').exists() and not (root/'cameras.txt').exists():
        root = root/'0'
    if (root/'cameras.bin').is_file():
        intrinsics = read_intrinsics_binary(str(root/'cameras.bin'))
    elif (root/'cameras.txt').is_file():
        intrinsics = read_intrinsics_text(str(root/'cameras.txt'))
    else:
        return
    for camera in cameras:

        intr = intrinsics[camera.colmap_id]
        if intr.model == 'SIMPLE_PINHOLE':
            fx, cx, cy = intr.params
            fy = fx
        elif intr.model == 'PINHOLE':
            fx, fy, cx, cy = intr.params
        else:
            raise ValueError(f'Undistorted pinhole cameras required, got {intr.model}')
        sx,sy = camera.image_width/intr.width,camera.image_height/intr.height
        camera.Fx,camera.Fy = float(fx*sx),float(fy*sy)
        camera.Cx,camera.Cy = float(cx*sx),float(cy*sy)
        camera.projection_matrix = getProjectionMatrixCenterShift(camera.znear,camera.zfar,
            camera.Cx,camera.Cy,camera.Fx,camera.Fy,camera.image_width,camera.image_height).T.to(camera.world_view_transform)
        camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix


class LazyCamera(Camera):

    @property
    def instance_mask(self):
        if self.mask_path is None:
            return None
        from pipeline.masks import load_mask_path
        key=(self.mask_path,self.image_width,self.image_height)
        cache=self._mask_cache
        if key not in cache:
            cache[key]=load_mask_path(self.mask_path,(self.image_width,self.image_height))
            while len(cache)>self._cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def get_image(self):
        from scene.cameras import process_image
        key=(self.image_path,self.resolution,self.ncc_scale)
        cache=self._image_cache
        if key not in cache:
            rgb,gray,_=process_image(self.image_path,self.resolution,self.ncc_scale)
            cache[key]=(rgb,gray)
            while len(cache)>self._cache_size:
                cache.popitem(last=False)
        cache.move_to_end(key)
        rgb,gray=cache[key]
        return rgb,gray


def configure_lazy_cameras(cameras, mask_root, model_path, cache_size, min_mask_area=1, io_workers=0):

    from collections import OrderedDict
    from pathlib import Path
    import json
    from PIL import Image
    from pipeline.masks import find_mask_path
    cache_path=Path(model_path)/'instance_mask_labels.json'
    metadata=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    image_cache,mask_cache=OrderedDict(),OrderedDict()
    from pipeline.setup_io import ordered_prefetch
    def prepare(camera):
        camera.__class__=LazyCamera
        camera._image_cache=image_cache
        camera._mask_cache=mask_cache
        camera._cache_size=cache_size
        camera.mask_path=str(find_mask_path(mask_root,camera)) if mask_root else None
        camera.available_instance_ids=None
        if camera.mask_path:
            path=Path(camera.mask_path)
            stat=path.stat()
            signature=[stat.st_size,stat.st_mtime_ns]
            metadata_key=f'{path}|{camera.image_width}x{camera.image_height}'
            entry=metadata.get(metadata_key)
            if not entry or entry['signature'] != signature or 'counts' not in entry:
                with Image.open(path) as image:
                    if image.mode not in ('L','P','I','I;16'):
                        raise ValueError(f'Expected integer labels: {path}')
                    resized=image.resize((camera.image_width,camera.image_height),Image.Resampling.NEAREST)
                    labels,counts=np.unique(np.asarray(resized),return_counts=True)
                    labels,counts=labels.tolist(),counts.tolist()
                entry={'signature':signature,'labels':labels,'counts':counts}

            camera.instance_pixel_counts=dict(zip(entry['labels'],entry['counts']))
            camera.available_instance_ids={label for label,count in camera.instance_pixel_counts.items() if count>=min_mask_area}
            return metadata_key, entry
        return None
    for result in ordered_prefetch(prepare, cameras, io_workers):
        if result is not None:
            metadata[result[0]] = result[1]
    cache_path.write_text(json.dumps(metadata))
