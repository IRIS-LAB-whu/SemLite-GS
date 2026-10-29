from argparse import ArgumentParser
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch

from arguments import ModelParams, PipelineParams
from pipeline.masks import mask_roi
from pipeline.model import GaussianModel
from pipeline.rendering import render
from pipeline.scene import Scene


def make_parser():
    parser = ArgumentParser(description='Fuse instance depth into a signed, weighted TSDF field.')
    model, pipeline = ModelParams(parser), PipelineParams(parser)
    parser.add_argument('--instance_masks', required=True,
                        help='Directory of integer PNG/TIFF labels, one per source image')
    parser.add_argument('--instance_id', required=True, type=int,
                        help='Label in the camera masks (not the DOM one-based ID)')
    parser.add_argument('--model_instance_id', type=int, default=None,
                        help='Optional Gaussian owner ID in the checkpoint')
    parser.add_argument('--iteration', type=int, default=-1)
    parser.add_argument('--voxel_size', type=float, default=0.3,
                        help='TSDF voxel edge length in scene/world units')
    parser.add_argument('--depth_sampling_stride', type=int, default=1,
                        help='Open3D depth stride for TSDF volume block discovery')
    parser.add_argument('--sdf_trunc', type=float, default=0,
                        help='TSDF truncation distance; default is 4 times voxel_size')
    parser.add_argument('--max_depth', type=float, default=1000,
                        help='Maximum camera-space depth in scene/world units')
    parser.add_argument('--min_alpha', type=float, default=0.5)
    parser.add_argument('--min_mask_area', type=int, default=200)
    parser.add_argument('--mask_erosion', type=int, default=0,
                        help='Erode mask this many pixels before fusion')
    parser.add_argument('--use_depth_filter', action='store_true',
                        help='Reject PGSR depths with grazing depth normals')
    parser.add_argument('--max_views', type=int, default=0,
                        help='Maximum number of eligible views; zero uses all')
    parser.add_argument('--output_dir', required=True)
    return parser, model, pipeline


def process_depth(depth, alpha, labels, instance_id, max_depth, min_alpha,
                  erosion=0, depth_normal=None, ray_dirs=None):

    depth = np.asarray(depth, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    mask = np.asarray(labels) == instance_id
    if depth.shape != mask.shape or alpha.shape != mask.shape:
        raise ValueError('Depth, alpha and instance mask dimensions differ')
    if erosion:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=erosion).astype(bool)
    valid = mask & np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    valid &= np.isfinite(alpha) & (alpha >= min_alpha)
    if depth_normal is not None:
        normal = np.asarray(depth_normal, dtype=np.float32)
        rays = np.asarray(ray_dirs, dtype=np.float32)
        if normal.shape != depth.shape + (3,) or rays.shape != normal.shape:
            raise ValueError('Normal/ray dimensions differ from depth')
        normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-8)
        valid &= np.abs(np.sum(normal * rays, axis=-1)) >= np.cos(np.deg2rad(80))
    return np.ascontiguousarray(np.where(valid, depth, 0).astype(np.float32)), int(valid.sum())


def fuse(model, scene, pipeline, args):
    if args.voxel_size <= 0 or args.max_depth <= 0 or not 0 <= args.min_alpha <= 1:
        raise ValueError('voxel_size and max_depth must be positive; min_alpha must be in [0,1]')
    if args.mask_erosion < 0 or args.max_views < 0 or args.min_mask_area < 1:
        raise ValueError('mask_erosion/max_views must be nonnegative; min_mask_area positive')
    if args.depth_sampling_stride < 1:
        raise ValueError('depth_sampling_stride must be positive')
    trunc = args.sdf_trunc or 4 * args.voxel_size
    if trunc <= 0:
        raise ValueError('sdf_trunc must be positive')
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=args.depth_sampling_stride)
    background = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0],
                              dtype=torch.float32, device='cuda')
    eligible = [cam for cam in scene.getTrainCameras()
                if cam.instance_pixel_counts.get(args.instance_id, 0) >= args.min_mask_area]
    if args.max_views:
        eligible = eligible[:args.max_views]
    if not eligible:
        raise ValueError(f'No train camera has at least {args.min_mask_area} pixels of instance {args.instance_id}')
    render_model = model
    if model.context_mode == 'independent':
        from pipeline.active import ActiveSetCache, CompactModel
        owner = args.instance_id if args.model_instance_id is None else args.model_instance_id
        with torch.no_grad():
            render_model = CompactModel(model, ActiveSetCache().select(model, owner), owner)
    owner = args.instance_id if args.model_instance_id is None else args.model_instance_id
    if not bool((model.instance_ids == owner).any()):
        raise ValueError(f'Checkpoint has no Gaussian owner {owner}; check the mask label mapping')
    records = []
    for index, camera in enumerate(eligible, 1):
        mask = camera.instance_mask == args.instance_id
        roi = mask_roi(mask)
        x, y, width, height = roi
        with torch.no_grad():
            result = render(camera, render_model, pipeline, background, roi=roi,
                            return_plane=True, return_depth_normal=args.use_depth_filter, halo=0)
        depth = result['plane_depth'].squeeze(0).detach().cpu().numpy()
        alpha = result['rendered_alpha'].squeeze(0).detach().cpu().numpy()
        color = (result['render'].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        normal = result['depth_normal'].permute(1, 2, 0).cpu().numpy() if args.use_depth_filter else None
        rays = None
        if args.use_depth_filter:
            from pipeline.rendering import pixel_rays
            rays = pixel_rays(camera, roi, 'cpu').numpy()
        fused_depth, valid_count = process_depth(
            depth, alpha, mask[y:y+height, x:x+width].numpy(), True,
            args.max_depth, args.min_alpha, args.mask_erosion, normal, rays)
        if valid_count:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(fused_depth), depth_scale=1.0,
                depth_trunc=args.max_depth, convert_rgb_to_intensity=False)
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width, height, camera.Fx, camera.Fy, camera.Cx-x, camera.Cy-y)
            world_to_camera = camera.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
            volume.integrate(rgbd, intrinsics, world_to_camera)
        records.append({'image': camera.image_name, 'roi': list(roi),
                        'mask_pixels': int(mask.sum()), 'valid_depth_pixels': valid_count})
        print(f'[{index}/{len(eligible)}] {camera.image_name}: {valid_count} valid depth pixels', flush=True)
        del result
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if o3d.__version__ not in ('0.18.0','0.19.0'):
        raise RuntimeError('Legacy field bridge requires Open3D 0.18/0.19')
    from semlite_gs.reconstruction import native_field
    field = native_field.export_field(volume)
    field['backend'] = np.array('open3d_legacy_scalable')
    field['minimum_surface_weight'] = np.array(0.)
    field['open3d_version'] = np.array(o3d.__version__)
    np.savez_compressed(output/'observed_tsdf.npz', **field)
    if not len(field['phi']):
        raise RuntimeError('Fusion produced no observed voxels')
    report = {'model': str(Path(args.model_path).resolve()), 'iteration': scene.loaded_iter,
              'instance_id': args.instance_id, 'model_instance_id': args.model_instance_id,
              'context_mode': model.context_mode, 'render_points': len(render_model.get_xyz),
              'voxel_size': args.voxel_size, 'sdf_trunc': trunc,
              'depth_sampling_stride': args.depth_sampling_stride,
              'max_depth': args.max_depth, 'min_alpha': args.min_alpha,
              'eligible_views': len(eligible),
              'integrated_views': sum(bool(row['valid_depth_pixels']) for row in records),
              'observed_voxels': len(field['phi']), 'views': records,
              'dense_mesh_extracted': False, 'field': str(output/'observed_tsdf.npz')}
    (output/'tsdf_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != 'views'}, indent=2))
    return report


def main():
    parser, model_args, pipe_args = make_parser()
    args = parser.parse_args()
    args.instance_ids = str(args.instance_id)
    args.image_cache_size = 4
    args.min_instance_mask_area = args.min_mask_area
    args.mask_background_id = 0
    args.building_ply = ''
    args.freeze_global_outside_instances = False
    masks = Path(args.instance_masks)
    if not masks.is_absolute():
        masks = Path(args.source_path)/masks
    if not masks.is_dir():
        parser.error(f'Instance mask directory does not exist: {masks}')
    dataset = model_args.extract(args)
    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, args, load_iteration=args.iteration, shuffle=False)
    fuse(model, scene, pipe_args.extract(args), args)


if __name__ == '__main__':
    main()
