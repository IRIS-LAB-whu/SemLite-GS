import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from pipeline.model import GaussianModel
from pipeline.scene import Scene
from pipeline.train import make_parser
from pipeline.masks import mask_roi
from pipeline.rendering import render, render_many
from pipeline.setup_io import ordered_prefetch, bounded_writer, camera_batches


def subset_model(source, selected):

    result = GaussianModel(source.max_sh_degree)
    for name in ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation'):
        setattr(result, name, getattr(source, name).detach()[selected])
    result.active_sh_degree = source.active_sh_degree
    result.use_app = False
    return result


def make_guard(edge, outer, frozen_rgb, frozen_alpha, frozen_depth,
               train_alpha, train_depth, train_roi, ground_truth, args):

    candidate = ((frozen_alpha >= args.min_frozen_alpha) &
                 ((frozen_rgb-ground_truth).abs().mean(0) <= args.max_rgb_error))
    guard = outer & candidate
    if train_roi is not None:
        x, y, w, h = train_roi
        in_front = ((train_alpha > .01) & torch.isfinite(train_depth) &
                    (frozen_depth[y:y+h, x:x+w] + args.depth_margin < train_depth))
        inner = (edge[y:y+h, x:x+w] & candidate[y:y+h, x:x+w] &
                 ((train_alpha < args.max_trainable_alpha) | in_front))
        guard[y:y+h, x:x+w] |= inner
    return guard


def generate(args, model_params, pipeline_params):
    if not args.init_ply or not args.instance_masks or not args.freeze_global_outside_instances:
        raise ValueError('Guard generation requires init_ply, instance_masks and freeze_global_outside_instances')
    if args.edge_pixels < 0 or args.guard_outer_pixels < 0 or args.max_cameras < 0:
        raise ValueError('Guard radii and max_cameras must be nonnegative')
    if not (0 <= args.min_frozen_alpha <= 1 and 0 <= args.max_trainable_alpha <= 1 and
            args.max_rgb_error >= 0 and args.depth_margin >= 0):
        raise ValueError('Invalid guard alpha, RGB or depth threshold')
    batch_size = getattr(args, 'render_batch_size', 1)
    io_workers = getattr(args, 'io_workers', 0)
    camera_workers = getattr(args, 'camera_io_workers', 0)
    if batch_size < 1 or min(io_workers, camera_workers) < 0:
        raise ValueError('Positive render batch size and nonnegative IO workers required')
    started = time.perf_counter()
    model = GaussianModel(args.sh_degree)
    scene = Scene(model_params.extract(args), model, args)
    frozen = ~model.trainable_mask
    if not frozen.any() or not model.trainable_mask.any():
        raise ValueError('Guard generation needs both frozen and target Gaussians')
    cameras = sorted((c for c in scene.getTrainCameras()
                      if any(owner in c.available_instance_ids for owner in scene.instance_ids)),
                     key=lambda c: c.image_name)
    if args.max_cameras:
        cameras = cameras[:args.max_cameras]
    if not cameras:
        raise ValueError('No camera contains the selected instance mask labels')
    frozen_model = subset_model(model, frozen)
    train_model = subset_model(model, ~frozen)
    pipe = pipeline_params.extract(args)
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    output = Path(args.output_dir)
    for name in ('guard', 'frozen_rgb'):
        (output/name).mkdir(parents=True, exist_ok=True)
    rows = []

    def read_camera(camera):
        if not camera_workers:
            return camera, camera.instance_mask, camera.get_image()[0]
        from pipeline.masks import load_mask_path
        from scene.cameras import process_image
        labels = load_mask_path(camera.mask_path,(camera.image_width,camera.image_height))
        rgb = process_image(camera.image_path,camera.resolution,camera.ncc_scale)[0]
        return camera, labels, rgb
    def save_images(filename, guard, rgb):
        Image.fromarray(guard).save(output/'guard'/filename)
        Image.fromarray(rgb, mode='RGB').save(output/'frozen_rgb'/filename)
    torch.cuda.synchronize()
    render_started = time.perf_counter()
    stream = ordered_prefetch(read_camera,cameras,camera_workers)
    with torch.no_grad(), bounded_writer(io_workers) as submit:
        for batch in camera_batches(stream,batch_size,lambda p:p[0]):
            cams = [p[0] for p in batch]
            fixed_batch = ([render(cams[0],frozen_model,pipe,background,
                                  return_plane=True,return_depth_normal=False)] if batch_size == 1 else
                           render_many(cams,frozen_model,pipe,background,
                                       return_plane=True,return_depth_normal=False))
            for (camera, labels, gt), fixed in zip(batch,fixed_batch):
                labels = labels.to('cuda')
                mask = torch.zeros_like(labels,dtype=torch.bool)
                for owner in scene.instance_ids:
                    mask |= labels == owner
                edge = (mask & (F.max_pool2d((~mask)[None,None].float(),
                        2*args.edge_pixels+1,1,args.edge_pixels)[0,0]>0)
                        if args.edge_pixels else torch.zeros_like(mask))
                outer = ((F.max_pool2d(mask[None,None].float(),
                         2*args.guard_outer_pixels+1,1,args.guard_outer_pixels)[0,0]>0)&~mask
                         if args.guard_outer_pixels else torch.zeros_like(mask))
                moving = (render(camera,train_model,pipe,background,roi=mask_roi(edge),
                                 return_plane=True,return_depth_normal=False)
                          if args.edge_pixels and bool(edge.any()) else None)
                guard = make_guard(edge,outer,fixed['render'],fixed['rendered_alpha'][0],
                    fixed['plane_depth'][0],
                    moving['rendered_alpha'][0] if moving is not None else None,
                    moving['plane_depth'][0] if moving is not None else None,
                    moving['roi'] if moving is not None else None,gt.to('cuda'),args)
                filename = Path(camera.mask_path).name
                guard_array = guard.byte().mul(255).cpu().numpy()
                rgb = fixed['render'].clamp(0,1).permute(1,2,0).mul(255).byte().cpu().numpy()
                submit(save_images,filename,guard_array,rgb)
                counts = torch.stack((mask.sum(),edge.sum(),guard.sum(),(guard&outer).sum())).cpu().tolist()
                row = dict(zip(('mask_pixels','edge_pixels','guard_pixels','outer_guard_pixels'),counts))
                row['camera'] = camera.image_name
                rows.append(row)
                print(f'Guard {len(rows)}/{len(cameras)} {camera.image_name}: {row["guard_pixels"]} pixels',flush=True)
                del moving, guard, gt
            del fixed_batch, fixed
    torch.cuda.synchronize()
    render_io_seconds = time.perf_counter()-render_started
    summary = {'render_batch_size': batch_size, 'io_workers': io_workers,
               'camera_io_workers': camera_workers, 'render_io_seconds': render_io_seconds,
               'process_seconds': time.perf_counter()-started, 'renderer': 'PGSR three-scale native rasterizer',
               'source_path': str(Path(args.source_path).resolve()),
               'instance_masks': str(Path(args.instance_masks).resolve()),
               'building_ply': str(Path(args.building_ply).resolve()) if args.building_ply else '',
               'resolution': args.resolution, 'white_background': args.white_background,
               'init_ply': str(Path(args.init_ply).resolve()),
               'init_ply_translation': args.init_ply_translation, 'instance_ids': scene.instance_ids,
               'cameras': len(rows), 'trainable_points': int((~frozen).sum()),
               'frozen_points': int(frozen.sum()), 'guard_pixels': sum(r['guard_pixels'] for r in rows),
               'edge_pixels': args.edge_pixels, 'guard_outer_pixels': args.guard_outer_pixels,
               'min_frozen_alpha': args.min_frozen_alpha, 'max_rgb_error': args.max_rgb_error,
               'max_trainable_alpha': args.max_trainable_alpha, 'depth_margin': args.depth_margin,
               'per_camera': rows}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Wrote PGSR guard targets: {output} ({len(rows)} cameras, {summary["guard_pixels"]} pixels)', flush=True)
    return summary


def main():
    parser, model, _, pipeline = make_parser()
    parser.description = __doc__
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--render_batch_size', type=int, default=1)
    parser.add_argument('--io_workers', type=int, default=0)
    parser.add_argument('--edge_pixels', type=int, default=0)
    parser.add_argument('--guard_outer_pixels', type=int, default=16)
    parser.add_argument('--min_frozen_alpha', type=float, default=.6)
    parser.add_argument('--max_rgb_error', type=float, default=.1)
    parser.add_argument('--max_trainable_alpha', type=float, default=.5)
    parser.add_argument('--depth_margin', type=float, default=2.)
    parser.add_argument('--max_cameras', type=int, default=0)
    args = parser.parse_args()
    generate(args, model, pipeline)


if __name__ == '__main__':
    main()
