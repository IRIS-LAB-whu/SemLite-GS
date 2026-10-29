#!/usr/bin/env python3


import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from pipeline.generate_guard import subset_model
from pipeline.guard import load_guard_targets, union_roi
from pipeline.masks import mask_roi
from pipeline.model import GaussianModel
from pipeline.rendering import render
from pipeline.scene import Scene
from pipeline.train import make_parser


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser,model_args,_,pipe_args=make_parser()
    parser.add_argument('--guard_dir',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--max_pairs',type=int,default=0)
    args=parser.parse_args()
    output=Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('Use an empty output directory')
    output.mkdir(parents=True,exist_ok=True)
    guard_summary=json.loads((Path(args.guard_dir)/'summary.json').read_text())
    if (guard_summary.get('edge_pixels'),guard_summary.get('guard_outer_pixels'))!=(0,16):
        raise ValueError('Collector requires 0/16 guard files')
    model=GaussianModel(args.sh_degree)
    scene=Scene(model_args.extract(args),model,args)
    frozen=~model.trainable_mask
    if not bool(frozen.any()) or not bool((~frozen).any()):raise ValueError('Expected both frozen and trainable points')
    cameras=sorted((c for c in scene.getTrainCameras() if any(i in c.available_instance_ids for i in scene.instance_ids)),key=lambda c:c.image_name)
    load_guard_targets(cameras,args.guard_dir,args.guard_io_workers)
    pairs=[(c,i) for c in cameras for i in scene.instance_ids if i in c.available_instance_ids]
    if args.max_pairs:pairs=pairs[:args.max_pairs]
    freeze_idx=torch.where(frozen)[0]
    frozen_model=subset_model(model,frozen)
    colors=torch.ones((len(freeze_idx),3),device='cuda',requires_grad=True)
    contributors=torch.zeros(len(freeze_idx),device='cuda',dtype=torch.bool)
    tiles=torch.zeros_like(contributors)
    pipe=pipe_args.extract(args)
    bg=torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    rows=[]
    started=time.monotonic()
    for n,(camera,iid) in enumerate(pairs,1):
        mask=camera.instance_mask==iid
        roi=union_roi(mask_roi(mask),getattr(getattr(camera,'frozen_guard',None),'roi',None))
        pkg=render(camera,frozen_model,pipe,bg,roi=roi,return_plane=False,
                   return_depth_normal=False,override_color=colors)
        derivative,=torch.autograd.grad(pkg['raw']['render'].sum(),colors)
        if not torch.isfinite(derivative).all() or (derivative<0).any():
            raise RuntimeError(f'Invalid colour derivative on {camera.image_name}')
        hits=derivative.ne(0).any(-1)
        contributors|=hits
        tile=pkg['radii']>0
        tiles|=tile
        rows.append(dict(camera=camera.image_name,instance=iid,contributors=int(hits.sum()),
                         tile_candidates=int(tile.sum()),roi=roi))
        del pkg,derivative,hits,tile
        if n%50==0 or n==len(pairs):
            state=dict(state='collecting',pairs_done=n,pairs=len(pairs),
                       frozen_contributors=int(contributors.sum()),
                       elapsed_seconds=time.monotonic()-started)
            (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
            print('COLLECT',n,len(pairs),state['frozen_contributors'],flush=True)
    keep=frozen.clone()
    keep[freeze_idx]=contributors
    keep|=~frozen
    indices=torch.where(keep)[0].cpu().numpy()
    np.save(output/'keep_indices.npy',indices)
    source=PlyData.read(args.init_ply,mmap=True)['vertex'].data
    if len(source)!=len(keep):raise ValueError('Source PLY length changed')
    xyz=np.column_stack([source[k] for k in ('x','y','z')])
    if not np.array_equal(xyz,model.get_xyz.detach().cpu().numpy()):
        raise ValueError('Source PLY XYZ differs from loaded PGSR model')
    target=output/'candidate_3d.ply'
    PlyData([PlyElement.describe(source[indices].copy(),'vertex')],text=False).write(str(target))
    result=dict(state='candidate_needs_pgsr_validation',source=str(Path(args.init_ply).resolve()),
                source_sha256=sha(args.init_ply),guard_dir=str(Path(args.guard_dir).resolve()),
                guard_summary_sha256=sha(Path(args.guard_dir)/'summary.json'),
                output=str(target.resolve()),output_sha256=sha(target),
                total_points=len(model.get_xyz),trainable_points=int((~frozen).sum()),
                frozen_before=int(frozen.sum()),frozen_kept=int(contributors.sum()),
                frozen_removed=int((~contributors).sum()),selected_points=len(indices),
                conservative_tile_union=int(tiles.sum()),pairs=len(pairs),
                cameras=len(cameras),halo_pixels=5,elapsed_seconds=time.monotonic()-started,
                rule='Frozen-only native PGSR positive RGB colour derivatives on target+0/16 guard ROI with five-pixel render halo',
                per_pair=rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'state.json').write_text(json.dumps(dict(state='candidate_complete',pairs=len(pairs),selected_points=len(indices)),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('total_points','trainable_points','frozen_before','frozen_kept','frozen_removed','selected_points','pairs','elapsed_seconds')},indent=2),flush=True)

if __name__=='__main__':main()
