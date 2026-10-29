#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from plyfile import PlyData, PlyElement
import numpy as np
import torch

from pipeline.model import GaussianModel
from pipeline.scene import Scene
from pipeline.train import make_parser
from pipeline.guard import load_guard_targets, union_roi
from pipeline.masks import mask_roi
from pipeline.rendering import render, render_many
from pipeline.setup_io import camera_batches


def main():
    parser, model_args, _, pipe_args = make_parser()
    parser.add_argument('--candidate_ply', required=True)
    parser.add_argument('--keep_indices', required=True)
    parser.add_argument('--guard_dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--repair_dir', default='')
    parser.add_argument('--max_pairs', type=int, default=0)
    parser.add_argument('--validation_batch_size', type=int, default=1)
    parser.add_argument('--validation_batch_order', choices=['input','roi'], default='input')
    parser.add_argument('--audit_batched_against_single', action='store_true')
    args = parser.parse_args()
    if args.validation_batch_size < 1:
        parser.error('validation_batch_size must be positive')
    process_started = time.perf_counter()
    original = GaussianModel(args.sh_degree)
    scene = Scene(model_args.extract(args), original, args)
    candidate = GaussianModel(args.sh_degree)
    candidate.load_ply(args.candidate_ply)
    candidate.active_sh_degree = original.active_sh_degree
    keep = torch.as_tensor(np.load(args.keep_indices), device='cuda', dtype=torch.long)
    keep_mask = torch.zeros(len(original.get_xyz), dtype=torch.bool, device='cuda')
    keep_mask[keep] = True
    repair = torch.zeros_like(keep_mask)
    fields = ('_xyz','_features_dc','_features_rest','_opacity','_scaling','_rotation')
    if len(candidate.get_xyz) != len(keep) or any(not torch.equal(getattr(original,k)[keep],getattr(candidate,k)) for k in fields):
        raise ValueError('Candidate PLY does not preserve the selected PGSR parameters exactly')
    cameras = sorted((c for c in scene.getTrainCameras() if any(i in c.available_instance_ids for i in scene.instance_ids)), key=lambda c:c.image_name)
    load_guard_targets(cameras,args.guard_dir,args.guard_io_workers)
    pairs = [(c,i) for c in cameras for i in scene.instance_ids if i in c.available_instance_ids]
    if args.max_pairs:
        positions = np.linspace(0,len(pairs)-1,min(args.max_pairs,len(pairs)),dtype=int)
        pairs = [pairs[int(k)] for k in positions]
    pipe = pipe_args.extract(args)
    bg = torch.tensor([1.,1.,1.] if args.white_background else [0.,0.,0.],device='cuda')
    keys = ('render','rendered_alpha','plane_depth','rendered_normal','depth_normal')
    maxima = {k:0. for k in keys}
    failures = []

    prepared = [(camera, iid, union_roi(mask_roi(camera.instance_mask == iid),
                getattr(getattr(camera, 'frozen_guard', None), 'roi', None)))
                for camera, iid in pairs]
    if args.validation_batch_order == 'roi':
        prepared.sort(key=lambda p: (p[0].image_height, p[0].image_width,
                      (p[2][3]+63)//64, (p[2][2]+63)//64, p[2][3], p[2][2]))
    batches = list(camera_batches(prepared, args.validation_batch_size, lambda p: p[0]))
    audit_maxima = {key: 0. for key in keys}
    audit_seconds = 0.
    torch.cuda.synchronize()
    validation_started = time.perf_counter()
    completed = 0
    with torch.no_grad():
        for batch in batches:
            cams, rois = [p[0] for p in batch], [p[2] for p in batch]
            def render_batch(model):
                if args.validation_batch_size == 1:
                    return [render(cams[0],model,pipe,bg,roi=rois[0],return_plane=True,return_depth_normal=True)]
                return render_many(cams,model,pipe,bg,rois=rois,return_plane=True,return_depth_normal=True)
            fulls, crops = render_batch(original), render_batch(candidate)
            errors = torch.stack([torch.stack([(a[k]-b[k]).abs().max() for k in keys])
                                  for a,b in zip(fulls,crops)]).cpu().tolist()
            if args.audit_batched_against_single:
                torch.cuda.synchronize()
                audit_started = time.perf_counter()
                for (camera,iid,roi),a,b in zip(batch,fulls,crops):
                    for model,actual in ((original,a),(candidate,b)):
                        expected = render(camera,model,pipe,bg,roi=roi,return_plane=True,return_depth_normal=True)
                        for key in keys:
                            delta = float((actual[key]-expected[key]).abs().max())
                            audit_maxima[key] = max(audit_maxima[key],delta)
                            if not torch.equal(actual[key],expected[key]):
                                raise RuntimeError(f'Batch/single audit differs: {camera.image_name} {iid} {key}: {delta}')
                        if not torch.equal(actual['radii'],expected['radii']):
                            raise RuntimeError('Batch/single radii differ; repair set would change')
                        del expected
                torch.cuda.synchronize()
                audit_seconds += time.perf_counter()-audit_started
            for (camera,iid,roi),full,values in zip(batch,fulls,errors):
                delta = dict(zip(keys,values))
                for key in keys:
                    maxima[key] = max(maxima[key],delta[key])
                if any(not np.isfinite(v) or v > (1e-3 if k == 'plane_depth' else 1e-5)
                       for k,v in delta.items()):
                    if args.repair_dir:
                        repair |= (full['radii']>0) & (~original.trainable_mask) & (~keep_mask)
                    failures.append(dict(camera=camera.image_name,instance=iid,roi=roi,difference=delta,
                                         tile_candidates=int((full['radii']>0).sum()),
                                         positive_observe=int((full['out_observe']>0).sum())))
            completed += len(batch)
            if completed%50 < len(batch) or completed == len(pairs):
                print('VALIDATE',completed,len(pairs),len(failures),maxima,flush=True)
            del fulls,crops,full
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter()-validation_started-audit_seconds
    if args.repair_dir and failures:
        repair_dir = Path(args.repair_dir)
        repair_dir.mkdir(parents=True, exist_ok=True)
        if any(repair_dir.iterdir()):
            raise ValueError('Repair directory must be empty')
        repaired = torch.where(keep_mask | repair)[0].cpu().numpy()
        source = PlyData.read(args.init_ply, mmap=True)['vertex'].data
        PlyData([PlyElement.describe(source[repaired].copy(),'vertex')], text=False).write(str(repair_dir/'candidate_3d.ply'))
        np.save(repair_dir/'keep_indices.npy', repaired)
    result=dict(state='passed' if not failures else 'failed',pairs=len(pairs),failures=len(failures),
                repair_added=int(repair.sum()),repair_dir=str(Path(args.repair_dir).resolve()) if args.repair_dir else '',
                validation_batch_size=args.validation_batch_size,
                validation_batch_order=args.validation_batch_order,
                validation_seconds=validation_seconds, audit_seconds=audit_seconds,
                batch_single_audit=audit_maxima if args.audit_batched_against_single else None,
                process_seconds=time.perf_counter()-process_started,
                validation_scope='sample' if args.max_pairs else 'all',
                maxima=maxima,failed_pairs=failures,source_ply=str(Path(args.init_ply).resolve()),
                candidate_ply=str(Path(args.candidate_ply).resolve()),guard_dir=str(Path(args.guard_dir).resolve()))
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('state','pairs','failures','maxima')},indent=2),flush=True)
    if failures:raise SystemExit(2)

if __name__=='__main__':main()
