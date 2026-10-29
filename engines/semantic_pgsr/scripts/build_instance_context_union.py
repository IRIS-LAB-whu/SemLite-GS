#!/usr/bin/env python3
"""Assemble validated instance contexts for grouped refinement."""
import argparse,json
from pathlib import Path
import numpy as np
from plyfile import PlyData,PlyElement


def build(jobs_root, instance_ids, output):
    specs={};keeps=[];source_path=None;source=None
    for owner in instance_ids:
        job=Path(jobs_root)/f'instance_{owner:03d}'
        state=json.loads((job/'state.json').read_text())
        candidate=Path(state['final_init_ply']).resolve()
        validation=candidate.parent/'validation.json'
        report=json.loads(validation.read_text())
        if report.get('state')!='passed' or report.get('failures')!=0:
            raise ValueError(f'Owner {owner}: candidate has no passing validation')
        if Path(report['candidate_ply']).resolve()!=candidate:
            raise ValueError('Validation refers to another candidate')
        path=Path(report['source_ply']).resolve()
        if source_path is None:
            source_path=path;source=PlyData.read(str(path))['vertex'].data
            if not all(f'scale_{i}' in source.dtype.names for i in range(3)):
                raise ValueError('Native 3D FastGS scales required')
        if path!=source_path:raise ValueError('All candidates must share one scene-filtered source PLY')
        keep=np.load(candidate.parent/'keep_indices.npy')
        rows=PlyData.read(str(candidate))['vertex'].data
        if not np.array_equal(source[keep],rows):
            raise ValueError(f'Owner {owner}: candidate no longer matches validated source rows')
        keeps.append(keep)
        specs[str(owner)]={'candidate_ply':str(candidate),'validation':str(validation),'guard_dir':report['guard_dir']}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name in ('union_3d.ply','context_manifest.json'):
        if (output/name).exists():raise FileExistsError(output/name)
    keep=np.unique(np.concatenate(keeps))
    PlyData([PlyElement.describe(source[keep],'vertex')]).write(str(output/'union_3d.ply'))
    np.save(output/'union_source_indices.npy',keep)
    (output/'context_manifest.json').write_text(json.dumps({'instances':specs},indent=2)+'\n')
    return {'source_ply':str(source_path),'source_points':len(source),'union_points':len(keep),'instance_ids':instance_ids,'output':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs-root',required=True,type=Path)
    p.add_argument('--instance-ids',required=True)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();ids=[int(i) for i in a.instance_ids.split(',')]
    if not ids or len(ids)!=len(set(ids)):p.error('Provide distinct instance IDs')
    print(json.dumps(build(a.jobs_root,ids,a.output_dir),indent=2))
