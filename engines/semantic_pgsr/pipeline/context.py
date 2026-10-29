import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
from pipeline.guard import load_guard_targets


def _keys(xyz):
    return np.ascontiguousarray(xyz, dtype=np.float32).view('V12').reshape(-1)


@torch.no_grad()
def install_instance_context(model, cameras, sampler, manifest, resume=False, mode='independent', guard_io_workers=0):
    if mode not in ('independent', 'joint'):
        raise ValueError('Unknown instance context mode')
    if resume and model.context_mode != mode:
        raise ValueError('Checkpoint context mode differs; old joint checkpoints require --instance_context_mode joint. Independent training must start from initialization.')
    document = json.loads(Path(manifest).read_text())
    specs = document['instances']
    signature = hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest()
    if resume and model.context_signature != signature:
        raise ValueError('Checkpoint context manifest differs; use the original manifest')
    model.context_signature = signature
    if set(map(int, specs)) != set(sampler.eligible):
        raise ValueError('Context manifest must cover exactly the requested owners')
    xyz_keys = _keys(model.get_xyz.detach().cpu().numpy()) if not resume else None
    for text_owner, spec in specs.items():
        owner = int(text_owner)
        candidate = Path(spec['candidate_ply']).resolve()
        report = json.loads(Path(spec['validation']).read_text())
        if (report.get('state') != 'passed' or report.get('failures') != 0 or
                Path(report['candidate_ply']).resolve() != candidate or
                Path(report['guard_dir']).resolve() != Path(spec['guard_dir']).resolve()):
            raise ValueError(f'Owner {owner}: context candidate/guard is not validated')
        if resume:
            if owner not in model.context_membership:
                raise ValueError('Cannot attach a new context subset to an old trained checkpoint; initialize from PLY')
        else:
            rows = PlyData.read(str(candidate))['vertex'].data
            keys = _keys(np.column_stack([rows[a] for a in ('x','y','z')]))


            if not np.isin(keys, xyz_keys).all():
                raise ValueError(f'Owner {owner}: initialization is missing validated candidate points; build their union first')
            model.context_membership[owner] = torch.as_tensor(np.isin(xyz_keys, keys), device=model.get_xyz.device)
        selected = [cameras[i] for i in sampler.eligible[owner]]
        previous = [getattr(c, 'frozen_guard', None) for c in selected]
        load_guard_targets(selected, spec['guard_dir'], guard_io_workers)
        for cam, old in zip(selected, previous):
            if not hasattr(cam, '_instance_guards'): cam._instance_guards = {}
            cam._instance_guards[owner] = cam.frozen_guard
            cam.frozen_guard = old
    if not resume and mode == 'independent':
        print('Independent context snapshots:', model.make_context_independent(), flush=True)
    from pipeline.active import ActiveSetCache
    cache = ActiveSetCache()
    for owner in sampler.eligible:
        count = len(cache.select(model, owner))
        print(f'Instance context: owner={owner} selected={count}/{len(model.get_xyz)} mode={model.context_mode}', flush=True)
