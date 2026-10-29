import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .geometry import save
from .envelope import reconstruct


def main(argv=None):
    parser = argparse.ArgumentParser(description='Reconstruct lightweight planar envelopes from signed, weighted TSDF fields.')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--field-root', type=Path, help='Directory containing instance_XXX/observed_tsdf.npz')
    source.add_argument('--tsdf-field', type=Path, help='Signed, weighted TSDF NPZ field')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', default='scene', help='Scene label used for the fused output filename')
    parser.add_argument('--ids', nargs='+', type=int, help='Instance IDs for --field-root; default: all')
    parser.add_argument('--instance-id', type=int, default=1, help='Output ID for --tsdf-field')
    parser.add_argument('--workers', type=int, default=1, help='Parallel instance processes; each uses the same algorithm')
    parser.add_argument('--fuse', action='store_true', help='Also combine successful meshes, retaining coordinates and face instance IDs')
    parser.add_argument('--max-faces', type=int, help='Maximum output triangle count')
    parser.add_argument('--sdf-weight', type=float, default=.25, help='Local SDF term weight')
    args = parser.parse_args(argv)
    if args.workers < 1 or args.instance_id < 1:
        parser.error('--workers and --instance-id must be positive')
    if args.max_faces is not None and args.max_faces<4:
        parser.error('--max-faces must be at least 4')
    if not 0 < args.sdf_weight < float('inf'):
        parser.error('--sdf-weight must be finite and positive')
    if not args.scene or Path(args.scene).name != args.scene or args.scene in {'.', '..'}:
        parser.error('--scene must be a single filename component')
    output = args.output.resolve()
    if args.tsdf_field is not None:
        if args.ids is not None:
            parser.error('--ids is only valid with --field-root')
        input_path = args.tsdf_field.resolve()
        inputs = [(args.instance_id, input_path)]
        source_root = input_path.parent
    elif args.field_root is not None:
        source_root = args.field_root.resolve()
        paths = {}
        for path in sorted(source_root.glob('instance_*/observed_tsdf.npz')):
            paths[int(path.parent.name.split('_')[-1])] = path
        ids = args.ids if args.ids is not None else sorted(paths)
        if not ids or len(ids) != len(set(ids)) or min(ids) < 1:
            parser.error('No instances found, or instance IDs are duplicated/nonpositive')
        if any(i not in paths for i in ids):
            parser.error('A requested instance has no observed TSDF field')
        inputs = [(i, paths[i]) for i in ids]
    for iid, path in inputs:
        if path.suffix.lower() != '.npz':
            parser.error('Expected a signed, weighted TSDF NPZ field')
        if not path.is_file():
            parser.error(f'Input TSDF source does not exist: {path}')

    inputs.sort(key=lambda item: item[1].stat().st_size, reverse=True)
    sdf_context=dict(strength=args.sdf_weight,max_vertex_displacement_m=.03)
    jobs = [(i, str(source_root), str(output), True, True, str(path),args.max_faces,sdf_context) for i, path in inputs]
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.workers == 1:
        rows = [reconstruct(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            rows = list(pool.map(reconstruct, jobs))
    rows.sort(key=lambda row: row['instance'])
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}
    batch = dict(scene=args.scene, source_root=str(source_root), instances=rows, workers=args.workers,
                 wall_seconds=time.perf_counter()-started, code_sha256=code_hashes,
                 successful_instances=sum(r['state']=='complete' for r in rows),
                 failed_instances=[r['instance'] for r in rows if r['state']!='complete'],
                 parameters=dict(local_roof_recovery=True, multi_direction_straight_edges=True,
                                 supported_height_cells=True, joint_plane_refit=True,
                                 joint_sdf_optimization=True,
                                 local_field_losses=True,
                                 joint_sdf_weight=args.sdf_weight,
                                 max_faces=args.max_faces))

    if args.fuse:
        from .fusion import fuse
        batch['fused_mesh'] = fuse(rows, output, args.scene)
    save(output/'batch.json', batch)
    print(json.dumps({k: batch[k] for k in ['scene','successful_instances','failed_instances','wall_seconds']}, indent=2))
    return 1 if batch['failed_instances'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
