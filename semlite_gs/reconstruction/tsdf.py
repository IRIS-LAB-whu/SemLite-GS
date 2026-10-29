from pathlib import Path
import numpy as np


def field_observations(path, minimum_weight=3.):
    from . import native_field
    if Path(path).suffix.lower() != '.npz':
        raise ValueError('Expected a signed, weighted TSDF NPZ field')
    with np.load(path, allow_pickle=False) as data:
        raw=np.asarray(data['coordinates'])
        phi=np.asarray(data['phi'],np.float64)
        weight=np.asarray(data['weight'],np.float32)
        voxel=float(data['voxel']);trunc=float(data.get('sdf_trunc',5*voxel))
        offset=np.asarray(data.get('lattice_offset',[0.,0.,0.]),np.float64)
        backend=str(data.get('backend',''))
    if (raw.ndim!=2 or raw.shape[1]!=3 or not len(raw) or
            not np.isfinite(raw).all() or not np.array_equal(raw,np.rint(raw)) or
            np.max(np.abs(raw))>np.iinfo(np.int32).max-16 or
            phi.shape!=(len(raw),) or weight.shape!=phi.shape or
            not np.isfinite(phi).all() or not np.isfinite(weight).all() or
            np.any(weight<=0) or not np.isfinite(voxel) or voxel<=0 or
            not np.isfinite(trunc) or trunc<=0 or np.any(np.abs(phi)>trunc*(1+1e-6)) or
            offset.shape!=(3,) or not np.isfinite(offset).all()):
        raise ValueError('Invalid signed, weighted TSDF field')
    coords=raw.astype(np.int32)
    if len(np.unique(coords,axis=0))!=len(coords):
        raise ValueError('Duplicate TSDF lattice coordinates')
    threshold=0. if backend=='open3d_legacy_scalable' else float(minimum_weight)
    if not np.isfinite(threshold) or threshold<0: raise ValueError('Invalid observation weight threshold')
    vertices,patches=native_field.zero_observations(coords,phi,weight,voxel,trunc,offset,threshold)
    if not len(patches): raise ValueError('No supported TSDF zero observations')
    vertices,patches=canonical_surface(vertices,patches)
    return vertices,patches,dict(source='weighted_tsdf_zero_crossings',observed_voxels=len(phi),
        voxel_m=voxel,sdf_trunc_m=trunc,minimum_weight=threshold,
        quadrature_points=len(patches),dense_mesh_extracted=False)


def canonical_surface(vertices, faces):
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    inverse = np.empty(len(order), dtype=np.int64); inverse[order] = np.arange(len(order))
    vertices = vertices[order]; faces = inverse[faces]
    first = faces.argmin(axis=1)
    faces = np.take_along_axis(faces, (first[:, None]+np.arange(3)) % 3, axis=1)
    faces = faces[np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))]
    return vertices, faces

