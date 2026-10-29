import argparse
from pathlib import Path
import open3d as o3d
from .reconstruction.tsdf import canonical_surface
import numpy as np


def extract_legacy_surface(data):
    if o3d.__version__ not in ('0.18.0','0.19.0'):
        raise RuntimeError('Legacy field bridge requires validated Open3D 0.18/0.19 ABI')
    from .reconstruction import native_field
    raw=np.asarray(data['coordinates'])
    phi=np.asarray(data['phi'],np.float64);weight=np.asarray(data['weight'],np.float32)
    voxel=float(data['voxel']);trunc=float(data['sdf_trunc'])
    if (raw.ndim!=2 or raw.shape[1]!=3 or not len(raw) or
            not np.isfinite(raw).all() or not np.array_equal(raw,np.rint(raw)) or
            np.max(np.abs(raw))>np.iinfo(np.int32).max-16 or
            phi.shape!=(len(raw),) or weight.shape!=phi.shape or
            not np.isfinite(phi).all() or not np.isfinite(weight).all() or
            np.any(weight<=0) or not np.isfinite(voxel) or voxel<=0 or
            not np.isfinite(trunc) or trunc<=0 or np.any(np.abs(phi)>trunc*(1+1e-6))):
        raise ValueError('Invalid measured legacy TSDF field')
    coords=raw.astype(np.int32)
    if len(np.unique(coords,axis=0))!=len(coords):
        raise ValueError('Duplicate legacy TSDF lattice coordinates')
    if not np.array_equal(np.asarray(data['lattice_offset']),[.5,.5,.5]):
        raise ValueError('Unsupported legacy sample lattice offset')
    I=o3d.pipelines.integration
    volume=I.ScalableTSDFVolume(voxel,trunc,I.TSDFVolumeColorType.NoColor)
    blocks,inverse=np.unique(coords//16,axis=0,return_inverse=True)
    order=np.argsort(inverse,kind='stable');bounds=np.r_[0,np.cumsum(np.bincount(inverse))]
    owners=[]
    for index,block in enumerate(blocks):
        ids=order[bounds[index]:bounds[index+1]];local=coords[ids]-block*16
        flat=local[:,0]*256+local[:,1]*16+local[:,2]
        values=np.zeros((4096,2));values[flat,0]=phi[ids]/trunc;values[flat,1]=weight[ids]
        unit=I.UniformTSDFVolume(voxel*16,16,trunc,I.TSDFVolumeColorType.NoColor,origin=block.astype(float)*(voxel*16))
        unit.inject_volume_tsdf(o3d.utility.Vector2dVector(values))
        owners.append(unit);native_field.attach_unit(volume,unit,*map(int,block))
    surface=volume.extract_triangle_mesh()

    del volume
    return surface,len(blocks)


def extract_surface_from_field(path, minimum_weight=3.0):
    with np.load(Path(path)) as data:
        if str(data.get("backend", "")) == "open3d_legacy_scalable":
            surface, blocks = extract_legacy_surface(data)
            vertices, faces = canonical_surface(np.asarray(surface.vertices).copy(), np.asarray(surface.triangles).copy())
            return vertices, faces, dict(source="observed_sparse_tsdf_field", backend="open3d_legacy_scalable",
                observed_voxels=len(data["phi"]), activated_blocks=blocks, voxel_m=float(data["voxel"]),
                sdf_trunc_m=float(data["sdf_trunc"]), lattice_offset=[.5,.5,.5], minimum_weight=0.,
                surface_vertices=len(vertices), surface_faces=len(faces), canonical_order=True)
    with np.load(Path(path)) as data:
        coordinates = np.asarray(data["coordinates"], dtype=np.int32)
        phi = np.asarray(data["phi"], dtype=np.float32)
        weight = np.asarray(data["weight"], dtype=np.float32)
        voxel = float(data["voxel"])
    if (coordinates.ndim != 2 or coordinates.shape[1] != 3 or
            phi.shape != (len(coordinates),) or weight.shape != phi.shape or
            not len(coordinates) or not np.isfinite(phi).all() or
            not np.isfinite(weight).all() or not np.isfinite(voxel) or voxel <= 0):
        raise ValueError("Invalid sparse TSDF field")
    if np.any(weight <= 0):
        raise ValueError("Sparse TSDF contains unobserved voxel records")

    blocks = np.unique(np.floor_divide(coordinates, 8), axis=0).astype(np.int32)
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3d.core.Dtype.Float32, o3d.core.Dtype.Float32),
        attr_channels=((1,), (1,)), voxel_size=voxel, block_resolution=8,
        block_count=int(len(blocks) * 1.5) + 100,
        device=o3d.core.Device("CPU:0"),
    )
    volume.hashmap().activate(o3d.core.Tensor(blocks, o3d.core.Dtype.Int32))
    if volume.hashmap().size() != len(blocks):
        raise ValueError("Not all sparse TSDF blocks were activated")
    positions, flattened = volume.voxel_coordinates_and_flattened_indices()
    query = np.rint(positions.numpy() / voxel).astype(np.int32)
    flattened = flattened.numpy().reshape(-1)

    lower = np.minimum(coordinates.min(axis=0), query.min(axis=0))
    upper = np.maximum(coordinates.max(axis=0), query.max(axis=0))
    span = (upper.astype(np.int64) - lower + 1)

    def lattice_key(x):
        shifted = x.astype(np.int64) - lower
        return (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]

    source_keys = lattice_key(coordinates)
    order = np.argsort(source_keys)
    sorted_keys = source_keys[order]
    if np.any(np.diff(sorted_keys) == 0):
        raise ValueError("Duplicate TSDF lattice coordinates")
    target_keys = lattice_key(query)
    index = np.searchsorted(sorted_keys, target_keys)
    found = index < len(sorted_keys)
    found[found] &= sorted_keys[index[found]] == target_keys[found]
    source_index = order[index[found]]
    if len(source_index) != len(coordinates):
        raise ValueError("Sparse TSDF coordinate mapping is incomplete")
    volume.attribute("tsdf").numpy().reshape(-1)[flattened[found]] = \
        phi[source_index] / (5 * voxel)
    volume.attribute("weight").numpy().reshape(-1)[flattened[found]] = \
        weight[source_index]

    surface = volume.extract_triangle_mesh(
        weight_threshold=minimum_weight
    ).to_legacy()
    vertices = np.asarray(surface.vertices, dtype=np.float64).copy()
    faces = np.asarray(surface.triangles, dtype=np.int64).copy()
    if not len(vertices) or not len(faces):
        raise ValueError("No supported TSDF zero surface was extracted")


    vertices, faces = canonical_surface(vertices, faces)
    report = {
        "source": "observed_sparse_tsdf_field",
        "observed_voxels": int(len(coordinates)),
        "activated_blocks": int(len(blocks)),
        "voxel_m": voxel,
        "minimum_weight": float(minimum_weight),
        "surface_vertices": int(len(vertices)),
        "surface_faces": int(len(faces)),
        "canonical_order": True,
    }
    return vertices, faces, report





def main():
    p=argparse.ArgumentParser();p.add_argument('--field',required=True);p.add_argument('--output',required=True)
    p.add_argument('--largest-component',action='store_true');a=p.parse_args()
    v,f,_=extract_surface_from_field(a.field)
    mesh=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),o3d.utility.Vector3iVector(f))
    if a.largest_component:
        labels,counts,areas=mesh.cluster_connected_triangles()
        if len(counts): mesh.remove_triangles_by_mask(np.asarray(labels)!=int(np.argmax(areas)))
        mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals();Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    if not o3d.io.write_triangle_mesh(a.output,mesh): raise RuntimeError('Cannot write dense mesh')

if __name__=='__main__':main()
