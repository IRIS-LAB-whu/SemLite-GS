import hashlib

import numpy as np
from plyfile import PlyData

from .geometry import save


def fuse(rows, output, scene, kind="lightweight"):
    vertices, faces, instances = [], [], []
    nv = nf = 0
    for row in rows:
        if row['state'] != 'complete':
            continue
        mesh = PlyData.read(row['mesh'],known_list_len={'face':{'vertex_indices':3}})
        xyz = np.column_stack([mesh['vertex'][key] for key in ('x','y','z')])
        tri = np.asarray(mesh['face']['vertex_indices'])
        iid = row['instance']
        v = np.empty(len(xyz), dtype=[('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1')])
        for k, key in enumerate(('x','y','z')):
            v[key] = xyz[:, k]

        color = 64 + np.random.RandomState(iid).randint(0, 192, 3)
        for k, key in enumerate(('red','green','blue')):
            v[key] = color[k]
        f = np.empty(len(tri), dtype=[('vertex_indices','<i4',(3,)),('instance_id','<i4')])
        f['vertex_indices'], f['instance_id'] = tri + nv, iid
        vertices.append(v)
        faces.append(f)
        instances.append(dict(instance=iid,mesh=row['mesh'],sha256=row['mesh_sha256'],
                              vertex_start=nv,vertices=len(v),face_start=nf,faces=len(f)))
        nv += len(v)
        nf += len(f)
    path = output/f'{scene}_{kind}_fused.ply'
    if vertices:


        vv=np.concatenate(vertices);ff=np.concatenate(faces)
        packed=np.empty(len(ff),dtype=[('count','u1'),('vertex_indices','<i4',(3,)),('instance_id','<i4')])
        packed['count']=3;packed['vertex_indices']=ff['vertex_indices'];packed['instance_id']=ff['instance_id']
        temporary=path.with_suffix('.ply.tmp')
        with temporary.open('wb') as stream:
            stream.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(vv))+
                '\nproperty double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nelement face '+str(len(ff))+
                '\nproperty list uchar int vertex_indices\nproperty int instance_id\nend_header\n').encode())
            vv.tofile(stream);packed.tofile(stream)
        temporary.replace(path)
    result = dict(scene=scene,mesh=str(path) if vertices else None,instances=instances,vertices=nv,faces=nf,
                  failed_instances=[r['instance'] for r in rows if r['state']!='complete'],
                  original_coordinates_preserved=True)
    if vertices:
        result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(output/'fusion_manifest.json', result)
    return result['mesh']
