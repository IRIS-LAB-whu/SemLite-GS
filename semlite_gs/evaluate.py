import argparse
import csv
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from .runtime import write_json


def measure(mesh_path, lidar_path, samples=100000, seed=20261013):
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));pc=o3d.io.read_point_cloud(str(lidar_path))
    v=np.asarray(mesh.vertices);f=np.asarray(mesh.triangles);q=np.asarray(pc.points)
    if not len(f) or not len(q) or not np.isfinite(v).all() or not np.isfinite(q).all():
        raise ValueError('Empty or nonfinite mesh/reference: '+str(mesh_path))
    tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    rng=np.random.default_rng(seed);t=tri[rng.choice(len(tri),samples,p=areas/areas.sum())]
    uv=rng.random((samples,2));r=np.sqrt(uv[:,0]);s=uv[:,1]
    points=(1-r[:,None])*t[:,0]+(r*(1-s))[:,None]*t[:,1]+(r*s)[:,None]*t[:,2]
    m2l=float(cKDTree(q).query(points)[0].mean())
    ray=o3d.t.geometry.RaycastingScene();ray.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances=[]
    for start in range(0,len(q),100000):
        distances.append(ray.compute_distance(o3d.core.Tensor(q[start:start+100000].astype(np.float32))).numpy())
    l2m=float(np.concatenate(distances).mean())
    return dict(CD_m=(m2l+l2m)/2,mesh_to_lidar_m=m2l,lidar_to_mesh_m=l2m,vertices=len(v),faces=len(f),lidar_points=len(q))


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=100000);a=p.parse_args()
    rows=json.loads(Path(a.manifest).read_text());results=[]
    for row in rows:
        try: result=dict(row,**measure(row['mesh'],row['lidar'],a.samples,20261013+int(row['instance'])),state='complete')
        except Exception as exc: result=dict(row,state='failed',error=str(exc))
        results.append(result)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);write_json(out/'per_instance.json',results)
    keys=list(dict.fromkeys(k for r in results for k in r))
    with (out/'per_instance.csv').open('w',newline='') as s:
        w=csv.DictWriter(s,fieldnames=keys);w.writeheader();w.writerows(results)
    ok=[r for r in results if r['state']=='complete']
    summary={'expected_instances':len(rows),'evaluated_instances':len(ok),'failed':[r for r in results if r['state']!='complete'],
             'protocol':'100k area samples mesh-to-LiDAR; exact triangle distance LiDAR-to-mesh; arithmetic half-sum in world units; equal instance mean.',
             'averages':{k:float(np.mean([r[k] for r in ok])) for k in ('CD_m','vertices','faces')} if ok else {}}
    write_json(out/'summary.json',summary)
    if len(ok)!=len(rows): raise SystemExit('Incomplete evaluation; see per_instance.json; no silent exclusion.')

if __name__=='__main__':main()
