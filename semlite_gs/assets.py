import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from .runtime import ROOT, ENGINES

EXTENSIONS = {
 'fastgs': ['simple-knn','fused-ssim','diff-gaussian-rasterization_fastgs','diff-gaussian-rasterization_fastgs_ortho','diff-gaussian-rasterization-semantic'],
 'pgsr': ['simple-knn','diff-plane-rasterization','pgsr-pipeline-ops','fused-ssim','pgsr-view-batch']}


def build_field():
    import torch
    import open3d
    if open3d.__version__ not in ('0.18.0','0.19.0'):
        raise RuntimeError('The legacy TSDF bridge is validated only with Open3D 0.18.0/0.19.0.')
    target=ROOT/'semlite_gs/reconstruction'/('native_field'+sysconfig.get_config_var('EXT_SUFFIX'))
    includes=[sysconfig.get_path('include'),str(Path(torch.__file__).parent/'include'),
              str(ROOT/'third_party/open3d/include'),str(ROOT/'third_party/open3d/include/open3d/3rdparty'),os.environ.get('EIGEN_INCLUDE_DIR','/usr/include/eigen3')]
    command=[os.environ.get('CXX','c++'),'-O2','-shared','-std=c++14','-fPIC',
             str(ROOT/'semlite_gs/reconstruction/native_field.cpp'),'-o',str(target)]
    command += ['-I'+p for p in includes]
    subprocess.run(command,check=True)

    subprocess.run([sys.executable, '-c', """
import numpy as np
import open3d as o3d
from semlite_gs.reconstruction import native_field
from semlite_gs.dense import extract_legacy_surface
from semlite_gs.reconstruction.tsdf import canonical_surface
I = o3d.pipelines.integration
volume = I.ScalableTSDFVolume(.1, .4, I.TSDFVolumeColorType.NoColor, depth_sampling_stride=1)
rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    o3d.geometry.Image(np.zeros((40, 40, 3), np.uint8)),
    o3d.geometry.Image(np.full((40, 40), 1.03, np.float32)), depth_scale=1, depth_trunc=3)
camera = o3d.camera.PinholeCameraIntrinsic(40, 40, 40, 40, 20, 20)
for _ in range(4):
    volume.integrate(rgbd, camera, np.eye(4))
field = native_field.export_field(volume)
if not len(field['weight']) or not np.all(field['weight'] == 4):
    raise RuntimeError('Native TSDF bridge did not preserve measured weights')
surface, _ = extract_legacy_surface(field)
reference = volume.extract_triangle_mesh()
a = canonical_surface(np.asarray(surface.vertices), np.asarray(surface.triangles))
b = canonical_surface(np.asarray(reference.vertices), np.asarray(reference.triangles))
if not all(np.array_equal(x, y) for x, y in zip(a, b)):
    raise RuntimeError('Native TSDF bridge failed its surface round-trip check')
v, f = native_field.zero_observations(field['coordinates'], field['phi'], field['weight'],
    field['voxel'], field['sdf_trunc'], np.asarray(field['lattice_offset'], dtype=float), 0.)
c = canonical_surface(v, f)
if not np.allclose(c[0], b[0], rtol=0, atol=1e-12) or not np.array_equal(c[1], b[1]):
    raise RuntimeError('TSDF zero-crossing observation check failed')
print('Native TSDF bridge and zero-crossing observations passed')
"""], cwd=ROOT, check=True)
    print(target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('engine',choices=['field','fastgs','pgsr'])
    args=parser.parse_args()
    if args.engine=='field': return build_field()
    for name in EXTENSIONS[args.engine]:
        root=ENGINES[args.engine]/'submodules'/name
        subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=root,check=True)

if __name__=='__main__':main()
