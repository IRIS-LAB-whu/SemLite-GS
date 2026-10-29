import argparse
import importlib.util
from pathlib import Path
from .runtime import ROOT,write_json

def audit(source):
    spec=importlib.util.spec_from_file_location('semlite_colmap_io',ROOT/'engines/fastgs/scene/colmap_loader.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=Path(source);sparse=source/'sparse/0'
    if not sparse.exists(): sparse=source/'sparse'
    if (sparse/'cameras.bin').exists():
        cams=module.read_intrinsics_binary(str(sparse/'cameras.bin'));images=module.read_extrinsics_binary(str(sparse/'images.bin'))
    else:
        cams=module.read_intrinsics_text(str(sparse/'cameras.txt'));images=module.read_extrinsics_text(str(sparse/'images.txt'))
    if not cams or not images:raise ValueError('Empty registered camera set')
    unsupported={c.model for c in cams.values()}-{'SIMPLE_PINHOLE','PINHOLE'}
    if unsupported:raise ValueError('Undistort with COLMAP first: '+str(unsupported))
    missing=[i.name for i in images.values() if not (source/'images'/i.name).is_file()]
    if missing:raise ValueError('Missing registered image files: '+str(missing[:10]))
    stems=[Path(i.name).stem for i in images.values()]
    if len(set(stems))!=len(stems):raise ValueError('Duplicate image basenames would collide in mask filenames')
    return dict(state='passed',cameras=len(cams),registered_images=len(images),units='input world units',vertical_axis='world Z')
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    write_json(a.output,audit(a.source))
if __name__=='__main__':main()
