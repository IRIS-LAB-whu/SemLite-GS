import argparse
import json
import subprocess
import sys
from pathlib import Path
from .pipeline import load_config
from .runtime import ROOT,ENGINES,engine_environment,write_json

IMPORTS={
 'fastgs':['torch','diff_gaussian_rasterization_fastgs','diff_gaussian_rasterization_fastgs_ortho','diff_gaussian_rasterization','simple_knn._C','fused_ssim_cuda'],
 'pgsr':['torch','pipeline.train','pipeline.field','diff_plane_rasterization','simple_knn._C','pgsr_compact_batch','pgsr_pipeline_ops._C','pgsr_pipeline_ops._pair','pgsr_pipeline_ops._geometry','pgsr_pipeline_ops._roi','fused_ssim_cuda'],
 'sam3':['torch','sam3.model_builder','sam3.model.sam3_image_processor'],
 'mesh':['numpy','scipy','open3d','manifold3d','shapely','semlite_gs.reconstruction.native_field','semlite_gs.reconstruction.sdf']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output');a=p.parse_args()
    c=load_config(a.config);rows=[]
    for engine,names in IMPORTS.items():
        script='import importlib,json; out={}\n'
        script+='\n'.join('m=importlib.import_module('+repr(name)+');out['+repr(name)+']=getattr(m,"__file__",None)' for name in names)
        script+='\nprint(json.dumps(out))\n'
        command=[c['python'][engine],'-c',script]
        result=subprocess.run(command,cwd=ENGINES[engine],env=engine_environment(engine),capture_output=True,text=True)
        row={'engine':engine,'passed':result.returncode==0,'stdout':result.stdout,'stderr':result.stderr}
        if row['passed']:
            locations=json.loads(result.stdout)
            required=('diff_','simple_knn','pipeline.','pgsr_','fused_ssim','sam3.','semlite_gs.reconstruction')
            wrong={name:path for name,path in locations.items() if name.startswith(required) and not Path(path).resolve().is_relative_to(ROOT)} if sys.version_info>=(3,9) else {name:path for name,path in locations.items() if name.startswith(required) and ROOT not in Path(path).resolve().parents}
            if wrong: row.update(passed=False,stderr='External project code imported: '+str(wrong))
        rows.append(row);print(engine,'PASS' if row['passed'] else 'FAIL')
        if not row['passed']:print(result.stderr[-1500:])
    if a.output:write_json(a.output,rows)
    if not all(r['passed'] for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
