import argparse
from pathlib import Path
from semlite_gs.reconstruction.envelope import reconstruct
from .runtime import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--field',required=True);p.add_argument('--output',required=True)
    p.add_argument('--instance',type=int,required=True);p.add_argument('--max-faces',type=int)
    p.add_argument('--largest-component',action='store_true')
    a=p.parse_args()
    if Path(a.field).suffix.lower()!='.npz': p.error('Expected a signed, weighted TSDF NPZ field')
    if a.instance<1 or (a.max_faces is not None and a.max_faces<4): p.error('Invalid ID or face budget')
    context={'strength':.25,'max_vertex_displacement_m':.03}
    job=(a.instance,str(Path(a.field).parent),a.output,True,True,a.field,a.max_faces,context,
         {'largest_component':a.largest_component})
    row=reconstruct(job);write_json(Path(a.output)/f'instance_{a.instance:03d}'/'result.json',row)
    if row['state']!='complete': raise SystemExit('Envelope failed: '+str(row))
if __name__=='__main__':main()
