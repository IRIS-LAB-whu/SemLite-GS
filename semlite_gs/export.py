import argparse,json,hashlib
from pathlib import Path
from semlite_gs.reconstruction.fusion import fuse

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--scene',required=True)
    p.add_argument('--kind',choices=['lightweight','dense'],default='lightweight')
    a=p.parse_args();out=Path(a.output);ids=json.loads((out/'instances.json').read_text())['valid_ids'];rows=[]
    for iid in ids:
        if a.kind=='lightweight':
            row=json.loads((out/'08_lightweight'/f'instance_{iid:03d}/result.json').read_text())
        else:
            path=out/'09_exports/dense'/f'instance_{iid:03d}.ply'
            row=dict(instance=iid,state='complete',mesh=str(path),mesh_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if row['state']!='complete':raise ValueError('Missing/failed instance '+str(iid))
        rows.append(row)
    target=out/'09_exports'/a.kind;target.mkdir(parents=True,exist_ok=True);fuse(rows,target,a.scene,kind=a.kind)
if __name__=='__main__':main()
