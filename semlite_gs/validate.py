import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from .runtime import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    config=json.loads((out/'resolved_config.json').read_text());ids=json.loads((out/'instances.json').read_text())
    it=config.get('initialization',{}).get('iterations',30000)
    masks=out/'04_semantics/04_render_mask'/f'ours_{it}'/'instance_masks/labels'
    counts={i:0 for i in ids['valid_ids']};views={i:0 for i in counts};files=list(masks.glob('*.png'))
    if not files: raise ValueError('No propagated masks')
    allowed=set(ids['all_ids'])|{0}
    for f in files:
        values,n=np.unique(np.asarray(Image.open(f)),return_counts=True)
        if not set(map(int,values))<=allowed: raise ValueError('Unknown labels in '+str(f))
        for i,k in zip(values,n):
            if int(i) in counts:
                counts[int(i)]+=int(k)
                views[int(i)]+=int(k>=config.get('min_mask_area',200))
    if any(v==0 for v in views.values()): raise ValueError('Selected IDs with no eligible training cameras: '+str(views))
    write_json(out/'mask_audit.json',dict(state='passed',images=len(files),pixels=counts,eligible_views=views,
        background=0,training_ids='positive Level2 IDs; original Gaussian IDs plus one'))
if __name__=='__main__':main()
