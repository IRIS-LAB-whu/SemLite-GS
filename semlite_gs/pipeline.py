import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import queue
import sys
from .runtime import ROOT, ENGINES, run_command, write_json

STAGES=('prepare','initialize','orthophoto','segment','propagate','guard','train','field','envelope','export')


def load_config(path):
    path=Path(path).resolve();c=json.loads(path.read_text())
    for key in ('source','output'):
        p=Path(c[key]).expanduser();c[key]=str((path.parent/p).resolve() if not p.is_absolute() else p.resolve())
    if c['source']==c['output']: raise ValueError('Input and output directories must differ')
    c.setdefault('scene',Path(c['source']).name)
    if Path(c['scene']).name!=c['scene']: raise ValueError('Scene must be a filename component')
    for name in ('instances_per_group','processes_per_gpu','cpu_threads','mesh_workers'):
        if c.get(name,1)<1: raise ValueError(name+' must be positive')
    return c


class SemLiteGS:
    def __init__(self,c):
        self.c=c;self.out=Path(c['output']);self.source=Path(c['source'])
        self.fast=self.out/'01_initialization';self.semantic=self.out/'04_semantics'
        self.iteration=c.get('initialization',{}).get('iterations',30000)
        self.ply=self.fast/'point_cloud'/f'iteration_{self.iteration}'/'point_cloud.ply'
        self.masks=self.semantic/'04_render_mask'/f'ours_{self.iteration}'/'instance_masks/labels'
        self.gpus=c.get('gpus',[0]);self.gpu=self.gpus[0]
        self.steps=c.get('training',{}).get('iterations',20000)
    def command(self,name,engine,args,outputs=(),gpu=None):
        return run_command(self.c,name,engine,args,outputs,self.gpu if gpu is None else gpu)
    def ids(self):
        manifest=json.loads((self.out/'instances.json').read_text())
        selected=self.c.get('valid_ids')
        ids=manifest['valid_ids'] if selected is None else selected
        if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(manifest['all_ids']):
            raise ValueError('valid_ids must be a nonempty distinct subset of generated IDs')
        if ids!=manifest['valid_ids']:
            manifest.update(valid_ids=ids,excluded_ids=sorted(set(manifest['all_ids'])-set(ids)))
            write_json(self.out/'instances.json',manifest)
        return ids
    def groups(self):
        ids=self.ids();n=self.c.get('instances_per_group',4)
        return [ids[i:i+n] for i in range(0,len(ids),n)]
    def parallel(self,items,fn,per_gpu=None):
        slots=queue.Queue()
        for gpu in self.gpus:
            for _ in range(per_gpu or self.c.get('processes_per_gpu',1)): slots.put(gpu)
        def run(item):
            gpu=slots.get()
            try: return fn(item,gpu)
            finally: slots.put(gpu)
        with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
            return list(pool.map(run,items))
    def common(self,ids,model,init=None):
        return ['-s',self.source,'-m',model,'-r',self.c.get('training',{}).get('resolution',1),
                '--instance_masks',self.masks,'--instance_ids',','.join(map(str,ids)),
                '--mask_background_id',0,'--init_ply',init or self.ply,
                '--building_ply',self.semantic/'02_point_cloud/building_only_instances_color.ply',
                '--building_transform',self.out/'identity_transform.txt',
                '--building_colors',self.semantic/'03_json/instance_mask_colors.json',
                '--building_values',self.out/'training_values.json',
                '--freeze_global_outside_instances','--skip_initial_mask_prune','--skip_occluded_frozen_prune',
                '--min_instance_mask_area',self.c.get('min_mask_area',200),
                '--max_points_per_instance',self.c.get('training',{}).get('max_points_per_instance',100000)]
    def prepare(self):
        if not (self.source/'images').is_dir(): raise ValueError('Expected registered, undistorted images/ input')
        sparse=self.source/'sparse/0'
        if not sparse.exists(): sparse=self.source/'sparse'
        if not any((sparse/('cameras'+s)).is_file() and (sparse/('images'+s)).is_file() for s in ('.bin','.txt')):
            raise ValueError('Expected COLMAP cameras/images in sparse/0 or sparse')
        self.out.mkdir(parents=True,exist_ok=True)
        write_json(self.out/'resolved_config.json',self.c)
        self.command('00_input_audit','mesh',['-m','semlite_gs.validate_input','--source',self.source,'--output',self.out/'input_audit.json'],[self.out/'input_audit.json'])

        (self.out/'identity_transform.txt').write_text('1 0 0\n0 1 0\n0 0 1\n')
        write_json(self.out/'input_contract.json',dict(source=str(self.source),units='input world units (metres required for reported CD)',
                   vertical_axis='world Z',camera_model='undistorted PINHOLE / SIMPLE_PINHOLE',lidar_used_for_reconstruction=False))
    def initialize(self):
        args=['train.py','-s',self.source,'-m',self.fast,'-r',self.c.get('initialization',{}).get('resolution',4),
              '--iterations',self.iteration,'--save_iterations',self.iteration,'--test_iterations',self.iteration]
        args+=self.c.get('initialization',{}).get('extra_args',[])
        self.command('01_initialize','fastgs',args,[self.ply,self.fast/'cfg_args'])
    def orthophoto(self):
        size=self.c.get('orthophoto',{}).get('size',6000)
        self.command('02_orthophoto','fastgs',['render_DOM.py','--ply',self.ply,'--out',self.out/'02_dom/DOM.png',
            '--width',size,'--height',size,'--sh-degree',0,'--orientation-mode','world_z'],[self.out/'02_dom/DOM.png'])
    def segment(self):
        args=[ENGINES['fastgs']/'sam3_scripts/run_sam3_building.py','--image',self.out/'02_dom/DOM.png',
              '--out-dir',self.out/'03_sam3','--sam3-root',ENGINES['sam3']]
        if self.c.get('sam3_checkpoint'): args+=['--checkpoint',self.c['sam3_checkpoint']]
        self.command('03_segment','sam3',args,[self.out/'03_sam3/TOM_mask_sam3.png'])
    def propagate(self):
        args=['render_from_DOM_mask.py','--model-dir',self.fast,'--source-dir',self.source,
              '--dom-mask',self.out/'03_sam3/TOM_mask_sam3.png','--dom-image',self.out/'02_dom/DOM.png',
              '--out-dir',self.semantic,'--iteration',self.iteration,'--orientation-mode','world_z',
              '--floating-high-percentile',100,'--floating-low-percentile',0,'--floating-height-margin',1]
        self.command('04_propagate','fastgs',args,[self.semantic/'pipeline_summary.json',self.masks])

        info=json.loads((self.semantic/'03_json/training_label_ids.json').read_text());all_ids=info['label_ids']
        valid=all_ids if self.c.get('valid_ids') is None else self.c['valid_ids']
        if not valid or len(set(valid))!=len(valid) or not set(valid)<=set(all_ids):
            raise ValueError('valid_ids must be a nonempty, distinct subset of generated positive labels')
        write_json(self.out/'instances.json',{'all_ids':all_ids,'valid_ids':valid,
            'excluded_ids':sorted(set(all_ids)-set(valid)), 'background':0,
            'mapping':[{'level2_id':i,'training_mask_value':i,'gaussian_instance_id':i-1} for i in all_ids]})
        write_json(self.out/'training_values.json',{str(i-1):i for i in all_ids})
        self.command('04_validate_masks','mesh',['-m','semlite_gs.validate','--output',self.out],
                     [self.out/'mask_audit.json'])
    def guard_one(self,iid,gpu):
        root=self.out/'05_context'/f'instance_{iid:03d}';guard=root/'guard'
        common=self.common([iid],root/'scratch')
        self.command(f'05_guard_{iid:03d}','pgsr',['-m','pipeline.generate_guard']+common+
                     ['--output_dir',guard,'--edge_pixels',0,'--guard_outer_pixels',16],[guard/'summary.json'],gpu)
        candidate=root/'candidate'
        self.command(f'05_collect_{iid:03d}','pgsr',['scripts/collect_pgsr_frozen_union.py']+common+
                     ['--guard_dir',guard,'--output_dir',candidate],[candidate/'candidate_3d.ply'],gpu)

        self.command(f'05_validate_{iid:03d}','pgsr',['scripts/validate_pgsr_frozen_union.py']+common+
                     ['--guard_dir',guard,'--candidate_ply',candidate/'candidate_3d.ply',
                      '--keep_indices',candidate/'keep_indices.npy','--report',candidate/'validation.json'],
                     [candidate/'validation.json'],gpu)
        write_json(root/'state.json',{'final_init_ply':str(candidate/'candidate_3d.ply')})
    def guard(self):
        self.parallel(self.ids(),self.guard_one)
        for n,ids in enumerate(self.groups()):
            group=self.out/'05_context'/f'group_{n:03d}'
            self.command(f'05_union_{n:03d}','pgsr',['scripts/build_instance_context_union.py','--jobs-root',self.out/'05_context',
                         '--instance-ids',','.join(map(str,ids)),'--output-dir',group],
                         [group/'context_manifest.json',group/'union_3d.ply'])
    def train(self):
        def one(item,gpu):
            n,ids=item;group=self.out/'05_context'/f'group_{n:03d}';model=self.out/'06_training'/f'group_{n:03d}'
            args=['-m','pipeline.train']+self.common(ids,model,group/'union_3d.ply')+[
                '--instance_context_manifest',group/'context_manifest.json','--instance_context_mode','independent',
                '--compact_active_gaussians','--camera_batch_size',len(ids),
                '--frozen_context_guard_dir',self.out/'05_context'/f'instance_{ids[0]:03d}'/'guard',
                '--frozen_context_guard_weight',self.c.get('training',{}).get('guard_weight',1.),
                '--iterations',self.steps,'--save_iterations',self.steps,'--checkpoint_iterations',self.steps,
                '--seed',22,'--multi_view_min_dis',self.c.get('training',{}).get('multi_view_min_dis',.01),
                '--multi_view_max_dis',self.c.get('training',{}).get('multi_view_max_dis',20.)]
            if len(ids)>1: args+=['--multi_instance_batch']
            args+=self.c.get('training',{}).get('extra_args',[])
            self.command(f'06_train_{n:03d}','pgsr',args,[model/'point_cloud'/f'iteration_{self.steps}'/'point_cloud.ply'],gpu)
        self.parallel(list(enumerate(self.groups())),one)
    def field(self):
        owners=[(iid,n) for n,ids in enumerate(self.groups()) for iid in ids]
        def one(item,gpu):
            iid,n=item;target=self.out/'07_fields'/f'instance_{iid:03d}'
            args=['-m','pipeline.field','-s',self.source,'-m',self.out/'06_training'/f'group_{n:03d}',
                  '-r',self.c.get('training',{}).get('resolution',1),'--instance_masks',self.masks,
                  '--instance_id',iid,'--model_instance_id',iid,'--iteration',self.steps,
                  '--voxel_size',self.c.get('tsdf',{}).get('voxel_size',.3),
                  '--depth_sampling_stride',self.c.get('tsdf',{}).get('stride',1),
                  '--max_depth',self.c.get('tsdf',{}).get('max_depth',1000),
                  '--min_alpha',self.c.get('tsdf',{}).get('min_alpha',.5),'--output_dir',target,'--use_depth_filter']
            self.command(f'07_field_{iid:03d}','pgsr',args,[target/'observed_tsdf.npz',target/'tsdf_report.json'],gpu)
        self.parallel(owners,one)
    def envelope(self):

        def one(iid):
            args=['-m','semlite_gs.envelope','--field',self.out/'07_fields'/f'instance_{iid:03d}/observed_tsdf.npz',
                  '--output',self.out/'08_lightweight','--instance',iid]
            opts=self.c.get('lightweight',{})
            if opts.get('max_faces'): args+=['--max-faces',opts['max_faces']]
            if opts.get('largest_component',True): args+=['--largest-component']
            self.command(f'08_envelope_{iid:03d}','mesh',args,[self.out/'08_lightweight'/f'instance_{iid:03d}/mesh.ply'])
        with ThreadPoolExecutor(max_workers=self.c.get('mesh_workers',3)) as pool:list(pool.map(one,self.ids()))
    def export(self):
        self.command('09_export','mesh',['-m','semlite_gs.export','--output',self.out,'--scene',self.c['scene']],
                     [self.out/'09_exports/lightweight/fusion_manifest.json'])
        if self.c.get('export_dense',False):
            for iid in self.ids():
                self.command(f'09_dense_{iid:03d}','mesh',['-m','semlite_gs.dense','--field',
                    self.out/'07_fields'/f'instance_{iid:03d}/observed_tsdf.npz','--output',
                    self.out/'09_exports/dense'/f'instance_{iid:03d}.ply','--largest-component'],
                    [self.out/'09_exports/dense'/f'instance_{iid:03d}.ply'])
            self.command('09_dense_fusion','mesh',['-m','semlite_gs.export','--output',self.out,'--scene',self.c['scene'],'--kind','dense'],[self.out/'09_exports/dense/fusion_manifest.json'])
        if self.c.get('evaluation_manifest'):
            self.command('10_evaluate','mesh',['-m','semlite_gs.evaluate','--manifest',self.c['evaluation_manifest'],
                         '--output',self.out/'10_evaluation'],[self.out/'10_evaluation/summary.json'])


def main():
    p=argparse.ArgumentParser(prog='SemLite-GS', description='Reconstruct lightweight building meshes from registered UAV images.');p.add_argument('command',choices=['run','plan']);p.add_argument('--config',required=True)
    p.add_argument('--from-stage',choices=STAGES,default=STAGES[0]);p.add_argument('--to-stage',choices=STAGES,default=STAGES[-1])
    a=p.parse_args();c=load_config(a.config)
    stages=STAGES[STAGES.index(a.from_stage):STAGES.index(a.to_stage)+1]
    if not stages: p.error('Stage interval is reversed')
    if a.command=='plan':
        print(json.dumps({'pipeline':'SemLite-GS','stages':stages,'config':c,'lightweight_dependency':'signed, weighted TSDF NPZ field',
                          'optional_dense_output':c.get('export_dense',False)},indent=2));return
    run=SemLiteGS(c)
    for stage in stages: getattr(run,stage)()

if __name__=='__main__':main()
