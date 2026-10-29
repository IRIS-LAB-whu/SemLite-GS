import hashlib,json,subprocess,time,fcntl
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .geometry import audit,write_ply

def binary():
    src=Path(__file__).with_name('plane_budget.cpp');digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    dest=Path('/tmp')/('envelope_plane_budget_'+digest)
    with dest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not dest.exists():
            tmp=dest.with_suffix('.tmp')
            subprocess.run(['g++','-O2','-std=c++17','-I/usr/include/eigen3',str(src),'-o',str(tmp),'-lgmp','-lmpfr'],check=True,timeout=180)
            tmp.replace(dest)
    return dest

def planes(mesh,coverage):

    adjacency=mesh.face_adjacency
    good=np.einsum('ij,ij->i',mesh.face_normals[adjacency[:,0]],mesh.face_normals[adjacency[:,1]])>1-1e-10
    a=adjacency[good];n=len(mesh.faces)
    _,labels=connected_components(coo_matrix((np.ones(len(a)),(a[:,0],a[:,1])),shape=(n,n)).tocsr(),directed=False)
    mass=np.bincount(labels,weights=mesh.area_faces);order=np.argsort(-mass,kind='stable')
    if coverage<=0:chosen=[]
    else:chosen=order[:int(np.searchsorted(np.cumsum(mass[order]),coverage*mass.sum()))+1]
    eq=[];tags=[set() for _ in mesh.vertices]
    for label in chosen:
        faces=np.flatnonzero(labels==label);normal=mesh.face_normals[faces[np.argmax(mesh.area_faces[faces])]]
        vids=np.unique(mesh.faces[faces]);d=-float(np.mean(mesh.vertices[vids]@normal))
        if np.max(abs(mesh.vertices[vids]@normal+d))>1e-5:continue
        index=len(eq);eq.append(np.r_[normal,d])
        for v in vids:tags[v].add(index)
    return eq,tags

def reduce_to_budget(mesh,target,output,executable=None):
    if target<4:raise ValueError('Target must be at least 4 faces')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    original=mesh;current=mesh.copy();checks=audit(current.vertices,current.faces)
    if not checks['valid']:raise ValueError('Invalid source topology')
    started=time.monotonic();records=[];bounds=output/'bounds.txt'
    np.savetxt(bounds,original.bounds.reshape(1,-1),fmt='%.17g')


    stages=[(c,0) for c in [1.,.98,.95,.90,.80,.60,.30,0.]]+[(0.,m) for m in [1,2,3,1,2]]
    for index,(coverage,mode) in enumerate(stages):
        if len(current.faces)<=target:break
        exe=executable or binary();eq,tags=planes(current,coverage)
        src=output/('stage_%02d.off'%index);label=output/('stage_%02d.planes.txt'%index);dst=output/('stage_%02d.ply'%index)
        current.export(src,digits=17)
        with label.open('w') as f:
            f.write('%d %d\n'%(len(eq),len(current.vertices)))
            for p in eq:f.write(' '.join('%.17g'%x for x in p)+'\n')
            for p,ids in zip(current.vertices,tags):f.write(' '.join('%.17g'%x for x in p)+' '+str(len(ids))+' '+ ' '.join(map(str,sorted(ids)))+'\n')
        proc=subprocess.run([str(exe),str(src),str(label),str(target),str(dst),'45',str(bounds),str(mode)],capture_output=True,text=True,timeout=60)
        record=dict(coverage=coverage,placement_mode=mode,protected_planes=len(eq),before_faces=len(current.faces),returncode=proc.returncode)
        if proc.returncode:record['error']=proc.stderr[-2000:];records.append(record);continue
        candidate=trimesh.load(dst,process=False);quality=audit(candidate.vertices,candidate.faces)
        accepted=quality['valid'] and quality['components']==checks['components'] and quality['open_boundary_loops']==checks['open_boundary_loops']
        record.update(after_faces=len(candidate.faces),accepted=accepted,topology=quality,native=json.loads(proc.stdout))
        records.append(record)
        if accepted:current=candidate


    if 0<len(current.faces)-target<=max(16,int(.02*target)):
        removed_area=0.;removed=0
        while len(current.faces)>target:
            e=np.sort(current.faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
            _,inv,counts=np.unique(e,axis=0,return_inverse=True,return_counts=True)
            boundary=(counts[inv]==1).reshape(-1,3).any(axis=1)
            choices=np.flatnonzero(boundary);choices=choices[np.argsort(current.area_faces[choices])]
            changed=False
            for i in choices:
                trial=current.copy();trial.update_faces(np.arange(len(trial.faces))!=i);trial.remove_unreferenced_vertices()
                q=audit(trial.vertices,trial.faces)
                if q['valid'] and q['components']==checks['components'] and q['open_boundary_loops']==checks['open_boundary_loops']:
                    removed_area+=float(current.area_faces[i]);removed+=1;current=trial;changed=True;break
            if not changed:break
        records.append(dict(operation='smallest_existing_boundary_ear_trim',removed_faces=removed,removed_area_m2=removed_area,after_faces=len(current.faces)))
    report=dict(target_faces=target,source_faces=len(original.faces),faces=len(current.faces),budget_met=len(current.faces)<=target,
                stages=records,topology=audit(current.vertices,current.faces),seconds=time.monotonic()-started,
                method='fitted envelope plus non-QEM Lindstrom-Turk contractions with progressively relaxed exact plane constraints',uses_lidar=False)
    if not report['budget_met']:report['state']='budget_unmet'
    else:report['state']='complete'
    write_ply(output/'mesh.ply',np.asarray(current.vertices),np.asarray(current.faces))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return current,report
