import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import time
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize,LinearConstraint
from scipy.sparse import coo_matrix
from .geometry import closest
from .geometry import weighted_quantile

PARAMETERS=dict(max_support_distance_m=2.,normal_alignment=.7,tangential_tolerance_m=.04,
 minimum_support_m2=.5,minimum_support_fraction=.08,max_plane_shift_m=1.5,
 vertex_anchor_weight=.01,edge_smoothness_weight=.2,max_solver_iterations=200,
 roof_coverage_margin_m=.15,version='multidirection_joint_v5_block_incidence')


def incidence_null_basis(normals,incidence,vertex_count,base_ids):


    P=len(normals);owners=[[] for _ in range(vertex_count)]
    for p,ids in enumerate(incidence):
        for i in ids:owners[int(i)].append(p)
    base=set(map(int,base_ids));local=[];compatibility=[];free_count=0
    for i,ids in enumerate(owners):
        ids=np.asarray(ids,dtype=int);B=normals[ids]
        if i in base:B=np.vstack([B,[0.,0.,1.]])
        u,s,vh=np.linalg.svd(B,full_matrices=True)
        rank=int(np.sum(s>s[0]*1e-9))
        inverse=(vh[:rank].T/s[:rank])@u[:,:rank].T
        tangent=vh[rank:].T
        for coefficients in u[:,rank:].T:
            row=np.zeros(P);row[ids]=coefficients[:len(ids)]
            size=np.linalg.norm(row)
            if size>1e-10:compatibility.append(row/size)
        local.append((ids,inverse[:,:len(ids)],tangent,free_count))
        free_count+=tangent.shape[1]
    R=null_space(np.asarray(compatibility),rcond=1e-9) if compatibility else np.eye(P)
    K=R.shape[1];Z=np.zeros((3*vertex_count+P,K+free_count));Z[3*vertex_count:,:K]=R
    for i,(ids,inverse,tangent,start) in enumerate(local):
        Z[3*i:3*i+3,:K]=inverse@R[ids]
        Z[3*i:3*i+3,K+start:K+start+tangent.shape[1]]=tangent
    return Z


def roof_boundary_guards(v,f,c,normals,area,pn,pd,owner,fn):


    edges={}
    for face,tri in enumerate(f):
        for i in range(3):edges.setdefault(tuple(sorted((int(tri[i]),int(tri[(i+1)%3])))),[]).append(face)
    records=[]
    for edge,faces in edges.items():
        if len(faces)!=2:continue
        for roof,wall in [faces,faces[::-1]]:
            if fn[roof,2]<.65 or abs(fn[wall,2])>.5:continue
            p=int(owner[wall]);sign=1. if fn[wall]@pn[p]>0 else -1.;out=sign*pn[p];offset=sign*pd[p]
            if v[f[roof]].mean(0)@out-offset>=-1e-6:continue
            x,y=v[list(edge)];length=np.linalg.norm(y-x)
            if length<1:continue
            tangent=(y-x)/length;along=(c-x)@tangent
            roof_normal=fn[roof];height_error=abs((c-v[f[roof,0]])@roof_normal)
            valid=(along>.1)&(along<length-.1)&(height_error<.65)&(abs(c@out-offset)<1.5)&(normals@roof_normal>.65)
            weights=area[valid]*(normals[valid]@roof_normal)
            if weights.sum()<max(.5,length*.12):continue
            bins=np.clip((along[valid]/length*10).astype(int),0,9)
            coverage=float(np.mean(np.bincount(bins,weights=weights,minlength=10)>.025))
            if coverage<.5:continue
            bound=weighted_quantile(c[valid]@out,weights,.95)
            lower=min(0.,bound-offset-PARAMETERS['roof_coverage_margin_m'])
            records.append(dict(plane=p,sign=sign,minimum_outward_shift_m=float(lower),edge=list(edge),support_m2=float(weights.sum()),edge_coverage=coverage))
    return records


def plane_groups(vertices,faces):
    triangles=vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=np.linalg.norm(cross,axis=1)/2
    normals=cross/np.maximum(2*area[:,None],1e-20)


    directions=normals.copy()
    signs=np.sign(directions[np.arange(len(faces)),np.argmax(abs(directions),axis=1)])
    directions*=signs[:,None]
    offsets=np.einsum('ij,ij->i',directions,triangles.mean(1))
    keys=np.c_[np.round(directions,7),np.round(offsets,6)]
    _,face_plane=np.unique(keys,axis=0,return_inverse=True)
    normals_p=[];offsets_p=[];incidences=[];areas=[]
    for p in range(int(face_plane.max())+1):
        ids=np.flatnonzero(face_plane==p);best=ids[np.argmax(area[ids])]
        normal=directions[best];verts=np.unique(faces[ids]);position=float(np.mean(vertices[verts]@normal))
        normals_p.append(normal);offsets_p.append(position);incidences.append(verts);areas.append(area[ids].sum())
    return np.array(normals_p),np.array(offsets_p),face_plane,incidences,np.array(areas),normals,area


def joint_refit(vertices,faces,centers,normals,area,checkpoint_path=None,runtime=None,sdf_context=None):
    from .fit_runtime import guarded_refit
    return guarded_refit(vertices,faces,centers,normals,area,
                         checkpoint_path=checkpoint_path,runtime=runtime,sdf_context=sdf_context)


def _solve_joint_planes(vertices,faces,centers,normals,area,_checkpoint=None,_runtime=None,_sdf=None):
    started=time.perf_counter();v=np.asarray(vertices,float);f=np.asarray(faces,int)

    origin=v.mean(0);local=v-origin;c=np.asarray(centers)-origin
    pn,pd,owner,incidence,pa,fn,fa=plane_groups(local,f)
    V,P=len(v),len(pn);D=3*V+P
    projection,nearest=closest(local,f,c)
    delta=c-projection;alignment=np.sum(normals*fn[nearest],axis=1)
    n=pn[owner[nearest]];tangent=delta-n*np.sum(delta*n,axis=1)[:,None]
    valid=(alignment>PARAMETERS['normal_alignment'])&(np.linalg.norm(delta,axis=1)<PARAMETERS['max_support_distance_m'])&(np.linalg.norm(tangent,axis=1)<PARAMETERS['tangential_tolerance_m'])
    target=np.zeros(P);weights=np.full(P,.1);records=[]
    for p in range(P):
        selected=valid&(owner[nearest]==p);w=area[selected]*alignment[selected]
        support=float(w.sum());supported=support>=max(PARAMETERS['minimum_support_m2'],PARAMETERS['minimum_support_fraction']*pa[p])
        record=dict(plane=p,normal=pn[p].tolist(),initial_offset_m=float(pd[p]+pn[p]@origin),patch_area_m2=float(pa[p]),support_m2=support,supported=bool(supported))
        if supported:
            values=c[selected]@pn[p]-pd[p]
            target[p]=weighted_quantile(values,w,.5);weights[p]+=support
            record.update(target_shift_m=float(target[p]),q10_shift_m=weighted_quantile(values,w,.1),q90_shift_m=weighted_quantile(values,w,.9))
        records.append(record)
    supported=sum(r['supported'] for r in records)
    report=dict(parameters=PARAMETERS,planes=P,supported_planes=supported,vertices=V,faces=len(f),plane_fit_records=records)
    guards=roof_boundary_guards(local,f,c,normals,area,pn,pd,owner,fn)
    report['roof_coverage_guards']=guards
    sdf_terms=None
    if _sdf is not None:
        from .sdf import JointSDFTerms
        sdf_terms=JointSDFTerms(v,f,owner,_sdf)
        report['joint_sdf']=dict(sdf_terms.metadata, **sdf_terms.evaluate(np.zeros_like(v)))
    if _checkpoint is not None:
        _checkpoint(v,dict(report,state='complete',solver_converged=False,
            solver_message='Retained validated pre-fit mesh',stop_reason='preparation',
            retained_candidate='initial',accepted_step=0.,max_vertex_shift_m=0.,
            max_plane_displacement_residual_m=0.,solver_iterations=0,
            seconds=time.perf_counter()-started))
    if supported==0 and (sdf_terms is None or not len(sdf_terms.ids)):
        report.update(max_plane_displacement_residual_m=0.,accepted_step=0.,state='no_supported_planes',
                      solver_converged=False,stop_reason='no_supported_planes',retained_candidate='initial',seconds=time.perf_counter()-started)
        return v.copy(),report


    rr=[];cc=[];data=[];row=0
    for p,ids in enumerate(incidence):
        for i in ids:
            for axis in range(3):rr.append(row);cc.append(3*i+axis);data.append(pn[p,axis])
            rr.append(row);cc.append(3*V+p);data.append(-1.);row+=1
    base=float(local[:,2].min())
    from .geometry import boundary_vertices


    base_ids=np.union1d(np.flatnonzero(abs(local[:,2]-base)<1e-7),boundary_vertices(f))
    for i in base_ids:
        rr.append(row);cc.append(3*i+2);data.append(1.);row+=1
    C=coo_matrix((data,(rr,cc)),shape=(row,D)).tocsr()
    Z=incidence_null_basis(pn,incidence,V,base_ids)
    incidence_error=float(np.max(abs(C@Z))) if Z.size else 0.
    if incidence_error>1e-7:raise ValueError('Block incidence basis residual exceeds tolerance')
    report['incidence_basis_residual']=incidence_error
    report['degrees_of_freedom']=Z.shape[1]
    if not Z.shape[1]:raise ValueError('No movable degrees of freedom')
    Zv=Z[:3*V];Zp=Z[3*V:];K=Z.shape[1]
    edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)


    blocks=Zv.reshape(V,3,K)
    length=np.linalg.norm(local[edges[:,0]]-local[edges[:,1]],axis=1)
    E=((blocks[edges[:,0]]-blocks[edges[:,1]])/np.maximum(length[:,None,None],1.)).reshape(-1,K)
    H=Zp.T@(weights[:,None]*Zp)+PARAMETERS['vertex_anchor_weight']*(Zv.T@Zv)+PARAMETERS['edge_smoothness_weight']*(E.T@E)
    g=Zp.T@(weights*target)
    if sdf_terms is not None and len(sdf_terms.ids):
        field_H,field_g=sdf_terms.quadratic(Zv)
        H+=field_H;g+=field_g
    scale=max(weights.max(),1.);H/=scale;g/=scale


    transform=np.linalg.solve(np.linalg.cholesky(H).T,np.eye(K))
    Z=Z@transform;g=transform.T@g;H=np.eye(K)
    Zv=Z[:3*V];Zp=Z[3*V:];blocks=Zv.reshape(V,3,K)


    distances=local@pn.T-pd
    nearby=(abs(distances)>1e-9)&(abs(distances)<3.1)
    for p,ids in enumerate(incidence):nearby[ids,p]=False
    ii,pp=np.nonzero(nearby)
    signs=np.sign(distances[ii,pp]);gap=np.minimum(abs(distances[ii,pp])*.15,.02)
    G=(np.einsum('ij,ijk->ik',pn[pp],blocks[ii])-Zp[pp])*signs[:,None]
    lower=gap-abs(distances[ii,pp])
    max_shift=PARAMETERS['max_plane_shift_m']
    if sdf_terms is not None and len(sdf_terms.ids):max_shift=min(max_shift,sdf_terms.max_step)
    A=np.vstack([Z,G]);lb=np.r_[np.full(D,-max_shift),lower]
    ub=np.r_[np.full(D,max_shift),np.full(len(G),np.inf)]
    if guards:
        A=np.vstack([A,np.array([r['sign']*Zp[r['plane']] for r in guards])])
        lb=np.r_[lb,[r['minimum_outward_shift_m'] for r in guards]]
        ub=np.r_[ub,np.full(len(guards),np.inf)]
    initial=local[f];cross0=np.cross(initial[:,1]-initial[:,0],initial[:,2]-initial[:,0])
    unit0=cross0/(2*fa[:,None]);area_scale=np.maximum(fa,1e-3)
    minimum_area=np.maximum(1.01e-10,fa*.01)
    face_blocks=blocks[f]
    def signed_areas(y):
        moved=initial+np.einsum('fvck,k->fvc',face_blocks,y)
        e1=moved[:,1]-moved[:,0];e2=moved[:,2]-moved[:,0]
        return moved,.5*np.sum(np.cross(e1,e2)*unit0,axis=1)
    def geometry_fun(y):
        return (signed_areas(y)[1]-minimum_area)/area_scale
    def geometry_jac(y):
        moved,_=signed_areas(y)
        gb=.5*np.cross(moved[:,2]-moved[:,0],unit0)
        gc=.5*np.cross(unit0,moved[:,1]-moved[:,0]);ga=-gb-gc
        return (np.einsum('fc,fck->fk',ga,face_blocks[:,0])+np.einsum('fc,fck->fk',gb,face_blocks[:,1])+np.einsum('fc,fck->fk',gc,face_blocks[:,2]))/area_scale[:,None]
    unconstrained=np.linalg.solve(H,g)
    within=np.all(A@unconstrained>=lb-1e-9) and np.all(A@unconstrained<=ub+1e-9) and np.all(geometry_fun(unconstrained)>=0)
    def field_accepts(candidate):
        return sdf_terms is None or sdf_terms.evaluate((Zv@candidate).reshape(V,3))['acceptable']
    within=within and field_accepts(unconstrained)
    stop_reason=None
    if within:
        y=unconstrained;iterations=0;message='Unconstrained quadratic optimum is feasible';rounds=0;fallbacks=0;converged=True
    else:
        def objective(y):return float(y@H@y-2*g@y),2*(H@y-g)
        best_y=np.zeros(K);best_value=0.;budget=PARAMETERS['max_solver_iterations'];converged=False
        from .fit_runtime import ProgressGuard,FitStopped
        progress=ProgressGuard(_runtime) if _runtime is not None else None
        def checkpoint_best():
            if _checkpoint is None:return
            moved=Z@best_y
            if sdf_terms is not None:
                report['joint_sdf']=dict(sdf_terms.metadata, **sdf_terms.evaluate(moved[:3*V].reshape(V,3)))
            checkpoint_vertices=v+moved[:3*V].reshape(V,3)
            checkpoint_records=[dict(r,shift_m=float(moved[3*V+p]),
                final_offset_m=float(pd[p]+pn[p]@origin+moved[3*V+p])) for p,r in enumerate(records)]
            _checkpoint(checkpoint_vertices,dict(report,state='complete',solver_converged=False,
                stop_reason='in_progress',retained_candidate='best_feasible' if best_value<0 else 'initial',
                solver_message='Best fully feasible SLSQP candidate',
                plane_fit_records=checkpoint_records,accepted_step=1.,
                solver_callbacks=progress.callbacks if progress else None,
                max_vertex_shift_m=float(np.linalg.norm(moved[:3*V].reshape(V,3),axis=1).max()),
                max_plane_displacement_residual_m=float(abs(C@moved).max()),
                weighted_plane_target_error_before_m=float(np.sum(weights*abs(target))/weights.sum()),
                weighted_plane_target_error_after_m=float(np.sum(weights*abs(moved[3*V:]-target))/weights.sum()),
                seconds=time.perf_counter()-started))
        def consider(candidate,backtrack=False):
            nonlocal best_y,best_value
            candidate=np.asarray(candidate)
            for _ in range(25 if backtrack else 1):
                value=objective(candidate)[0]
                if value>=best_value:return
                values=A@candidate
                if np.all(values>=lb-1e-9) and np.all(values<=ub+1e-9) and np.all(geometry_fun(candidate)>=0) and field_accepts(candidate):
                    best_y=candidate.copy();best_value=value;checkpoint_best();return
                candidate=candidate*.5
        class ExpandConstraints(Exception):
            def __init__(self, candidate):self.candidate=np.asarray(candidate).copy()
        separation_events=[];separation_callbacks=0
        def callback(candidate):
            nonlocal separation_callbacks
            separation_callbacks+=1


            consider(candidate,backtrack=True)
            av=A@candidate
            missing=((av<lb-1e-8)|(av>ub+1e-8))&~active
            missing_faces=(geometry_fun(candidate)<-1e-10)&~active_faces
            if missing.any() or missing_faces.any():
                raise ExpandConstraints(candidate)
            if progress is not None:progress.check(best_value)
        consider(unconstrained,backtrack=True)


        trial=A@unconstrained
        active=(trial<lb+1e-9)|(trial>ub-1e-9)
        active_faces=geometry_fun(unconstrained)<1e-9
        y=np.zeros(K);iterations=0;fallbacks=0
        try:
            for rounds in range(1,21):
                if iterations>=budget:break


                face_ids=np.flatnonzero(active_faces)
                constraints=[]
                if len(face_ids):constraints.append(dict(type='ineq',fun=lambda q:geometry_fun(q)[face_ids],jac=lambda q:geometry_jac(q)[face_ids]))
                if active.any():constraints.insert(0,LinearConstraint(A[active],lb[active],ub[active]))
                callbacks_before=separation_callbacks
                try:
                    result=minimize(objective,y,jac=True,method='SLSQP',constraints=constraints,callback=callback,options=dict(maxiter=min(100,budget-iterations),ftol=1e-10))
                except ExpandConstraints as expansion:
                    values=A@expansion.candidate
                    added=((values<lb-1e-8)|(values>ub+1e-8))&~active
                    added_faces=(geometry_fun(expansion.candidate)<-1e-10)&~active_faces
                    separation_events.append(dict(round=rounds,added_planes=int(added.sum()),added_faces=int(added_faces.sum())))
                    active|=added;active_faces|=added_faces
                    iterations+=max(1,separation_callbacks-callbacks_before);y=best_y.copy()
                    continue
                iterations+=int(result.nit)
                consider(result.x,backtrack=True)
                if not result.success and iterations<budget:


                    undo=np.linalg.inv(transform);fallbacks+=1
                    def original_objective(u):
                        value,gradient=objective(undo@u)
                        return value,undo.T@gradient
                    retry_constraints=[]
                    if active.any():retry_constraints.append(LinearConstraint(A[active]@undo,lb[active],ub[active]))
                    if len(face_ids):retry_constraints.append(dict(type='ineq',fun=lambda u:geometry_fun(undo@u)[face_ids],jac=lambda u:geometry_jac(undo@u)[face_ids]@undo))
                    retry=minimize(original_objective,transform@best_y,jac=True,method='SLSQP',constraints=retry_constraints,callback=lambda u:(consider(undo@u,backtrack=True),progress.check(best_value) if progress is not None else None),options=dict(maxiter=budget-iterations,ftol=1e-10))
                    iterations+=int(retry.nit);consider(undo@retry.x,backtrack=True)
                    if not retry.success:break
                    y=undo@retry.x;message=retry.message
                elif result.success:y=result.x;message=result.message
                else:break
                values=A@y;violated=(values<lb-1e-8)|(values>ub+1e-8)
                violated_faces=geometry_fun(y)<-1e-10
                if not violated.any() and not violated_faces.any():converged=True;break
                if not np.any(violated&~active) and not np.any(violated_faces&~active_faces):break
                active|=violated;active_faces|=violated_faces
        except FitStopped as stopped:
            stop_reason=str(stopped)
            iterations=progress.callbacks if progress is not None else iterations
        if not converged:
            y=best_y;message=(str(stop_reason)+': retained best fully feasible fit' if stop_reason else 'Iteration limit or numerical stall: retained best fully feasible joint fit')
        report.update(early_separation_events=separation_events,total_plane_inequality_rows=len(A),active_plane_inequality_rows=int(active.sum()),
                      total_face_inequality_rows=len(f),active_face_inequality_rows=int(active_faces.sum()))
    if not field_accepts(y):
        y=best_y;converged=False;stop_reason='nonlinear_field_guard'
        message='Retained best fully feasible candidate with non-increasing field residual'
    x=Z@y
    if np.min(A@y-lb)<-1e-6 or np.max(A@y-ub)>1e-6:raise ValueError('Plane arrangement constraint violation')
    displacement=x[:3*V].reshape(V,3)


    alpha=1.
    rejected=[]
    for _ in range(25):
        fitted=v+alpha*displacement;t=fitted[f]
        cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);new_area=np.linalg.norm(cross,axis=1)/2
        if np.all(np.sum(cross*cross0,axis=1)>0) and np.all(new_area>=np.maximum(1e-10,fa*1e-5)):break
        if not rejected:
            rejected=[dict(face=int(j),initial_area_m2=float(fa[j]),trial_area_m2=float(new_area[j]),plane=int(owner[j])) for j in np.flatnonzero((np.sum(cross*cross0,axis=1)<=0)|(new_area<np.maximum(1e-10,fa*1e-5)))]
        alpha*=.5
    else:raise ValueError('Unable to retain face orientation/area')
    x*=alpha
    if sdf_terms is not None:
        report['joint_sdf']=dict(sdf_terms.metadata, **sdf_terms.evaluate(fitted-v))
        if not report['joint_sdf']['acceptable']:

            fitted=v.copy();x*=0.;alpha=0.;converged=False
            stop_reason='field_guard_rejected';message='Retained initial feasible mesh after nonlinear field validation'
            report['joint_sdf']=dict(sdf_terms.metadata, **sdf_terms.evaluate(np.zeros_like(v)))
    for p,r in enumerate(records):r.update(shift_m=float(x[3*V+p]),final_offset_m=float(pd[p]+pn[p]@origin+x[3*V+p]))
    report.update(state='complete',accepted_step=float(alpha),solver_iterations=iterations,solver_converged=converged,solver_message=message,constraint_rounds=rounds,solver_coordinate_fallbacks=fallbacks,backtracking_faces=rejected,
        stop_reason=stop_reason or ('converged' if converged else 'iteration_limit_or_numerical_stall'),
        retained_candidate='optimized' if converged else ('best_feasible' if np.any(x) else 'initial'),
        max_vertex_shift_m=float(np.linalg.norm(fitted-v,axis=1).max()),max_plane_displacement_residual_m=float(abs(C@x).max()),
        weighted_plane_target_error_before_m=float(np.sum(weights*abs(target))/weights.sum()),
        weighted_plane_target_error_after_m=float(np.sum(weights*abs(x[3*V:]-target))/weights.sum()),
        seconds=time.perf_counter()-started)
    return fitted,report
