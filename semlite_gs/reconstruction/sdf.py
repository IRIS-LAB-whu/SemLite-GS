from pathlib import Path
import hashlib
import numpy as np
import json
import time


class SparseTSDF:
    def __init__(self, path, minimum_weight=3.):
        self.path = str(Path(path).resolve())
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data['coordinates'])
            self.phi = np.asarray(data['phi'], dtype=float)
            self.weight = np.asarray(data['weight'], dtype=float)
            self.voxel = float(data['voxel'])
            self.offset = np.asarray(data.get('lattice_offset', [0.,0.,0.]), dtype=float)
            self.trunc = float(data.get('sdf_trunc', 5*self.voxel))
        if (raw.ndim != 2 or raw.shape[1] != 3 or not len(raw) or
                not np.isfinite(raw).all() or not np.array_equal(raw, np.rint(raw)) or
                self.phi.shape != (len(raw),) or self.weight.shape != self.phi.shape or
                not np.isfinite(self.phi).all() or not np.isfinite(self.weight).all() or
                np.any(self.weight < 0) or not np.isfinite(self.voxel) or self.voxel <= 0):
            raise ValueError('Invalid metric sparse TSDF field')
        if (self.offset.shape != (3,) or not np.isfinite(self.offset).all() or
                not np.isfinite(self.trunc) or self.trunc <= 0):
            raise ValueError('Invalid TSDF lattice origin or truncation distance')
        coords = raw.astype(np.int64)
        self.lower = coords.min(0); self.upper = coords.max(0)
        self.span = self.upper - self.lower + 1
        if np.prod(self.span.astype(object)) >= np.iinfo(np.int64).max:
            raise ValueError('TSDF lattice key would overflow')
        keys = self.key(coords)
        self.order = np.argsort(keys); self.keys = keys[self.order]
        if np.any(np.diff(self.keys) == 0):
            raise ValueError('Duplicate TSDF lattice coordinates')
        self.minimum_weight = minimum_weight

    def key(self, coords):
        c = coords - self.lower
        return (c[:, 0] * self.span[1] + c[:, 1]) * self.span[2] + c[:, 2]

    def query(self, xyz):

        xyz = np.asarray(xyz, float); grid = xyz / self.voxel - self.offset
        base = np.floor(grid).astype(np.int64); frac = grid-base
        valid = np.all((base >= self.lower) & (base+1 <= self.upper), axis=1)
        value = np.zeros(len(xyz)); gradient = np.zeros_like(xyz)
        confidence = np.full(len(xyz), np.inf)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    offset = np.array([dx, dy, dz]); key = self.key(base+offset)
                    pos = np.searchsorted(self.keys, key)
                    safe = np.minimum(pos, len(self.keys)-1)
                    found = (pos < len(self.keys)) & (self.keys[safe] == key)
                    ids = self.order[safe]; phi = self.phi[ids]; weight = self.weight[ids]
                    valid &= found & (weight >= self.minimum_weight)

                    valid &= np.abs(phi) < .99*self.trunc
                    confidence = np.minimum(confidence, weight)
                    factors = np.where(offset, frac, 1-frac)
                    value += phi*np.prod(factors, axis=1)
                    for axis in range(3):
                        others = [a for a in range(3) if a != axis]
                        gradient[:, axis] += (phi*(1 if offset[axis] else -1) *
                            np.prod(factors[:, others], axis=1)/self.voxel)
        return value, gradient, confidence, valid


class JointSDFTerms:

    def __init__(self, vertices, faces, owner, context):
        self.vertices = np.asarray(vertices, float); self.faces = np.asarray(faces, int)
        self.basis = np.asarray(context['basis'], float)
        if self.basis.shape != (2, 2) or not np.allclose(self.basis.T@self.basis, np.eye(2), atol=1e-8):
            raise ValueError('SDF world/local basis must be orthonormal')
        self.strength = float(context.get('strength', .25))
        self.max_step = float(context.get('max_step_m', .15))
        if not np.isfinite(self.strength) or self.strength <= 0 or not 0 < self.max_step <= .5:
            raise ValueError('Invalid joint SDF strength or trust region')
        self.field = SparseTSDF(context['field_path'])
        count = int(context.get('samples', 12000))
        if not 30 <= count <= 100000:
            raise ValueError('Joint SDF samples must be between 30 and 100000')
        tri = self.vertices[self.faces]
        cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        area = np.linalg.norm(cross, axis=1)/2
        rng = np.random.default_rng(20261029)
        ids = rng.choice(len(tri), size=count, p=area/area.sum())
        uv = rng.random((count, 2)); root = np.sqrt(uv[:, 0])
        bary = np.c_[1-root, root*(1-uv[:, 1]), root*uv[:, 1]]
        samples = np.einsum('ni,nij->nj', bary, tri[ids])
        world = self.to_world(samples)
        phi, gradient, confidence, valid = self.field.query(world)
        gradient[:, :2] = gradient[:, :2]@self.basis
        length = np.linalg.norm(gradient, axis=1)
        normal = cross[ids]/np.maximum(2*area[ids, None], 1e-30)
        alignment = np.abs(np.einsum('ni,ni->n', gradient, normal))/np.maximum(length, 1e-30)
        valid &= (length >= .2) & (length <= 5.) & (alignment >= .8)
        residual = phi/np.maximum(length, .2)
        valid &= np.abs(residual) <= min(.6, 3*self.field.voxel)

        plane = np.asarray(owner)[ids]
        counts = np.bincount(plane[valid], minlength=int(np.max(owner))+1)
        valid &= counts[plane] >= 8
        if valid.sum() < 30:
            valid[:] = False
        self.ids = self.faces[ids[valid]]; self.bary = bary[valid]
        self.samples = samples[valid]; self.gradient = gradient[valid]/length[valid, None]
        self.gradient_norm = length[valid]; self.residual = residual[valid]

        robust = np.minimum(1., .15/np.maximum(np.abs(self.residual), 1e-12))
        self.weights = (area.sum()/count * np.minimum(confidence[valid]/30., 1.) *
                        alignment[valid]**2 * robust)
        self.base_energy = float(np.dot(self.weights, self.residual**2))
        kinds = np.abs(normal[valid, 2])
        self.metadata = dict(enabled=True, model='local_gradient_linearized_TSDF_zero_surface',
            field=self.field.path, field_sha256=self._hash(), strength=self.strength,
            samples_requested=count, samples_supported=int(valid.sum()),
            supported_planes=int(len(np.unique(plane[valid]))),
            roof_samples=int((kinds > .8).sum()), wall_samples=int((kinds < .5).sum()),
            other_samples=int(((kinds >= .5)&(kinds <= .8)).sum()),
            minimum_corner_weight=3., max_coordinate_step_m=self.max_step,
            gradient_normalized=True, unknown_voxels_filled=False,
            face_indices_fixed=True, plane_directions_fixed=True,
            status='supported' if valid.any() else 'no_reliable_field_support')

    def _hash(self):
        h=hashlib.sha256()
        with open(self.field.path, 'rb') as f:
            for block in iter(lambda:f.read(4194304), b''): h.update(block)
        return h.hexdigest()

    def to_world(self, local):
        return np.c_[local[:, :2]@self.basis.T, local[:, 2]]

    def quadratic(self, vertex_basis):
        blocks = vertex_basis.reshape(len(self.vertices), 3, -1); k=blocks.shape[2]
        H=np.zeros((k,k)); g=np.zeros(k)
        for first in range(0, len(self.ids), 512):
            s=slice(first, first+512)
            derivative=np.einsum('ni,nicj,nc->nj', self.bary[s], blocks[self.ids[s]], self.gradient[s])
            w=self.strength*self.weights[s]
            H += derivative.T@(w[:, None]*derivative)
            g -= derivative.T@(w*self.residual[s])
        return H,g

    def evaluate(self, displacement):
        if not len(self.ids):
            return dict(acceptable=True, valid_samples=0, rms_before_m=None, rms_after_m=None)
        moved=self.samples+np.einsum('ni,nij->nj', self.bary, displacement[self.ids])
        phi,_,_,valid=self.field.query(self.to_world(moved))
        if not valid.all():
            return dict(acceptable=False, valid_samples=int(valid.sum()), reason='field_support_lost')
        residual=phi/self.gradient_norm
        energy=float(np.dot(self.weights, residual**2)); w=float(self.weights.sum())
        return dict(acceptable=bool(energy <= self.base_energy+max(1e-12,self.base_energy*1e-8)),
                    valid_samples=len(phi), rms_before_m=float(np.sqrt(self.base_energy/w)),
                    rms_after_m=float(np.sqrt(energy/w)), energy_before=self.base_energy, energy_after=energy)


def refine_local_envelope(vertices, faces, centers, normals, area, native_vertices,
                          native_faces, context, checkpoint_path=None):


    import time
    import trimesh
    import open3d as o3d
    from .plane_fit import joint_refit
    started=time.monotonic(); baseline=np.asarray(vertices,float).copy()
    max_displacement=float(context.get("max_vertex_displacement_m", .075))
    if not 0 < max_displacement <= .075:
        raise ValueError("Local envelope displacement must be in (0, 0.075] metres")
    config=dict(context, max_step_m=max_displacement/np.sqrt(3.))
    candidate,fit=joint_refit(baseline,faces,centers,normals,area,
                            sdf_context=config,checkpoint_path=checkpoint_path)
    field=fit.get('joint_sdf',{})
    report=dict(enabled=True, stage=context.get('stage','after_joint_plane_fit'), accepted=False,
                max_vertex_displacement_m=max_displacement, fit=fit,
                field_source=context['field_path'])
    if not field.get('samples_supported',0):
        report.update(reason='no_reliable_field_support_or_fit_timeout',seconds=time.monotonic()-started)
        return baseline,report
    if (not field.get('acceptable',False) or
            field.get('energy_after',np.inf)>=field.get('energy_before',0.)-1e-10):
        report.update(reason='no_field_improvement',seconds=time.monotonic()-started)
        return baseline,report
    displacement=float(np.linalg.norm(candidate-baseline,axis=1).max())
    report['actual_max_vertex_displacement_m']=displacement
    if displacement>max_displacement+1e-8:
        report.update(reason='displacement_guard',seconds=time.monotonic()-started)
        return baseline,report

    def scene(v,f):
        s=o3d.t.geometry.RaycastingScene(nthreads=1)
        s.add_triangles(o3d.core.Tensor(np.asarray(v,np.float32)),o3d.core.Tensor(np.asarray(f,np.uint32)))
        return s
    def distance(s,points):
        return s.compute_distance(o3d.core.Tensor(np.asarray(points,np.float32))).numpy()
    native=trimesh.Trimesh(native_vertices,native_faces,process=False)
    fixed,_=trimesh.sample.sample_surface(native,20000,seed=20261029)
    native_scene=scene(native_vertices,native_faces)
    def proxy(v):
        mesh=trimesh.Trimesh(v,faces,process=False)
        points,_=trimesh.sample.sample_surface(mesh,20000,seed=20261029)
        forward=distance(native_scene,points);reverse=distance(scene(v,faces),fixed)
        return dict(mesh_to_TSDF_m=float(forward.mean()),TSDF_to_mesh_m=float(reverse.mean()),
                    mesh_to_TSDF_p95_m=float(np.quantile(forward,.95)),TSDF_to_mesh_p95_m=float(np.quantile(reverse,.95)))
    before=proxy(baseline);after=proxy(candidate)
    limits={k:max(.005 if 'p95' in k else .001,.01*value) for k,value in before.items()}
    report.update(surface_proxy_before=before,surface_proxy_after=after,surface_guard_allowance_m=limits,
                  source_samples=20000,metric='zero_surface_distance')
    if any(after[k]>before[k]+limits[k] for k in before):
        report.update(reason='TSDF_surface_guard',seconds=time.monotonic()-started)
        return baseline,report
    report.update(accepted=True,reason='field_improved_and_surface_guard_passed',seconds=time.monotonic()-started)
    return candidate,report


def stencil(field, points):

    grid=np.asarray(points,float)/field.voxel-field.offset
    base=np.floor(grid).astype(np.int64);frac=grid-base
    ids=[];basis=[];gradient=[]
    for x in (0,1):
        for y in (0,1):
            for z in (0,1):
                offset=np.array([x,y,z]);key=field.key(base+offset)
                pos=np.searchsorted(field.keys,key)
                if np.any(pos>=len(field.keys)) or not np.array_equal(field.keys[np.minimum(pos,len(field.keys)-1)],key):
                    raise ValueError('Stencil contains unknown field nodes')
                ids.append(field.order[pos]);factors=np.where(offset,frac,1-frac)
                basis.append(np.prod(factors,axis=1));g=[]
                for axis in range(3):
                    g.append((1 if offset[axis] else -1)*np.prod(factors[:,[a for a in range(3) if a!=axis]],axis=1)/field.voxel)
                gradient.append(np.stack(g,axis=1))
    return np.stack(ids,axis=1),np.stack(basis,axis=1),np.stack(gradient,axis=1)


def optimize_local_field(field_path, vertices, faces, output, *, iterations=12,
                         maximum_correction_m=.03, lambda_eik=.02,
                         lambda_smooth=.1, samples=6000):
    import torch
    import trimesh
    import open3d as o3d
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not 1<=iterations<=50 or not 0<maximum_correction_m<=.1 or lambda_eik<=0 or lambda_smooth<=0:
        raise ValueError('Invalid short local field optimization policy')
    torch.set_num_threads(1)
    field=SparseTSDF(field_path)
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    points,face_ids=trimesh.sample.sample_surface(mesh,samples,seed=20261029)
    phi,grad,weight,valid=field.query(points)
    length=np.linalg.norm(grad,axis=1)
    align=np.abs(np.einsum('ij,ij->i',grad,mesh.face_normals[face_ids]))/np.maximum(length,1e-12)
    valid &= (length>=.2)&(length<=5)&(align>=.8)&(np.abs(phi)/np.maximum(length,.2)<min(.6,2*field.voxel))
    plane_points=points[valid];plane_weights=np.minimum(weight[valid]/30.,1)*align[valid]**2
    report=dict(enabled=True,parameterization='phi = phi0 + eta * delta',nonnegative_delta=True,
        losses=['L_plane','L_eik','L_smooth_delta'],lambda_eik=lambda_eik,lambda_smooth=lambda_smooth,
        maximum_correction_m=maximum_correction_m,iterations_requested=iterations,
        eta='fixed confidence times proximity to finite envelope faces; zero outside reliable narrow band',
        initial_field_source='measured weighted TSDF',field_source=str(Path(field_path).resolve()),
        field_sha256=hashlib.sha256(Path(field_path).read_bytes()).hexdigest(),plane_samples=len(plane_points),
        uses_LiDAR=False,weights_unchanged=True,unknown_voxels_filled=False)
    if len(plane_points)<30:
        report.update(accepted=False,reason='insufficient_reliable_plane_support',iterations_run=0,seconds=time.monotonic()-started)
        (output/'field_loss_report.json').write_text(json.dumps(report,indent=2));return str(field_path),report
    rng=np.random.default_rng(20261029)
    eik_points=plane_points+rng.uniform(-.25,.25,plane_points.shape)*field.voxel
    ep,eg,ew,ev=field.query(eik_points);ev &= np.abs(ep)<min(.6,2*field.voxel)
    eik_points=eik_points[ev];eik_weights=np.minimum(ew[ev]/30.,1.)
    if len(eik_points)<30:
        report.update(accepted=False,reason='insufficient_reliable_eikonal_support',iterations_run=0,seconds=time.monotonic()-started)
        (output/'field_loss_report.json').write_text(json.dumps(report,indent=2));return str(field_path),report
    pi,pb,pg=stencil(field,plane_points);ei,eb,eg=stencil(field,eik_points)
    used,inverse=np.unique(np.r_[pi.ravel(),ei.ravel()],return_inverse=True)
    pidx=inverse[:pi.size].reshape(pi.shape);eidx=inverse[pi.size:].reshape(ei.shape)
    with np.load(field_path,allow_pickle=False) as data:coords=np.asarray(data['coordinates'])[used]
    xyz=(coords+field.offset)*field.voxel
    ray=o3d.t.geometry.RaycastingScene(nthreads=1)
    ray.add_triangles(o3d.core.Tensor(np.asarray(vertices,np.float32)),o3d.core.Tensor(np.asarray(faces,np.uint32)))
    closest=ray.compute_closest_points(o3d.core.Tensor(xyz.astype(np.float32)))
    q=closest['points'].numpy();face=closest['primitive_ids'].numpy().astype(int)
    distance=np.linalg.norm(q-xyz,axis=1)
    qp,qg,qw,qv=field.query(q);ql=np.linalg.norm(qg,axis=1)
    qa=np.abs(np.einsum('ij,ij->i',qg,mesh.face_normals[face]))/np.maximum(ql,1e-12)
    radius=1.5*field.voxel
    active=qv&(ql>=.2)&(ql<=5)&(qa>=.8)&(distance<radius)&(np.abs(qp)/np.maximum(ql,.2)<min(.6,2*field.voxel))
    active &= (field.weight[used]>=3)&(np.abs(field.phi[used])<min(.6,2*field.voxel,field.trunc-maximum_correction_m))
    eta=np.where(active,np.minimum(np.minimum(field.weight[used],qw)/30.,1)*qa**2*np.exp(-2*(distance/radius)**2),0.)
    eta=np.clip(eta,0,1)
    keys=field.key(coords);order=np.argsort(keys);sorted_keys=keys[order]
    edge_a=[];edge_b=[];boundary=[]
    for axis in range(3):
        for direction in (-1,1):
            neighbor=coords.copy();neighbor[:,axis]+=direction;key=field.key(neighbor)
            pos=np.searchsorted(sorted_keys,key);safe=np.minimum(pos,len(keys)-1)
            found=(pos<len(keys))&(sorted_keys[safe]==key)&np.all((neighbor>=field.lower)&(neighbor<=field.upper),axis=1)
            aa=np.flatnonzero(found);bb=order[safe[found]]
            take=active[aa]|active[bb];edge_a.extend(aa[take]);edge_b.extend(bb[take]);boundary.extend(np.flatnonzero(~found&active))
    def tensor(x,dtype=torch.float64):return torch.as_tensor(np.asarray(x),dtype=dtype)
    base=tensor(field.phi[used]);influence=tensor(eta);mask=tensor(active)
    pi_t=tensor(pidx,torch.long);ei_t=tensor(eidx,torch.long)
    pb_t=tensor(pb);eg_t=tensor(eg);pw=tensor(plane_weights);ew_t=tensor(eik_weights)
    ia=tensor(edge_a,torch.long);ib=tensor(edge_b,torch.long);boundary_t=tensor(boundary,torch.long)
    delta=torch.zeros(len(used),dtype=torch.float64,requires_grad=True)
    def losses():
        corrected=base+influence*delta
        plane=(corrected[pi_t]*pb_t).sum(1)
        gradient=(corrected[ei_t][:,:,None]*eg_t).sum(1)
        lp=(pw*(plane/field.voxel)**2).sum()/pw.sum().clamp_min(1e-12)
        le=(ew_t*(torch.linalg.vector_norm(gradient,dim=1)-1)**2).sum()/ew_t.sum().clamp_min(1e-12)
        diffs=torch.cat([delta[ia]-delta[ib],delta[boundary_t]])/field.voxel
        ls=(diffs**2).mean() if diffs.numel() else delta.sum()*0
        return lp+lambda_eik*le+lambda_smooth*ls,lp,le,ls
    def record(values):
        a=[float(x.detach()) for x in values];return dict(total=a[0],L_plane=a[1],L_eik=a[2],L_smooth_delta=a[3])
    initial=record(losses());best=initial['total'];best_delta=np.zeros(len(used));history=[dict(iteration=0,**initial)]
    optimizer=torch.optim.Adam([delta],lr=min(.005,maximum_correction_m/4))
    for step in range(1,iterations+1):
        optimizer.zero_grad();values=losses();values[0].backward();optimizer.step()
        with torch.no_grad():delta.clamp_(0,maximum_correction_m);delta.mul_(mask)
        row=record(losses());history.append(dict(iteration=step,**row))
        if np.isfinite(row['total']) and row['total']<best:
            best=row['total'];best_delta=delta.detach().numpy().copy()
    with torch.no_grad():delta.copy_(tensor(best_delta))
    final=record(losses());correction=eta*best_delta
    accepted=bool(best<initial['total']-1e-12 and np.any(correction>1e-10))
    report.update(accepted=accepted,reason='bounded_combined_field_loss_decreased' if accepted else 'unchanged_field_retained',
        eikonal_samples=len(eik_points),candidate_nodes=len(used),influenced_nodes=int(active.sum()),
        corrected_nodes=int(np.count_nonzero(correction>1e-10)),maximum_actual_correction_m=float(correction.max()),
        delta_min_m=float(best_delta.min()),delta_max_m=float(best_delta.max()),eta_min=float(eta.min()),eta_max=float(eta.max()),
        iterations_run=iterations,loss_before=initial,loss_after=final,loss_history=history)
    path=str(field_path)
    if accepted:
        with np.load(field_path,allow_pickle=False) as data:payload={k:data[k].copy() for k in data.files}
        original=np.asarray(payload['phi'],float);updated=original.copy();updated[used]+=correction
        assert np.all(updated>=original) and np.max(updated-original)<=maximum_correction_m+1e-12
        payload['phi']=updated;payload['correction_source']=np.array(str(Path(field_path).resolve()))
        path=str(output/'corrected_tsdf.npz');np.savez_compressed(path,**payload)
    np.savez_compressed(output/'correction.npz',field_row_indices=used,coordinates=coords,eta=eta,delta=best_delta,correction=correction)
    report.update(output_field=path,seconds=time.monotonic()-started)
    (output/'field_loss_report.json').write_text(json.dumps(report,indent=2))
    return path,report

