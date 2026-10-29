import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import hashlib, time, traceback
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely import contains_xy
import open3d as o3d
import manifold3d as manifold
from . import geometry


def orientation_families(normals,area):
    mask=abs(normals[:,2])<.35
    angle=np.arctan2(normals[mask,1],normals[mask,0])%(np.pi/2)
    weights=area[mask]
    hist,edges=np.histogram(angle,bins=180,range=(0,np.pi/2),weights=weights)
    hist=ndimage.gaussian_filter1d(hist,2,mode='wrap')
    peaks,_=find_peaks(np.tile(hist,3),distance=16,prominence=max(hist.max()*.08,1e-8))
    peaks=peaks[(peaks>=180)&(peaks<360)]-180
    modes=[]
    for p in sorted(peaks,key=lambda p:hist[p],reverse=True)[:6]:
        a=(edges[p]+edges[p+1])/2;delta=(angle-a+np.pi/4)%(np.pi/2)-np.pi/4
        keep=abs(delta)<np.deg2rad(5)
        if weights[keep].sum()<.025*weights.sum():continue
        a+=np.arctan2(np.sum(weights[keep]*np.sin(4*delta[keep])),np.sum(weights[keep]*np.cos(4*delta[keep])))/4
        modes.append(a)
    return np.array(modes)


def straighten_polygon(poly,modes,support):

    simple=poly.simplify(.65,preserve_topology=True)
    candidates=np.r_[modes,modes+np.pi/2]
    records=[]
    def ring(coords):
        xy=np.asarray(coords)[:-1];reference=Polygon(xy)


        while len(xy)>4:
            options=[]
            for i in range(len(xy)):
                proposal=np.delete(xy,i,axis=0);p=Polygon(proposal)
                if not p.is_valid:continue
                error=p.boundary.hausdorff_distance(reference.boundary)
                if error<=.9 and p.symmetric_difference(reference).area<max(4.,reference.area*.10):
                    options.append((error,i,proposal))
            if not options:break
            xy=min(options,key=lambda t:(t[0],t[1]))[2]


        if len(candidates):
            while len(xy)>4:
                d=np.roll(xy,-1,axis=0)-xy;lengths=np.linalg.norm(d,axis=1)
                angles=np.arctan2(d[:,1],d[:,0]);diff=abs((angles[:,None]-candidates[None,:]+np.pi/2)%np.pi-np.pi/2)
                owner=diff.argmin(1);changed=False
                for i in np.argsort(lengths):
                    a=(i-1)%len(xy);b=(i+1)%len(xy)
                    orth=abs(abs((candidates[owner[a]]-candidates[owner[b]]+np.pi/2)%np.pi-np.pi/2)-np.pi/2)<1e-6
                    if lengths[i]<2.4 and diff[i].min()>np.deg2rad(12) and diff[a].min()<np.deg2rad(12) and diff[b].min()<np.deg2rad(12) and orth:

                        ta=np.array([np.cos(candidates[owner[a]]),np.sin(candidates[owner[a]])]);tb=np.array([np.cos(candidates[owner[b]]),np.sin(candidates[owner[b]])])
                        na=np.array([-ta[1],ta[0]]);nb=np.array([-tb[1],tb[0]])
                        corner=np.linalg.solve(np.array([na,nb]),[((xy[a]+xy[i])/2)@na,((xy[b]+xy[(b+1)%len(xy)])/2)@nb])
                        proposal=xy.copy();proposal[i]=corner;proposal=np.delete(proposal,b,axis=0);p=Polygon(proposal)
                        if p.is_valid and p.boundary.hausdorff_distance(reference.boundary)<1.5:
                            xy=proposal;changed=True;break
                if not changed:break
        lines=[]
        for p,q in zip(xy,np.roll(xy,-1,axis=0)):
            delta=q-p;length=np.linalg.norm(delta);angle=np.arctan2(delta[1],delta[0])
            if len(candidates):
                differences=(candidates-angle+np.pi/2)%np.pi-np.pi/2
                j=np.argmin(abs(differences))
                if abs(differences[j])<np.deg2rad(12):angle+=differences[j]
            tangent=np.array([np.cos(angle),np.sin(angle)]);normal=np.array([-tangent[1],tangent[0]])
            offset=((p+q)/2)@normal;support_area=0.
            if support is not None and length>1.:
                c,n,a,z=support
                along=(c[:,:2]-p)@tangent
                good=(abs(c[:,:2]@normal-offset)<1.1)&(along>-.25)&(along<length+.25)&(abs(n[:,:2]@normal)>.85)&(c[:,2]<z+1.)
                support_area=float(a[good].sum())
                if support_area>max(.8,length*.3):
                    proposal=geometry.weighted_quantile(c[good,:2]@normal,a[good],.5)
                    offset+=np.clip(proposal-offset,-.65,.65)
            lines.append((normal,offset))
            records.append(dict(length_m=float(length),direction_deg=float(np.degrees(angle)%180),support_m2=support_area))
        result=[]
        for i,(normal,offset) in enumerate(lines):
            previous,po=lines[i-1];matrix=np.array([previous,normal])
            if abs(np.linalg.det(matrix))<1e-5:
                result.append(xy[i]);continue
            intersection=np.linalg.solve(matrix,[po,offset])
            if np.linalg.norm(intersection-xy[i])>1.5:return xy
            result.append(intersection)
        return np.array(result)
    candidate=Polygon(ring(simple.exterior.coords),[ring(r.coords) for r in simple.interiors])
    accepted=candidate.is_valid and candidate.area>0 and candidate.hausdorff_distance(poly)<1.5 and candidate.symmetric_difference(poly).area<max(4.,poly.area*.18)
    return (candidate if accepted else simple),dict(regularized=bool(accepted),edges=records)


def complete_supported_holes(layers,base,vertices,faces):


    footprint=unary_union([p for p,_,_ in layers]);holes=[Polygon(r) for p in geometry.polygons(footprint) for r in p.interiors]


    closed=footprint.buffer(1.,join_style=2).buffer(-1.,join_style=2)
    seams=closed.difference(footprint)
    holes=list(geometry.polygons(unary_union(holes+[seams])))
    holes=[p for p in holes if p.area>1e-6]
    if not holes:return list(layers),[]
    scene=o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(vertices.astype('float32')),o3d.core.Tensor(faces.astype('uint32')))
    top=float(vertices[:,2].max()+2);upper=max(z for _,z,_ in layers);result=list(layers);records=[]
    for hole in holes:
        a,b,c,d=hole.bounds
        step=max(.1,np.sqrt(hole.area/12000))
        nx=max(2,int(np.ceil((c-a)/step)));ny=max(2,int(np.ceil((d-b)/step)))
        x,y=np.meshgrid(a+(np.arange(nx)+.5)*(c-a)/nx,b+(np.arange(ny)+.5)*(d-b)/ny)
        q=np.c_[x.ravel(),y.ravel()];q=q[contains_xy(hole,*q.T)]
        row=dict(area_m2=float(hole.area),bounds=list(hole.bounds),filled=False,samples=len(q))
        if not len(q):row['reason']='no_interior_samples';records.append(row);continue
        rays=np.c_[q,np.full(len(q),top),np.zeros((len(q),2)),-np.ones(len(q))].astype('float32')
        hits=scene.cast_rays(o3d.core.Tensor(rays));height=top-hits['t_hit'].numpy();normal=hits['primitive_normals'].numpy()
        finite=np.isfinite(height);up=finite&(normal[:,2]>.5)
        row.update(hit_fraction=float(finite.mean()),upward_fraction=float(up.mean()))
        if finite.mean()<.95:
            row['reason']='insufficient_surface_coverage';records.append(row);continue
        adjacent=[z for p,z,_ in layers if p.boundary.intersection(hole.boundary.buffer(1e-4)).length>1e-5]
        if up.mean()<.5:


            z=min(adjacent) if adjacent else base
            narrow=2*hole.area/max(hole.length,1e-9)<2.
            if not narrow or np.quantile(height[finite],.05)<z-1.:
                row['reason']='insufficient_surface_coverage';records.append(row);continue
            cluster=height[finite];row['support_type']='narrow_seam_at_adjacent_roof'
        else:
            values=height[up];low=float(np.quantile(values,.25,method='nearest'))
            cluster=values[abs(values-low)<1.25];z=float(np.median(cluster))
            row['support_type']='upward_surface'
        row.update(cap_z_m=z,cluster_fraction=float(len(cluster)/len(q)),height_above_base_m=z-base)
        if z<base+max(2.,.18*(upper-base)):
            row['reason']='ground_or_low_floor';records.append(row);continue
        if len(cluster)/len(q)<.25:


            low=float(np.quantile(height[finite],.05))
            narrow=2*hole.area/max(hole.length,1e-9)<2.
            if not narrow or not adjacent or low<base+max(2.,.18*(upper-base)):
                row['reason']='no_coherent_height_support';records.append(row);continue
            z=min(low,min(adjacent))
            if z<base+max(2.,.18*(upper-base)):
                row['reason']='ground_or_low_floor';records.append(row);continue
            row.update(cap_z_m=z,height_above_base_m=z-base,support_type='continuous_elevated_riser')

        result.append((hole.buffer(1e-5,join_style=2),z,float(hole.area*len(cluster)/len(q))))
        row.update(filled=True,reason='observed_elevated_surface')
        records.append(row)
    return result,records


def clean_numerical_faces(vertices,faces):

    v=vertices.copy();f=faces.copy()
    for _ in range(max(100,2*len(f))):
        tri=v[f];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        bad=np.flatnonzero(area<1e-10)
        if not len(bad):break
        changed=False
        for index in bad:
            ids=f[index];lengths=np.linalg.norm(v[ids]-v[np.roll(ids,-1)],axis=1)
            for k in np.argsort(lengths):
                if lengths[k]>=1e-4:break
                a,b=int(ids[k]),int(ids[(k+1)%3]);proposal=f.copy();proposal[proposal==b]=a
                proposal=proposal[(proposal[:,0]!=proposal[:,1])&(proposal[:,1]!=proposal[:,2])&(proposal[:,2]!=proposal[:,0])]


                _,inverse,counts=np.unique(np.sort(proposal,axis=1),axis=0,return_inverse=True,return_counts=True)
                proposal=proposal[counts[inverse]==1]
                edges=np.sort(proposal[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
                _,counts=np.unique(edges,axis=0,return_counts=True)
                if np.any(counts>2):continue
                f=proposal;changed=True;break
            if changed:break
            k=int(lengths.argmax());a,b=int(ids[k]),int(ids[(k+1)%3]);mid=int(ids[(k+2)%3])
            delta=v[b]-v[a];t=(v[mid]-v[a])@delta/(delta@delta);project=v[a]+t*delta
            if not 0<t<1 or np.linalg.norm(project-v[mid])>1e-5:continue
            neighbors=np.flatnonzero(np.any(f==a,axis=1)&np.any(f==b,axis=1));neighbors=neighbors[neighbors!=index]
            if len(neighbors)!=1:continue
            other=int(neighbors[0]);face=f[other]
            for j in range(3):
                p,q=int(face[j]),int(face[(j+1)%3]);last=int(face[(j+2)%3])
                if {p,q}=={a,b}:break
            v[mid]=project
            f=np.concatenate([np.delete(f,[index,other],axis=0),[[p,mid,last],[mid,q,last]]])
            changed=True;break
        if not changed:raise ValueError('Could not repair degenerate boolean triangle')
    else:raise ValueError('Numerical triangle cleanup did not converge')
    ids,inverse=np.unique(f,return_inverse=True)
    return v[ids],inverse.reshape(-1,3)


def assemble_union(layers,base,slabs=(),cells=None):

    origin=np.asarray(unary_union([p for p,_,_ in layers]).centroid.coords[0])
    solids=[]
    for polygon,z,_ in ([] if cells is not None else layers):
        if z<=base:continue
        p=orient(polygon,sign=1.)
        rings=[np.asarray(r.coords)[:-1]-origin for r in [p.exterior,*p.interiors]]
        solids.append(manifold.CrossSection(rings,manifold.FillRule.EvenOdd).extrude(z-base))
    for polygon,lo,hi in slabs:
        if hi<=lo or lo<base:raise ValueError('Invalid supported slab interval')
        p=orient(polygon,sign=1.)
        rings=[np.asarray(r.coords)[:-1]-origin for r in [p.exterior,*p.interiors]]
        solids.append(manifold.CrossSection(rings,manifold.FillRule.EvenOdd).extrude(hi-lo).translate((0,0,lo-base)))
    if cells is not None:
        from .architecture import cell_solid
        solids.extend(cell_solid(cell,origin,base) for cell in cells)
    solid=manifold.Manifold.batch_boolean(solids,manifold.OpType.Add)
    if str(solid.status())!='Error.NoError':raise ValueError(str(solid.status()))
    solid=solid.simplify(1e-4)
    mesh=solid.to_mesh64();v=np.asarray(mesh.vert_properties)[:,:3].copy();f=np.asarray(mesh.tri_verts,dtype=np.int64)


    v,f=clean_numerical_faces(v,f)

    f=f[~np.all(abs(v[f,2])<1e-7,axis=1)]
    ids,inv=np.unique(f,return_inverse=True);v=v[ids];f=inv.reshape(-1,3)
    v[:,:2]+=origin;v[:,2]+=base
    t=v[f];cross=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=np.linalg.norm(cross,axis=1)
    kinds=np.where(abs(cross[:,2])>norm*.99,'roof','wall')
    return v,f,kinds


def reconstruct(job):
    iid,source,output=job[:3];fit_joint=job[3] if len(job)>3 else True
    fit_overhangs=job[4] if len(job)>4 else True
    started=time.perf_counter();target=Path(output)/f'instance_{iid:03d}';target.mkdir(parents=True,exist_ok=True)
    path=Path(job[5]) if len(job)>5 else Path(source)/f'instance_{iid:03d}/observed_tsdf.npz'
    row=dict(instance=iid,state='failed',source=str(path))
    try:
        joint_sdf=dict(job[7] or {}) if len(job)>7 else {}
        joint_sdf.setdefault('strength',.25)
        joint_sdf.setdefault('max_vertex_displacement_m',.03)
        if path.suffix.lower() != '.npz':
            raise ValueError('Envelope reconstruction requires a signed, weighted TSDF NPZ field')
        if not fit_joint:
            raise ValueError('SDF envelope reconstruction requires joint plane fitting')
        from .tsdf import field_observations
        nv,nf,field_surface=field_observations(path)
        row['field_surface']=field_surface
        if len(job)>8 and job[8].get('largest_component'):
            observations=o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(nv),o3d.utility.Vector3iVector(nf))
            labels,counts,areas=observations.cluster_connected_triangles()
            if not len(counts): raise ValueError('Field has no observed surface component')
            keep=int(np.argmax(areas))
            observations.remove_triangles_by_mask(np.asarray(labels)!=keep)
            observations.remove_unreferenced_vertices()
            nv=np.asarray(observations.vertices).copy();nf=np.asarray(observations.triangles).copy()
            row['observation_cleanup']=dict(rule='largest connected surface area',components=len(counts),kept_faces=len(nf))
        tri=nv[nf];c=tri.mean(1);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2;n=cross/np.maximum(2*area[:,None],1e-12)
        basis,c,n,layers,base,info=geometry.initialize(c,n,area,orthogonal=False)
        recovered=[]
        if fit_overhangs:
            from .architecture import recover_local_roof_layers
            layers,recovered=recover_local_roof_layers(layers,base,c,n,area)
        modes=orientation_families(n,area);records=[];straight=[]
        for p,z,s in layers:
            p,record=straighten_polygon(p,modes,(c,n,area,z));straight.append((p,z,s));records.append(record)
        slabs=[];overhangs=[]
        if fit_overhangs:
            from .architecture import separate_supported_overhangs
            straight,slabs,overhangs=separate_supported_overhangs(layers,straight,base,c,n,area,modes)
        local=np.c_[nv[:,:2]@basis,nv[:,2]]
        layers,holes=complete_supported_holes(straight,base,local,nf)
        cells=None;profiles=[]
        if fit_overhangs:
            from .architecture import build_height_cells
            cells,profiles=build_height_cells(layers,slabs,base,c,n,area,overhangs)
        v,f,k=assemble_union(layers,base,slabs=slabs if cells is None else (),cells=cells)
        v,f,fan_repairs=geometry.split_boundary_fans(v,f)
        row['boundary_fan_repairs']=fan_repairs
        from .geometry import remove_negligible_components
        v,f,k,component_cleanup=remove_negligible_components(v,f,k)
        row['component_cleanup']=component_cleanup
        checks=geometry.audit(v,f)
        if not checks['valid']:raise ValueError(f'Invalid topology: {checks}')
        from .geometry import fit_supported_bottom
        v,bottom_fit=fit_supported_bottom(v,f,c,n,area)
        geometry.save(target/'supported_bottom.json',bottom_fit)
        row['supported_bottom']=bottom_fit
        from .plane_fit import joint_refit
        v,fit=joint_refit(v,f,c,n,area,checkpoint_path=target/'joint_fit_checkpoint.npz')
        checks=geometry.audit(v,f)
        if not checks['valid']:raise ValueError(f'Invalid refitted topology: {checks}')
        geometry.save(target/'joint_fit.json',fit)
        row.update(joint_fit=fit,plane_fit_records=fit['plane_fit_records'],joint_fit_code_sha256=hashlib.sha256((Path(__file__).parent/'plane_fit.py').read_bytes()).hexdigest())
        world=np.c_[v[:,:2]@basis.T,v[:,2]]
        cap=job[6] if len(job)>6 else None
        if cap is not None:
            import trimesh
            from .face_budget import reduce_to_budget
            reduced,budget=reduce_to_budget(trimesh.Trimesh(vertices=world,faces=f,process=False),cap,target/'face_budget')
            row['face_budget']=budget
            if not budget['budget_met']:raise ValueError('Requested face budget could not be reached with valid topology')
            world=np.asarray(reduced.vertices);f=np.asarray(reduced.faces);v=np.c_[world[:,:2]@basis,world[:,2]]
            checks=geometry.audit(v,f)
            k=np.where(abs(reduced.face_normals[:,2])>.99,'roof','wall')
        from .sdf import refine_local_envelope
        baseline_world=world.copy()
        local_field_path=str(path.resolve())
        from .sdf import optimize_local_field
        local_field_path,field_fit=optimize_local_field(local_field_path,baseline_world,f,target/'local_field',
            iterations=12,maximum_correction_m=.03,lambda_eik=.02,lambda_smooth=.1)
        row['local_field_optimization']=field_fit
        sdf_context=dict(joint_sdf,field_path=local_field_path,basis=basis.tolist(),stage='local_sdf_refinement')
        v,sdf_report=refine_local_envelope(v,f,c,n,area,local,nf,sdf_context,
            checkpoint_path=target/'local_sdf_checkpoint.npz')
        checks=geometry.audit(v,f)
        if not checks['valid']:raise ValueError(f'Invalid local SDF topology: {checks}')
        geometry.save(target/'local_sdf_optimization.json',sdf_report)
        row['local_sdf_optimization']=sdf_report
        row['local_sdf_code_sha256']=hashlib.sha256((Path(__file__).parent/'sdf.py').read_bytes()).hexdigest()
        world=np.c_[v[:,:2]@basis.T,v[:,2]] if sdf_report['accepted'] else baseline_world
        geometry.write_ply(target/'mesh.ply',world,f)
        np.savez_compressed(target/'surface.npz',local_vertices=v,faces=f,basis=basis,face_kind=k)
        row.update(state='complete',mesh=str(target/'mesh.ply'),mesh_sha256=hashlib.sha256((target/'mesh.ply').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),vertices=len(v),faces=len(f),native_faces=len(nf),topology=checks,initialization=info,orientation_families_deg=np.degrees(modes).tolist(),line_fits=records,hole_completion=holes,overhangs=overhangs)
        row.update(wall_height_profiles=profiles,height_cells=len(cells) if cells is not None else None,recovered_roof_patches=recovered)
        if fit_overhangs:
            row['height_cell_code_sha256']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['architecture.py']}
    except Exception as error:row.update(error=repr(error),traceback=traceback.format_exc())
    row['reconstruction_seconds']=time.perf_counter()-started
    geometry.save(target/'report.json',row)
    print(iid,row['state'],row.get('faces'),round(row['reconstruction_seconds'],2),row.get('error',''),flush=True)
    return row
