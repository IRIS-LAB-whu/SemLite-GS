import json
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely import contains_xy
from scipy.spatial import cKDTree


PARAMS=dict(grid_m=.6,roof_band_m=.85,minimum_roof_area_m2=5.,outline_tolerance_m=.9,
            minimum_hole_area_m2=12.,normal_threshold=.65,roof_peak_prominence_m2=.3,
            initializer_version='ordered_local_candidates_v2')


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def polygons(geom):
    if geom.geom_type=='Polygon':return [geom]
    return [p for p in getattr(geom,'geoms',[]) if p.geom_type=='Polygon']


def simplify_ring(coords,tolerance):

    p=np.round(np.asarray(coords,float)[:-1],6)
    p=p[np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)>1e-7]
    before=p-np.roll(p,1,axis=0);after=np.roll(p,-1,axis=0)-p
    p=p[abs(before[:,0]*after[:,1]-before[:,1]*after[:,0])>1e-8]
    original=Polygon(p)
    if not original.is_valid:return np.asarray(coords,float)[:-1]

    for _ in range(len(p)):
        if len(p)<=4:break
        choices=[]
        for i in range(len(p)):
            q=np.roll(p,-i,axis=0);a,b,c,d=q[:4]
            for corner in [np.array([a[0],d[1]]),np.array([d[0],a[1]])]:
                if min(np.linalg.norm(corner-a),np.linalg.norm(corner-d))<1e-7:continue
                if max(np.linalg.norm(b-corner,ord=np.inf),np.linalg.norm(c-corner,ord=np.inf))>2*tolerance:continue
                candidate=np.vstack([a,corner,q[3:]])
                prev=candidate-np.roll(candidate,1,axis=0);nxt=np.roll(candidate,-1,axis=0)-candidate
                candidate=candidate[abs(prev[:,0]*nxt[:,1]-prev[:,1]*nxt[:,0])>1e-8]
                if len(candidate)<4 or len(candidate)>=len(p):continue


                change=np.array([a,b,c,d,corner])
                cost=abs(np.sum(change[:,0]*np.roll(change[:,1],-1)-change[:,1]*np.roll(change[:,0],-1)))/2
                choices.append((cost,candidate))
        if not choices:break
        accepted=False
        for _,candidate in sorted(choices,key=lambda item:item[0]):
            poly=Polygon(candidate)
            if not poly.is_valid:continue
            with np.errstate(invalid='ignore'):
                distance=original.boundary.hausdorff_distance(poly.boundary)
            if not np.isfinite(distance) or distance>tolerance:continue
            if original.symmetric_difference(poly).area>max(3.,.08*original.area):continue
            p=candidate;accepted=True;break
        if not accepted:break
    return p


def mask_geometry(mask,origin,step):

    rects=[]
    for i in range(mask.shape[0]):
        changes=np.diff(np.r_[False,mask[i],False].astype(np.int8))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            rects.append(box(origin[0]+i*step,origin[1]+a*step,
                             origin[0]+(i+1)*step,origin[1]+b*step))
    return unary_union(rects)


def initialize(centers,normals,area,orthogonal=True):
    vertical=abs(normals[:,2])<.35
    theta=np.arctan2(normals[vertical,1],normals[vertical,0])
    weights=area[vertical]
    hist,edges=np.histogram(np.mod(theta,np.pi/2),bins=180,range=(0,np.pi/2),weights=weights)
    peak=int(np.argmax(ndimage.gaussian_filter1d(hist,2,mode='wrap')))
    angle=(edges[peak]+edges[peak+1])/2
    delta=(theta-angle+np.pi/4)%(np.pi/2)-np.pi/4
    keep=abs(delta)<np.deg2rad(12)
    if not keep.any():raise ValueError('No dominant vertical wall support')
    angle+=np.arctan2(np.sum(weights[keep]*np.sin(4*delta[keep])),np.sum(weights[keep]*np.cos(4*delta[keep])))/4
    basis=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    c=np.c_[centers[:,:2]@basis,centers[:,2]]
    n=np.c_[normals[:,:2]@basis,normals[:,2]]
    ground=weighted_quantile(c[:,2],area,.01)
    upper=weighted_quantile(c[:,2],area,.995)
    roof=(n[:,2]>PARAMS['normal_threshold'])&(c[:,2]>ground+max(2.,.18*(upper-ground)))
    if np.sum(roof)<30:raise ValueError('Insufficient upward roof support')
    rw=area[roof]*n[roof,2];rc=c[roof]
    zmin=np.floor(rc[:,2].min()/.25)*.25
    bins=np.arange(zmin,rc[:,2].max()+.5,.25)
    zh,_=np.histogram(rc[:,2],bins=bins,weights=rw)
    smooth=ndimage.gaussian_filter1d(zh,1)


    peaks,_=find_peaks(np.r_[0,smooth,0],distance=4,prominence=PARAMS['roof_peak_prominence_m2'])
    peaks=peaks-1
    levels=bins[peaks]+.125
    if not len(levels):levels=np.array([weighted_quantile(rc[:,2],rw,.5)])
    distances=abs(rc[:,2,None]-levels[None,:]);owners=distances.argmin(axis=1)
    step=PARAMS['grid_m'];origin=np.floor(rc[:,:2].min(0)/step)*step-3*step
    ij=np.floor((rc[:,:2]-origin)/step).astype(int)
    shape=tuple(ij.max(0)+4);layers=[];raw_support=0.
    for k,z in enumerate(levels):
        selected=(owners==k)&(distances[:,k]<PARAMS['roof_band_m'])
        if rw[selected].sum()<PARAMS['minimum_roof_area_m2']:continue
        level=weighted_quantile(rc[selected,2],rw[selected],.5)
        acc=np.zeros(shape);np.add.at(acc,(ij[selected,0],ij[selected,1]),rw[selected])
        mask=acc>.025
        mask=ndimage.binary_closing(mask,structure=np.ones((3,3)))

        holes=ndimage.binary_fill_holes(mask)&~mask;labels,num=ndimage.label(holes)
        sizes=np.bincount(labels.ravel());fill=(sizes*step*step<PARAMS['minimum_hole_area_m2']);fill[0]=False
        mask|=fill[labels]
        labels,num=ndimage.label(mask)
        for label in range(1,num+1):
            component=labels==label
            supported=selected&component[ij[:,0],ij[:,1]]
            support=float(rw[supported].sum())
            if support<PARAMS['minimum_roof_area_m2']:continue
            raw_support+=support
            for poly in polygons(mask_geometry(component,origin,step)):
                if poly.area<PARAMS['minimum_roof_area_m2']:continue
                if orthogonal:
                    outer=simplify_ring(poly.exterior.coords,PARAMS['outline_tolerance_m'])
                    inner=[simplify_ring(r.coords,PARAMS['outline_tolerance_m']) for r in poly.interiors if Polygon(r).area>=PARAMS['minimum_hole_area_m2']]
                    regular=Polygon(outer,inner)
                else:


                    regular=poly.simplify(.65,preserve_topology=True)
                if not regular.is_valid:regular=poly
                layers.append((regular,float(level),support))
    if not layers:raise ValueError('No supported roof layers')
    footprint=unary_union([p for p,_,_ in layers])
    covered=contains_xy(footprint,rc[:,0],rc[:,1])
    info=dict(angle_deg=float(np.degrees(angle)),dominant_wall_fraction=float(weights[keep].sum()/max(weights.sum(),1e-9)),
              candidate_ground_m=float(ground),layers=[dict(height_m=z,area_m2=p.area,support_m2=s) for p,z,s in layers],
              retained_upward_support_fraction=float(raw_support/rw.sum()),roof_support_projection_fraction=float(rw[covered].sum()/rw.sum()))
    return basis,c,n,layers,ground,info


def audit(vertices,faces):
    directed=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    edges,inv,counts=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
    winding=np.bincount(inv,weights=np.where(directed[:,0]<directed[:,1],1,-1))
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(vertices),len(vertices))).tocsr()
    components=int(connected_components(graph,directed=False,return_labels=False))
    boundary=edges[counts==1];ids,remap=np.unique(boundary,return_inverse=True);be=remap.reshape(-1,2)
    bg=coo_matrix((np.ones(len(be)),(be[:,0],be[:,1])),shape=(len(ids),len(ids))).tocsr()
    loops=int(connected_components(bg,directed=False,return_labels=False)) if len(ids) else 0
    areas=np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)/2
    result=dict(components=components,open_boundary_loops=loops,nonmanifold_edges=int((counts>2).sum()),
                inconsistent_winding=int(((counts==2)&(winding!=0)).sum()),bad_boundary_vertices=int((np.bincount(be.ravel())!=2).sum()),
                degenerate_faces=int((areas<1e-10).sum()),minimum_triangle_area_m2=float(areas.min()))
    result['valid']=not any(result[k] for k in ['nonmanifold_edges','inconsistent_winding','bad_boundary_vertices','degenerate_faces'])
    return result


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    return float(np.interp(q, np.cumsum(weights[order])/weights.sum(), values[order]))


def closest(vertices, faces, points):
    scene = o3d.t.geometry.RaycastingScene(nthreads=2)
    scene.add_triangles(o3d.core.Tensor(vertices.astype(np.float32)),
                        o3d.core.Tensor(faces.astype(np.uint32)))
    chunks = [scene.compute_closest_points(o3d.core.Tensor(p.astype(np.float32)))
              for p in np.array_split(points, max(1, int(np.ceil(len(points)/200000))))]
    return (np.concatenate([p['points'].numpy() for p in chunks]),
            np.concatenate([p['primitive_ids'].numpy() for p in chunks]))


def write_ply(path, vertices, faces):
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(vertices))+
                 '\nproperty double x\nproperty double y\nproperty double z\nelement face '+str(len(faces))+
                 '\nproperty list uchar uint vertex_indices\nend_header\n').encode())
        np.asarray(vertices, dtype='<f8').tofile(f)
        data = np.empty(len(faces), dtype=[('n','u1'),('v','<u4',(3,))])
        data['n'], data['v'] = 3, faces
        data.tofile(f)


def split_boundary_fans(vertices,faces):
    v=vertices.copy();f=faces.copy();records=[]
    edges,counts=np.unique(np.sort(np.vstack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0,return_counts=True)
    ids,degree=np.unique(edges[counts==1],return_counts=True)
    for vertex in ids[degree>2]:
        incident=np.flatnonzero(np.any(f==vertex,axis=1)).tolist()
        remaining=set(incident);fans=[]
        while remaining:
            group={min(remaining)};remaining-=group;queue=list(group)
            while queue:
                face=queue.pop();neighbors=set(f[face])-{vertex}
                linked={j for j in remaining if neighbors.intersection(set(f[j])-{vertex})}
                remaining-=linked;group|=linked;queue.extend(sorted(linked))
            fans.append(sorted(group))
        if len(fans)<2:continue
        new_ids=[]
        for fan in fans[1:]:
            new=len(v);v=np.vstack([v,v[vertex]]);new_ids.append(new)
            for face in fan:f[face,f[face]==vertex]=new
        records.append(dict(original_vertex=int(vertex),duplicates=new_ids,face_fans=fans))
    assert np.array_equal(vertices[faces],v[f]),'Repair changed triangle geometry'
    return v,f,records


def boundary_vertices(faces):
    edges,counts=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    return np.unique(edges[counts==1])


def fit_supported_bottom(vertices,faces,centers,normals,area):
    v=np.asarray(vertices,dtype=float);out=v.copy();base=float(v[:,2].min())
    ids=boundary_vertices(faces);ids=ids[abs(v[ids,2]-base)<1e-6]
    report=dict(version='local_coherent_TSDF_bottom_v1',reference='TSDF only',base_m=base,
                boundary_vertices=len(ids),moved_vertices=0,accepted=False,
                xy_cell_m=1.,z_cell_m=.5,minimum_bin_area_m2=.12,continuous_bins=3,margin_m=.25)
    good=(abs(normals[:,2])<.5)&(area>0)
    if not len(ids) or good.sum()<30:return out,report
    c=centers[good];a=area[good]
    keys=np.floor(c/np.array([1.,1.,.5])).astype(np.int64)
    unique,inv=np.unique(keys,axis=0,return_inverse=True);mass=np.bincount(inv,weights=a)
    active=unique[mass>=.12]


    occupied=set(map(tuple,active));low={}
    for x,y,z in active:
        key=(x,y)
        if (x,y,z+1) in occupied and (x,y,z+2) in occupied:
            low[key]=min(low.get(key,np.inf),z*.5)
    if not low:return out,report
    xy=np.array(list(low),float)+.5;z=np.array(list(low.values()))
    tree=cKDTree(xy);cross=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
    normal=cross/np.maximum(np.linalg.norm(cross,axis=1)[:,None],1e-20)
    records=[]
    for i in ids:
        near=tree.query_ball_point(v[i,:2],3.)
        if len(near)<2:continue
        floor=float(np.quantile(z[near],.2)-.25)
        shift=floor-v[i,2]
        if shift<=.2:continue
        incident=np.flatnonzero(np.any(faces==i,axis=1));nn=normal[incident]
        if np.any(abs(nn[:,2])>.65):continue

        dxdy=np.linalg.lstsq(nn[:,:2],-nn[:,2],rcond=None)[0]
        direction=np.r_[dxdy,1.]
        if np.max(abs(nn@direction))>1e-7:continue
        other=np.unique(faces[incident]);above=v[other,2][v[other,2]>base+1e-5]
        if not len(above):continue

        if floor>=np.min(above)-.5 or shift>.35*(v[other,2].max()-base):continue
        if np.linalg.norm(dxdy*shift)>2.:continue
        out[i]+=direction*shift
        records.append(dict(vertex=int(i),old_z_m=float(v[i,2]),new_z_m=float(out[i,2]),support_columns=len(near)))
    if records:
        new_cross=np.cross(out[faces[:,1]]-out[faces[:,0]],out[faces[:,2]]-out[faces[:,0]])
        valid=audit(out,faces)['valid'] and np.all(np.einsum('ij,ij->i',new_cross,cross)>0)
        if not valid:
            report['reason']='joint_boundary_geometry_guard_rejected';return v.copy(),report
        report.update(accepted=True,moved_vertices=len(records),max_raise_m=float((out[:,2]-v[:,2]).max()))
    report['records']=records
    report['support_columns']=len(low)
    return out,report


def remove_negligible_components(vertices, faces, kinds,
                                  absolute_area_m2=1e-3, relative_area=1e-6,
                                  maximum_area_m2=1e-2):


    v = np.asarray(vertices)
    f = np.asarray(faces)
    k = np.asarray(kinds)
    if len(f) != len(k):
        raise ValueError("Face-kind array does not match the mesh")
    edges = f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(v), len(v)),
    ).tocsr()
    _, labels = connected_components(graph, directed=False)
    owners = labels[f[:, 0]]
    if not np.all(labels[f] == owners[:, None]):
        raise ValueError("A face spans disconnected vertex components")
    triangle = v[f]
    area = np.linalg.norm(
        np.cross(triangle[:, 1] - triangle[:, 0],
                 triangle[:, 2] - triangle[:, 0]), axis=1
    ) / 2
    component_area = np.bincount(owners, weights=area, minlength=len(v))
    component_faces = np.bincount(owners, minlength=len(v))
    present = np.flatnonzero(component_faces)
    threshold = min(maximum_area_m2,
                    max(absolute_area_m2, relative_area * float(area.sum())))
    discarded = [int(i) for i in present if component_area[i] < threshold]
    if len(discarded) == len(present):
        discarded.remove(int(present[np.argmax(component_area[present])]))
    report = {
        "area_threshold_m2": float(threshold),
        "removed": [
            {"faces": int(component_faces[i]), "area_m2": float(component_area[i])}
            for i in discarded
        ],
    }
    if not discarded:
        return v, f, k, report
    keep = ~np.isin(owners, discarded)
    remaining_faces = f[keep]
    used, inverse = np.unique(remaining_faces, return_inverse=True)
    return v[used], inverse.reshape(-1, 3), k[keep], report

