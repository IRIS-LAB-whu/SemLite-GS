import numpy as np
from scipy import ndimage
from shapely import contains_xy
from .geometry import mask_geometry,polygons,weighted_quantile
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from dataclasses import dataclass
import manifold3d as manifold
from scipy.spatial import cKDTree


def recover_local_roof_layers(layers,base,c,n,a):
    result=list(layers);records=[]
    upper=weighted_quantile(c[:,2],a,.995)
    selected=(n[:,2]>.65)&(c[:,2]>base+max(2.,.18*(upper-base)))
    points=c[selected];weights=a[selected]*n[selected,2]
    if not len(points):return result,records
    covered=np.zeros(len(points),bool)
    for p,z,_ in result:
        covered|=contains_xy(p.buffer(.25),points[:,0],points[:,1])&(abs(points[:,2]-z)<.85)
    points=points[~covered];weights=weights[~covered]
    if weights.sum()<5:return result,records
    step=.4;origin=np.floor(points[:,:2].min(0)/step)*step-3*step
    ij=np.floor((points[:,:2]-origin)/step).astype(int);shape=tuple(ij.max(0)+4)
    acc=np.zeros(shape);np.add.at(acc,tuple(ij.T),weights)
    mask=ndimage.binary_closing(acc>.015,structure=np.ones((3,3)))
    labels,num=ndimage.label(mask)
    for label in range(1,num+1):
        ids=labels[tuple(ij.T)]==label
        if weights[ids].sum()<5:continue
        local=points[ids];w=weights[ids]

        remaining=np.ones(len(local),bool)
        for _ in range(12):
            if w[remaining].sum()<5:break
            bins=np.arange(np.floor(local[remaining,2].min()/.2)*.2,local[remaining,2].max()+.4,.2)
            hist,_=np.histogram(local[remaining,2],bins=bins,weights=w[remaining])
            peak=int(np.argmax(ndimage.gaussian_filter1d(hist,1)))
            z=(bins[peak]+bins[peak+1])/2
            band=remaining&(abs(local[:,2]-z)<.6)
            remaining[band]=False
            if w[band].sum()<5:continue
            level=weighted_quantile(local[band,2],w[band],.5)
            cells=np.floor((local[band,:2]-origin)/step).astype(int)
            grid=np.zeros(shape);np.add.at(grid,tuple(cells.T),w[band])
            footprint=ndimage.binary_closing(grid>.015,structure=np.ones((3,3)))
            for p in polygons(mask_geometry(footprint,origin,step)):
                core=p.buffer(-.4)

                if p.area<5 or core.area<max(1.,.1*p.area):continue
                inside=contains_xy(p,local[band,0],local[band,1]);support=float(w[band][inside].sum())
                if support<max(5.,p.area*.35):continue
                p=p.simplify(.35,preserve_topology=True)
                result.append((p,float(level),support))
                records.append(dict(height_m=float(level),area_m2=float(p.area),support_m2=support,bounds=list(p.bounds),reason='coherent_local_upward_support_missed_by_global_levels'))
    return result,records


PARAMETERS=dict(min_inset_m=.3,max_inset_m=2.5,roof_band_m=.85,
               min_edge_coverage=.55,min_underside_fraction=.15,
               version='supported_height_transition_v4_clipped_concavities')


def clip_body_to_roof(roof,body):


    if roof is None or body is None:return None
    if roof.buffer(.02).covers(body):return body
    clipped=body.intersection(roof)
    if (clipped.geom_type!='Polygon' or not clipped.is_valid or
            clipped.area<body.area*.95 or clipped.area<1):return None
    return clipped


def supported_body_contour(roof,xy,normals,roof_offsets,body_offsets):


    body=clip_body_to_roof(roof,intersect_lines(xy,normals,body_offsets))
    if body is not None:return body
    if roof is None:return None
    strips=[]
    for i,(p,q,d) in enumerate(zip(xy,np.roll(xy,-1,axis=0),normals)):
        width=roof_offsets[i]-body_offsets[i]
        if width<1e-6:continue
        p=p+d*(roof_offsets[i]-p@d);q=q+d*(roof_offsets[i]-q@d)
        strips.append(Polygon([p+d*.02,q+d*.02,q-d*width,p-d*width]))
    candidate=roof.difference(unary_union(strips))
    if candidate.geom_type!='Polygon' or not candidate.is_valid or candidate.area<roof.area*.55:return None
    return candidate


def intersect_lines(xy,normals,offsets):
    corners=[]
    for i in range(len(xy)):
        matrix=normals[[i-1,i]]
        if abs(np.linalg.det(matrix))<1e-5:
            corners.append(xy[i]);continue
        corner=np.linalg.solve(matrix,offsets[[i-1,i]])
        corners.append(corner)
    p=Polygon(corners)


    return p if p.is_valid and p.area>1 and p.boundary.hausdorff_distance(Polygon(xy).boundary)<3. else None


def separate_supported_overhangs(raw_layers,body_layers,base,c,n,a,modes):

    from .envelope import straighten_polygon
    bodies=list(body_layers);slabs=[];records=[]
    for index,(raw,z,support) in enumerate(raw_layers):
        row=dict(layer=index,roof_z_m=float(z),accepted=False);records.append(row)
        if raw.area<20 or raw.interiors:
            row['reason']='small_or_courtyard_layer';continue


        simple=raw.simplify(1.2,preserve_topology=True)
        if raw.symmetric_difference(simple).area>raw.area*.04:simple=raw
        roof,_=straighten_polygon(simple,modes,None);roof=orient(roof,sign=1.)
        xy=np.asarray(roof.exterior.coords)[:-1];delta=np.roll(xy,-1,axis=0)-xy
        lengths=np.linalg.norm(delta,axis=1);tangent=delta/lengths[:,None]
        outward=np.c_[tangent[:,1],-tangent[:,0]]
        offsets=np.sum(xy*outward,axis=1);ro=offsets.copy();bo=offsets.copy()
        rows=[];qualified=[]
        for i,(p,t,d,length) in enumerate(zip(xy,tangent,outward,lengths)):
            along=(c[:,:2]-p)@t;across=c[:,:2]@d
            near=(along>.25)&(along<length-.25)&(abs(across-offsets[i])<2.5)
            top=near&(n[:,2]>.65)&(abs(c[:,2]-z)<PARAMETERS['roof_band_m'])
            wall=near&(n[:,:2]@d>.8)&(c[:,2]>base+.5)&(c[:,2]<z-.5)
            r=dict(edge=i,length_m=float(length),roof_support_m2=float(a[top].sum()),wall_support_m2=float(a[wall].sum()),accepted=False)
            rows.append(r)
            if length<2. or a[top].sum()<max(1.,length*.15):continue


            ro[i]=weighted_quantile(across[top],a[top]*n[top,2],.95)
            ro[i]=np.clip(ro[i],offsets[i]-.9,offsets[i]+.9)
            bo[i]=ro[i]
            if a[wall].sum()<max(2.,length*.3):continue
            bins=np.clip((along[wall]/length*10).astype(int),0,9)
            acc=np.bincount(bins,weights=a[wall],minlength=10)
            coverage=float(np.mean(acc>.1));r['wall_edge_coverage']=coverage
            if coverage<PARAMETERS['min_edge_coverage']:continue


            values=across[wall];wa=a[wall]
            bins_d=np.arange(offsets[i]-2.55,offsets[i]+2.7,.15)
            hist,_=np.histogram(values,bins=bins_d,weights=wa)
            smoothed=gaussian_filter1d(hist,1)
            peaks=np.flatnonzero((smoothed>=np.roll(smoothed,1))&(smoothed>=np.roll(smoothed,-1))&(smoothed>smoothed.max()*.04))
            wall_offset=weighted_quantile(values,wa,.5)
            candidates=[]
            for peak in peaks:
                candidate=(bins_d[peak]+bins_d[peak+1])/2;cluster=abs(values-candidate)<.25
                if wa[cluster].sum()<max(2.,length*.3):continue
                if np.ptp(c[wall,2][cluster])<min(1.5,.2*(z-base)):continue
                ab=np.clip((along[wall][cluster]/length*10).astype(int),0,9)
                if np.mean(np.bincount(ab,weights=wa[cluster],minlength=10)>.1)<PARAMETERS['min_edge_coverage']:continue
                offset=weighted_quantile(values[cluster],wa[cluster],.5)
                if PARAMETERS['min_inset_m']<=ro[i]-offset<=PARAMETERS['max_inset_m']:
                    top_support=weighted_quantile(c[wall,2][cluster],wa[cluster],.9)
                    candidates.append((top_support,offset))
            if candidates:


                top_support,wall_offset=max(candidates)
                r['selected_wall_upper_support_z_m']=float(top_support)
            inset=float(ro[i]-wall_offset);r['inset_m']=inset
            if PARAMETERS['min_inset_m']<=inset<=PARAMETERS['max_inset_m']:
                bo[i]=wall_offset;r['accepted']=True;qualified.append(i)
        row['edges']=rows
        if not qualified:row['reason']='no_continuous_inset_wall_support';continue
        rp=intersect_lines(xy,outward,ro);bp=intersect_lines(xy,outward,bo)
        bp=supported_body_contour(rp,xy,outward,ro,bo)
        if rp is None or bp is None:
            row['reason']='invalid_or_non_nested_contours';continue
        strip=rp.difference(bp)
        if strip.area<3. or strip.area>rp.area*.45:
            row['reason']='insufficient_or_excessive_overhang_area';continue
        under=(n[:,2]<-.5)&(c[:,2]>base+.5)&(c[:,2]<z-.12)&contains_xy(strip,c[:,0],c[:,1])
        row['overhang_area_m2']=float(strip.area);row['underside_support_m2']=float(a[under].sum())
        if a[under].sum()<max(1.,PARAMETERS['min_underside_fraction']*strip.area):
            row['reason']='no_observed_underside';continue


        values=c[under,2];weights=a[under]*(-n[under,2])
        edges=np.arange(base+.5,z+.15,.15)
        hist,_=np.histogram(values,bins=edges,weights=weights)
        smooth=gaussian_filter1d(hist,1)
        peaks=np.flatnonzero((smooth>=np.roll(smooth,1))&(smooth>=np.roll(smooth,-1)))
        peak=int(np.argmax(smooth));selected=None
        for candidate in peaks[::-1]:
            mode=(edges[candidate]+edges[candidate+1])/2;mask=abs(values-mode)<.35
            if weights[mask].sum()>=max(1.,.1*strip.area):
                peak=int(candidate);selected=mask;break
        if selected is None:
            row['reason']='no_coherent_underside_height';continue
        lo=weighted_quantile(values[selected],weights[selected],.5)
        if weights[selected].sum()<max(1.,.1*strip.area):
            row['reason']='no_coherent_underside_height';continue


        supported_edges=0
        for i in qualified:
            along=(c[under,:2]-xy[i])@tangent[i];across=c[under,:2]@outward[i]
            edge=(along>=0)&(along<=lengths[i])&(across>bo[i])&(across<ro[i]+.1)&(abs(c[under,2]-lo)<.65)
            if a[under][edge].sum()>max(.4,lengths[i]*(ro[i]-bo[i])*.1):supported_edges+=1
            else:
                bo[i]=ro[i];rows[i]['accepted']=False
                rows[i]['reason']='no_underside_at_this_height'
        row['underside_supported_edges']=supported_edges
        if supported_edges==0:row['reason']='no_corresponding_edge_underside';continue
        bp=supported_body_contour(rp,xy,outward,ro,bo)
        if bp is None:
            row['reason']='invalid_certified_body_contour';continue
        strip=rp.difference(bp)
        if strip.area<3.:
            row['reason']='insufficient_certified_overhang_area';continue
        row['overhang_area_m2']=float(strip.area)
        bodies[index]=(bp,z,support);slabs.append((rp,float(lo),float(z)))
        row.update(accepted=True,reason='separate_roof_wall_and_observed_soffit',underside_z_m=float(lo),roof_area_m2=float(rp.area),body_area_m2=float(bp.area))
    return bodies,slabs,records


@dataclass
class HeightCell:
    lower: Polygon
    upper: Polygon
    bottom: float
    top: float
    evidence: str = 'supported_horizontal_layer'


def cell_solid(cell,origin,base):

    if cell.top<=cell.bottom:raise ValueError('Nonpositive cell height')
    lo=orient(cell.lower,1.);hi=orient(cell.upper,1.)
    lower=[np.asarray(r.coords)[:-1]-origin for r in [lo.exterior,*lo.interiors]]
    upper=[np.asarray(r.coords)[:-1]-origin for r in [hi.exterior,*hi.interiors]]
    if [len(r) for r in lower]!=[len(r) for r in upper]:raise ValueError('Loft ring topology mismatch')
    if all(np.allclose(a,b,atol=1e-10,rtol=0) for a,b in zip(lower,upper)):
        return manifold.CrossSection(lower,manifold.FillRule.EvenOdd).extrude(cell.top-cell.bottom).translate((0,0,cell.bottom-base))
    count=sum(map(len,lower));v=np.vstack([np.c_[np.vstack(lower),np.full(count,cell.bottom-base)],np.c_[np.vstack(upper),np.full(count,cell.top-base)]])
    faces=[np.asarray(manifold.triangulate(lower))[:,::-1],np.asarray(manifold.triangulate(upper))+count]
    offset=0;sides=[]
    for ring in lower:
        for j in range(len(ring)):
            a=offset+j;b=offset+(j+1)%len(ring)
            sides.extend([[a,b,b+count],[a,b+count,a+count]])
        offset+=len(ring)
    faces.append(np.array(sides));f=np.vstack(faces)
    solid=manifold.Manifold(manifold.Mesh64(np.ascontiguousarray(v,dtype=np.float64),np.ascontiguousarray(f,dtype=np.uint64)))
    if str(solid.status())!='Error.NoError':raise ValueError('Invalid supported height cell: '+str(solid.status()))
    return solid


def wall_loft(poly,bottom,top,c,n,a):


    if poly.interiors or top-bottom<3:return poly,poly,[]
    p=orient(poly,1.);xy=np.asarray(p.exterior.coords)[:-1]
    delta=np.roll(xy,-1,axis=0)-xy;length=np.linalg.norm(delta,axis=1)
    tangent=delta/length[:,None];outward=np.c_[tangent[:,1],-tangent[:,0]]
    offsets=np.sum(xy*outward,axis=1);low=offsets.copy();high=offsets.copy();records=[]
    for i,(x,t,d,L) in enumerate(zip(xy,tangent,outward,length)):
        along=(c[:,:2]-x)@t;across=c[:,:2]@d
        supported=(along>.15)&(along<L-.15)&(abs(across-offsets[i])<max(2.5,.25*(top-bottom)))&(n[:,:2]@d>.7)&(abs(n[:,2])<.65)&(c[:,2]>bottom+.2)&(c[:,2]<top-.2)
        if a[supported].sum()<max(3.,L*.5):continue
        cc=c[supported];aa=a[supported];dd=across[supported];nn=n[supported]
        step=max(.5,(top-bottom)/30);bins=np.floor((cc[:,2]-bottom)/step).astype(int)
        zs=[];ds=[];ws=[]
        for k in np.unique(bins):
            mask=bins==k
            if aa[mask].sum()<max(.3,L*step*.025):continue
            zs.append(weighted_quantile(cc[mask,2],aa[mask],.5));ds.append(weighted_quantile(dd[mask],aa[mask],.5));ws.append(aa[mask].sum())
        if len(zs)<4 or np.ptp(zs)<max(2.,.45*(top-bottom)):continue
        z=np.array(zs);values=np.array(ds);weights=np.array(ws)
        X=np.c_[z-top,np.ones(len(z))];w=weights.copy()
        for _ in range(6):
            fit=np.linalg.lstsq(X*np.sqrt(w[:,None]),values*np.sqrt(w),rcond=None)[0]
            residual=abs(X@fit-values);w=weights*np.minimum(1.,.15/np.maximum(residual,1e-9))
        slope,intercept=fit
        error=weighted_quantile(residual,weights,.9)
        normal_slope=weighted_quantile(-nn[:,2]/np.maximum(nn[:,:2]@d,1e-8),aa,.5)
        row=dict(edge=i,slope_m_per_m=float(slope),profile_error_m=float(error),normal_slope=float(normal_slope),accepted=False)
        records.append(row)
        if not .015<abs(slope)<.65 or error>.22 or abs(slope-normal_slope)>.08:continue

        ab=np.clip((along[supported]/L*10).astype(int),0,9)
        if np.mean(np.bincount(ab,weights=aa,minlength=10)>.2)<.6:continue
        if abs(intercept-offsets[i])>1.5:continue
        low[i]=intercept+slope*(bottom-top);high[i]=intercept;row['accepted']=True
    if not any(r['accepted'] for r in records):return poly,poly,records
    lp=intersect_lines(xy,outward,low);hp=intersect_lines(xy,outward,high)
    if lp is None or hp is None or lp.area<p.area*.25 or hp.area<p.area*.25:
        for r in records:r['accepted']=False
        return poly,poly,records
    return lp,hp,records


def build_height_cells(bodies,slabs,base,c,n,a,overhang_records):


    bottoms={r['layer']:r['underside_z_m'] for r in overhang_records if r.get('accepted')}
    cells=[];profiles=[]
    support_index=cKDTree(c[:,:2])
    for i,(poly,z,_) in enumerate(bodies):
        ceiling=bottoms.get(i,z)


        margin=max(2.5,.25*(ceiling-base))+1e-6
        bounds=np.array(poly.bounds,dtype=float);bounds[:2]-=margin;bounds[2:]+=margin
        center=(bounds[:2]+bounds[2:])/2;radius=np.linalg.norm(bounds[2:]-center)+1e-7
        ids=np.sort(np.asarray(support_index.query_ball_point(center,radius),dtype=np.int64))
        xy=c[ids,:2];ids=ids[((xy>=bounds[:2])&(xy<=bounds[2:])).all(axis=1)]
        lp,hp,records=wall_loft(poly,base,ceiling,c[ids],n[ids],a[ids])
        cells.append(HeightCell(lp,hp,float(base),float(ceiling),'observed_wall_profile'))
        profiles.append(dict(layer=i,planes=records))
    for poly,lo,hi in slabs:
        cells.append(HeightCell(poly,poly,float(lo),float(hi),'observed_underside_and_upper_surface'))
    return cells,profiles

