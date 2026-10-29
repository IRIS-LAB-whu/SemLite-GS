import os
import torch
import torch.nn.functional as F
from pipeline.rendering import pixel_rays
from pipeline.losses import local_mask, _intrinsic
from pipeline.supervision import supervision_gray


USE_RAGGED_GEOMETRY = os.environ.get("PGSR_EXPERIMENTAL_RAGGED", "1") == "1"
USE_LAYOUT_STABLE_PAIR = True
USE_CACHED_GEOMETRY = True
USE_FUSED_NCC = True
USE_FUSED_POINT_TRANSFORM = True
USE_FUSED_RIGID_CHAIN = True


class _PointMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix):
        from pipeline.loss_batch import _get_c_module
        ctx.save_for_backward(matrix)
        return _get_c_module().point_matmul(points.contiguous(),matrix,False)

    @staticmethod
    def backward(ctx, grad):
        from pipeline.loss_batch import _get_c_module
        matrix,=ctx.saved_tensors
        return _get_c_module().point_matmul(grad.contiguous(),matrix,True),None


def point_matmul(points,matrix):
    if USE_FUSED_POINT_TRANSFORM and points.is_cuda and points.dtype==torch.float32 and not matrix.requires_grad:
        return _PointMatmul.apply(points,matrix)
    return points.bmm(matrix)


class _RigidChain(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,first,second):
        from pipeline.loss_batch import _get_c_module
        first,second=first.contiguous(),second.contiguous()
        ctx.save_for_backward(first,second)
        return _get_c_module().rigid_chain(points.contiguous(),first,second,False)

    @staticmethod
    def backward(ctx,grad):
        from pipeline.loss_batch import _get_c_module
        first,second=ctx.saved_tensors
        return _get_c_module().rigid_chain(grad.contiguous(),first,second,True),None,None


def rigid_chain(points,first,second):
    if USE_FUSED_RIGID_CHAIN and points.is_cuda and points.dtype==torch.float32 and not (first.requires_grad or second.requires_grad):
        return _RigidChain.apply(points,first,second)
    world=point_matmul(points-first[:,None,3,:3],first[:,:3,:3].transpose(1,2))
    return point_matmul(world,second[:,:3,:3])+second[:,None,3,:3]


class _PatchNCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ref, near, centers, homography, bounds, valid, radius):
        from pipeline.loss_batch import _get_c_module
        native = _get_c_module()
        if USE_LAYOUT_STABLE_PAIR:
            from pgsr_pipeline_ops import _pair
            native = _pair
        ncc, good, jac, critical = native.patch_ncc_forward(
            ref.contiguous(),near.contiguous(),centers.contiguous(),
            homography.contiguous(),bounds.contiguous(),valid.contiguous(),radius)
        ctx.save_for_backward(jac)
        ctx.mark_non_differentiable(good,critical)
        return ncc, good, critical

    @staticmethod
    def backward(ctx, grad, unused, unused_critical):
        jac, = ctx.saved_tensors
        return None,None,None,jac*grad[...,None,None],None,None,None


def fused_patch_ncc(ref, near, centers, homography, bounds, valid, radius):

    ncc,good,critical=_PatchNCC.apply(ref,near,centers,homography,bounds,valid,radius)


    count=int(critical.sum(-1).max())
    if count:
        ids=critical.to(torch.int32).topk(count,dim=1).indices
        selected=critical.gather(1,ids)
        cs=centers.gather(1,ids[...,None].expand(-1,-1,2))
        hs=homography.gather(1,ids[...,None,None].expand(-1,-1,3,3))
        exact,accepted=reference_patch_ncc(ref,near,cs,hs,bounds,valid.gather(1,ids)&selected,radius)
        ncc=ncc.scatter(1,ids,torch.where(selected,exact,ncc.gather(1,ids)))
        good=good.scatter(1,ids,torch.where(selected,accepted,good.gather(1,ids)))
    return ncc,good


def reference_patch_ncc(ref,near,centers,homography,bounds,valid,radius):
    b,k=centers.shape[:2];device=centers.device;dtype=centers.dtype
    a=torch.arange(-radius,radius+1,device=device,dtype=dtype)
    oy,ox=torch.meshgrid(a,a,indexing='ij')
    pixels=centers[:,:,None]+torch.stack((ox,oy),-1).reshape(1,1,-1,2)
    hp=torch.cat((pixels,torch.ones_like(pixels[...,:1])),-1)
    if USE_LAYOUT_STABLE_PAIR:
        from pipeline.stable_pair import project_patches
        warped=project_patches(pixels,homography)
    else:warped=hp@homography.transpose(-1,-2)
    z=warped[...,2:]
    near_pixels=warped[...,:2]/torch.where(z.abs()>1e-8,z,torch.full_like(z,1e-8))
    inside=((pixels>=0).all(-1)&(pixels<=bounds[:,None,None,:2]).all(-1)&
        (near_pixels>=0).all(-1)&(near_pixels<=bounds[:,None,None,2:]).all(-1)&
        torch.isfinite(near_pixels).all(-1)).all(-1)&valid
    safe_ref=torch.where(inside[:,:,None,None],pixels,torch.zeros_like(pixels))
    safe_near=torch.where(inside[:,:,None,None],near_pixels,torch.zeros_like(near_pixels))
    count=(2*radius+1)**2
    x=_sample(ref,safe_ref,bounds[:,:2]).reshape(b,k,count);y=_sample(near,safe_near,bounds[:,2:]).reshape(b,k,count)
    sx,sy=x.sum(-1),y.sum(-1)
    cross=(x*y).sum(-1)-sy*sx/count
    vx=x.square().sum(-1)-sx.square()/count;vy=y.square().sum(-1)-sy.square()/count
    ncc=(1-cross.square()/(vx*vy+1e-8)).clamp(0,2)
    return ncc,inside&(ncc<.9)

def prepare_pair_constants(cameras, device):

    cameras=[c for c in cameras if all(hasattr(c,k) for k in ("Fx","Fy","Cx","Cy","ncc_scale"))]
    if not cameras:return
    values=[];intrinsics=[];rois=[];owners=[]
    for c in cameras:
        gray=supervision_gray(c)
        values.append([c.Fx,c.Fy,c.Cx,c.Cy,c.ncc_scale,gray.shape[-1]-1,gray.shape[-2]-1])
        scale=float(c.ncc_scale)
        intrinsics.append([[c.Fx/scale,0,c.Cx/scale],[0,c.Fy/scale,c.Cy/scale],[0,0,1]])
        c._pair_roi_constants={}
        for entry in getattr(c,"_supervision",{}).values():
            x,y,w,h=entry.roi;rois.append([x,y,w-1,h-1]);owners.append((c,entry.roi))
    values=torch.tensor(values,device=device,dtype=torch.float32)
    intrinsics=torch.tensor(intrinsics,device=device,dtype=torch.float32)
    inverses=torch.linalg.inv(intrinsics)
    for i,c in enumerate(cameras):c._pair_constants=(values[i],intrinsics[i],inverses[i])
    if rois:
        tensor=torch.tensor(rois,device=device,dtype=torch.float32)
        for i,(c,roi) in enumerate(owners):c._pair_roi_constants[roi]=tensor[i]


def _camera_constants(c,device):
    cached=getattr(c,"_pair_constants",None)
    if cached is None or cached[0].device!=device:
        prepare_pair_constants([c],device);cached=c._pair_constants
    return cached


def _roi_constants(c,roi,device,dtype):
    cache=getattr(c,"_pair_roi_constants",{})
    value=cache.get(roi)
    if value is None or value.device!=device:
        x,y,w,h=roi;value=torch.tensor([x,y,w-1,h-1],device=device,dtype=dtype)
        cache[roi]=value;c._pair_roi_constants=cache
    return value


def _stack_images(images):
    h=max(t.shape[-2] for t in images); w=max(t.shape[-1] for t in images)
    return torch.stack([F.pad(t,(0,w-t.shape[-1],0,h-t.shape[-2])) for t in images])


def _sample(images,pixels,bounds=None):
    if USE_LAYOUT_STABLE_PAIR and bounds is not None and images.is_cuda and images.dtype==torch.float32:
        from pipeline.stable_pair import sample
        return sample(images,pixels,bounds)

    b,c,h,w=images.shape
    grid=torch.stack((2*pixels[...,0]/max(w-1,1)-1,2*pixels[...,1]/max(h-1,1)-1),-1)
    return F.grid_sample(images,grid.reshape(b,-1,1,2),align_corners=True).squeeze(-1).transpose(1,2)


def _mean(values,mask):
    return torch.where(mask,values,0.).sum(-1)/mask.sum(-1).clamp_min(1)


def pair_losses(cameras,near_cameras,packages,near_packages,opt,masks=None,virtual=None):
    b=len(cameras)
    if not b: raise ValueError('At least one multiview pair required')
    masks=[None]*b if masks is None else masks
    virtual=[False]*b if virtual is None else virtual
    device=packages[0]['plane_depth'].device;dtype=packages[0]['plane_depth'].dtype
    sizes=[p['plane_depth'].numel() for p in packages]; n=max(sizes)
    def padpoints(values,columns=False):
        return torch.stack([F.pad(v,(0,0,0,n-len(v))) if columns else F.pad(v,(0,n-len(v))) for v in values])
    use_ragged=(USE_RAGGED_GEOMETRY and USE_LAYOUT_STABLE_PAIR and USE_CACHED_GEOMETRY
                and USE_FUSED_RIGID_CHAIN and device.type=='cuda' and dtype==torch.float32 and b<=8
                and not any(c.world_view_transform.requires_grad for c in (*cameras,*near_cameras)))
    if use_ragged:
        from pipeline.ragged_geometry import geometry
        rc=[_camera_constants(c,device) for c in cameras]
        nc=[_camera_constants(cameras[i] if virtual[i] else c,device) for i,c in enumerate(near_cameras)]
        rv=torch.stack([v[0] for v in rc]);nv_constants=torch.stack([v[0] for v in nc])
        noise,valid=geometry(cameras,near_cameras,packages,near_packages,masks,rc,nc)
    else:
        depths=padpoints([p['plane_depth'].reshape(-1) for p in packages])
        rays=padpoints([pixel_rays(c,p['roi'],device,dtype).reshape(-1,3) for c,p in zip(cameras,packages)],True)
        pixels=[]
        for p in packages:
            x,y,w,h=p['roi']
            yy,xx=torch.meshgrid(torch.arange(y,y+h,device=device,dtype=dtype),torch.arange(x,x+w,device=device,dtype=dtype),indexing='ij')
            pixels.append(torch.stack((xx,yy),-1).reshape(-1,2))
        pixels=padpoints(pixels,True)
        selected=padpoints([local_mask(m,p).reshape(-1) for m,p in zip(masks,packages)])
        view=torch.stack([c.world_view_transform for c in cameras]);nv=torch.stack([c.world_view_transform for c in near_cameras])
        points=rays*depths[...,None]
        near=rigid_chain(points,view,nv)
        if USE_CACHED_GEOMETRY:
            rc=[_camera_constants(c,device) for c in cameras]
            nc=[_camera_constants(cameras[i] if virtual[i] else c,device) for i,c in enumerate(near_cameras)]
            rv=torch.stack([v[0] for v in rc]);nv_constants=torch.stack([v[0] for v in nc])

            focal,center=rv[:,None,:2],rv[:,None,2:4]
            nfocal,ncenter=nv_constants[:,None,:2],nv_constants[:,None,2:4]
            regions=torch.stack([_roi_constants(c,p['roi'],device,dtype) for c,p in zip(near_cameras,near_packages)])
            origins,bounds=regions[:,None,:2],regions[:,None,2:]
        else:
            focal=torch.tensor([[c.Fx,c.Fy] for c in cameras],device=device,dtype=dtype)[:,None]
            center=torch.tensor([[c.Cx,c.Cy] for c in cameras],device=device,dtype=dtype)[:,None]
            nfocal=torch.tensor([[c.Fx,c.Fy] for c in near_cameras],device=device,dtype=dtype)[:,None]
            ncenter=torch.tensor([[c.Cx,c.Cy] for c in near_cameras],device=device,dtype=dtype)[:,None]
            origins=torch.tensor([p['roi'][:2] for p in near_packages],device=device,dtype=dtype)[:,None]
            bounds=torch.tensor([[p['roi'][2]-1,p['roi'][3]-1] for p in near_packages],device=device,dtype=dtype)[:,None]
        projected=near[...,:2]/near[...,2:3].clamp_min(1e-8)*nfocal+ncenter
        local=projected-origins
        mapped=_sample(_stack_images([p['plane_depth'] for p in near_packages]),local,bounds)[...,0]
        valid=(local>=0).all(-1)&(local<=bounds).all(-1)&(near[...,2]>0)&(mapped>0)&(depths>0)&selected
        near_masks=_stack_images([p['loss_mask'].float().reshape(1,*p['plane_depth'].shape[-2:]) if 'loss_mask' in p else torch.ones_like(p['plane_depth']) for p in near_packages])
        valid=valid&(_sample(near_masks,local,bounds)[...,0]>.999)
        back=near/near[...,2:3].clamp_min(1e-8)*mapped[...,None]
        reference=rigid_chain(back,nv,view)
        roundtrip=reference[...,:2]/reference[...,2:3].clamp_min(1e-8)*focal+center
        noise=(roundtrip-pixels).norm(dim=-1)
        valid=valid&(reference[...,2]>0)&torch.isfinite(noise)
    if not opt.wo_use_geo_occ_aware:
        valid=valid&(noise<opt.multi_view_pixel_noise_th)
        weights=torch.exp(-noise).detach()
    else: weights=torch.ones_like(noise)
    loss=opt.multi_view_geo_weight*_mean(weights*noise,valid)
    if opt.multi_view_ncc_weight==0: return loss
    max_valid=int(valid.sum(-1).max())
    k=min(opt.multi_view_sample_num,max_valid)
    if k<=0: return loss

    scores=torch.rand_like(noise) if max_valid>opt.multi_view_sample_num else torch.ones_like(noise)
    ids=torch.where(valid,scores,-torch.ones_like(noise)).topk(k,dim=1).indices
    chosen_valid=valid.gather(1,ids)
    def gather(v):return v.gather(1,ids[...,None].expand(-1,-1,v.shape[-1]))
    if use_ragged:
        from pipeline.ragged_geometry import gather as packed_gather
        normals,distance,chosen_pixels=packed_gather(packages,ids)
    else:
        normals=gather(padpoints([p['rendered_normal'].permute(1,2,0).reshape(-1,3) for p in packages],True))
        distance=padpoints([p['rendered_distance'].reshape(-1) for p in packages]).gather(1,ids)
    refscale=(rv[:,4] if USE_CACHED_GEOMETRY else torch.tensor([c.ncc_scale for c in cameras],device=device,dtype=dtype))[:,None,None,None]
    r=opt.multi_view_patch_size

    if USE_CACHED_GEOMETRY:
        kn=torch.stack([v[1] for v in nc]);kr_inv=torch.stack([v[2] for v in rc])
    else:
        kr=torch.stack([_intrinsic(c,device,dtype,c.ncc_scale) for c in cameras])
        kn=torch.stack([_intrinsic(c,device,dtype,c.ncc_scale) for c in near_cameras])
        kr_inv=torch.linalg.inv(kr)
    if USE_LAYOUT_STABLE_PAIR and device.type=='cuda' and dtype==torch.float32:
        from pipeline.stable_pair import homography as stable_homography
        from pipeline.stable_pair import camera_pair
        calibration=[camera_pair(c,n) for c,n in zip(cameras,near_cameras)]
        rotations=torch.stack([v[0] for v in calibration]);translations=torch.stack([v[1] for v in calibration])
        homography=stable_homography(normals,distance,rotations,translations,kn,kr_inv)
    else:
        rotation=nv[:,:3,:3].transpose(1,2).bmm(view[:,:3,:3])
        translation=-rotation.bmm(view[:,3,:3,None]).squeeze(-1)+nv[:,3,:3]
        homography=rotation[:,None]-translation[:,None,:,None]*normals[:,:,None,:]/distance.clamp_min(1e-8)[:,:,None,None]
        homography=kn[:,None]@homography@kr_inv[:,None]
    rg=[supervision_gray(c).to(device) for c in cameras]
    ng=[supervision_gray(cameras[i] if virtual[i] else c).to(device) for i,c in enumerate(near_cameras)]
    if USE_FUSED_NCC and device.type=='cuda' and dtype==torch.float32 and 0<=r<=5 and all(min(v.shape[-2:])>1 for v in rg+ng):
        limits=(torch.cat((rv[:,5:7],nv_constants[:,5:7]),-1) if USE_CACHED_GEOMETRY else
                torch.tensor([[a.shape[-1]-1,a.shape[-2]-1,b.shape[-1]-1,b.shape[-2]-1] for a,b in zip(rg,ng)],device=device,dtype=dtype))
        chosen_valid=chosen_valid&(distance>1e-8)
        for i,is_virtual in enumerate(virtual):
            if is_virtual:chosen_valid[i]=False
        centers=(chosen_pixels if use_ragged else gather(pixels))/refscale.reshape(b,1,1)
        ncc,good=fused_patch_ncc(_stack_images(rg),_stack_images(ng),centers,homography,limits,chosen_valid,r)
        return loss+opt.multi_view_ncc_weight*_mean(ncc*weights.gather(1,ids),good)
    a=torch.arange(-r,r+1,device=device,dtype=dtype);oy,ox=torch.meshgrid(a,a,indexing='ij')
    offsets=torch.stack((ox,oy),-1).reshape(1,1,-1,2)
    ref_pixels=(chosen_pixels if use_ragged else gather(pixels))[:,:,None]/refscale+offsets
    hp=torch.cat((ref_pixels,torch.ones_like(ref_pixels[...,:1])),-1)
    warped=hp@homography.transpose(-1,-2)
    denom=warped[...,2:3]
    near_pixels=warped[...,:2]/torch.where(denom.abs()>1e-8,denom,torch.full_like(denom,1e-8))
    rb=(rv[:,5:7] if USE_CACHED_GEOMETRY else torch.tensor([[v.shape[-1]-1,v.shape[-2]-1] for v in rg],device=device,dtype=dtype))[:,None,None]
    nb=(nv_constants[:,5:7] if USE_CACHED_GEOMETRY else torch.tensor([[v.shape[-1]-1,v.shape[-2]-1] for v in ng],device=device,dtype=dtype))[:,None,None]
    inside=((ref_pixels>=0).all(-1)&(ref_pixels<=rb).all(-1)&(near_pixels>=0).all(-1)&(near_pixels<=nb).all(-1)&torch.isfinite(near_pixels).all(-1)).all(-1)
    inside=inside&(distance>1e-8)&chosen_valid
    for i,is_virtual in enumerate(virtual):
        if is_virtual:inside[i]=False

    safe_ref=torch.where(inside[:,:,None,None],ref_pixels,torch.zeros_like(ref_pixels))
    safe_near=torch.where(inside[:,:,None,None],near_pixels,torch.zeros_like(near_pixels))
    count=(2*r+1)**2
    ref_values=_sample(_stack_images(rg),safe_ref).reshape(b,k,count)
    near_values=_sample(_stack_images(ng),safe_near).reshape(b,k,count)


    rs,ns=ref_values.sum(-1),near_values.sum(-1)
    cross=(ref_values*near_values).sum(-1)-ns*rs/count
    rv=ref_values.square().sum(-1)-rs.square()/count
    nv_=near_values.square().sum(-1)-ns.square()/count
    ncc=(1-cross.square()/(rv*nv_+1e-8)).clamp(0,2)
    good=inside&(ncc<.9)
    return loss+opt.multi_view_ncc_weight*_mean(ncc*weights.gather(1,ids),good)
