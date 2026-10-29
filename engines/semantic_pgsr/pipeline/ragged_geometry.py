import torch
from pgsr_pipeline_ops import _geometry
USE_NATIVE_RAYS = True

class _Geometry(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rois, *inputs):

        inputs=tuple(x.contiguous() for x in inputs)
        ctx.rois=rois;ctx.save_for_backward(*inputs)
        noise,valid=_geometry.geometry(inputs,rois,inputs[0].new_empty(0),False)
        ctx.mark_non_differentiable(valid)
        return noise,valid
    @staticmethod
    def backward(ctx, grad, unused):
        gs=_geometry.geometry(ctx.saved_tensors,ctx.rois,grad.contiguous(),True)
        result=[]
        for i in range(len(ctx.rois)):
            result.extend([gs[2*i],None,None,gs[2*i+1],None,None,None,None,None])
        return (None,*result)

def geometry(cameras,near_cameras,packages,near_packages,masks,rc,nc):
    from pipeline.rendering import pixel_rays
    from pipeline.losses import local_mask
    inputs=[];rois=[]
    for c,nc_,p,np,m,r,n in zip(cameras,near_cameras,packages,near_packages,masks,rc,nc):
        depth=p['plane_depth'];device,dtype=depth.device,depth.dtype
        nm=np.get('loss_mask');nm=nm.to(dtype=dtype) if nm is not None else depth.new_empty(0)
        inputs.extend([depth,(depth.new_empty(0) if USE_NATIVE_RAYS else pixel_rays(c,p['roi'],device,dtype)),local_mask(m,p),np['plane_depth'],nm,c.world_view_transform,nc_.world_view_transform,r[0],n[0]])
        rois.append(list(p['roi'])+list(np['roi']))
    return _Geometry.apply(rois,*inputs)


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rois,ids,*inputs):
        inputs=tuple(t.contiguous() for t in inputs);ids=ids.contiguous()
        ctx.rois=rois;ctx.save_for_backward(ids,*inputs)
        out=tuple(_geometry.gather(inputs,rois,ids,[],False))
        ctx.mark_non_differentiable(out[2])
        return out
    @staticmethod
    def backward(ctx,gn,gd,gp):
        ids,*inputs=ctx.saved_tensors
        shape=ids.shape
        if gn is None:gn=inputs[0].new_zeros((*shape,3))
        if gd is None:gd=inputs[0].new_zeros(shape)
        grads=_geometry.gather(inputs,ctx.rois,ids,[gn.contiguous(),gd.contiguous()],True)
        return (None,None,*grads)

def gather(packages,ids):
    inputs=[];rois=[]
    for p in packages:
        inputs.extend([p['rendered_normal'],p['rendered_distance']]);rois.append(list(p['roi']))
    return _Gather.apply(rois,ids,*inputs)
