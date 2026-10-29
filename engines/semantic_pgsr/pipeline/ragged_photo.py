import torch
from pgsr_pipeline_ops import _roi
class _Photo(torch.autograd.Function):
    @staticmethod
    def forward(ctx,rois,guard,normals,image_weight,counts,guard_counts,ssim,guard_prediction,*inputs):
        inputs=tuple(x.contiguous() for x in inputs)
        counts,guard_counts,ssim=counts.contiguous(),guard_counts.contiguous(),ssim.contiguous()
        ctx.args=(rois,guard,normals,image_weight)
        ctx.save_for_backward(counts,guard_counts,ssim,guard_prediction,*inputs)
        return _roi.photo(inputs,rois,counts,guard_counts,ssim,guard_prediction,guard,normals,image_weight,counts.new_empty(0),False)[0]
    @staticmethod
    def backward(ctx,grad):


        counts,guard_counts,ssim,guard_prediction,*inputs=ctx.saved_tensors;rois,guard,normals,image_weight=ctx.args
        gs=_roi.photo(inputs,rois,counts,guard_counts,ssim,guard_prediction,guard,normals,image_weight,grad.contiguous(),True)
        result=[]
        for i in range(len(rois)):result.extend([*gs[4*i:4*i+4],None,None,None,None,None])
        return (None,None,None,None,None,None,None,gs[-1] if guard else None,*result)

def terms(packages,targets,counts,ssim,opt,iteration,guard_weight,guard_prediction=None):
    normal=iteration>opt.single_view_weight_from_iter;inputs=[];rois=[]
    for p,t in zip(packages,targets):
        raw=p['raw'];rgb=raw['render'];empty=rgb.new_empty(0)
        inputs.extend([rgb,raw.get('app_image',empty),p['depth_normal'] if normal else empty,p['rendered_normal'] if normal else empty,*t])
        x,y,w,h=p['roi'];rx,ry,rw,rh=p['raw_roi'];rois.append([x-rx,y-ry,w,h,rw,rh])
    gc=torch.stack([t[4].sum() for t in targets]) if guard_weight else torch.zeros_like(counts)
    if guard_prediction is None:
        h=max(p['raw']['render'].shape[-2] for p in packages);w=max(p['raw']['render'].shape[-1] for p in packages)
        guard_prediction=torch.stack([torch.nn.functional.pad(p['raw']['render'],(0,w-p['raw']['render'].shape[-1],0,h-p['raw']['render'].shape[-2])) for p in packages])
    return _Photo.apply(rois,bool(guard_weight),normal,not opt.wo_image_weight,counts,gc,ssim.detach(),guard_prediction,*inputs)
