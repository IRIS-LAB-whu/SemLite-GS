import torch
from torch.autograd.function import once_differentiable
from semlite_gs.vendor.rotation_conversions import quaternion_to_matrix


USE_FUSED_AXIS = False
USE_FUSED_MAPS = False

def configure(mode):

    if mode not in ('reference', 'axis', 'fused'):
        raise ValueError('render_inputs_mode must be reference, axis or fused')
    global USE_FUSED_AXIS, USE_FUSED_MAPS
    USE_FUSED_AXIS = mode in ('axis', 'fused')
    USE_FUSED_MAPS = mode == 'fused'


def _native():
    from pipeline.loss_batch import _get_c_module
    return _get_c_module()

def reference_axis(rotation, scales):
    index = scales.min(-1).indices
    return quaternion_to_matrix(rotation).gather(2,index[:,None,None].expand(-1,3,1)).squeeze(-1), index

class _Axis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rotation, scales):
        normal, index = _native().axis_forward(rotation, scales)
        ctx.save_for_backward(rotation,index);ctx.mark_non_differentiable(index)
        return normal,index
    @staticmethod
    @once_differentiable
    def backward(ctx, grad, unused):
        return _native().axis_backward(*ctx.saved_tensors,grad.contiguous()),None

def smallest_axis(rotation, scales):

    if rotation.is_cuda and rotation.is_contiguous() and rotation.dtype==torch.float32 and scales.device==rotation.device and scales.dtype==rotation.dtype:
        return _Axis.apply(rotation.contiguous(),scales.contiguous())
    return reference_axis(rotation,scales)

def reference_maps(xyz, normal, view, center):
    negative=(normal*(center-xyz)).sum(-1,keepdim=True)<0
    normal=torch.where(negative,-normal,normal) @ view[:3,:3]
    points=xyz @ view[:3,:3]+view[3,:3]
    distance=(normal*points).sum(-1,keepdim=True).abs()
    return torch.cat((normal,torch.ones_like(distance),distance),-1)

class _Maps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, normal, view, center, mask):
        out=_native().camera_maps_forward(xyz,normal,view,center)
        ctx.save_for_backward(xyz,normal,view,center,mask)
        return out
    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        dx,dn=_native().camera_maps_backward(*ctx.saved_tensors,grad.contiguous())
        return dx,dn,None,None,None

def camera_maps(xyz, normal, view, center, gradient_mask=None):
    tensors=(xyz,normal,view,center)
    if all(t.is_cuda and t.dtype==torch.float32 and t.device==xyz.device for t in tensors) and not (view.requires_grad or center.requires_grad):
        mask=(torch.empty(0,device=xyz.device,dtype=torch.bool) if gradient_mask is None else gradient_mask.contiguous())
        return _Maps.apply(*(t.contiguous() for t in tensors),mask)
    return reference_maps(xyz,normal,view,center)
