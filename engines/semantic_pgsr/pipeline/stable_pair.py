import torch
from pgsr_pipeline_ops import _pair

USE_FIXED_GRAY_GRAD_ONLY = True

class _Sample(torch.autograd.Function):
 @staticmethod
 def forward(ctx,images,pixels,bounds):
  images,pixels,bounds=images.contiguous(),pixels.contiguous(),bounds.contiguous()
  ctx.save_for_backward(images,pixels,bounds)
  return _pair.sample_forward(images,pixels,bounds)
 @staticmethod
 def backward(ctx,grad):
  if USE_FIXED_GRAY_GRAD_ONLY and not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
   gp=_pair.sample_pixel_backward(*ctx.saved_tensors,grad.contiguous())
   return None,gp,None
  gi,gp=_pair.sample_backward(*ctx.saved_tensors,grad.contiguous())
  return gi,gp,None

def sample(images,pixels,bounds):
 return _Sample.apply(images,pixels.reshape(pixels.shape[0],-1,2),bounds.reshape(-1,2))

class _Homography(torch.autograd.Function):
 @staticmethod
 def forward(ctx,normals,distance,view,near,kn,ki):
  inputs=tuple(v.contiguous() for v in (normals,distance,view,near,kn,ki))
  ctx.save_for_backward(*inputs)
  return _pair.homography(*inputs,normals.new_empty(0),False)[0]
 @staticmethod
 def backward(ctx,grad):
  gn,gd=_pair.homography(*ctx.saved_tensors,grad.contiguous(),True)
  return gn,gd,None,None,None,None

def homography(normals,distance,view,near,kn,ki):
 if any(v.requires_grad for v in (view,near,kn,ki)):
  raise ValueError('Camera calibration must be fixed for fused homographies')
 return _Homography.apply(normals,distance,view,near,kn,ki)


def camera_pair(camera,near_camera):

    v,nv=camera.world_view_transform,near_camera.world_view_transform
    key=(id(near_camera),v.data_ptr(),v._version,nv.data_ptr(),nv._version)
    cache=getattr(camera,'_stable_pair_calibration',{})
    if key not in cache:
        view,near=v[None].contiguous(),nv[None].contiguous()
        rotation=near[:,:3,:3].transpose(1,2).bmm(view[:,:3,:3])
        translation=-rotation.bmm(view[:,3,:3,None]).squeeze(-1)+near[:,3,:3]
        cache[key]=(rotation[0],translation[0])
        camera._stable_pair_calibration=cache
    return cache[key]


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx,pixels,matrix):
        if pixels.requires_grad:raise ValueError('Patch centers must be fixed')
        pixels,matrix=pixels.contiguous(),matrix.contiguous()
        ctx.save_for_backward(pixels,matrix)
        return _pair.project(pixels,matrix,pixels.new_empty(0),False)
    @staticmethod
    def backward(ctx,grad):
        return None,_pair.project(*ctx.saved_tensors,grad.contiguous(),True)

def project_patches(pixels,matrix):
    return _Project.apply(pixels,matrix)
