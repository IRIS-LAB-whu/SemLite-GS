import torch
from torch.autograd.function import once_differentiable
from pipeline.native_compacts import native

class _Crops(torch.autograd.Function):
    @staticmethod
    def forward(ctx,specs,*values):
        ctx.set_materialize_grads(False)
        ctx.sizes=[[*v.shape,y,x] for v,(y,x,h,w) in zip(values,specs)]
        return tuple(v[:,y:y+h,x:x+w] for v,(y,x,h,w) in zip(values,specs))
    @staticmethod
    @once_differentiable
    def backward(ctx,*grads):
        result=[None]*len(grads)
        ids=[i for i,g in enumerate(grads) if g is not None and ctx.needs_input_grad[i+1]]
        for start in range(0,len(ids),32):
            group=ids[start:start+32]
            gs=native.crop_backward([grads[i] for i in group],[ctx.sizes[i] for i in group])
            for i,g in zip(group,gs):result[i]=g
        return None,*result

def crops(values,specs):
    assert len(values)==len(specs)
    if not values:return ()
    if any(v.ndim!=3 or not v.is_cuda or v.dtype!=torch.float32 for v in values):
        return tuple(v[...,y:y+h,x:x+w] for v,(y,x,h,w) in zip(values,specs))
    return _Crops.apply(specs,*values)

class _Maps(torch.autograd.Function):
    @staticmethod
    def forward(ctx,*values):
        ctx.set_materialize_grads(False);ctx.save_for_backward(*values)
        return tuple(native.maps_forward(list(values)))
    @staticmethod
    @once_differentiable
    def backward(ctx,*grads):
        values=ctx.saved_tensors;result=[None]*len(values)
        ids=[i for i,g in enumerate(grads) if g is not None]
        if ids:
            gs=native.maps_backward([v for i in ids for v in values[i*5:i*5+5]],
                                   [grads[i].contiguous() for i in ids])
            for j,i in enumerate(ids):result[5*i:5*i+2]=gs[2*j:2*j+2]
        return tuple(result)

def maps_many(cameras,models):
    values=[]
    for c,m in zip(cameras,models):
        values.extend([m.get_xyz,m.get_smallest_axis(),c.world_view_transform,
                       c.camera_center,m.raster_gradient_mask])
    return _Maps.apply(*(v.contiguous() for v in values))

def crop_packages(packages):
    refs=[];values=[];specs=[]
    for p in packages:
        x,y,w,h=p['roi'];rx,ry,_,_=p['raw_roi']
        for key,v in p['raw'].items():
            refs.append((p,key));values.append(v);specs.append((y-ry,x-rx,h,w))
    for (p,key),v in zip(refs,crops(values,specs)):p[key]=v
