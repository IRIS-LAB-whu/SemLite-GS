import copy
import sys
from pathlib import Path
import torch
from pipeline.active import CompactModel, PARAMETERS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'submodules/pgsr-view-batch'))
import pgsr_compact_batch as native

class Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, locals_, *leaves):
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*locals_)
        return tuple(v.view_as(v) for v in values)

    @staticmethod
    def backward(ctx, *gradients):
        positions=[i for i,g in enumerate(gradients) if g is not None and ctx.needs_input_grad[i+2]]
        result=[None]*len(gradients)
        if positions:
            local=ctx.saved_tensors
            selected=native.backward([gradients[i] for i in positions],[local[i] for i in positions])
            for i,g in zip(positions,selected):result[i]=g
        return None,None,*result

class NativeCompacts:
    def __init__(self):
        self.key=None

    def prepare(self, model, indices):
        if not indices:return {}
        if len(indices)>4 or any(not getattr(model,a).is_cuda or getattr(model,a).dtype!=torch.float32
                                or not getattr(model,a).is_contiguous() for a in PARAMETERS.values()):
            return {o:CompactModel(model,i,o) for o,i in indices.items()}
        key=(id(model.instance_ids),model.instance_ids._version,
             tuple((o,id(i),i._version,len(i)) for o,i in indices.items()))
        if key!=self.key:
            with torch.no_grad():
                templates={o:CompactModel(model,i,o) for o,i in indices.items()}
            for c in templates.values():
                for attr in PARAMETERS.values():delattr(c,attr)
                c.leaves={}
            self.templates=templates
            self.ids_ref=model.instance_ids
            self.key=key
        entries=[(o,n,a,c) for o,c in self.templates.items() for n,a in PARAMETERS.items()]
        raw=native.forward([getattr(model,a).detach() for o,n,a,c in entries],
                           [c.global_indices for o,n,a,c in entries],
                           [c.optimizer_indices for o,n,a,c in entries])
        leaves=[v.requires_grad_(torch.is_grad_enabled()) for v in raw[1::2]]
        values=Route.apply(tuple(raw[::2]),tuple(c.local_trainable for o,n,a,c in entries),*leaves)
        result={o:copy.copy(c) for o,c in self.templates.items()}
        for c in result.values():
            c.active_sh_degree=model.active_sh_degree;c.use_app=model.use_app;c.leaves={}
        for (o,n,a,_),leaf,value in zip(entries,leaves,values):
            result[o].leaves[n]=leaf;setattr(result[o],a,value)
        return result

def prepare(model,indices):
    cache=getattr(model,'_native_compacts',None)
    if cache is None:cache=model._native_compacts=NativeCompacts()
    return cache.prepare(model,indices)
