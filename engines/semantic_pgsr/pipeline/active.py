import torch
from functools import cached_property
from pipeline.model import GaussianModel

USE_FUSED_ADAM = True
USE_COMPACT_METADATA = True

PARAMETERS = {'xyz':'_xyz', 'f_dc':'_features_dc', 'f_rest':'_features_rest',
              'opacity':'_opacity', 'scaling':'_scaling', 'rotation':'_rotation'}


class ActiveSetCache:
    def __init__(self, padding=-1.):
        if not torch.isfinite(torch.tensor(padding)):
            raise ValueError('context padding must be finite')
        self.padding = padding
        self.population_key = None
        self.indices_by_instance = {}

    @torch.no_grad()
    def select(self, model, instance):
        xyz, ids = model._xyz, model.instance_ids


        position_version = xyz._version if self.padding >= 0 else None
        population_key = (xyz.data_ptr(), position_version, ids.data_ptr(), ids._version,
                          len(xyz), self.padding, model.context_mode, tuple((k,v.data_ptr(),v._version) for k,v in model.context_membership.items()))
        if population_key != self.population_key:
            self.indices_by_instance.clear()
            self.population_key = population_key
        if instance not in self.indices_by_instance:
            target = ids == instance
            if not target.any():
                raise ValueError(f'No Gaussians owned by instance {instance}')
            if instance in model.context_membership:
                keep = (target if model.context_mode == 'independent' else ids >= 0) | model.context_membership[instance]
                indices = keep.nonzero().flatten()
                if model.context_mode == 'independent':
                    indices = indices[model.render_order[indices].argsort(stable=True)]
            elif self.padding < 0:
                indices = torch.arange(len(xyz), device=xyz.device)
            else:
                points = xyz[target]
                lo, hi = points.amin(0)-self.padding, points.amax(0)+self.padding
                keep = target | ((xyz >= lo).all(1) & (xyz <= hi).all(1))
                indices = keep.nonzero().flatten()
            self.indices_by_instance[instance] = indices
        return self.indices_by_instance[instance]


def _compact_metadata(source, indices, instance):
    key = (indices.data_ptr(), indices._version, source.instance_ids.data_ptr(),
           source.instance_ids._version, len(indices))
    cache = getattr(source, '_compact_metadata_cache', {})
    entry = cache.get(instance)
    if entry is not None and entry[0] == key and entry[3] is source.instance_ids:
        return entry[2]
    ids = source.instance_ids[indices]
    target = ids == instance
    local = target.nonzero().flatten()
    result = (torch.where(target, ids, -1), target.contiguous(), local, indices[local])

    cache[instance] = (key, indices, result, source.instance_ids)
    source._compact_metadata_cache = cache
    return result


class CompactModel(GaussianModel):
    def __init__(self, source, indices, instance):
        super().__init__(source.max_sh_degree)
        self.target_instance = int(instance)
        self.global_indices = indices
        self.active_sh_degree = source.active_sh_degree
        self.use_app = source.use_app
        self.freeze_global = True
        if USE_COMPACT_METADATA:
            (self.instance_ids, self.raster_gradient_mask, self.local_trainable,
             self.optimizer_indices) = _compact_metadata(source, indices, instance)
        else:
            self.instance_ids = source.instance_ids[indices]
            target = self.instance_ids == instance
            self.raster_gradient_mask = target.contiguous()
            self.instance_ids = torch.where(target, self.instance_ids, -1)
            self.local_trainable = target.nonzero().flatten()
            self.optimizer_indices = indices[self.local_trainable]
        self.leaves = {}
        for name, attr in PARAMETERS.items():


            values = getattr(source, attr).detach()[indices]
            leaf = values[self.local_trainable].clone().requires_grad_(torch.is_grad_enabled())
            self.leaves[name] = leaf
            setattr(self, attr, values.index_copy(0, self.local_trainable, leaf))


    @cached_property
    def get_scaling(self):
        return super().get_scaling

    @cached_property
    def get_rotation(self):
        return super().get_rotation

    @cached_property
    def get_opacity(self):
        return super().get_opacity

    @cached_property
    def get_features(self):
        return super().get_features

    @cached_property
    def _smallest_axis(self):
        from pipeline import render_inputs
        if render_inputs.USE_FUSED_AXIS:
            return render_inputs.smallest_axis(self.get_rotation,self.get_scaling)
        return super().get_smallest_axis(return_idx=True)

    def get_smallest_axis(self, return_idx=False):
        return self._smallest_axis if return_idx else self._smallest_axis[0]

    def get_normal(self, view_cam):
        normal = self.get_smallest_axis()
        negative = (normal*(view_cam.camera_center-self._xyz)).sum(-1,keepdim=True)<0
        return torch.where(negative,-normal,normal)


@torch.no_grad()
def active_adam_step_many(model, compacts):

    compacts = list(compacts)
    if len({c.target_instance for c in compacts}) != len(compacts):
        raise ValueError('Mixed compact models must have distinct target owners')
    for group in model.optimizer.param_groups:
        entries = [(c.optimizer_indices, c.leaves[group['name']].grad)
                   for c in compacts if group['name'] in c.leaves and c.leaves[group['name']].grad is not None]
        if not entries:
            continue
        if group.get('weight_decay',0) or group.get('amsgrad',False) or group.get('maximize',False):
            raise ValueError('Active Adam supports native PGSR Adam settings only')
        p = group['params'][0]
        state = model.optimizer.state[p]
        if not state:
            state['step'] = torch.tensor(0.)
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)


        if state['step'].device.type != 'cpu':
            state['step'] = state['step'].cpu()
        state['step'] += 1
        step = float(state['step'])
        beta1, beta2 = group['betas']


        indices, grad = entries[0] if len(entries)==1 else (
            torch.cat([e[0] for e in entries]), torch.cat([e[1] for e in entries]))
        if USE_FUSED_ADAM and p.is_cuda and p.dtype==torch.float32 and p.is_contiguous():
            from pipeline.loss_batch import _get_c_module
            _get_c_module().indexed_adam(p,state['exp_avg'],state['exp_avg_sq'],
                indices.contiguous(),grad.contiguous(),beta1,beta2,(1-beta2**step)**.5,
                -group['lr']/(1-beta1**step),group['eps'])
            continue
        avg = state['exp_avg'][indices]
        sq = state['exp_avg_sq'][indices]
        avg.lerp_(grad, 1-beta1)
        sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
        denom = sq.sqrt().div_((1-beta2**step)**.5).add_(group['eps'])
        value = p[indices].addcdiv_(avg, denom, value=-group['lr']/(1-beta1**step))
        p[indices] = value
        state['exp_avg'][indices] = avg
        state['exp_avg_sq'][indices] = sq
    model.optimizer.zero_grad(set_to_none=True)


@torch.no_grad()
def add_compact_stats(model, package, indices, batch_size):
    trainable = package.get('active_trainable_mask')
    if trainable is None:
        trainable = model.trainable_mask[indices]
    visible = package['visibility_filter'] & trainable
    observed = visible & (package['out_observe'].reshape(-1)>0)
    global_observed = indices[observed]
    model.max_radii2D[global_observed] = torch.maximum(model.max_radii2D[global_observed],package['radii'][observed])
    signed, absolute = package['viewspace_points'], package['viewspace_points_abs']
    if signed.grad is not None and absolute.grad is not None:
        global_visible = indices[visible]
        model.xyz_gradient_accum[global_visible] += signed.grad[visible,:2].norm(dim=-1,keepdim=True)*batch_size
        model.xyz_gradient_accum_abs[global_visible] += absolute.grad[visible,:2].norm(dim=-1,keepdim=True)*batch_size
        model.denom[global_visible] += 1
        model.denom_abs[global_visible] += 1


USE_FUSED_STATS = True


@torch.no_grad()
def add_compact_stats_many(model, packages, compacts, batch_size, fused=None):

    if len(packages) != len(compacts):
        raise ValueError('One compact model is required per statistics package')
    if not packages:
        return
    if fused is None:
        fused = USE_FUSED_STATS
    if not fused:
        for package, compact in zip(packages, compacts):
            package['active_trainable_mask'] = compact.trainable_mask
            add_compact_stats(model, package, compact.global_indices, batch_size)
        return
    groups = {}
    compatible = model._xyz.is_cuda and model._xyz.dtype == torch.float32
    for package, compact in zip(packages, compacts):
        owner = compact.target_instance
        if owner in groups and groups[owner][0] is not compact:
            compatible = False
        groups.setdefault(owner, (compact, []))[1].append(package)
        compatible = compatible and package['radii'].dtype == torch.int32 and package['out_observe'].dtype == torch.int32
    if not compatible or len(groups) > 16 or len(packages) > 32:
        for package, compact in zip(packages, compacts):
            package['active_trainable_mask'] = compact.trainable_mask
            add_compact_stats(model, package, compact.global_indices, batch_size)
        return
    from pipeline.loss_batch import _get_c_module
    native = _get_c_module()
    if not hasattr(native, 'compact_stats'):
        raise RuntimeError('Rebuild pgsr-pipeline-ops for fused compact statistics')
    indices=[];local=[];counts=[];visible=[];radii=[];observed=[];signed=[];absolute=[];has_grad=[]
    for compact, views in groups.values():
        indices.append(compact.optimizer_indices)
        local.append(compact.local_trainable)
        counts.append(len(views))
        for package in views:
            sg=package['viewspace_points'].grad;ag=package['viewspace_points_abs'].grad
            valid=sg is not None and ag is not None

            signed.append(sg.contiguous() if valid else package['viewspace_points'])
            absolute.append(ag.contiguous() if valid else package['viewspace_points_abs'])
            has_grad.append(valid)
            visible.append(package['visibility_filter'].contiguous())
            radii.append(package['radii'].contiguous())
            observed.append(package['out_observe'].contiguous())
    native.compact_stats([model.max_radii2D,model.xyz_gradient_accum,model.xyz_gradient_accum_abs,
                         model.denom,model.denom_abs],indices,local,counts,visible,radii,observed,
                         signed,absolute,has_grad,batch_size)
