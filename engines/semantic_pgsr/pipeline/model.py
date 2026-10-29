import torch
from scene.gaussian_model import GaussianModel as NativeGaussianModel
from utils.general_utils import build_rotation


class GaussianModel(NativeGaussianModel):
    def __init__(self, sh_degree):
        super().__init__(sh_degree)
        self.instance_ids = torch.empty(0, dtype=torch.long)
        self.freeze_global = False
        self.max_points_per_instance = 0
        self.context_membership = {}
        self.context_signature = None
        self.context_mode = "joint"
        self.render_order = torch.empty(0, dtype=torch.long)
        self.next_render_order = 0
        self.fast_point_budget = False

    def initialize_metadata(self, ids=None, freeze_global=False, budget=0):
        self.context_membership = {}
        self.context_signature = None
        self.instance_ids = (torch.full((len(self._xyz),), -1, device=self._xyz.device, dtype=torch.long)
                             if ids is None else ids.to(self._xyz.device, dtype=torch.long))
        if len(self.instance_ids) != len(self._xyz):
            raise ValueError('Instance metadata must match Gaussian count')
        self.context_mode = "joint"
        self.render_order = torch.arange(len(self._xyz), device=self._xyz.device)
        self.next_render_order = len(self._xyz)
        self.freeze_global = freeze_global
        self.max_points_per_instance = budget
        self.plane_normals = self._xyz.new_zeros((len(self._xyz),3))
        self.plane_offsets = self._xyz.new_zeros(len(self._xyz))
        self.plane_assigned = torch.zeros(len(self._xyz), dtype=torch.bool, device=self._xyz.device)

    @property
    def trainable_mask(self):
        if not self.freeze_global:
            return torch.ones(len(self._xyz), dtype=torch.bool, device=self._xyz.device)
        return self.instance_ids >= 0

    def capture(self):
        return {'native': super().capture(), 'instance_ids': self.instance_ids,
                'freeze_global': self.freeze_global, 'budget': self.max_points_per_instance,
                'context_membership': self.context_membership, 'context_signature': self.context_signature,
                'context_mode': self.context_mode, 'render_order': self.render_order,
                'next_render_order': self.next_render_order,
                'planes': {k: getattr(self,k) for k in ('plane_normals','plane_offsets','plane_assigned')}}

    def restore(self, state, training_args):
        if isinstance(state, dict):
            super().restore(state['native'], training_args)
            self.initialize_metadata(state['instance_ids'], state['freeze_global'], state['budget'])
            self.context_mode = state.get('context_mode', 'joint')
            self.render_order = state.get('render_order', self.render_order).to(self._xyz.device)
            self.next_render_order = state.get('next_render_order', len(self._xyz))
            self.context_signature = state.get('context_signature')
            self.context_membership = {int(k): v.to(self._xyz.device) for k,v in state.get('context_membership', {}).items()}
            for key,value in state.get('planes', {}).items():
                setattr(self,key,value.to(self._xyz.device))
        else:
            super().restore(state, training_args)
            self.initialize_metadata()

    def save_ply(self, path, mask=None):
        super().save_ply(path)
        selected = slice(None) if mask is None else mask.to(self._xyz.device).bool()
        if mask is not None:

            from plyfile import PlyData, PlyElement
            cloud = PlyData.read(path)
            rows = cloud['vertex'].data[selected.cpu().numpy()]
            PlyData([PlyElement.describe(rows,'vertex')]).write(path)
        torch.save({'instance_ids': self.instance_ids[selected].cpu(), 'freeze_global': self.freeze_global,
                    'frozen_mask': (~self.trainable_mask[selected]).cpu(),
                    'budget': self.max_points_per_instance,
                    'context_membership': {k:v[selected].cpu() for k,v in self.context_membership.items()},
                    'context_signature': self.context_signature,
                    'context_mode': self.context_mode, 'render_order': self.render_order[selected].cpu(),
                    'next_render_order': self.next_render_order,
                    'planes': {k: getattr(self,k)[selected].cpu() for k in ('plane_normals','plane_offsets','plane_assigned')}},
                   path + '.instances.pt')

    def load_ply(self, path):
        import os


        with open(path, 'rb') as stream:
            in_vertices = False
            scale_fields = []
            for raw_line in stream:
                line = raw_line.decode('ascii').strip()
                if line.startswith('element '):
                    in_vertices = line.startswith('element vertex ')
                elif in_vertices and line.startswith('property '):
                    name = line.rsplit(' ', 1)[-1]
                    if name.startswith('scale_'):
                        scale_fields.append(name)
                elif line == 'end_header':
                    break
        if sorted(scale_fields) != ['scale_0', 'scale_1', 'scale_2']:
            raise ValueError(
                f'PGSR requires three scale fields (scale_0..2); {path} has {scale_fields}. '
                'Use the original 3D FastGS point_cloud.ply, not point_cloud_2dgs.ply '
                'or visible_mask_fastgs.ply.')
        super().load_ply(path)
        self._knn_f = torch.nn.Parameter(torch.randn(len(self._xyz), 6, device=self._xyz.device))
        self.max_radii2D = self._xyz.new_zeros(len(self._xyz))
        self.max_weight = self._xyz.new_zeros(len(self._xyz))
        if os.path.exists(path + '.instances.pt'):
            data = torch.load(path + '.instances.pt', map_location=self._xyz.device, weights_only=True)
            self.initialize_metadata(data['instance_ids'], data['freeze_global'], data['budget'])
            self.context_mode = data.get('context_mode', 'joint')
            self.render_order = data.get('render_order', self.render_order).to(self._xyz.device)
            self.next_render_order = data.get('next_render_order', len(self._xyz))
            self.context_signature = data.get('context_signature')
            self.context_membership = {int(k): v.to(self._xyz.device) for k,v in data.get('context_membership', {}).items()}
            for key,value in data.get('planes', {}).items():
                setattr(self,key,value.to(self._xyz.device))
        else:
            self.initialize_metadata()

    @torch.no_grad()
    def make_context_independent(self):


        if self.context_mode != 'joint' or not self.freeze_global or not self.context_membership:
            raise ValueError('Independent context requires a fresh frozen model and validated memberships')
        if self.optimizer is None or any(self.optimizer.state.values()):
            raise ValueError('Create immutable context before the first optimizer update')
        ids = self.instance_ids
        snapshot = torch.zeros_like(ids, dtype=torch.bool)
        for owner, membership in self.context_membership.items():
            snapshot |= membership & (ids >= 0) & (ids != owner)
        indices = snapshot.nonzero().flatten()
        original_count = len(ids)
        membership = {owner: torch.cat((mask & (ids < 0), mask[indices] & (ids[indices] != owner)))
                      for owner, mask in self.context_membership.items()}
        if len(indices):
            super().densification_postfix(self._xyz[indices].detach().clone(),
                self._knn_f[indices].detach().clone(), self._features_dc[indices].detach().clone(),
                self._features_rest[indices].detach().clone(), self._opacity[indices].detach().clone(),
                self._scaling[indices].detach().clone(), self._rotation[indices].detach().clone())
            self.instance_ids = torch.cat((ids, torch.full_like(indices, -1)))
            self.render_order = torch.cat((self.render_order, self.render_order[indices]))
            for key in ('plane_normals', 'plane_offsets', 'plane_assigned'):
                value = getattr(self, key)
                setattr(self, key, torch.cat((value, torch.zeros_like(value[indices]))))
        self.context_membership = membership
        self.context_mode = 'independent'
        return {'source_points': original_count, 'shared_snapshot_points': len(indices),
                'population_points': len(self._xyz)}

    @torch.no_grad()
    def optimizer_step(self):

        frozen = ~self.trainable_mask
        for group in self.optimizer.param_groups:
            p = group['params'][0]
            if p.grad is not None:
                p.grad[frozen] = 0
            state = self.optimizer.state.get(p, {})
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in state:
                    state[key][frozen] = 0
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def prune_points(self, mask):
        self._prune_population(mask & self.trainable_mask)

    def prune_initial_points(self, mask):


        if self.optimizer is None or any(self.optimizer.state.values()):
            raise RuntimeError('initial point pruning requires setup before any optimizer step')
        self._prune_population(mask)

    def _prune_population(self, mask):
        ids = self.instance_ids[~mask]
        self.render_order = self.render_order[~mask]
        self.context_membership = {k:v[~mask] for k,v in self.context_membership.items()}
        planes = {k: getattr(self,k)[~mask] for k in ('plane_normals','plane_offsets','plane_assigned')}
        super().prune_points(mask)
        self.instance_ids = ids
        for key,value in planes.items():
            setattr(self,key,value)

    def reset_opacity(self):
        frozen = ~self.trainable_mask
        before = self._opacity.detach()[frozen].clone()
        super().reset_opacity()
        with torch.no_grad():
            self._opacity[frozen] = before

    @torch.no_grad()
    def enforce_initial_budgets(self):
        if self.max_points_per_instance <= 0:
            return torch.ones(len(self._xyz),dtype=torch.bool,device=self._xyz.device)
        remove = torch.zeros(len(self._xyz),dtype=torch.bool,device=self._xyz.device)
        for owner in self.instance_ids.unique():
            if owner < 0:
                continue
            indices = (self.instance_ids == owner).nonzero().flatten()
            if self.fast_point_budget and len(indices) <= self.max_points_per_instance:
                continue
            order = self.get_opacity[indices,0].argsort(descending=True,stable=True)
            remove[indices[order[self.max_points_per_instance:]]] = True
        if remove.any():
            self.prune_points(remove)
        return ~remove

    def _select(self, scores, candidates, cost=1, limit=None, reserved=None):

        candidates = candidates & self.trainable_mask
        if self.context_mode == 'independent':
            chosen = torch.zeros_like(candidates)


            counts = None
            if self.fast_point_budget and self.context_membership:
                owner_masks = [self.instance_ids == owner for owner in self.context_membership]
                counts = torch.stack([torch.stack((target.sum(), self.context_membership[owner].sum(),
                    target.new_zeros((), dtype=torch.long) if reserved is None else (reserved & target).sum()))
                    for owner, target in zip(self.context_membership, owner_masks)]).cpu().tolist()
            for slot, owner in enumerate(self.context_membership):
                target = owner_masks[slot] if counts is not None else self.instance_ids == owner
                indices = (candidates & target).nonzero().flatten()
                indices = indices[scores[indices].argsort(descending=True, stable=True)]
                if counts is None:
                    population = int(target.sum())
                    context = int(self.context_membership[owner].sum())
                    reserved_count = 0 if reserved is None else int((reserved & target).sum())
                else:
                    population, context, reserved_count = counts[slot]
                capacity = max(0, (self.max_all_points-population-context)//cost-reserved_count)
                if limit is not None:
                    capacity = min(capacity, int(limit))
                if self.max_points_per_instance > 0:
                    capacity = min(capacity, max(0, (self.max_points_per_instance-population)//cost-reserved_count))
                chosen[indices[:capacity]] = True
            return chosen
        capacity = max(0, (self.max_all_points - len(self._xyz)) // cost)
        if reserved is not None:
            capacity = max(0,capacity-int(reserved.sum()))
        if limit is not None:
            capacity = min(capacity, int(limit))
        chosen = torch.zeros_like(candidates)
        indices = candidates.nonzero().flatten()
        indices = indices[torch.argsort(scores[indices], descending=True)]
        for owner in self.instance_ids[indices].unique():
            group = indices[self.instance_ids[indices] == owner]
            if self.max_points_per_instance > 0 and owner >= 0:
                current = int((self.instance_ids == owner).sum())
                if reserved is not None:
                    current += int((reserved & (self.instance_ids == owner)).sum())*cost
                remaining = max(0, (self.max_points_per_instance - current) // cost)
                group = group[:remaining]
            chosen[group] = True
        indices = indices[chosen[indices]][:capacity]
        chosen.zero_()
        chosen[indices] = True
        return chosen

    def _append_children(self, selected, copies, shrink):
        parents = selected.nonzero().flatten().repeat(copies)
        self.render_order = torch.cat((self.render_order, torch.arange(
            self.next_render_order, self.next_render_order + len(parents), device=self._xyz.device)))
        self.next_render_order += len(parents)
        self.context_membership = {k:torch.cat((v,v[parents])) for k,v in self.context_membership.items()}
        ids = torch.cat((self.instance_ids, self.instance_ids[parents]))
        planes = {k: torch.cat((getattr(self,k),getattr(self,k)[parents])) for k in ('plane_normals','plane_offsets','plane_assigned')}
        scales = self.get_scaling[parents]
        samples = torch.normal(torch.zeros_like(scales), scales)
        rotations = build_rotation(self._rotation[parents])
        xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[parents]
        super().densification_postfix(xyz, self._knn_f[parents], self._features_dc[parents],
            self._features_rest[parents], self._opacity[parents],
            torch.log(scales / shrink), self._rotation[parents])
        self.instance_ids = ids
        for key,value in planes.items():
            setattr(self,key,value)

    def densify_and_clone(self, grads, threshold, extent):
        score = grads.norm(dim=-1)
        selected = self._select(score, (score >= threshold) &
            (self.get_scaling.max(dim=1).values <= self.percent_dense * extent))
        if selected.any():
            self._append_children(selected, 1, 1.0)

    def densify_and_split(self, grads, threshold, grads_abs, abs_threshold, extent, radii, N=2):
        count = len(self._xyz)
        def pad(x):
            out = self._xyz.new_zeros(count)
            out[:len(x)] = x.reshape(-1)
            return out
        score, absolute, radius = pad(grads), pad(grads_abs), pad(radii)
        large = self.get_scaling.max(dim=1).values > self.percent_dense * extent
        signed = self._select(score, (score >= threshold) & large, N - 1)

        absolute_candidates = (absolute >= abs_threshold) & large & ~signed & (radius > self.abs_split_radii2D_threshold)
        selected = signed | self._select(absolute,absolute_candidates,N-1,
                                        self.max_abs_split_points,reserved=signed)
        if selected.any():
            self._append_children(selected, N, 0.8*N)
            self.prune_points(torch.cat((selected, torch.zeros(N*int(selected.sum()), device=selected.device, dtype=torch.bool))))

    def add_densification_stats(self, signed, absolute, update_filter):
        super().add_densification_stats(signed, absolute, update_filter & self.trainable_mask)
