import numpy as np
import torch
from scipy.spatial import cKDTree
from semlite_gs.vendor.rotation_conversions import quaternion_to_matrix, matrix_to_quaternion


def assign_plane_constraints(model, prior, args):
    xyz = model.get_xyz.detach().cpu().numpy()
    owners = model.instance_ids.cpu().numpy()
    best = np.full(len(xyz), np.inf)
    assignments = np.full(len(xyz), -1, dtype=np.int64)
    multiplier = float(getattr(args, 'plane_hard_support_multiplier', 3.0))
    for index, (normal, offset, owner, spacing, support) in enumerate(zip(
            prior['plane_normals'],prior['plane_offsets'],prior['plane_instance_ids'],
            prior['plane_spacings'],prior['plane_support_points'])):
        selected = np.flatnonzero(owners == owner)
        if not len(selected) or not len(support):
            continue
        distance,_ = cKDTree(support).query(xyz[selected],k=1)
        residual = np.abs(xyz[selected] @ normal + offset)
        eligible = (distance <= spacing*multiplier) & (residual <= spacing*multiplier) & (distance < best[selected])
        ids = selected[eligible]
        best[ids] = distance[eligible]
        assignments[ids] = index
    mask = torch.as_tensor(assignments >= 0,device=model.get_xyz.device) & model.trainable_mask
    model.plane_assigned = mask
    if mask.any():
        indices = assignments[mask.cpu().numpy()]
        model.plane_normals[mask] = torch.as_tensor(np.asarray(prior['plane_normals'])[indices],device=model.get_xyz.device)
        model.plane_offsets[mask] = torch.as_tensor(np.asarray(prior['plane_offsets'])[indices],device=model.get_xyz.device)
    print(f'Hard plane assignments: {int(mask.sum())}/{len(mask)}')


@torch.no_grad()
def enforce_plane_constraints(model):
    assigned = model.plane_assigned & model.trainable_mask
    if not assigned.any():
        return
    normals = torch.nn.functional.normalize(model.plane_normals[assigned],dim=-1)
    xyz = model._xyz[assigned]
    model._xyz[assigned] = xyz - ((xyz*normals).sum(-1)+model.plane_offsets[assigned])[:,None]*normals
    matrices = quaternion_to_matrix(model.get_rotation[assigned])
    smallest = model.get_scaling[assigned].argmin(dim=-1)

    rows = torch.arange(len(matrices),device=matrices.device)
    next_axis = (smallest+1)%3
    tangent = matrices[rows,:,next_axis]
    tangent -= (tangent*normals).sum(-1)[:,None]*normals
    degenerate = tangent.norm(dim=-1)<1e-6
    fallback = torch.nn.functional.one_hot(normals.abs().argmin(-1),3).to(normals)
    fallback -= (fallback*normals).sum(-1)[:,None]*normals
    tangent = torch.nn.functional.normalize(torch.where(degenerate[:,None],fallback,tangent),dim=-1)
    matrices[rows,:,smallest] = normals
    matrices[rows,:,next_axis] = tangent
    matrices[rows,:,(smallest+2)%3] = torch.linalg.cross(normals,tangent)
    model._rotation[assigned] = matrix_to_quaternion(matrices)

    for group in model.optimizer.param_groups:
        if group['name'] not in ('xyz','rotation'):
            continue
        state = model.optimizer.state.get(group['params'][0],{})
        for key in ('exp_avg','exp_avg_sq'):
            if key in state:
                state[key][assigned] = 0
