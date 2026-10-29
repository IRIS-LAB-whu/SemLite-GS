from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ROIMask:
    tensor: torch.Tensor
    roi: tuple


@dataclass(frozen=True)
class Supervision:
    roi: tuple
    raw_roi: tuple
    rgb: torch.Tensor
    mask: ROIMask
    weight: torch.Tensor
    mask_roi: tuple = None


@dataclass(frozen=True)
class CacheStatus:
    device: str
    fallback: bool
    bytes: int


def supervision_for(camera, instance):
    return getattr(camera, '_supervision', {}).get(instance)


def supervision_gray(camera):
    gray = getattr(camera, '_supervision_gray', None)
    return camera.get_image()[1] if gray is None else gray


def _transfer(tensor, device):
    return tensor.to(device)


def _upload(payloads, device):
    result = []
    for entries, gray in payloads:
        promoted = {}
        for instance, entry in entries.items():
            promoted[instance] = Supervision(entry.roi, entry.raw_roi,
                _transfer(entry.rgb, device),
                ROIMask(_transfer(entry.mask.tensor, device), entry.roi),
                _transfer(entry.weight, device), entry.mask_roi)
        result.append((promoted, _transfer(gray, device)))
    return result


def _bytes(payloads, guards=()):
    targets = sum(t.numel()*t.element_size()
                  for entries, gray in payloads
                  for t in [gray]+[t for e in entries.values()
                                    for t in (e.rgb, e.mask.tensor, e.weight)])
    return targets + sum(t.numel()*t.element_size()
                         for guard in guards if guard is not None
                         for t in (guard.mask, guard.rgb))


def fallback_supervision(cameras, log=print):

    cameras = list(cameras)
    payloads, guards = [], []
    for camera in cameras:
        camera._loss_targets = {}
        camera._pair_constants = None
        camera._pair_roi_constants = {}
        payload = getattr(camera, '_supervision_cpu', None)
        if payload is not None:
            camera._supervision, camera._supervision_gray = payload
            payloads.append(payload)
        if hasattr(camera, '_frozen_guard_cpu'):
            camera.frozen_guard = camera._frozen_guard_cpu
            guards.append(camera.frozen_guard)
        if hasattr(camera, '_instance_guards_cpu'):
            camera._instance_guards = camera._instance_guards_cpu
            guards.extend(camera._instance_guards.values())
    torch.cuda.empty_cache()
    status = CacheStatus('cpu', True, _bytes(payloads, guards))
    if log:
        log(f'Supervision cache: CPU fallback ({status.bytes/2**20:.1f} MiB); GPU targets released')
    return status


@torch.no_grad()
def prepare_supervision(cameras, instance_ids, device='cuda', halo=5, guarded=False, log=print):


    from pipeline.losses import crop, _image_weight
    from pipeline.masks import mask_roi
    from pipeline.guard import GuardTarget, union_roi, guard_for
    cameras, instance_ids = list(cameras), list(instance_ids)
    device = torch.device(device)
    guards = [getattr(camera, '_frozen_guard_cpu', getattr(camera, 'frozen_guard', None))
              if guarded else None for camera in cameras]
    payloads = []
    for camera, guard in zip(cameras, guards):
        rgb, gray = camera.get_image()
        rgb, gray = rgb.detach().cpu(), gray.detach().cpu().clone()
        labels = getattr(camera, 'instance_mask', None)
        if labels is not None:
            labels = labels.detach().cpu().reshape(rgb.shape[-2:])
        weight = _image_weight(rgb)
        height, width = rgb.shape[-2:]
        entries = {}
        for instance in instance_ids:
            mask = torch.ones((height,width),dtype=torch.bool) if labels is None else labels == instance
            if not mask.any():
                continue
            roi = mask_roi(mask)
            foreground_roi = roi
            instance_guard = guard_for(camera, instance) if guarded else None
            if instance_guard is not None:
                roi = union_roi(roi, instance_guard.roi)
            x,y,w,h = roi
            rx,ry = max(0,x-halo),max(0,y-halo)
            raw_roi = (rx,ry,min(width,x+w+halo)-rx,min(height,y+h+halo)-ry)
            entries[instance] = Supervision(roi,raw_roi,crop(rgb,raw_roi).clone(),
                ROIMask(crop(mask,roi).clone(),roi),crop(weight,roi).clone(),foreground_roi)
        payloads.append((entries,gray))
    fallback = False
    promoted, promoted_guards = payloads, guards
    owner_guards = [getattr(c, '_instance_guards_cpu', getattr(c, '_instance_guards', {})) for c in cameras]
    promoted_owner_guards = owner_guards
    if device.type != 'cpu':
        try:
            promoted = _upload(payloads, device)
            promoted_owner_guards = [{i: None if g is None else GuardTarget(g.roi, _transfer(g.mask,device), _transfer(g.rgb,device)) for i,g in gs.items()} for gs in owner_guards]
            promoted_guards = [None if guard is None else GuardTarget(
                guard.roi, _transfer(guard.mask, device), _transfer(guard.rgb, device))
                for guard in guards]
        except torch.cuda.OutOfMemoryError:
            if device.type != 'cuda':
                raise
            fallback = True


        if fallback:
            promoted, promoted_guards = payloads, guards
            promoted_owner_guards = owner_guards
            torch.cuda.empty_cache()
    for camera, cpu, active, guard_cpu, guard_active in zip(
            cameras, payloads, promoted, guards, promoted_guards):
        camera._supervision_cpu = cpu
        camera._supervision,camera._supervision_gray = active
        if guarded:
            camera._frozen_guard_cpu = guard_cpu
            camera.frozen_guard = guard_active
    for camera, cpu, active in zip(cameras, owner_guards, promoted_owner_guards):
        camera._instance_guards_cpu, camera._instance_guards = cpu, active
    from pipeline.pair_batch import prepare_pair_constants
    try:
        prepare_pair_constants(cameras,'cpu' if fallback else device)
    except torch.cuda.OutOfMemoryError:
        if device.type != 'cuda':
            raise
        fallback_supervision(cameras,log=None)
        fallback=True
        prepare_pair_constants(cameras,'cpu')
    status = CacheStatus('cpu' if fallback else str(device),fallback,_bytes(payloads, guards + [g for gs in owner_guards for g in gs.values()]))
    if log:
        suffix = ' (CUDA OOM; CPU fallback)' if fallback else ''
        log(f'Supervision cache: {status.device}, {status.bytes/2**20:.1f} MiB{suffix}')
    return status
