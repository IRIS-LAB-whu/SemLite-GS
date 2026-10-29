from pathlib import Path
import sys
import os
import torch
import torch.nn.functional as F
from pipeline.losses import crop, local_mask, _image_weight, _mean
from pipeline.guard import guard_for
from pipeline.supervision import supervision_for
_root = Path(__file__).resolve().parents[1]
for _sub in ('fused-ssim', 'pgsr-pipeline-ops'):
    sys.path.insert(0, str(_root/'submodules'/_sub))
from fused_ssim import packed_masked_photo_loss
from pgsr_pipeline_ops import _get_c_module
_C = _get_c_module()


USE_RAGGED_PHOTO = os.environ.get("PGSR_EXPERIMENTAL_RAGGED", "1") == "1"


class _PackedL1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, weights, denom):
        ctx.save_for_backward(pred, target, weights, denom)
        return _C.packed_l1_forward(pred, target, weights, denom)

    @staticmethod
    def backward(ctx, grad):
        return _C.packed_l1_backward(*ctx.saved_tensors, grad.contiguous()), None, None, None


def _pad(t, h, w):
    return F.pad(t, (0, w-t.shape[-1], 0, h-t.shape[-2]))


def _raw_pad(t, package):
    x,y,w,h = package['roi']; rx,ry,rw,rh = package['raw_roi']
    return F.pad(t, (x-rx, rw-w-x+rx, y-ry, rh-h-y+ry))


def _targets(camera, package, mask):
    entry = None if package.get('_disable_supervision_cache') else supervision_for(camera, package.get('instance_id'))
    guard = guard_for(camera, package.get('instance_id'))
    key = (package['roi'], package['raw_roi'], id(entry), id(guard), id(mask))
    cache = getattr(camera, '_loss_targets', {})
    if key in cache: return cache[key]
    rgb = package['raw']['render']; rx,ry,rw,rh = package['raw_roi']; x,y,w,h = package['roi']
    if entry is None:
        full = camera.get_image()[0]
        gt = crop(full, package['raw_roi']).to(rgb.device)
        weight = crop(_image_weight(full), package['roi']).to(rgb.device)
    else:
        ex,ey,_,_ = entry.raw_roi
        gt = crop(entry.rgb,(rx-ex,ry-ey,rw,rh)).to(rgb.device)
        ex,ey,_,_ = entry.roi
        weight = crop(entry.weight,(x-ex,y-ey,w,h)).to(rgb.device)
    selected = _raw_pad(local_mask(mask,package).float(), package)
    weights = _raw_pad(weight,package)
    guard_mask = torch.zeros_like(selected)
    guard_rgb = torch.zeros_like(gt)
    if guard is not None:
        gx,gy,gw,gh=guard.roi
        if gx<rx or gy<ry or gx+gw>rx+rw or gy+gh>ry+rh:
            raise ValueError('Render ROI excludes frozen guard')
        guard_mask[gy-ry:gy-ry+gh,gx-rx:gx-rx+gw]=guard.mask.to(rgb.device)
        guard_rgb[:,gy-ry:gy-ry+gh,gx-rx:gx-rx+gw]=guard.rgb.to(rgb.device)
    result=(gt.contiguous(),selected,weights,guard_rgb,guard_mask)


    if entry is not None:
        camera._loss_targets={key:result}
    return result


def per_view_losses(cameras, packages, models, opt, iteration, masks, guard_weight=0.):
    count=len(cameras)
    targets=[_targets(c,p,m) for c,p,m in zip(cameras,packages,masks)]
    hs=[p['raw']['render'].shape[-2] for p in packages]
    ws=[p['raw']['render'].shape[-1] for p in packages]
    h,w=max(hs),max(ws)
    predictions=torch.stack([_pad(p['raw']['render'],h,w) for p in packages])
    gt_pool=torch.cat([t[0].reshape(-1) for t in targets])
    mask_pool=torch.cat([t[1].reshape(-1) for t in targets])
    areas=[a*b for a,b in zip(hs,ws)]
    starts=[0]
    for area in areas: starts.append(starts[-1]+area)
    offsets=torch.tensor(starts[:-1],device=predictions.device,dtype=torch.long)
    counts=torch.stack([t[1].sum() for t in targets])
    _,similarity,_=packed_masked_photo_loss(predictions,None,gt_pool,mask_pool,offsets,
                                          counts,range(count),hs,ws)
    has_pixels=(counts>0).to(similarity.dtype)
    ssim_loss=has_pixels-similarity
    use_ragged=USE_RAGGED_PHOTO and count<=8 and predictions.is_cuda and predictions.dtype==torch.float32
    if use_ragged:
        from pipeline.ragged_photo import terms as ragged_terms
        all_terms=ragged_terms(packages,targets,counts,ssim_loss,opt,iteration,guard_weight,predictions)
        normals=iteration>opt.single_view_weight_from_iter
        groups=[all_terms[:count]]
        if guard_weight:groups.append(all_terms[count:2*count])
        if normals:groups.append(all_terms[2*count:])
        terms=torch.cat(groups)
    else:
        prediction_groups=[];target_groups=[];weight_groups=[];denom_groups=[]
        exposed=[]
        for i,p in enumerate(packages):
            raw=p['raw']
            value=torch.where(ssim_loss[i].detach()<.5,raw['app_image'],raw['render']) if 'app_image' in raw else raw['render']
            exposed.append(_pad(value,h,w))
        prediction_groups.append(torch.stack(exposed))
        target_groups.append(torch.stack([_pad(t[0],h,w) for t in targets]))
        weight_groups.append(torch.stack([_pad(t[1],h,w) for t in targets]))
        denom_groups.append(counts*3)
        if guard_weight:
            prediction_groups.append(predictions)
            target_groups.append(torch.stack([_pad(t[3],h,w) for t in targets]))
            weight_groups.append(torch.stack([_pad(t[4],h,w) for t in targets]))
            denom_groups.append(torch.stack([t[4].sum()*3 for t in targets]))
        normals=iteration>opt.single_view_weight_from_iter
        if normals:
            prediction_groups.append(torch.stack([_pad(_raw_pad(p['depth_normal']-p['rendered_normal'],p),h,w) for p in packages]))
            target_groups.append(torch.zeros_like(predictions))
            weight_groups.append(torch.stack([_pad(t[1] if opt.wo_image_weight else t[1]*t[2],h,w) for t in targets]))
            denom_groups.append(counts)
        terms=_PackedL1.apply(torch.cat(prediction_groups).flatten(2).contiguous(),
                             torch.cat(target_groups).flatten(2).contiguous(),
                             torch.cat(weight_groups).flatten(1).contiguous(),torch.cat(denom_groups))
    loss=(1-opt.lambda_dssim)*terms[:count]+opt.lambda_dssim*ssim_loss
    pos=count
    if guard_weight:
        loss=loss+guard_weight*terms[pos:pos+count];pos+=count
    if normals: loss=loss+opt.single_view_weight*terms[pos:pos+count]

    scales=[]
    for p,model in zip(packages,models):
        visible=p['visibility_filter']
        if hasattr(model,'trainable_mask'): visible=visible & model.trainable_mask
        if 'instance_id' in p and hasattr(model,'instance_ids'): visible=visible & (model.instance_ids==p['instance_id'])
        scales.append(_mean(model.get_scaling.min(-1).values,visible))
    return loss+opt.scale_loss_weight*torch.stack(scales)*has_pixels
