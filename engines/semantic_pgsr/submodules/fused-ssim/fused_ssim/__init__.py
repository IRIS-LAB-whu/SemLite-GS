from typing import NamedTuple
import torch.nn as nn
import torch
from fused_ssim_cuda import fusedssim, fusedssim_backward

try:
    from fused_ssim_cuda import (
        packed_masked_photo_forward,
        packed_masked_photo_backward,
    )
except ImportError:
    # Keep the legacy fused-SSIM API usable until the local extension has been
    # rebuilt.  The packed entry point below reports the actionable error.
    packed_masked_photo_forward = None
    packed_masked_photo_backward = None

PACKED_MASKED_PHOTO_LOSS_AVAILABLE = (
    packed_masked_photo_forward is not None
    and packed_masked_photo_backward is not None
)

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()


class PackedMaskedPhotoLoss(torch.autograd.Function):
    """Fused masked L1/SSIM/dist losses over a ragged packed ROI catalogue."""

    @staticmethod
    def forward(
        ctx,
        pred,
        dist,
        gt_pool,
        mask_pool,
        roi_offsets,
        valid_counts,
        pair_ids,
        roi_heights,
        roi_widths,
        train=True,
    ):
        if dist is None:
            dist_tensor = pred.new_empty((0,))
            has_dist = False
        else:
            dist_tensor = dist
            has_dist = True

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pair_ids = tuple(int(value) for value in pair_ids)
        roi_heights = tuple(int(value) for value in roi_heights)
        roi_widths = tuple(int(value) for value in roi_widths)
        outputs = packed_masked_photo_forward(
            C1,
            C2,
            pred,
            dist_tensor,
            gt_pool,
            mask_pool,
            roi_offsets,
            valid_counts,
            pair_ids,
            roi_heights,
            roi_widths,
            bool(train),
        )
        l1, ssim, dist_loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = outputs

        ctx.save_for_backward(
            pred.detach(),
            dist_tensor.detach(),
            gt_pool,
            mask_pool,
            roi_offsets,
            valid_counts,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        )
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.pair_ids = pair_ids
        ctx.roi_heights = roi_heights
        ctx.roi_widths = roi_widths
        ctx.has_dist = has_dist
        ctx.train = bool(train)
        return l1, ssim, dist_loss

    @staticmethod
    def backward(ctx, grad_l1, grad_ssim, grad_dist_loss):
        if not ctx.train:
            raise RuntimeError(
                "packed_masked_photo_loss(train=False) does not retain backward buffers"
            )

        (
            pred,
            dist,
            gt_pool,
            mask_pool,
            roi_offsets,
            valid_counts,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        ) = ctx.saved_tensors
        batch = pred.shape[0]
        if grad_l1 is None:
            grad_l1 = pred.new_zeros((batch,))
        if grad_ssim is None:
            grad_ssim = pred.new_zeros((batch,))
        if grad_dist_loss is None:
            grad_dist_loss = pred.new_zeros((batch,))

        grad_pred, grad_dist = packed_masked_photo_backward(
            ctx.C1,
            ctx.C2,
            pred,
            dist,
            gt_pool,
            mask_pool,
            roi_offsets,
            valid_counts,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
            grad_l1.contiguous(),
            grad_ssim.contiguous(),
            grad_dist_loss.contiguous(),
            ctx.pair_ids,
            ctx.roi_heights,
            ctx.roi_widths,
        )
        if not ctx.has_dist:
            grad_dist = None
        return (
            grad_pred,
            grad_dist,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def packed_masked_photo_loss(
    pred,
    dist,
    gt_pool,
    mask_pool,
    roi_offsets,
    valid_counts,
    pair_ids,
    roi_heights,
    roi_widths,
    train=True,
):
    """Return per-camera masked ``(L1, SSIM, distortion)`` losses.

    ``gt_pool`` stores each catalogue ROI as contiguous CHW data at
    ``3 * roi_offsets[pair]``; ``mask_pool`` uses pixel offsets directly.
    The CPU sequences describe the selected batch and are passed to CUDA as a
    fixed-size launch descriptor, so this call creates no GPU work tensor.
    """

    if packed_masked_photo_forward is None or packed_masked_photo_backward is None:
        raise RuntimeError(
            "packed_masked_photo_loss requires rebuilding submodules/fused-ssim"
        )

    return PackedMaskedPhotoLoss.apply(
        pred,
        dist,
        gt_pool,
        mask_pool,
        roi_offsets,
        valid_counts,
        pair_ids,
        roi_heights,
        roi_widths,
        train,
    )
