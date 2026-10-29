import torch
from PIL import ImageFilter
from gaussian_renderer import render_fastgs
from .loss_utils import l1_loss
from fused_ssim import fused_ssim as fast_ssim
import torchvision.transforms as transforms
import random


def sampling_cameras(my_viewpoint_stack):


    num_cams = 10
    camlist = []
    for _ in range(num_cams):
        loc = random.randint(0, len(my_viewpoint_stack) - 1)
        camlist.append(my_viewpoint_stack.pop(loc))

    return camlist

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def normalize(config_value, value_tensor):
    multiplier = config_value
    value_tensor[value_tensor.isnan()] = 0

    valid_indices = (value_tensor > 0)
    valid_value = value_tensor[valid_indices].to(torch.float32)

    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

    return ret_value

def compute_gaussian_score_fastgs(camlist, gaussians, pipe, bg, args, DENSIFY = False):


    full_metric_counts = None
    full_metric_score = None

    for view in range(len(camlist)):
        my_viewpoint_cam = camlist[view]
        render_image = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult)["render"]
        photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

        gt_image = my_viewpoint_cam.original_image.cuda()
        get_flag = True
        l1_loss_norm = get_loss(render_image, gt_image)

        metric_map = (l1_loss_norm > args.loss_thresh).int()

        render_pkg = render_fastgs(my_viewpoint_cam, gaussians, pipe, bg, args.mult, get_flag = get_flag, metric_map = metric_map)

        accum_loss_counts = render_pkg["accum_metric_counts"]

        if DENSIFY:
            if full_metric_counts is None:
                full_metric_counts = accum_loss_counts.clone()
            else:
                full_metric_counts += accum_loss_counts

        if full_metric_score is None:
            full_metric_score = photometric_loss * accum_loss_counts.clone()
        else:
            full_metric_score += photometric_loss * accum_loss_counts

    pruning_score = (full_metric_score - torch.min(full_metric_score)) / (torch.max(full_metric_score) - torch.min(full_metric_score))

    if DENSIFY:
        importance_score = torch.div(full_metric_counts, len(camlist), rounding_mode='floor')
    else:
        importance_score = None
    return importance_score, pruning_score
