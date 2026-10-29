#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>
#include <vector>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
packed_masked_photo_forward(
    float C1,
    float C2,
    const torch::Tensor &pred,
    const torch::Tensor &dist,
    const torch::Tensor &gt_pool,
    const torch::Tensor &mask_pool,
    const torch::Tensor &roi_offsets,
    const torch::Tensor &valid_counts,
    const std::vector<int64_t> &pair_ids,
    const std::vector<int64_t> &roi_heights,
    const std::vector<int64_t> &roi_widths,
    bool train
);

std::tuple<torch::Tensor, torch::Tensor>
packed_masked_photo_backward(
    float C1,
    float C2,
    const torch::Tensor &pred,
    const torch::Tensor &dist,
    const torch::Tensor &gt_pool,
    const torch::Tensor &mask_pool,
    const torch::Tensor &roi_offsets,
    const torch::Tensor &valid_counts,
    const torch::Tensor &dm_dmu1,
    const torch::Tensor &dm_dsigma1_sq,
    const torch::Tensor &dm_dsigma12,
    const torch::Tensor &grad_l1,
    const torch::Tensor &grad_ssim,
    const torch::Tensor &grad_dist,
    const std::vector<int64_t> &pair_ids,
    const std::vector<int64_t> &roi_heights,
    const std::vector<int64_t> &roi_widths
);
