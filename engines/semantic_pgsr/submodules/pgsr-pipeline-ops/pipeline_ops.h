#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> PackedPlaneDepthCorrectionCUDA(
    const torch::Tensor& depths,
    const torch::Tensor& inverse_views,
    const torch::Tensor& camera_centers,
    const torch::Tensor& tan_fovxy,
    const torch::Tensor& full_projections,
    const torch::Tensor& polygon_vertices,
    const torch::Tensor& polygon_offsets,
    const torch::Tensor& task_cameras,
    const torch::Tensor& task_normals,
    const torch::Tensor& task_plane_offsets,
    const torch::Tensor& task_bboxes,
    const float max_depth_delta,
    const float depth_trunc,
    const float blend,
    const int min_pixels_per_plane);

std::vector<torch::Tensor> FusedPackedTSDFFusionCUDA(
    const torch::Tensor& transforms,
    const torch::Tensor& depths,
    const torch::Tensor& origin,
    const int64_t flat_start,
    const int64_t point_count,
    const int resolution,
    const int camera_batch_size,
    const float voxel_size,
    const float sdf_trunc,
    const float depth_trunc);

std::vector<torch::Tensor> PackedLabelComponentsCPU(
    const torch::Tensor& labels,
    const torch::Tensor& neighbors,
    const int64_t target_mode);
