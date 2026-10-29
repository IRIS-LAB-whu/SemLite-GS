#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace {

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == torch::kFloat32, #x " must be float32")

constexpr int kVoxelLanes = 32;
constexpr int kCameraLanes = 8;
constexpr int kThreads = kVoxelLanes * kCameraLanes;

__device__ __forceinline__ float project_component(
    const float x,
    const float y,
    const float z,
    const float* __restrict__ transform,
    const int column) {
  // This explicit order is bit-identical to the full-FP32 K=4 cuBLAS dot
  // selected by torch.einsum("nf,bfg->bng") on the production GPU. Plain C++
  // arithmetic was contracted/reassociated differently by nvcc.
  float value = __fmul_rn(x, transform[column]);
  value = __fmaf_rn(y, transform[4 + column], value);
  value = __fmaf_rn(z, transform[8 + column], value);
  return __fadd_rn(value, transform[12 + column]);
}

// Match grid_sample(..., mode="bilinear", padding_mode="border",
// align_corners=True) for coordinates already known to lie strictly inside
// (-1, 1). Keeping this local avoids materialising a [C,N,2] grid per chunk.
__device__ __forceinline__ float sample_depth_align_corners(
    const float* __restrict__ image,
    const int width,
    const int height,
    const float ndc_x,
    const float ndc_y) {
  float ix = ((ndc_x + 1.0f) / 2.0f) * static_cast<float>(width - 1);
  float iy = ((ndc_y + 1.0f) / 2.0f) * static_cast<float>(height - 1);
  ix = fminf(static_cast<float>(width - 1), fmaxf(ix, 0.0f));
  iy = fminf(static_cast<float>(height - 1), fmaxf(iy, 0.0f));

  const int ix_nw = static_cast<int>(floorf(ix));
  const int iy_nw = static_cast<int>(floorf(iy));
  const int ix_ne = ix_nw + 1;
  const int iy_sw = iy_nw + 1;
  const float northwest_weight =
      (static_cast<float>(ix_ne) - ix) * (static_cast<float>(iy_sw) - iy);
  const float northeast_weight =
      (ix - static_cast<float>(ix_nw)) * (static_cast<float>(iy_sw) - iy);
  const float southwest_weight =
      (static_cast<float>(ix_ne) - ix) * (iy - static_cast<float>(iy_nw));
  const float southeast_weight =
      (ix - static_cast<float>(ix_nw)) * (iy - static_cast<float>(iy_nw));

  float sampled = image[iy_nw * width + ix_nw] * northwest_weight;
  if (ix_ne < width)
    sampled += image[iy_nw * width + ix_ne] * northeast_weight;
  if (iy_sw < height)
    sampled += image[iy_sw * width + ix_nw] * southwest_weight;
  if (ix_ne < width && iy_sw < height)
    sampled += image[iy_sw * width + ix_ne] * southeast_weight;
  return sampled;
}

__global__ void fused_packed_tsdf_fusion_kernel(
    const float* __restrict__ transforms,
    const float* __restrict__ depths,
    const float* __restrict__ origin,
    float* __restrict__ output_tsdf,
    float* __restrict__ output_weights,
    const int64_t flat_start,
    const int64_t point_count,
    const int resolution,
    const int camera_count,
    const int height,
    const int width,
    const int camera_batch_size,
    const float voxel_size,
    const float sdf_trunc,
    const float depth_trunc) {
  const int voxel_lane = threadIdx.x & (kVoxelLanes - 1);
  const int camera_lane = threadIdx.x / kVoxelLanes;
  const int64_t local_index =
      static_cast<int64_t>(blockIdx.x) * kVoxelLanes + voxel_lane;

  float point_x = 0.0f;
  float point_y = 0.0f;
  float point_z = 0.0f;
  if (local_index < point_count) {
    const int64_t flat = flat_start + local_index;
    const int64_t resolution_squared =
        static_cast<int64_t>(resolution) * resolution;
    const int64_t ix = flat / resolution_squared;
    const int64_t remainder = flat - ix * resolution_squared;
    const int64_t iy = remainder / resolution;
    const int64_t iz = remainder - iy * resolution;
    // Legacy materialises voxel_t * indices and the following origin add as
    // two eager kernels. Prevent contraction here so threshold decisions see
    // the same rounded world-space points.
    point_x = __fadd_rn(
        origin[0], __fmul_rn(voxel_size, static_cast<float>(ix)));
    point_y = __fadd_rn(
        origin[1], __fmul_rn(voxel_size, static_cast<float>(iy)));
    point_z = __fadd_rn(
        origin[2], __fmul_rn(voxel_size, static_cast<float>(iz)));
  }

  __shared__ float partial_sums[kCameraLanes * kVoxelLanes];
  __shared__ int partial_counts[kCameraLanes * kVoxelLanes];
  float output_sum = 0.0f;
  int output_count = 0;
  const int64_t image_pixels = static_cast<int64_t>(height) * width;
  for (int batch_start = 0; batch_start < camera_count;
       batch_start += camera_batch_size) {
    const int camera = batch_start + camera_lane;
    float value = 0.0f;
    int valid_sample = 0;
    if (local_index < point_count && camera < camera_count &&
        camera_lane < camera_batch_size) {
      const float* transform = transforms + 16 * camera;
      const float projected_z = project_component(
          point_x, point_y, point_z, transform, 3);
      const float denominator = fmaxf(projected_z, 1e-12f);
      const float ndc_x = project_component(
          point_x, point_y, point_z, transform, 0) / denominator;
      const float ndc_y = project_component(
          point_x, point_y, point_z, transform, 1) / denominator;
      if (!(ndc_x > -1.0f && ndc_x < 1.0f &&
            ndc_y > -1.0f && ndc_y < 1.0f && projected_z > 0.0f &&
            isfinite(ndc_x) && isfinite(ndc_y)))
        valid_sample = 0;
      else {
        const float sampled_depth = sample_depth_align_corners(
            depths + static_cast<int64_t>(camera) * image_pixels,
            width,
            height,
            ndc_x,
            ndc_y);
        const float sdf = sampled_depth - projected_z;
        if (sampled_depth > 0.0f && sampled_depth <= depth_trunc &&
            sdf > -sdf_trunc) {
          value = fminf(1.0f, fmaxf(-1.0f, sdf / sdf_trunc));
          valid_sample = 1;
        }
      }
    }
    partial_sums[threadIdx.x] = value;
    partial_counts[threadIdx.x] = valid_sample;
    __syncthreads();
    if (camera_lane == 0 && local_index < point_count) {
      const int batch_count = min(camera_batch_size, camera_count - batch_start);
      float batch_sum = 0.0f;
      int batch_weight = 0;
#pragma unroll
      for (int lane = 0; lane < kCameraLanes; ++lane) {
        if (lane < batch_count) {
          const int offset = lane * kVoxelLanes + voxel_lane;
          batch_sum += partial_sums[offset];
          batch_weight += partial_counts[offset];
        }
      }
      output_sum += batch_sum;
      output_count += batch_weight;
    }
    __syncthreads();
  }

  if (camera_lane == 0 && local_index < point_count) {
    output_weights[local_index] = static_cast<float>(output_count);
    output_tsdf[local_index] = output_count > 0
        ? output_sum / static_cast<float>(output_count)
        : 1.0f;
  }
}

}  // namespace

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
    const float depth_trunc) {
  CHECK_CUDA(transforms);
  CHECK_CUDA(depths);
  CHECK_CUDA(origin);
  CHECK_CONTIGUOUS(transforms);
  CHECK_CONTIGUOUS(depths);
  CHECK_CONTIGUOUS(origin);
  CHECK_FLOAT(transforms);
  CHECK_FLOAT(depths);
  CHECK_FLOAT(origin);
  TORCH_CHECK(transforms.dim() == 3 && transforms.size(1) == 4 &&
                  transforms.size(2) == 4,
              "transforms must have shape [C,4,4]");
  TORCH_CHECK(depths.dim() == 3, "depths must have shape [C,H,W]");
  TORCH_CHECK(transforms.size(0) == depths.size(0),
              "transforms and depths must have equal camera counts");
  TORCH_CHECK(origin.numel() == 3, "origin must contain three values");
  TORCH_CHECK(transforms.get_device() == depths.get_device() &&
                  transforms.get_device() == origin.get_device(),
              "all inputs must be on one CUDA device");
  TORCH_CHECK(depths.size(0) > 1, "fused packed TSDF fusion requires multiple cameras");
  TORCH_CHECK(depths.size(1) > 1 && depths.size(2) > 1,
              "depth maps must be larger than 1x1");
  TORCH_CHECK(resolution >= 2, "resolution must be at least 2");
  TORCH_CHECK(camera_batch_size >= 1 && camera_batch_size <= kCameraLanes,
              "camera_batch_size must be between 1 and 8");
  TORCH_CHECK(flat_start >= 0 && point_count >= 0,
              "flat_start and point_count must be non-negative");
  const int64_t total = static_cast<int64_t>(resolution) * resolution * resolution;
  TORCH_CHECK(flat_start <= total && point_count <= total - flat_start,
              "requested TSDF chunk lies outside the volume");
  TORCH_CHECK(voxel_size > 0.0f && sdf_trunc > 0.0f && depth_trunc > 0.0f,
              "voxel_size, sdf_trunc, and depth_trunc must be positive");

  c10::cuda::CUDAGuard device_guard(depths.device());
  auto output_tsdf = torch::empty({point_count}, depths.options());
  auto output_weights = torch::empty({point_count}, depths.options());
  if (point_count == 0)
    return {output_tsdf, output_weights};

  const int64_t block_count = (point_count + kVoxelLanes - 1) / kVoxelLanes;
  TORCH_CHECK(block_count <= static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
              "TSDF chunk is too large for one CUDA launch");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_packed_tsdf_fusion_kernel<<<
      static_cast<unsigned int>(block_count), kThreads, 0, stream>>>(
      transforms.data_ptr<float>(),
      depths.data_ptr<float>(),
      origin.data_ptr<float>(),
      output_tsdf.data_ptr<float>(),
      output_weights.data_ptr<float>(),
      flat_start,
      point_count,
      resolution,
      depths.size(0),
      depths.size(1),
      depths.size(2),
      camera_batch_size,
      voxel_size,
      sdf_trunc,
      depth_trunc);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_tsdf, output_weights};
}
