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
#define CHECK_INT(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

__device__ __forceinline__ bool point_on_segment(
    const float x,
    const float y,
    const float ax,
    const float ay,
    const float bx,
    const float by) {
  const float cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax);
  const float scale = fabsf(bx - ax) + fabsf(by - ay) + 1.0f;
  if (fabsf(cross) > 1e-5f * scale)
    return false;
  return x >= fminf(ax, bx) - 1e-5f && x <= fmaxf(ax, bx) + 1e-5f &&
         y >= fminf(ay, by) - 1e-5f && y <= fmaxf(ay, by) + 1e-5f;
}

__device__ __forceinline__ bool inside_polygon(
    const float x,
    const float y,
    const float* __restrict__ vertices,
    const int begin,
    const int end) {
  bool inside = false;
  int previous = end - 1;
  for (int current = begin; current < end; ++current) {
    const float ax = vertices[2 * previous];
    const float ay = vertices[2 * previous + 1];
    const float bx = vertices[2 * current];
    const float by = vertices[2 * current + 1];
    if (point_on_segment(x, y, ax, ay, bx, by))
      return true;
    const bool crosses = (ay > y) != (by > y);
    if (crosses) {
      const float intersection = ax + (y - ay) * (bx - ax) / (by - ay);
      if (x < intersection)
        inside = !inside;
    }
    previous = current;
  }
  return inside;
}

__global__ void count_polygon_pixels_kernel(
    const float* __restrict__ vertices,
    const int* __restrict__ polygon_offsets,
    const int* __restrict__ bboxes,
    int* __restrict__ counts,
    const int task_count) {
  const int task = blockIdx.x;
  if (task >= task_count)
    return;
  const int x0 = bboxes[4 * task];
  const int y0 = bboxes[4 * task + 1];
  const int x1 = bboxes[4 * task + 2];
  const int y1 = bboxes[4 * task + 3];
  const int box_width = x1 - x0 + 1;
  const int box_height = y1 - y0 + 1;
  const int pixel_count = box_width * box_height;
  const int vertex_begin = polygon_offsets[task];
  const int vertex_end = polygon_offsets[task + 1];
  int local_count = 0;
  for (int linear = threadIdx.x; linear < pixel_count; linear += blockDim.x) {
    const int x = x0 + linear % box_width;
    const int y = y0 + linear / box_width;
    local_count += inside_polygon(
        static_cast<float>(x),
        static_cast<float>(y),
        vertices,
        vertex_begin,
        vertex_end);
  }
  __shared__ int reduction[256];
  reduction[threadIdx.x] = local_count;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    counts[task] = reduction[0];
}

__device__ __forceinline__ bool plane_depth_for_pixel(
    const int camera,
    const int task,
    const int x,
    const int y,
    const int width,
    const int height,
    const float raw_depth,
    const float* __restrict__ inverse_views,
    const float* __restrict__ centers,
    const float* __restrict__ tan_fovxy,
    const float* __restrict__ full_projections,
    const float* __restrict__ normals,
    const float* __restrict__ plane_offsets,
    const float depth_trunc,
    const float max_delta,
    float* plane_depth_out,
    float* delta_out) {
  if (!(raw_depth > 0.0f && raw_depth <= depth_trunc))
    return false;
  const float ndc_x = static_cast<float>(x) * (2.0f / static_cast<float>(width - 1)) - 1.0f;
  const float ndc_y = static_cast<float>(y) * (2.0f / static_cast<float>(height - 1)) - 1.0f;
  const float camera_point[4] = {
      ndc_x * tan_fovxy[2 * camera],
      ndc_y * tan_fovxy[2 * camera + 1],
      1.0f,
      1.0f};
  const float* inverse_view = inverse_views + 16 * camera;
  float world_h[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int column = 0; column < 4; ++column) {
#pragma unroll
    for (int row = 0; row < 4; ++row)
      world_h[column] += camera_point[row] * inverse_view[4 * row + column];
  }
  const float inverse_w = 1.0f / fmaxf(world_h[3], 1e-12f);
  const float* center = centers + 3 * camera;
  const float ray[3] = {
      world_h[0] * inverse_w - center[0],
      world_h[1] * inverse_w - center[1],
      world_h[2] * inverse_w - center[2]};
  const float* normal = normals + 3 * task;
  const float denominator = ray[0] * normal[0] + ray[1] * normal[1] + ray[2] * normal[2];
  if (fabsf(denominator) <= 1e-10f)
    return false;
  const float numerator = -(
      center[0] * normal[0] + center[1] * normal[1] +
      center[2] * normal[2] + plane_offsets[task]);
  const float lambda = numerator / denominator;
  if (!(lambda > 0.0f) || !isfinite(lambda))
    return false;
  const float plane_point[4] = {
      center[0] + lambda * ray[0],
      center[1] + lambda * ray[1],
      center[2] + lambda * ray[2],
      1.0f};
  const float* projection = full_projections + 16 * camera;
  float plane_depth = 0.0f;
#pragma unroll
  for (int row = 0; row < 4; ++row)
    plane_depth += plane_point[row] * projection[4 * row + 3];
  const float delta = fabsf(plane_depth - raw_depth);
  if (!(isfinite(plane_depth) && plane_depth > 0.0f &&
        plane_depth <= depth_trunc && delta <= max_delta))
    return false;
  *plane_depth_out = plane_depth;
  *delta_out = delta;
  return true;
}

__global__ void select_best_plane_kernel(
    const float* __restrict__ depths,
    const float* __restrict__ inverse_views,
    const float* __restrict__ centers,
    const float* __restrict__ tan_fovxy,
    const float* __restrict__ full_projections,
    const float* __restrict__ vertices,
    const int* __restrict__ polygon_offsets,
    const int* __restrict__ task_cameras,
    const float* __restrict__ normals,
    const float* __restrict__ plane_offsets,
    const int* __restrict__ bboxes,
    const int* __restrict__ task_counts,
    unsigned long long* __restrict__ best_keys,
    const int task_count,
    const int width,
    const int height,
    const int min_pixels,
    const float depth_trunc,
    const float max_delta) {
  const int task = blockIdx.x;
  if (task >= task_count || task_counts[task] < min_pixels)
    return;
  const int camera = task_cameras[task];
  const int x0 = bboxes[4 * task];
  const int y0 = bboxes[4 * task + 1];
  const int x1 = bboxes[4 * task + 2];
  const int y1 = bboxes[4 * task + 3];
  const int box_width = x1 - x0 + 1;
  const int pixel_count = box_width * (y1 - y0 + 1);
  const int vertex_begin = polygon_offsets[task];
  const int vertex_end = polygon_offsets[task + 1];
  const int image_pixels = width * height;
  for (int linear = threadIdx.x; linear < pixel_count; linear += blockDim.x) {
    const int x = x0 + linear % box_width;
    const int y = y0 + linear / box_width;
    if (!inside_polygon(
            static_cast<float>(x), static_cast<float>(y), vertices,
            vertex_begin, vertex_end))
      continue;
    const int pixel = camera * image_pixels + y * width + x;
    float plane_depth = 0.0f;
    float delta = 0.0f;
    if (!plane_depth_for_pixel(
            camera, task, x, y, width, height, depths[pixel], inverse_views,
            centers, tan_fovxy, full_projections, normals, plane_offsets,
            depth_trunc, max_delta, &plane_depth, &delta))
      continue;
    const unsigned long long key =
        (static_cast<unsigned long long>(__float_as_uint(delta)) << 32) |
        static_cast<unsigned int>(task);
    atomicMin(best_keys + pixel, key);
  }
}

__global__ void apply_best_plane_kernel(
    const float* __restrict__ depths,
    float* __restrict__ corrected,
    const float* __restrict__ inverse_views,
    const float* __restrict__ centers,
    const float* __restrict__ tan_fovxy,
    const float* __restrict__ full_projections,
    const int* __restrict__ task_cameras,
    const float* __restrict__ normals,
    const float* __restrict__ plane_offsets,
    const unsigned long long* __restrict__ best_keys,
    int64_t* __restrict__ corrected_count,
    const int total_pixels,
    const int width,
    const int height,
    const float depth_trunc,
    const float max_delta,
    const float blend) {
  const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
  if (pixel >= total_pixels)
    return;
  const unsigned long long key = best_keys[pixel];
  if (key == ~0ULL)
    return;
  const int task = static_cast<int>(key & 0xffffffffULL);
  const int image_pixels = width * height;
  const int camera = pixel / image_pixels;
  const int local_pixel = pixel - camera * image_pixels;
  const int y = local_pixel / width;
  const int x = local_pixel - y * width;
  float plane_depth = 0.0f;
  float delta = 0.0f;
  if (!plane_depth_for_pixel(
          camera, task, x, y, width, height, depths[pixel], inverse_views,
          centers, tan_fovxy, full_projections, normals, plane_offsets,
          depth_trunc, max_delta, &plane_depth, &delta))
    return;
  corrected[pixel] = (1.0f - blend) * depths[pixel] + blend * plane_depth;
  atomicAdd(
      reinterpret_cast<unsigned long long*>(corrected_count),
      static_cast<unsigned long long>(1));
}

}  // namespace

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
    const int min_pixels_per_plane) {
  CHECK_CUDA(depths);
  CHECK_CUDA(inverse_views);
  CHECK_CUDA(camera_centers);
  CHECK_CUDA(tan_fovxy);
  CHECK_CUDA(full_projections);
  CHECK_CUDA(polygon_vertices);
  CHECK_CUDA(polygon_offsets);
  CHECK_CUDA(task_cameras);
  CHECK_CUDA(task_normals);
  CHECK_CUDA(task_plane_offsets);
  CHECK_CUDA(task_bboxes);
  CHECK_CONTIGUOUS(depths);
  CHECK_CONTIGUOUS(inverse_views);
  CHECK_CONTIGUOUS(camera_centers);
  CHECK_CONTIGUOUS(tan_fovxy);
  CHECK_CONTIGUOUS(full_projections);
  CHECK_CONTIGUOUS(polygon_vertices);
  CHECK_CONTIGUOUS(polygon_offsets);
  CHECK_CONTIGUOUS(task_cameras);
  CHECK_CONTIGUOUS(task_normals);
  CHECK_CONTIGUOUS(task_plane_offsets);
  CHECK_CONTIGUOUS(task_bboxes);
  CHECK_FLOAT(depths);
  CHECK_FLOAT(inverse_views);
  CHECK_FLOAT(camera_centers);
  CHECK_FLOAT(tan_fovxy);
  CHECK_FLOAT(full_projections);
  CHECK_FLOAT(polygon_vertices);
  CHECK_INT(polygon_offsets);
  CHECK_INT(task_cameras);
  CHECK_FLOAT(task_normals);
  CHECK_FLOAT(task_plane_offsets);
  CHECK_INT(task_bboxes);
  TORCH_CHECK(depths.dim() == 3, "depths must have shape [B,H,W]");
  TORCH_CHECK(inverse_views.sizes() == torch::IntArrayRef({depths.size(0), 4, 4}),
              "inverse_views must have shape [B,4,4]");
  TORCH_CHECK(full_projections.sizes() == torch::IntArrayRef({depths.size(0), 4, 4}),
              "full_projections must have shape [B,4,4]");
  TORCH_CHECK(camera_centers.sizes() == torch::IntArrayRef({depths.size(0), 3}),
              "camera_centers must have shape [B,3]");
  TORCH_CHECK(tan_fovxy.sizes() == torch::IntArrayRef({depths.size(0), 2}),
              "tan_fovxy must have shape [B,2]");
  const int task_count = task_cameras.size(0);
  TORCH_CHECK(task_count < static_cast<long long>(std::numeric_limits<unsigned int>::max()),
              "too many packed plane tasks");
  TORCH_CHECK(polygon_offsets.numel() == task_count + 1,
              "polygon_offsets must have task_count + 1 elements");
  TORCH_CHECK(task_normals.size(0) == task_count && task_normals.size(1) == 3,
              "task_normals must have shape [T,3]");
  TORCH_CHECK(task_plane_offsets.numel() == task_count,
              "task_plane_offsets must have T elements");
  TORCH_CHECK(task_bboxes.size(0) == task_count && task_bboxes.size(1) == 4,
              "task_bboxes must have shape [T,4]");

  c10::cuda::CUDAGuard device_guard(depths.device());
  auto corrected = depths.clone();
  auto task_counts = torch::zeros(
      {task_count}, depths.options().dtype(torch::kInt32));
  auto best_keys = torch::full(
      depths.sizes(), -1, depths.options().dtype(torch::kInt64));
  auto corrected_count = torch::zeros(
      {1}, depths.options().dtype(torch::kInt64));
  if (task_count == 0)
    return {corrected, task_counts, corrected_count};

  const int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  count_polygon_pixels_kernel<<<task_count, threads, 0, stream>>>(
      polygon_vertices.data_ptr<float>(),
      polygon_offsets.data_ptr<int>(),
      task_bboxes.data_ptr<int>(),
      task_counts.data_ptr<int>(),
      task_count);
  select_best_plane_kernel<<<task_count, threads, 0, stream>>>(
      depths.data_ptr<float>(),
      inverse_views.data_ptr<float>(),
      camera_centers.data_ptr<float>(),
      tan_fovxy.data_ptr<float>(),
      full_projections.data_ptr<float>(),
      polygon_vertices.data_ptr<float>(),
      polygon_offsets.data_ptr<int>(),
      task_cameras.data_ptr<int>(),
      task_normals.data_ptr<float>(),
      task_plane_offsets.data_ptr<float>(),
      task_bboxes.data_ptr<int>(),
      task_counts.data_ptr<int>(),
      reinterpret_cast<unsigned long long*>(best_keys.data_ptr<int64_t>()),
      task_count,
      depths.size(2),
      depths.size(1),
      min_pixels_per_plane,
      depth_trunc,
      max_depth_delta);
  const int total_pixels = depths.numel();
  const int blocks = (total_pixels + threads - 1) / threads;
  apply_best_plane_kernel<<<blocks, threads, 0, stream>>>(
      depths.data_ptr<float>(),
      corrected.data_ptr<float>(),
      inverse_views.data_ptr<float>(),
      camera_centers.data_ptr<float>(),
      tan_fovxy.data_ptr<float>(),
      full_projections.data_ptr<float>(),
      task_cameras.data_ptr<int>(),
      task_normals.data_ptr<float>(),
      task_plane_offsets.data_ptr<float>(),
      reinterpret_cast<const unsigned long long*>(best_keys.data_ptr<int64_t>()),
      corrected_count.data_ptr<int64_t>(),
      total_pixels,
      depths.size(2),
      depths.size(1),
      depth_trunc,
      max_depth_delta,
      blend);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {corrected, task_counts, corrected_count};
}
