#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace cg = cooperative_groups;

#define G_00 0.001028380123898387f
#define G_01 0.0075987582094967365f
#define G_02 0.036000773310661316f
#define G_03 0.10936068743467331f
#define G_04 0.21300552785396576f
#define G_05 0.26601171493530273f
#define G_06 0.21300552785396576f
#define G_07 0.10936068743467331f
#define G_08 0.036000773310661316f
#define G_09 0.0075987582094967365f
#define G_10 0.001028380123898387f

// block size
#define BX 32
#define BY 32

// shared memory size
#define SX (BX + 10)
#define SSX (BX + 10)
#define SY (BY + 10)

// convolution scratchpad size
#define CX (BX)
#define CCX (BX + 0)
#define CY (BY + 10)


__device__ float get_pix_value(const float* img, const int b, const int c, const int y, const int x, const int CH, const int H, const int W) {
  if (x >= W || y >= H || x < 0 || y < 0) {
    return 0.0f;
  } else {
    return img[b * CH * H * W + c * H * W + y * W + x];
  }
}

__device__ void load_into_shared(float pixels[SY][SSX], const float *inp, const int CH, const int H, const int W, const int i) {
  auto block = cg::this_thread_block();
  const int batch = block.group_index().z;
  const int start_y = block.group_index().y * BY;
  const int start_x = block.group_index().x * BX;

  const int cnt = SY * SX;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX;
      int local_x = tid % SX;
      int y = start_y + local_y;
      int x = start_x + local_x;
      float one = get_pix_value(inp, batch, i, y - 5, x - 5, CH, H, W);
      pixels[local_y][local_x] = one;
    }
  }
}

__device__ void multiply_shared_mem(float pix1[SY][SSX], float pix2[SY][SSX]) {
  auto block = cg::this_thread_block();
  const int cnt = SY * SX;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      int local_y = tid / SX;
      int local_x = tid % SX;
      float one = pix1[local_y][local_x];
      float two = pix2[local_y][local_x];
      pix1[local_y][local_x] = one * two;
    }
  }
}

__device__ inline float do_sq(float val) {
  return val * val;
}

__device__ void
flush_conv_scratch(float buf[CY][CCX]) {
  auto block = cg::this_thread_block();
  const int cnt = CY * CX;
  const int num_blocks = (cnt + BX * BY - 1) / (BX * BY);
  for (int b = 0; b < num_blocks; ++b) {
    const int tid = b * (BX * BY) + block.thread_rank();
    if (tid < cnt) {
      const int local_y = tid / CX;
      const int local_x = tid % CX;
      buf[local_y][local_x] = 0.0f;
    }
  }
}

__device__ void do_separable_conv_x(float pixels[SY][SSX], float opt[CY][CCX], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();

  int local_y = block.thread_index().y;
  int local_x = block.thread_index().x + 5;
  float val = 0.0f;

  if (sq) {
    val += G_00 * do_sq(pixels[local_y][local_x - 5]);
    val += G_01 * do_sq(pixels[local_y][local_x - 4]);
    val += G_02 * do_sq(pixels[local_y][local_x - 3]);
    val += G_03 * do_sq(pixels[local_y][local_x - 2]);
    val += G_04 * do_sq(pixels[local_y][local_x - 1]);
    val += G_05 * do_sq(pixels[local_y][local_x    ]);
    val += G_06 * do_sq(pixels[local_y][local_x + 1]);
    val += G_07 * do_sq(pixels[local_y][local_x + 2]);
    val += G_08 * do_sq(pixels[local_y][local_x + 3]);
    val += G_09 * do_sq(pixels[local_y][local_x + 4]);
    val += G_10 * do_sq(pixels[local_y][local_x + 5]);
  } else {
    val += G_00 * pixels[local_y][local_x - 5];
    val += G_01 * pixels[local_y][local_x - 4];
    val += G_02 * pixels[local_y][local_x - 3];
    val += G_03 * pixels[local_y][local_x - 2];
    val += G_04 * pixels[local_y][local_x - 1];
    val += G_05 * pixels[local_y][local_x    ];
    val += G_06 * pixels[local_y][local_x + 1];
    val += G_07 * pixels[local_y][local_x + 2];
    val += G_08 * pixels[local_y][local_x + 3];
    val += G_09 * pixels[local_y][local_x + 4];
    val += G_10 * pixels[local_y][local_x + 5];
  }
  opt[local_y][local_x] = val;

  val = 0.0f;
  local_y = block.thread_index().y + BY;
  if (local_y < SY) {
    if (sq) {
      val += G_00 * do_sq(pixels[local_y][local_x - 5]);
      val += G_01 * do_sq(pixels[local_y][local_x - 4]);
      val += G_02 * do_sq(pixels[local_y][local_x - 3]);
      val += G_03 * do_sq(pixels[local_y][local_x - 2]);
      val += G_04 * do_sq(pixels[local_y][local_x - 1]);
      val += G_05 * do_sq(pixels[local_y][local_x    ]);
      val += G_06 * do_sq(pixels[local_y][local_x + 1]);
      val += G_07 * do_sq(pixels[local_y][local_x + 2]);
      val += G_08 * do_sq(pixels[local_y][local_x + 3]);
      val += G_09 * do_sq(pixels[local_y][local_x + 4]);
      val += G_10 * do_sq(pixels[local_y][local_x + 5]);
    } else {
      val += G_00 * pixels[local_y][local_x - 5];
      val += G_01 * pixels[local_y][local_x - 4];
      val += G_02 * pixels[local_y][local_x - 3];
      val += G_03 * pixels[local_y][local_x - 2];
      val += G_04 * pixels[local_y][local_x - 1];
      val += G_05 * pixels[local_y][local_x    ];
      val += G_06 * pixels[local_y][local_x + 1];
      val += G_07 * pixels[local_y][local_x + 2];
      val += G_08 * pixels[local_y][local_x + 3];
      val += G_09 * pixels[local_y][local_x + 4];
      val += G_10 * pixels[local_y][local_x + 5];
    }
    opt[local_y][local_x] = val;
  }
}

__device__ float do_separable_conv_y(float pixels[CY][CCX], int H, int W, bool sq = false) {
  auto block = cg::this_thread_block();
  int local_y = block.thread_index().y + 5;
  int local_x = block.thread_index().x + 5;
  float val = 0.0f;

  val += G_00 * pixels[local_y - 5][local_x];
  val += G_01 * pixels[local_y - 4][local_x];
  val += G_02 * pixels[local_y - 3][local_x];
  val += G_03 * pixels[local_y - 2][local_x];
  val += G_04 * pixels[local_y - 1][local_x];
  val += G_05 * pixels[local_y    ][local_x];
  val += G_06 * pixels[local_y + 1][local_x];
  val += G_07 * pixels[local_y + 2][local_x];
  val += G_08 * pixels[local_y + 3][local_x];
  val += G_09 * pixels[local_y + 4][local_x];
  val += G_10 * pixels[local_y + 5][local_x];

  return val;
}

__global__ void fusedssimCUDA(
  int H,
  int W,
  int CH,
  float C1,
  float C2,
  float* img1,
  float* img2,
  float* ssim_map,
  float* dm_dmu1 = nullptr,
  float* dm_dsigma1_sq = nullptr,
  float* dm_dsigma12 = nullptr
)
{
  auto block = cg::this_thread_block();
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;
  const int batch = block.group_index().z;

  // shared memory that will be used to load pixels temporarily
  __shared__ float buf1[SY][SSX];
  __shared__ float buf2[SY][SSX];
  __shared__ float buf3[CY][CCX];

  for (int i = 0; i < CH; ++i) {
    // load into shared
    load_into_shared(buf1, img1, CH, H, W, i);
    block.sync();

    // calculate mu1
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf1, buf3, H, W);
    block.sync();
    float mu1 = do_separable_conv_y(buf3, H, W);
    block.sync();

    // calculate sigma1_sq
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf1, buf3, H, W, true);
    block.sync();
    float sigma1_sq = do_separable_conv_y(buf3, H, W) - mu1 * mu1;
    block.sync();

    // calculate mu2
    load_into_shared(buf2, img2, CH, H, W, i);
    block.sync();
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    float mu2 = do_separable_conv_y(buf3, H, W);
    block.sync();

    // calculate sigma2_sq
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W, true);
    block.sync();
    float sigma2_sq = do_separable_conv_y(buf3, H, W) - mu2 * mu2;
    block.sync();

    // calculate sigma12
    multiply_shared_mem(buf1, buf2);
    block.sync();
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf1, buf3, H, W);
    block.sync();
    float sigma12 = do_separable_conv_y(buf3, H, W) - mu1 * mu2;
    block.sync();

    float mu1_sq = mu1 * mu1;
    float mu2_sq = mu2 * mu2;
    float mu1_mu2 = mu1 * mu2;
    float C = (2.0f * mu1_mu2 + C1);
    float D = (2.0f * sigma12 + C2);
    float A = (mu1_sq + mu2_sq + C1);
    float B = (sigma1_sq + sigma2_sq + C2);
    float m = (C * D) / (A * B);
    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      ssim_map[global_idx] = m;

      if (dm_dmu1) {
        dm_dmu1[global_idx] = (
          (mu2 * 2.0f * D) / (A * B)
          -(mu2 * 2.0f * C) / (A * B)
          -(mu1 * 2.0f * C * D) / ( A * A * B)
          +(mu1 * 2.0f * C * D) / (A * B * B)
        );
        dm_dsigma1_sq[global_idx] = ((-C * D) / (A * B * B));
        dm_dsigma12[global_idx] = ((2 * C) / (A * B));
      }
    }
  }
}

__global__ void fusedssim_backwardCUDA(
  int H,
  int W,
  int CH,
  float C1,
  float C2,
  float* img1,
  float* img2,
  float *dL_dmap,
  float *dL_dimg1,
  float* dm_dmu1 = nullptr,
  float* dm_dsigma1_sq = nullptr,
  float* dm_dsigma12 = nullptr
)
{
  auto block = cg::this_thread_block();
  const int pix_y = block.group_index().y * BY + block.thread_index().y;
  const int pix_x = block.group_index().x * BX + block.thread_index().x;
  const int pix_id = pix_y * W + pix_x;
  const int num_pix = H * W;
  const int batch = block.group_index().z;

  // shared memory that will be used to load pixels temporarily
  __shared__ float buf1[SY][SSX];
  __shared__ float buf2[SY][SSX];
  __shared__ float buf3[CY][CCX];

  for (int i = 0; i < CH; ++i) {
    float dL_dpix = 0.0f;
    float tmp = 0.0f;
    float pix1 = get_pix_value(img1, batch, i, pix_y, pix_x, CH, H, W);
    float pix2 = get_pix_value(img2, batch, i, pix_y, pix_x, CH, H, W);
    load_into_shared(buf1, dL_dmap, CH, H, W, i);

    // gradient from mu1
    load_into_shared(buf2, dm_dmu1, CH, H, W, i);
    block.sync();
    multiply_shared_mem(buf2, buf1);
    block.sync();
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    tmp = do_separable_conv_y(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma1_sq
    load_into_shared(buf2, dm_dsigma1_sq, CH, H, W, i);
    block.sync();
    multiply_shared_mem(buf2, buf1);
    block.sync();
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    tmp = pix1 * 2.0f * do_separable_conv_y(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    // gradient from sigma12
    load_into_shared(buf2, dm_dsigma12, CH, H, W, i);
    block.sync();
    multiply_shared_mem(buf2, buf1);
    block.sync();
    flush_conv_scratch(buf3);
    block.sync();
    do_separable_conv_x(buf2, buf3, H, W);
    block.sync();
    tmp = pix2 * do_separable_conv_y(buf3, H, W);
    block.sync();
    dL_dpix += tmp;

    if (pix_x < W && pix_y < H) {
      const int global_idx = batch * CH * num_pix + i * num_pix + pix_id;
      dL_dimg1[global_idx] = dL_dpix;
    }
  }
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
  float C1,
  float C2,
  torch::Tensor &img1,
  torch::Tensor &img2,
  bool train
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);
  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);

  torch::Tensor target = torch::zeros_like(img1).contiguous();
  torch::Tensor dm_dmu1 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  torch::Tensor dm_dsigma1_sq = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  torch::Tensor dm_dsigma12 = train ? torch::zeros_like(img1).contiguous() : torch::empty(0);
  fusedssimCUDA<<<grid,block>>>(
    H,
    W,
    CH,
    C1,
    C2,
    img1.contiguous().data<float>(),
    img2.contiguous().data<float>(),
    target.contiguous().data<float>(),
    dm_dmu1.contiguous().data<float>(),
    dm_dsigma1_sq.contiguous().data<float>(),
    dm_dsigma12.contiguous().data<float>()
  );

  return std::make_tuple(target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

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
)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
  int B = img1.size(0);
  int CH = img1.size(1);
  int H = img1.size(2);
  int W = img1.size(3);

  torch::Tensor dL_dimg1 = torch::zeros_like(img1).contiguous();

  dim3 grid((W + BX - 1) / BX, (H + BY - 1) / BY, B);
  dim3 block(BX, BY, 1);
  fusedssim_backwardCUDA<<<grid,block>>>(
    H,
    W,
    CH,
    C1,
    C2,
    img1.contiguous().data<float>(),
    img2.contiguous().data<float>(),
    dL_dmap.contiguous().data<float>(),
    dL_dimg1.contiguous().data<float>(),
    dm_dmu1.contiguous().data<float>(),
    dm_dsigma1_sq.contiguous().data<float>(),
    dm_dsigma12.contiguous().data<float>()
  );

  return dL_dimg1;
}

// Ragged masked photo loss -------------------------------------------------
//
// The regular fused SSIM API materialises a complete [B, 3, Hmax, Wmax]
// target and SSIM map.  The packed path below keeps the renderer's dense
// prediction (including its right/bottom fringe, which participates in SSIM)
// but reads GT and masks directly from the persistent ROI catalogue.  Only
// actual ROI tiles are scheduled in forward; backward additionally schedules
// the five-pixel SSIM halo.

#define PACKED_PHOTO_MAX_BATCH 64

struct PackedPhotoLaunchDescriptor {
  int batch_size;
  int pair_ids[PACKED_PHOTO_MAX_BATCH];
  int roi_heights[PACKED_PHOTO_MAX_BATCH];
  int roi_widths[PACKED_PHOTO_MAX_BATCH];
  int64_t compact_offsets[PACKED_PHOTO_MAX_BATCH];
  int forward_tile_prefix[PACKED_PHOTO_MAX_BATCH + 1];
  int backward_tile_prefix[PACKED_PHOTO_MAX_BATCH + 1];
};

static_assert(
    sizeof(PackedPhotoLaunchDescriptor) < 4096,
    "packed photo launch descriptor must fit CUDA's kernel parameter limit");

__device__ __forceinline__ int packed_photo_find_batch(
    const int work_index,
    const int *prefix,
    const int batch_size) {
  int lo = 0;
  int hi = batch_size;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (work_index < prefix[mid]) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ float packed_photo_dense_value(
    const float *image,
    const int batch,
    const int channel,
    const int y,
    const int x,
    const int channels,
    const int height,
    const int width) {
  if (x < 0 || x >= width || y < 0 || y >= height) {
    return 0.0f;
  }
  return image[
      static_cast<int64_t>(batch) * channels * height * width
      + static_cast<int64_t>(channel) * height * width
      + static_cast<int64_t>(y) * width + x];
}

__device__ __forceinline__ float packed_photo_gt_value(
    const float *gt_pool,
    const int64_t roi_offset,
    const int roi_height,
    const int roi_width,
    const int channel,
    const int y,
    const int x,
    const int64_t pool_pixels) {
  if (x < 0 || x >= roi_width || y < 0 || y >= roi_height || roi_offset < 0) {
    return 0.0f;
  }
  const int64_t area = static_cast<int64_t>(roi_height) * roi_width;
  const int64_t local = static_cast<int64_t>(y) * roi_width + x;
  if (area > pool_pixels || roi_offset > pool_pixels - area) {
    return 0.0f;
  }
  return gt_pool[3 * roi_offset + static_cast<int64_t>(channel) * area + local];
}

__device__ __forceinline__ float packed_photo_mask_value(
    const float *mask_pool,
    const int64_t roi_offset,
    const int roi_height,
    const int roi_width,
    const int y,
    const int x,
    const int64_t pool_pixels) {
  if (x < 0 || x >= roi_width || y < 0 || y >= roi_height || roi_offset < 0) {
    return 0.0f;
  }
  const int64_t area = static_cast<int64_t>(roi_height) * roi_width;
  if (area > pool_pixels || roi_offset > pool_pixels - area) {
    return 0.0f;
  }
  return mask_pool[roi_offset + static_cast<int64_t>(y) * roi_width + x];
}

__device__ void packed_photo_load_dense(
    float pixels[SY][SSX],
    const float *image,
    const int batch,
    const int channel,
    const int start_y,
    const int start_x,
    const int channels,
    const int height,
    const int width) {
  auto block = cg::this_thread_block();
  const int count = SY * SX;
  const int iterations = (count + BX * BY - 1) / (BX * BY);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int index = iteration * (BX * BY) + block.thread_rank();
    if (index < count) {
      const int local_y = index / SX;
      const int local_x = index % SX;
      pixels[local_y][local_x] = packed_photo_dense_value(
          image,
          batch,
          channel,
          start_y + local_y - 5,
          start_x + local_x - 5,
          channels,
          height,
          width);
    }
  }
}

__device__ void packed_photo_load_gt(
    float pixels[SY][SSX],
    const float *gt_pool,
    const int64_t roi_offset,
    const int roi_height,
    const int roi_width,
    const int channel,
    const int start_y,
    const int start_x,
    const int64_t pool_pixels) {
  auto block = cg::this_thread_block();
  const int count = SY * SX;
  const int iterations = (count + BX * BY - 1) / (BX * BY);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int index = iteration * (BX * BY) + block.thread_rank();
    if (index < count) {
      const int local_y = index / SX;
      const int local_x = index % SX;
      pixels[local_y][local_x] = packed_photo_gt_value(
          gt_pool,
          roi_offset,
          roi_height,
          roi_width,
          channel,
          start_y + local_y - 5,
          start_x + local_x - 5,
          pool_pixels);
    }
  }
}

__device__ __forceinline__ float packed_photo_warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__global__ void packed_masked_photo_forward_cuda(
    const int height,
    const int width,
    const float C1,
    const float C2,
    const float *__restrict__ pred,
    const float *__restrict__ dist,
    const int64_t dist_stride_batch,
    const int64_t dist_stride_y,
    const int64_t dist_stride_x,
    const float *__restrict__ gt_pool,
    const float *__restrict__ mask_pool,
    const int64_t *__restrict__ roi_offsets,
    const float *__restrict__ valid_counts,
    const PackedPhotoLaunchDescriptor descriptor,
    const int64_t pool_pixels,
    float *__restrict__ l1_output,
    float *__restrict__ ssim_output,
    float *__restrict__ dist_output,
    float *__restrict__ dm_dmu1,
    float *__restrict__ dm_dsigma1_sq,
    float *__restrict__ dm_dsigma12) {
  const int work_index = blockIdx.x;
  const int batch = packed_photo_find_batch(
      work_index,
      descriptor.forward_tile_prefix,
      descriptor.batch_size);
  const int roi_height = descriptor.roi_heights[batch];
  const int roi_width = descriptor.roi_widths[batch];
  const int tile_columns = (roi_width + BX - 1) / BX;
  const int local_tile = work_index - descriptor.forward_tile_prefix[batch];
  const int start_y = (local_tile / tile_columns) * BY;
  const int start_x = (local_tile % tile_columns) * BX;
  const int pixel_y = start_y + threadIdx.y;
  const int pixel_x = start_x + threadIdx.x;
  const bool in_roi = pixel_y < roi_height && pixel_x < roi_width;
  const int pair = descriptor.pair_ids[batch];
  const int64_t roi_offset = roi_offsets[pair];
  const int64_t roi_area = static_cast<int64_t>(roi_height) * roi_width;
  const int64_t roi_local = static_cast<int64_t>(pixel_y) * roi_width + pixel_x;

  __shared__ float buffer1[SY][SSX];
  __shared__ float buffer2[SY][SSX];
  __shared__ float buffer3[CY][CCX];
  __shared__ float warp_sums[3][BY];

  float l1_pixel = 0.0f;
  float ssim_pixel = 0.0f;
  auto block = cg::this_thread_block();

  for (int channel = 0; channel < 3; ++channel) {
    packed_photo_load_dense(
        buffer1, pred, batch, channel, start_y, start_x, 3, height, width);
    block.sync();

    flush_conv_scratch(buffer3);
    block.sync();
    do_separable_conv_x(buffer1, buffer3, height, width);
    block.sync();
    const float mu1 = do_separable_conv_y(buffer3, height, width);
    block.sync();

    flush_conv_scratch(buffer3);
    block.sync();
    do_separable_conv_x(buffer1, buffer3, height, width, true);
    block.sync();
    const float sigma1_sq =
        do_separable_conv_y(buffer3, height, width) - mu1 * mu1;
    block.sync();

    packed_photo_load_gt(
        buffer2,
        gt_pool,
        roi_offset,
        roi_height,
        roi_width,
        channel,
        start_y,
        start_x,
        pool_pixels);
    block.sync();
    flush_conv_scratch(buffer3);
    block.sync();
    do_separable_conv_x(buffer2, buffer3, height, width);
    block.sync();
    const float mu2 = do_separable_conv_y(buffer3, height, width);
    block.sync();

    flush_conv_scratch(buffer3);
    block.sync();
    do_separable_conv_x(buffer2, buffer3, height, width, true);
    block.sync();
    const float sigma2_sq =
        do_separable_conv_y(buffer3, height, width) - mu2 * mu2;
    block.sync();

    multiply_shared_mem(buffer1, buffer2);
    block.sync();
    flush_conv_scratch(buffer3);
    block.sync();
    do_separable_conv_x(buffer1, buffer3, height, width);
    block.sync();
    const float sigma12 =
        do_separable_conv_y(buffer3, height, width) - mu1 * mu2;
    block.sync();

    const float mu1_sq = mu1 * mu1;
    const float mu2_sq = mu2 * mu2;
    const float mu1_mu2 = mu1 * mu2;
    const float C = 2.0f * mu1_mu2 + C1;
    const float D = 2.0f * sigma12 + C2;
    const float A = mu1_sq + mu2_sq + C1;
    const float B = sigma1_sq + sigma2_sq + C2;
    const float ssim = (C * D) / (A * B);

    if (in_roi) {
      const float pred_value = packed_photo_dense_value(
          pred, batch, channel, pixel_y, pixel_x, 3, height, width);
      const float gt_value = packed_photo_gt_value(
          gt_pool,
          roi_offset,
          roi_height,
          roi_width,
          channel,
          pixel_y,
          pixel_x,
          pool_pixels);
      l1_pixel += fabsf(pred_value - gt_value);
      ssim_pixel += ssim;

      if (dm_dmu1 != nullptr) {
        const int64_t compact_index =
            3 * descriptor.compact_offsets[batch]
            + static_cast<int64_t>(channel) * roi_area + roi_local;
        dm_dmu1[compact_index] =
            (mu2 * 2.0f * D) / (A * B)
            - (mu2 * 2.0f * C) / (A * B)
            - (mu1 * 2.0f * C * D) / (A * A * B)
            + (mu1 * 2.0f * C * D) / (A * B * B);
        dm_dsigma1_sq[compact_index] = (-C * D) / (A * B * B);
        dm_dsigma12[compact_index] = (2.0f * C) / (A * B);
      }
    }
  }

  float mask_value = 0.0f;
  if (in_roi) {
    mask_value = packed_photo_mask_value(
        mask_pool,
        roi_offset,
        roi_height,
        roi_width,
        pixel_y,
        pixel_x,
        pool_pixels);
  }
  float photo_valid = valid_counts[pair];
  photo_valid = photo_valid < 1.0e-8f ? 1.0e-8f : photo_valid;
  float dist_valid = valid_counts[pair];
  dist_valid = dist_valid < 1.0f ? 1.0f : dist_valid;
  l1_pixel *= mask_value / (3.0f * photo_valid);
  ssim_pixel *= mask_value / (3.0f * photo_valid);
  float dist_pixel = 0.0f;
  if (dist != nullptr && in_roi) {
    const int64_t dist_index =
        static_cast<int64_t>(batch) * dist_stride_batch
        + static_cast<int64_t>(pixel_y) * dist_stride_y
        + static_cast<int64_t>(pixel_x) * dist_stride_x;
    dist_pixel = dist[dist_index] * mask_value / dist_valid;
  }

  l1_pixel = packed_photo_warp_sum(l1_pixel);
  ssim_pixel = packed_photo_warp_sum(ssim_pixel);
  dist_pixel = packed_photo_warp_sum(dist_pixel);
  if (threadIdx.x == 0) {
    warp_sums[0][threadIdx.y] = l1_pixel;
    warp_sums[1][threadIdx.y] = ssim_pixel;
    warp_sums[2][threadIdx.y] = dist_pixel;
  }
  block.sync();
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float block_l1 = 0.0f;
    float block_ssim = 0.0f;
    float block_dist = 0.0f;
    for (int warp = 0; warp < BY; ++warp) {
      block_l1 += warp_sums[0][warp];
      block_ssim += warp_sums[1][warp];
      block_dist += warp_sums[2][warp];
    }
    atomicAdd(l1_output + batch, block_l1);
    atomicAdd(ssim_output + batch, block_ssim);
    if (dist != nullptr) {
      atomicAdd(dist_output + batch, block_dist);
    }
  }
}

__device__ void packed_photo_load_weighted_derivative(
    float pixels[SY][SSX],
    const float *derivative,
    const float *mask_pool,
    const int64_t roi_offset,
    const int64_t compact_offset,
    const int roi_height,
    const int roi_width,
    const int channel,
    const int start_y,
    const int start_x,
    const float output_scale,
    const int64_t pool_pixels) {
  auto block = cg::this_thread_block();
  const int count = SY * SX;
  const int iterations = (count + BX * BY - 1) / (BX * BY);
  const int64_t area = static_cast<int64_t>(roi_height) * roi_width;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const int index = iteration * (BX * BY) + block.thread_rank();
    if (index < count) {
      const int local_y = index / SX;
      const int local_x = index % SX;
      const int y = start_y + local_y - 5;
      const int x = start_x + local_x - 5;
      float value = 0.0f;
      if (x >= 0 && x < roi_width && y >= 0 && y < roi_height) {
        const int64_t local = static_cast<int64_t>(y) * roi_width + x;
        const float mask_value = packed_photo_mask_value(
            mask_pool,
            roi_offset,
            roi_height,
            roi_width,
            y,
            x,
            pool_pixels);
        const int64_t derivative_index =
            3 * compact_offset + static_cast<int64_t>(channel) * area + local;
        value = derivative[derivative_index] * mask_value * output_scale;
      }
      pixels[local_y][local_x] = value;
    }
  }
}

__global__ __launch_bounds__(BX * BY, 1) void packed_masked_photo_backward_cuda(
    const int height,
    const int width,
    const float *__restrict__ pred,
    const float *__restrict__ gt_pool,
    const float *__restrict__ mask_pool,
    const int64_t *__restrict__ roi_offsets,
    const float *__restrict__ valid_counts,
    const float *__restrict__ dm_dmu1,
    const float *__restrict__ dm_dsigma1_sq,
    const float *__restrict__ dm_dsigma12,
    const float *__restrict__ grad_l1,
    const float *__restrict__ grad_ssim,
    const float *__restrict__ grad_dist_loss,
    const PackedPhotoLaunchDescriptor descriptor,
    const int64_t pool_pixels,
    float *__restrict__ grad_pred,
    float *__restrict__ grad_dist) {
  const int work_index = blockIdx.x;
  const int batch = packed_photo_find_batch(
      work_index,
      descriptor.backward_tile_prefix,
      descriptor.batch_size);
  const int roi_height = descriptor.roi_heights[batch];
  const int roi_width = descriptor.roi_widths[batch];
  const int halo_height = min(height, roi_height + 5);
  const int halo_width = min(width, roi_width + 5);
  const int tile_columns = (halo_width + BX - 1) / BX;
  const int local_tile = work_index - descriptor.backward_tile_prefix[batch];
  const int start_y = (local_tile / tile_columns) * BY;
  const int start_x = (local_tile % tile_columns) * BX;
  const int pixel_y = start_y + threadIdx.y;
  const int pixel_x = start_x + threadIdx.x;
  const bool in_halo = pixel_y < halo_height && pixel_x < halo_width;
  const bool in_roi = pixel_y < roi_height && pixel_x < roi_width;
  const int pair = descriptor.pair_ids[batch];
  const int64_t roi_offset = roi_offsets[pair];
  const int64_t compact_offset = descriptor.compact_offsets[batch];
  float photo_valid = valid_counts[pair];
  photo_valid = photo_valid < 1.0e-8f ? 1.0e-8f : photo_valid;
  float dist_valid = valid_counts[pair];
  dist_valid = dist_valid < 1.0f ? 1.0f : dist_valid;
  const float ssim_scale = grad_ssim[batch] / (3.0f * photo_valid);

  __shared__ float buffer1[SY][SSX];
  __shared__ float buffer2[CY][CCX];
  auto block = cg::this_thread_block();

  for (int channel = 0; channel < 3; ++channel) {
    float pixel_gradient = 0.0f;
    const float pred_value = packed_photo_dense_value(
        pred, batch, channel, pixel_y, pixel_x, 3, height, width);
    const float gt_value = packed_photo_gt_value(
        gt_pool,
        roi_offset,
        roi_height,
        roi_width,
        channel,
        pixel_y,
        pixel_x,
        pool_pixels);

    packed_photo_load_weighted_derivative(
        buffer1,
        dm_dmu1,
        mask_pool,
        roi_offset,
        compact_offset,
        roi_height,
        roi_width,
        channel,
        start_y,
        start_x,
        ssim_scale,
        pool_pixels);
    block.sync();
    flush_conv_scratch(buffer2);
    block.sync();
    do_separable_conv_x(buffer1, buffer2, height, width);
    block.sync();
    pixel_gradient += do_separable_conv_y(buffer2, height, width);
    block.sync();

    packed_photo_load_weighted_derivative(
        buffer1,
        dm_dsigma1_sq,
        mask_pool,
        roi_offset,
        compact_offset,
        roi_height,
        roi_width,
        channel,
        start_y,
        start_x,
        ssim_scale,
        pool_pixels);
    block.sync();
    flush_conv_scratch(buffer2);
    block.sync();
    do_separable_conv_x(buffer1, buffer2, height, width);
    block.sync();
    pixel_gradient +=
        pred_value * 2.0f * do_separable_conv_y(buffer2, height, width);
    block.sync();

    packed_photo_load_weighted_derivative(
        buffer1,
        dm_dsigma12,
        mask_pool,
        roi_offset,
        compact_offset,
        roi_height,
        roi_width,
        channel,
        start_y,
        start_x,
        ssim_scale,
        pool_pixels);
    block.sync();
    flush_conv_scratch(buffer2);
    block.sync();
    do_separable_conv_x(buffer1, buffer2, height, width);
    block.sync();
    pixel_gradient += gt_value * do_separable_conv_y(buffer2, height, width);
    block.sync();

    if (in_roi) {
      const float mask_value = packed_photo_mask_value(
          mask_pool,
          roi_offset,
          roi_height,
          roi_width,
          pixel_y,
          pixel_x,
          pool_pixels);
      const float difference = pred_value - gt_value;
      const float sign = (difference > 0.0f) - (difference < 0.0f);
      pixel_gradient +=
          grad_l1[batch] * mask_value * sign / (3.0f * photo_valid);
    }

    if (in_halo) {
      const int64_t dense_index =
          static_cast<int64_t>(batch) * 3 * height * width
          + static_cast<int64_t>(channel) * height * width
          + static_cast<int64_t>(pixel_y) * width + pixel_x;
      grad_pred[dense_index] = pixel_gradient;
    }
  }

  if (grad_dist != nullptr && in_roi) {
    const float mask_value = packed_photo_mask_value(
        mask_pool,
        roi_offset,
        roi_height,
        roi_width,
        pixel_y,
        pixel_x,
        pool_pixels);
    const int64_t dense_index =
        static_cast<int64_t>(batch) * height * width
        + static_cast<int64_t>(pixel_y) * width + pixel_x;
    grad_dist[dense_index] = grad_dist_loss[batch] * mask_value / dist_valid;
  }
}

static void check_packed_photo_tensor(
    const torch::Tensor &tensor,
    const char *name,
    const torch::Device &device,
    const bool require_contiguous = true) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.device() == device, name, " must be on the prediction device");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
  if (require_contiguous) {
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  }
}

static PackedPhotoLaunchDescriptor make_packed_photo_descriptor(
    const std::vector<int64_t> &pair_ids,
    const std::vector<int64_t> &roi_heights,
    const std::vector<int64_t> &roi_widths,
    const int64_t pair_count,
    const int height,
    const int width,
    int64_t *compact_pixels,
    int *forward_tiles,
    int *backward_tiles) {
  const int64_t batch_size = static_cast<int64_t>(pair_ids.size());
  TORCH_CHECK(batch_size > 0, "packed photo loss requires a non-empty batch");
  TORCH_CHECK(
      batch_size <= PACKED_PHOTO_MAX_BATCH,
      "packed photo loss supports at most ",
      PACKED_PHOTO_MAX_BATCH,
      " cameras, got ",
      batch_size);
  TORCH_CHECK(
      roi_heights.size() == pair_ids.size() && roi_widths.size() == pair_ids.size(),
      "pair_ids, roi_heights, and roi_widths must have equal lengths");

  PackedPhotoLaunchDescriptor descriptor{};
  descriptor.batch_size = static_cast<int>(batch_size);
  descriptor.forward_tile_prefix[0] = 0;
  descriptor.backward_tile_prefix[0] = 0;
  int64_t compact_total = 0;
  int64_t forward_total = 0;
  int64_t backward_total = 0;
  for (int64_t batch = 0; batch < batch_size; ++batch) {
    TORCH_CHECK(
        pair_ids[batch] >= 0 && pair_ids[batch] < pair_count,
        "pair_ids[",
        batch,
        "] is outside the ROI catalogue: ",
        pair_ids[batch]);
    TORCH_CHECK(
        roi_heights[batch] > 0 && roi_heights[batch] <= height,
        "roi_heights[",
        batch,
        "] must be in [1, ",
        height,
        "]");
    TORCH_CHECK(
        roi_widths[batch] > 0 && roi_widths[batch] <= width,
        "roi_widths[",
        batch,
        "] must be in [1, ",
        width,
        "]");
    TORCH_CHECK(
        pair_ids[batch] <= std::numeric_limits<int>::max()
            && roi_heights[batch] <= std::numeric_limits<int>::max()
            && roi_widths[batch] <= std::numeric_limits<int>::max(),
        "packed photo descriptor value exceeds int32 range");

    const int roi_height = static_cast<int>(roi_heights[batch]);
    const int roi_width = static_cast<int>(roi_widths[batch]);
    const int64_t area = static_cast<int64_t>(roi_height) * roi_width;
    descriptor.pair_ids[batch] = static_cast<int>(pair_ids[batch]);
    descriptor.roi_heights[batch] = roi_height;
    descriptor.roi_widths[batch] = roi_width;
    descriptor.compact_offsets[batch] = compact_total;
    compact_total += area;

    forward_total +=
        static_cast<int64_t>((roi_height + BY - 1) / BY)
        * ((roi_width + BX - 1) / BX);
    const int halo_height = std::min(height, roi_height + 5);
    const int halo_width = std::min(width, roi_width + 5);
    backward_total +=
        static_cast<int64_t>((halo_height + BY - 1) / BY)
        * ((halo_width + BX - 1) / BX);
    TORCH_CHECK(
        forward_total <= std::numeric_limits<int>::max()
            && backward_total <= std::numeric_limits<int>::max(),
        "packed photo tile count exceeds int32 range");
    descriptor.forward_tile_prefix[batch + 1] = static_cast<int>(forward_total);
    descriptor.backward_tile_prefix[batch + 1] = static_cast<int>(backward_total);
  }
  TORCH_CHECK(
      compact_total <= std::numeric_limits<int64_t>::max() / 3,
      "packed photo derivative buffer is too large");
  *compact_pixels = compact_total;
  *forward_tiles = static_cast<int>(forward_total);
  *backward_tiles = static_cast<int>(backward_total);
  return descriptor;
}

static PackedPhotoLaunchDescriptor check_packed_photo_common(
    const torch::Tensor &pred,
    const torch::Tensor &dist,
    const torch::Tensor &gt_pool,
    const torch::Tensor &mask_pool,
    const torch::Tensor &roi_offsets,
    const torch::Tensor &valid_counts,
    const std::vector<int64_t> &pair_ids,
    const std::vector<int64_t> &roi_heights,
    const std::vector<int64_t> &roi_widths,
    int64_t *compact_pixels,
    int *forward_tiles,
    int *backward_tiles) {
  TORCH_CHECK(pred.is_cuda(), "pred must be a CUDA tensor");
  TORCH_CHECK(pred.scalar_type() == torch::kFloat32, "pred must be float32");
  TORCH_CHECK(pred.is_contiguous(), "pred must be contiguous");
  TORCH_CHECK(
      pred.dim() == 4 && pred.size(1) == 3,
      "pred must have shape [B, 3, H, W]");
  TORCH_CHECK(
      pred.size(0) == static_cast<int64_t>(pair_ids.size()),
      "pred batch size must match pair_ids");
  TORCH_CHECK(pred.size(2) > 0 && pred.size(3) > 0, "pred spatial dimensions must be positive");
  TORCH_CHECK(
      pred.size(2) <= std::numeric_limits<int>::max()
          && pred.size(3) <= std::numeric_limits<int>::max(),
      "pred spatial dimensions exceed int32 range");

  const auto device = pred.device();
  check_packed_photo_tensor(gt_pool, "gt_pool", device);
  check_packed_photo_tensor(mask_pool, "mask_pool", device);
  check_packed_photo_tensor(valid_counts, "valid_counts", device);
  TORCH_CHECK(roi_offsets.is_cuda(), "roi_offsets must be a CUDA tensor");
  TORCH_CHECK(roi_offsets.device() == device, "roi_offsets must be on the prediction device");
  TORCH_CHECK(roi_offsets.scalar_type() == torch::kInt64, "roi_offsets must be int64");
  TORCH_CHECK(roi_offsets.is_contiguous(), "roi_offsets must be contiguous");
  TORCH_CHECK(gt_pool.dim() == 1, "gt_pool must be flat [3 * total_pixels]");
  TORCH_CHECK(mask_pool.dim() == 1, "mask_pool must be flat [total_pixels]");
  TORCH_CHECK(roi_offsets.dim() == 1, "roi_offsets must have shape [P]");
  TORCH_CHECK(valid_counts.dim() == 1, "valid_counts must have shape [P]");
  TORCH_CHECK(
      valid_counts.size(0) == roi_offsets.size(0),
      "valid_counts and roi_offsets must describe the same catalogue");
  TORCH_CHECK(
      mask_pool.numel() <= std::numeric_limits<int64_t>::max() / 3
          && gt_pool.numel() == 3 * mask_pool.numel(),
      "gt_pool must contain exactly three CHW values per mask_pool pixel");

  if (dist.numel() != 0) {
    check_packed_photo_tensor(dist, "dist", device, false);
    TORCH_CHECK(
        dist.dim() == 4 && dist.size(0) == pred.size(0) && dist.size(1) == 1
            && dist.size(2) == pred.size(2) && dist.size(3) == pred.size(3),
        "dist must have shape [B, 1, H, W]");
    TORCH_CHECK(
        dist.stride(0) >= 0 && dist.stride(2) >= 0 && dist.stride(3) >= 0,
        "dist must have non-negative strides");
  } else {
    check_packed_photo_tensor(dist, "dist", device, false);
  }

  return make_packed_photo_descriptor(
      pair_ids,
      roi_heights,
      roi_widths,
      roi_offsets.size(0),
      static_cast<int>(pred.size(2)),
      static_cast<int>(pred.size(3)),
      compact_pixels,
      forward_tiles,
      backward_tiles);
}

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
    bool train) {
  int64_t compact_pixels = 0;
  int forward_tiles = 0;
  int backward_tiles = 0;
  const PackedPhotoLaunchDescriptor descriptor = check_packed_photo_common(
      pred,
      dist,
      gt_pool,
      mask_pool,
      roi_offsets,
      valid_counts,
      pair_ids,
      roi_heights,
      roi_widths,
      &compact_pixels,
      &forward_tiles,
      &backward_tiles);
  (void)backward_tiles;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(pred));
  const int64_t batch = pred.size(0);
  auto l1_output = torch::zeros({batch}, pred.options());
  auto ssim_output = torch::zeros({batch}, pred.options());
  auto dist_output = torch::zeros({batch}, pred.options());
  auto dm_dmu1 = train
      ? torch::empty({3 * compact_pixels}, pred.options())
      : torch::empty({0}, pred.options());
  auto dm_dsigma1_sq = train
      ? torch::empty({3 * compact_pixels}, pred.options())
      : torch::empty({0}, pred.options());
  auto dm_dsigma12 = train
      ? torch::empty({3 * compact_pixels}, pred.options())
      : torch::empty({0}, pred.options());

  const bool has_dist = dist.numel() != 0;
  const float *dist_pointer = has_dist ? dist.data_ptr<float>() : nullptr;
  const int64_t dist_stride_batch = has_dist ? dist.stride(0) : 0;
  const int64_t dist_stride_y = has_dist ? dist.stride(2) : 0;
  const int64_t dist_stride_x = has_dist ? dist.stride(3) : 0;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(BX, BY, 1);
  packed_masked_photo_forward_cuda<<<forward_tiles, block, 0, stream>>>(
      static_cast<int>(pred.size(2)),
      static_cast<int>(pred.size(3)),
      C1,
      C2,
      pred.data_ptr<float>(),
      dist_pointer,
      dist_stride_batch,
      dist_stride_y,
      dist_stride_x,
      gt_pool.data_ptr<float>(),
      mask_pool.data_ptr<float>(),
      roi_offsets.data_ptr<int64_t>(),
      valid_counts.data_ptr<float>(),
      descriptor,
      mask_pool.numel(),
      l1_output.data_ptr<float>(),
      ssim_output.data_ptr<float>(),
      dist_output.data_ptr<float>(),
      train ? dm_dmu1.data_ptr<float>() : nullptr,
      train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
      train ? dm_dsigma12.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      l1_output,
      ssim_output,
      dist_output,
      dm_dmu1,
      dm_dsigma1_sq,
      dm_dsigma12);
}

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
    const torch::Tensor &grad_dist_loss,
    const std::vector<int64_t> &pair_ids,
    const std::vector<int64_t> &roi_heights,
    const std::vector<int64_t> &roi_widths) {
  (void)C1;
  (void)C2;
  int64_t compact_pixels = 0;
  int forward_tiles = 0;
  int backward_tiles = 0;
  const PackedPhotoLaunchDescriptor descriptor = check_packed_photo_common(
      pred,
      dist,
      gt_pool,
      mask_pool,
      roi_offsets,
      valid_counts,
      pair_ids,
      roi_heights,
      roi_widths,
      &compact_pixels,
      &forward_tiles,
      &backward_tiles);
  (void)forward_tiles;
  const auto device = pred.device();
  check_packed_photo_tensor(dm_dmu1, "dm_dmu1", device);
  check_packed_photo_tensor(dm_dsigma1_sq, "dm_dsigma1_sq", device);
  check_packed_photo_tensor(dm_dsigma12, "dm_dsigma12", device);
  check_packed_photo_tensor(grad_l1, "grad_l1", device);
  check_packed_photo_tensor(grad_ssim, "grad_ssim", device);
  check_packed_photo_tensor(grad_dist_loss, "grad_dist", device);
  const int64_t derivative_values = 3 * compact_pixels;
  TORCH_CHECK(
      dm_dmu1.dim() == 1 && dm_dmu1.numel() == derivative_values
          && dm_dsigma1_sq.dim() == 1 && dm_dsigma1_sq.numel() == derivative_values
          && dm_dsigma12.dim() == 1 && dm_dsigma12.numel() == derivative_values,
      "packed SSIM derivative buffers have the wrong size");
  TORCH_CHECK(
      grad_l1.dim() == 1 && grad_l1.size(0) == pred.size(0)
          && grad_ssim.dim() == 1 && grad_ssim.size(0) == pred.size(0)
          && grad_dist_loss.dim() == 1 && grad_dist_loss.size(0) == pred.size(0),
      "packed photo output gradients must have shape [B]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(pred));
  auto grad_pred = torch::zeros_like(pred).contiguous();
  const bool has_dist = dist.numel() != 0;
  auto grad_dist = has_dist
      ? torch::zeros(dist.sizes(), dist.options())
      : torch::empty({0}, pred.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 block(BX, BY, 1);
  packed_masked_photo_backward_cuda<<<backward_tiles, block, 0, stream>>>(
      static_cast<int>(pred.size(2)),
      static_cast<int>(pred.size(3)),
      pred.data_ptr<float>(),
      gt_pool.data_ptr<float>(),
      mask_pool.data_ptr<float>(),
      roi_offsets.data_ptr<int64_t>(),
      valid_counts.data_ptr<float>(),
      dm_dmu1.data_ptr<float>(),
      dm_dsigma1_sq.data_ptr<float>(),
      dm_dsigma12.data_ptr<float>(),
      grad_l1.data_ptr<float>(),
      grad_ssim.data_ptr<float>(),
      grad_dist_loss.data_ptr<float>(),
      descriptor,
      mask_pool.numel(),
      grad_pred.data_ptr<float>(),
      has_dist ? grad_dist.data_ptr<float>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(grad_pred, grad_dist);
}
