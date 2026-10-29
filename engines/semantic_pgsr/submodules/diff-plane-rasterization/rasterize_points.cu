/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/backward.h"
#include "cuda_rasterizer/rasterizer_impl.h"
#include "rasterize_points.h"
#include <fstream>
#include <string>
#include <functional>

// Experimental batch zero initialization; output storage remains independent.
#include <cstdlib>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
namespace {
struct ZeroItem { uint32_t* data; int64_t words; };
struct ZeroItems { ZeroItem values[128]; };
__global__ void batchZero(ZeroItems items) {
 const auto item=items.values[blockIdx.y];
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<item.words;i+=int64_t(gridDim.x)*blockDim.x)item.data[i]=0;
}
bool batchZeroEnabled() { const char* value=std::getenv("PGSR_BATCH_ZERO"); return value && std::string(value)=="1"; }
torch::Tensor zeroOrQueue(at::IntArrayRef shape, const torch::TensorOptions& options, std::vector<torch::Tensor>* queue) {
 if(!queue)return torch::zeros(shape,options);
 auto out=torch::empty(shape,options);queue->push_back(out);return out;
}
torch::Tensor fullOrQueue(at::IntArrayRef shape, const torch::TensorOptions& options, std::vector<torch::Tensor>* queue) {
 if(!queue)return torch::full(shape,0,options);
 return zeroOrQueue(shape,options,queue);
}
void flushZeros(const std::vector<torch::Tensor>& tensors) {
 for(size_t start=0;start<tensors.size();start+=128) {
  ZeroItems items{};int count=0;int64_t largest=0;
  for(size_t k=start;k<std::min(start+128,tensors.size());++k) {
   const auto& t=tensors[k];TORCH_CHECK(t.is_contiguous()&&t.element_size()==4,"batch zero requires contiguous 32-bit tensors");
   if(!t.numel())continue;
   items.values[count++]={reinterpret_cast<uint32_t*>(t.data_ptr()),t.numel()};largest=std::max(largest,t.numel());
  }
  if(count)batchZero<<<dim3(std::min<int64_t>((largest+255)/256,1024),count),256,0,at::cuda::getCurrentCUDAStream()>>>(items);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
 }
}
}

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDAImpl(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& all_map,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool render_geo,
	const bool debug, const int roi_x, const int roi_y, const int roi_width, const int roi_height, const float principal_x, const float principal_y, CudaRasterizer::BatchContext* batch, std::vector<torch::Tensor>* zero_queue = nullptr)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const c10::cuda::CUDAGuard device_guard(means3D.device());
  const int P = means3D.size(0);
  const int H = roi_height;
  const int W = roi_width;

  TORCH_CHECK(roi_x >= 0 && roi_y >= 0 && W > 0 && H > 0 && roi_x + W <= image_width && roi_y + H <= image_height, "invalid ROI");
  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = fullOrQueue({NUM_CHANNELS, H, W}, float_opts, P ? zero_queue : nullptr);
  // An empty scene has full transmittance, just like an all-culled tile.
  if (P == 0)
    out_color.copy_(background.reshape({NUM_CHANNELS, 1, 1}));
  torch::Tensor radii = fullOrQueue({P}, means3D.options().dtype(torch::kInt32), zero_queue);
  torch::Tensor out_observe = fullOrQueue({P}, means3D.options().dtype(torch::kInt32), zero_queue);
  torch::Tensor out_all_map = fullOrQueue({NUM_ALL_MAP, H, W}, float_opts, zero_queue);
  torch::Tensor out_plane_depth = fullOrQueue({1, H, W}, float_opts, zero_queue);
  
  torch::Device device = means3D.device();
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.dim() > 1)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H, image_width, image_height, roi_x, roi_y, principal_x, principal_y,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		all_map.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		radii.contiguous().data<int>(),
		out_observe.contiguous().data<int>(),
		out_all_map.contiguous().data<float>(),
		out_plane_depth.contiguous().data<float>(),
		render_geo,
		debug, batch);
  }
  return std::make_tuple(rendered, out_color, radii, out_observe, out_all_map, out_plane_depth, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDAImpl(
	const torch::Tensor& background,
	const torch::Tensor& all_map_pixels,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& all_maps,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_all_map,
	const torch::Tensor& dL_dout_plane_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool render_geo,
	const bool debug, const int image_width, const int image_height, const int roi_x, const int roi_y, const int roi_width, const int roi_height, const float principal_x, const float principal_y, BACKWARD::Batch* batch, std::vector<torch::Tensor>* temporaries = nullptr, std::vector<torch::Tensor>* zero_queue = nullptr)
{
  const c10::cuda::CUDAGuard device_guard(means3D.device());
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.dim() > 1)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = zeroOrQueue({P, 3}, means3D.options(), zero_queue);
  torch::Tensor dL_dmeans2D = zeroOrQueue({P, 3}, means3D.options(), zero_queue);
  torch::Tensor dL_dmeans2D_abs = zeroOrQueue({P, 3}, means3D.options(), zero_queue);
  torch::Tensor dL_dcolors = zeroOrQueue({P, NUM_CHANNELS}, means3D.options(), zero_queue);
  torch::Tensor dL_dall_map = zeroOrQueue({P, NUM_ALL_MAP}, means3D.options(), zero_queue);
  torch::Tensor dL_dconic = zeroOrQueue({P, 2, 2}, means3D.options(), zero_queue);
  if (temporaries) temporaries->push_back(dL_dconic);
  torch::Tensor dL_dopacity = zeroOrQueue({P, 1}, means3D.options(), zero_queue);
  torch::Tensor dL_dcov3D = zeroOrQueue({P, 6}, means3D.options(), zero_queue);
  torch::Tensor dL_dsh = zeroOrQueue({P, M, 3}, means3D.options(), zero_queue);
  torch::Tensor dL_dscales = zeroOrQueue({P, 3}, means3D.options(), zero_queue);
  torch::Tensor dL_drotations = zeroOrQueue({P, 4}, means3D.options(), zero_queue);
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  all_map_pixels.contiguous().data<float>(),
	  W, H, image_width, image_height, roi_x, roi_y, principal_x, principal_y,
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  all_maps.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_all_map.contiguous().data<float>(),
	  dL_dout_plane_depth.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dmeans2D_abs.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  dL_dall_map.contiguous().data<float>(),
	  render_geo,
	  debug, batch);
  }

  return std::make_tuple(dL_dmeans2D, dL_dmeans2D_abs, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dall_map);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& all_map_pixels,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& all_maps,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_all_map,
	const torch::Tensor& dL_dout_plane_depth,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool render_geo,
	const bool debug, const int image_width, const int image_height, const int roi_x, const int roi_y, const int roi_width, const int roi_height, const float principal_x, const float principal_y) {
    return RasterizeGaussiansBackwardCUDAImpl(background, all_map_pixels, means3D, radii, colors, all_maps, scales, rotations, scale_modifier, cov3D_precomp, viewmatrix, projmatrix, tan_fovx, tan_fovy, dL_dout_color, dL_dout_all_map, dL_dout_plane_depth, sh, degree, campos, geomBuffer, R, binningBuffer, imageBuffer, render_geo, debug, image_width, image_height, roi_x, roi_y, roi_width, roi_height, principal_x, principal_y, nullptr);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const c10::cuda::CUDAGuard device_guard(means3D.device());
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& all_map,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool render_geo,
	const bool debug, const int roi_x, const int roi_y, const int roi_width, const int roi_height, const float principal_x, const float principal_y) {
return RasterizeGaussiansCUDAImpl(background, means3D, colors, opacity, scales, rotations, scale_modifier, cov3D_precomp, all_map, viewmatrix, projmatrix, tan_fovx, tan_fovy, image_height, image_width, sh, degree, campos, prefiltered, render_geo, debug, roi_x, roi_y, roi_width, roi_height, principal_x, principal_y, nullptr);
}

template<size_t... I>
ForwardResult batchForward(const ForwardArgs& args, CudaRasterizer::BatchContext* batch, std::vector<torch::Tensor>* zero_queue, std::index_sequence<I...>) {
    return RasterizeGaussiansCUDAImpl(std::get<I>(args)..., batch, zero_queue);
}
template<size_t... I>
BackwardResult batchBackward(const BackwardArgs& args, BACKWARD::Batch* batch, std::vector<torch::Tensor>* temporaries, std::vector<torch::Tensor>* zero_queue, std::index_sequence<I...>) {
    return RasterizeGaussiansBackwardCUDAImpl(std::get<I>(args)..., batch, temporaries, zero_queue);
}
std::vector<ForwardResult> RasterizeGaussiansBatchCUDA(const std::vector<ForwardArgs>& args, bool fused) {
    if (args.empty()) return {};
    const auto& means = std::get<1>(args.front());
    const c10::cuda::CUDAGuard device_guard(means.device());
    CudaRasterizer::BatchContext batch;
    batch.fused = fused;
    std::vector<torch::Tensor> zeros;
    auto* zero_queue = batchZeroEnabled() && fused ? &zeros : nullptr;
    std::vector<ForwardResult> outputs;
    outputs.reserve(args.size());
    // All contiguous inputs are held by args through all deferred launches.
    for (const auto& camera : args) {
        TORCH_CHECK(std::get<1>(camera).device() == means.device(), "batch cameras must share a device");
        outputs.push_back(batchForward(camera, &batch, zero_queue, std::make_index_sequence<std::tuple_size<ForwardArgs>::value>{}));
    }
    auto scratch = torch::empty({0}, means.options().dtype(torch::kByte));
    auto binning = torch::empty({0}, means.options().dtype(torch::kByte));
    flushZeros(zeros);
    int rendered = batch.finish(resizeFunctional(scratch), resizeFunctional(binning));
    for (auto& out : outputs) {
        std::get<0>(out) = rendered;
        std::get<7>(out) = binning;
    }
    return outputs;
}
std::vector<BackwardResult> RasterizeGaussiansBatchBackwardCUDA(const std::vector<BackwardArgs>& args, const std::vector<torch::Tensor>& masks) {
    if (args.empty()) return {};
    const auto& means = std::get<2>(args.front());
    const c10::cuda::CUDAGuard guard(means.device());
    TORCH_CHECK(masks.size() == args.size(), "one gradient mask per camera required");
    BACKWARD::Batch batch;
    std::vector<torch::Tensor> zeros;
    auto* zero_queue = batchZeroEnabled() ? &zeros : nullptr;
    std::vector<torch::Tensor> temporaries;
    std::vector<BackwardResult> outputs;
    // Keep scratch conic gradients alive until all three batch kernels launch.
    // They are allocated by the implementation and retained below.
    outputs.reserve(args.size());
    for (const auto& camera : args) {
        TORCH_CHECK(std::get<2>(camera).device() == means.device(), "batch cameras must share a device");
        const auto& mask = masks[outputs.size()];
        const auto& xyz = std::get<2>(camera);
        TORCH_CHECK(mask.device() == xyz.device() && mask.scalar_type() == torch::kBool && mask.is_contiguous() && mask.dim() == 1 && (mask.numel() == 0 || mask.numel() == xyz.size(0)), "invalid gradient mask");
        size_t before = batch.renders.size();
        outputs.push_back(batchBackward(camera, &batch, &temporaries, zero_queue, std::make_index_sequence<std::tuple_size<BackwardArgs>::value>{}));
        if (batch.renders.size() > before) {
            batch.renders.back().gradient_mask = mask.numel() ? mask.data_ptr<bool>() : nullptr;
            batch.preprocesses.back().gradient_mask = mask.numel() ? mask.data_ptr<bool>() : nullptr;
        }
    }
    auto scratch = torch::empty({0}, means.options().dtype(torch::kByte));
    flushZeros(zeros);
    batch.finish(resizeFunctional(scratch));
    return outputs;
}
