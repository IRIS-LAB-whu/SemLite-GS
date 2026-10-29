#include <ATen/cuda/CUDAContext.h>
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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__device__ __forceinline__ void duplicateWithKeysImpl(unsigned int idx,
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid, int tile_offset_x, int tile_offset_y, int point_base = 0, uint32_t tile_base = 0)
{
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx + point_base == 0) ? 0 : offsets[idx + point_base - 1];
		uint2 rect_min, rect_max;

		getRect({points_xy[idx].x + tile_offset_x, points_xy[idx].y + tile_offset_y}, radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = tile_base + y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid, int tile_offset_x, int tile_offset_y, int point_base = 0, uint32_t tile_base = 0) {
    duplicateWithKeysImpl(blockIdx.x*blockDim.x+threadIdx.x,P,points_xy,depths,offsets,gaussian_keys_unsorted,gaussian_values_unsorted,radii,grid,tile_offset_x,tile_offset_y,point_base,tile_base);
}
__global__ void duplicateForwardBatchCUDA(const FORWARD::BatchCamera* cameras,int count,const uint32_t* offsets,uint64_t* keys,uint32_t* values) {
    int c=0;
    while(c+1<count && int(blockIdx.x)>=cameras[c].point_end) ++c;
    const auto& camera=cameras[c]; const auto& a=camera.pre;
    unsigned int idx=(int(blockIdx.x)-camera.point_begin)*blockDim.x+threadIdx.x;
    duplicateWithKeysImpl(idx,a.P,a.points_xy_image,a.depths,offsets,keys,values,a.radii,a.grid,
        a.roi_x%BLOCK_X,a.roi_y%BLOCK_Y,camera.point_base,camera.tile_begin);
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum <<<(P + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>> (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P, at::cuda::getCurrentCUDAStream());
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P, 0, 64, at::cuda::getCurrentCUDAStream());
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height, const int full_width, const int full_height, const int roi_x, const int roi_y, const float principal_x, const float principal_y,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* all_map,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	int* out_observe,
	float* out_all_map,
	float* out_plane_depth,
	const bool render_geo,
	bool debug, BatchContext* batch)
{
	const float focal_y = full_height / (2.0f * tan_fovy);
	const float focal_x = full_width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + roi_x % BLOCK_X + BLOCK_X - 1) / BLOCK_X, (height + roi_y % BLOCK_Y + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	if (!batch || !batch->fused) CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		full_width, full_height, roi_x, roi_y,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	if (batch != nullptr)
	{
		if (P > INT_MAX - batch->points || uint64_t(batch->tiles) + uint64_t(tile_grid.x) * tile_grid.y > INT_MAX)
			throw std::overflow_error("camera batch exceeds 32-bit CUB index capacity");
		const int point_base = batch->points;
		const uint32_t tile_base = batch->tiles;
		batch->points += P;
		batch->tiles += tile_grid.x * tile_grid.y;
        if (batch->fused) {
            int point_begin = batch->cameras.empty() ? 0 : batch->cameras.back().point_end;
            const float* features = colors_precomp ? colors_precomp : geomState.rgb;
            batch->cameras.push_back({
                {P,D,M,means3D,(glm::vec3*)scales,scale_modifier,(glm::vec4*)rotations,opacities,shs,
                 geomState.clamped,cov3D_precomp,colors_precomp,viewmatrix,projmatrix,(glm::vec3*)cam_pos,
                 full_width,full_height,roi_x,roi_y,tan_fovx,tan_fovy,focal_x,focal_y,radii,
                 geomState.means2D,geomState.depths,geomState.cov3D,geomState.rgb,geomState.conic_opacity,
                 tile_grid,geomState.tiles_touched,prefiltered},
                {imgState.ranges,nullptr,width,height,focal_x,focal_y,principal_x-roi_x,principal_y-roi_y,
                 roi_x%BLOCK_X,roi_y%BLOCK_Y,viewmatrix,cam_pos,geomState.means2D,features,all_map,
                 geomState.conic_opacity,imgState.accum_alpha,imgState.n_contrib,background,out_color,
                 out_observe,out_all_map,out_plane_depth,render_geo},
                point_begin,point_begin+(P+255)/256,int(tile_base),int(batch->tiles),point_base});
            return 0;
        }
		batch->counts.emplace_back(P, geomState.tiles_touched);
		batch->duplicate.emplace_back([=](const uint32_t* offsets, BinningState& binningState) {
			duplicateWithKeys<<<(P + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
				P, geomState.means2D, geomState.depths, offsets,
				binningState.point_list_keys_unsorted, binningState.point_list_unsorted,
				radii, tile_grid, roi_x % BLOCK_X, roi_y % BLOCK_Y, point_base, tile_base);
		});
		batch->render.emplace_back([=](const uint2* ranges, BinningState& binningState) {
			CHECK_CUDA(cudaMemcpyAsync(imgState.ranges, ranges + tile_base,
				tile_grid.x * tile_grid.y * sizeof(uint2), cudaMemcpyDeviceToDevice,
				at::cuda::getCurrentCUDAStream()), debug);
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		principal_x - roi_x, principal_y - roi_y, roi_x % BLOCK_X, roi_y % BLOCK_Y,
		viewmatrix,
		cam_pos,
		geomState.means2D,
		feature_ptr,
		all_map,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_observe,
		out_all_map,
		out_plane_depth,
		render_geo), debug)

		});
		return 0;
	}

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P, at::cuda::getCurrentCUDAStream()), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpyAsync(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream()), debug);

	CHECK_CUDA(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys <<<(P + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>> (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid, roi_x % BLOCK_X, roi_y % BLOCK_Y)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit, at::cuda::getCurrentCUDAStream()), debug)

	CHECK_CUDA(cudaMemsetAsync(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2), at::cuda::getCurrentCUDAStream()), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges <<<(num_rendered + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>> (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		principal_x - roi_x, principal_y - roi_y, roi_x % BLOCK_X, roi_y % BLOCK_Y,
		viewmatrix,
		cam_pos,
		geomState.means2D,
		feature_ptr,
		all_map,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		out_observe,
		out_all_map,
		out_plane_depth,
		render_geo), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const float* all_map_pixels,
	const int width, int height, const int full_width, const int full_height, const int roi_x, const int roi_y, const float principal_x, const float principal_y,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* all_maps,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	const float* dL_dout_all_map,
	const float* dL_dout_plane_depth,
	float* dL_dmean2D,
	float* dL_dmean2D_abs,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dall_map,
	const bool render_geo,
	bool debug, BACKWARD::Batch* batch)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = full_height / (2.0f * tan_fovy);
	const float focal_x = full_width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + roi_x % BLOCK_X + BLOCK_X - 1) / BLOCK_X, (height + roi_y % BLOCK_Y + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	if (batch) batch->renders.push_back({
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, full_width, full_height, principal_x - roi_x, principal_y - roi_y, roi_x % BLOCK_X, roi_y % BLOCK_Y,
		focal_x, focal_y,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		all_maps,
		all_map_pixels,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dout_all_map,
		dL_dout_plane_depth,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_abs,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dall_map,
		render_geo});
	else CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height, full_width, full_height, principal_x - roi_x, principal_y - roi_y, roi_x % BLOCK_X, roi_y % BLOCK_Y,
		focal_x, focal_y,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		all_maps,
		all_map_pixels,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dout_all_map,
		dL_dout_plane_depth,
		(float3*)dL_dmean2D,
		(float3*)dL_dmean2D_abs,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dall_map,
		render_geo), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	if (batch) batch->preprocesses.push_back({P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot});
	else CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}
int CudaRasterizer::BatchContext::finish(
	std::function<char*(size_t)> scratchBuffer,
	std::function<char*(size_t)> binningBuffer)
{
	if (points == 0) return 0;
	size_t scan_size = 0;
	uint32_t *counts_device = nullptr, *offsets = nullptr;
	cub::DeviceScan::InclusiveSum(nullptr, scan_size, counts_device, offsets, points, at::cuda::getCurrentCUDAStream());
	char* scratch = scratchBuffer(size_t(points) * 2 * sizeof(uint32_t) + scan_size + size_t(tiles) * sizeof(uint2) + cameras.size()*sizeof(FORWARD::BatchCamera) + 640);
	obtain(scratch, counts_device, points, 128);
	obtain(scratch, offsets, points, 128);
	char* scanning_space;
	obtain(scratch, scanning_space, scan_size, 128);
	uint2* ranges;
	obtain(scratch, ranges, tiles, 128);
    FORWARD::BatchCamera* camera_device;
    obtain(scratch,camera_device,cameras.size(),128);
    if (fused) {
        CHECK_CUDA(cudaMemcpyAsync(camera_device,cameras.data(),cameras.size()*sizeof(FORWARD::BatchCamera),cudaMemcpyHostToDevice,at::cuda::getCurrentCUDAStream()),false);
        FORWARD::preprocessBatch(camera_device,cameras.size(),cameras.back().point_end,counts_device);
    }
	int base = 0;
	for (auto count : counts) {
		CHECK_CUDA(cudaMemcpyAsync(counts_device + base, count.second,
			count.first * sizeof(uint32_t), cudaMemcpyDeviceToDevice, at::cuda::getCurrentCUDAStream()), false);
		base += count.first;
	}
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(scanning_space, scan_size,
		counts_device, offsets, points, at::cuda::getCurrentCUDAStream()), false);
	int rendered = 0;
	// Exactly one allocation-count readback/synchronization for the entire batch.
	CHECK_CUDA(cudaMemcpyAsync(&rendered, offsets + points - 1,
		sizeof(int), cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream()), false);
	CHECK_CUDA(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()), false);
	char* binning = binningBuffer(required<BinningState>(rendered));
	BinningState sorted = BinningState::fromChunk(binning, rendered);
    if(fused) duplicateForwardBatchCUDA<<<cameras.back().point_end,256,0,at::cuda::getCurrentCUDAStream()>>>(camera_device,cameras.size(),offsets,sorted.point_list_keys_unsorted,sorted.point_list_unsorted);
	for (auto& launch : duplicate) launch(offsets, sorted);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(sorted.list_sorting_space, sorted.sorting_size,
		sorted.point_list_keys_unsorted, sorted.point_list_keys,
		sorted.point_list_unsorted, sorted.point_list, rendered,
		0, 32 + getHigherMsb(tiles), at::cuda::getCurrentCUDAStream()), false);
	CHECK_CUDA(cudaMemsetAsync(ranges, 0, size_t(tiles) * sizeof(uint2), at::cuda::getCurrentCUDAStream()), false);
	if (rendered > 0)
		identifyTileRanges<<<(rendered + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(rendered, sorted.point_list_keys, ranges);
    if(fused) FORWARD::renderBatch(camera_device,cameras.size(),tiles,ranges,sorted.point_list);
	for (auto& launch : render) launch(ranges, sorted);
    CHECK_CUDA(cudaGetLastError(),false);
	return rendered;
}
