#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    void preprocess(int P, int D, int M,
        const float* orig_points,
        const glm::vec3* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* shs,
        bool* clamped,
        const float* cov3D_precomp,
        const float* colors_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* cam_pos,
        const int W, int H,
        const float focal_x, float focal_y,
        const float tan_fovx, float tan_fovy,
        int* radii,
        float2* points_xy_image,
        float* depths,
        float* cov3Ds,
        float* colors,
        float4* conic_opacity,
        const dim3 grid,
        uint32_t* tiles_touched,
        bool prefiltered,
        bool antialiasing);

    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* points_xy_image,
        const float* features,
        const float4* conic_opacity,
        float* final_T,
        uint32_t* n_contrib,
        const float* bg_color,
        float* out_color,
	        float* depths,
	        float* depth,
	        const int* instance_ids,
	        const float* instance_confidences,
	        int instance_none,
        int* out_instance,
        float alpha_thresh,
        float building_abs_thresh,
        float building_ratio_thresh,
        float winner_ratio_thresh,
        float semantic_alpha_min,
        int* out_winner,
        float* out_semantic_stats);
}

#endif
