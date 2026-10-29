from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="pgsr-pipeline-ops",
    version="0.1.0",
    packages=["pgsr_pipeline_ops"],
    ext_modules=[CUDAExtension(
        "pgsr_pipeline_ops._C",
        sources=["ext.cpp", "roi_pointwise.cu", "patch_ncc.cu", "point_transform.cu", "indexed_adam.cu", "compact_stats.cu", "render_inputs.cu", "plane_depth_correction.cu", "tsdf_fusion.cu", "label_components.cpp"],
    ), CUDAExtension(
        "pgsr_pipeline_ops._pair",
        sources=["pair_ext.cpp", "pair_stable.cu", "pair_patch_ncc.cu"],
    ), CUDAExtension(
        "pgsr_pipeline_ops._geometry",
        sources=["geometry_ext.cpp", "geometry_ragged.cu"],
        extra_compile_args={"nvcc": ["--fmad=false"]},
    ), CUDAExtension(
        "pgsr_pipeline_ops._roi",
        sources=["roi_ext.cpp", "roi_ragged.cu"],
        extra_compile_args={"nvcc": ["--fmad=false"]},
    )],
    cmdclass={"build_ext": BuildExtension},
)
