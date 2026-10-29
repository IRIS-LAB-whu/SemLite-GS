from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name='pgsr_compact_batch', ext_modules=[CUDAExtension('pgsr_compact_batch', ['compact_batch.cu','view_batch.cu'])], cmdclass={'build_ext': BuildExtension})
