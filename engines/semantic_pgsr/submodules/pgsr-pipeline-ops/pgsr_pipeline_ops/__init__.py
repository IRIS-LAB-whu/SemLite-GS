#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

"""Auxiliary mesh/depth operations; contains no Gaussian rasterizer."""
import importlib
import torch  # Load libtorch before the extension's shared-library dependencies.

_C = None


def _get_c_module():
    global _C
    if _C is None:
        try:
            _C = importlib.import_module("._C", __name__)
        except ImportError as exc:
            raise RuntimeError(
                "PGSR pipeline operations are not built. Run setup.py build_ext --inplace "
                "in submodules/pgsr-pipeline-ops using the pgsr_parallel Python and CUDA 11.8."
            ) from exc
    return _C


def packed_plane_depth_correction(*args):
    return _get_c_module().packed_plane_depth_correction(*args)


def fused_packed_tsdf_fusion(*args):
    return _get_c_module().fused_packed_tsdf_fusion(*args)


def packed_label_components(*args):
    return _get_c_module().packed_label_components(*args)
