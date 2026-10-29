#include <torch/extension.h>
#include <pybind11/stl.h>
#include "pipeline_ops.h"

torch::Tensor PackedL1Forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor PackedL1Backward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

std::vector<torch::Tensor> PatchNCCForward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
void IndexedAdam(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,double,double,double,double);

torch::Tensor PointMatmul(torch::Tensor,torch::Tensor,bool);

torch::Tensor RigidChain(torch::Tensor,torch::Tensor,torch::Tensor,bool);

void CompactStats(std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<int64_t>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<bool>,double);

std::vector<torch::Tensor> AxisForward(torch::Tensor,torch::Tensor);
torch::Tensor AxisBackward(torch::Tensor,torch::Tensor,torch::Tensor);
torch::Tensor CameraMapsForward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> CameraMapsBackward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axis_forward", &AxisForward);
  m.def("axis_backward", &AxisBackward);
  m.def("camera_maps_forward", &CameraMapsForward);
  m.def("camera_maps_backward", &CameraMapsBackward);
  m.def("compact_stats", &CompactStats);
  m.def("rigid_chain", &RigidChain);
  m.def("patch_ncc_forward", &PatchNCCForward);
  m.def("point_matmul", &PointMatmul);
  m.def("indexed_adam", &IndexedAdam);
  m.def("packed_l1_forward", &PackedL1Forward);
  m.def("packed_l1_backward", &PackedL1Backward);
  m.def("packed_plane_depth_correction", &PackedPlaneDepthCorrectionCUDA,
        "Correct packed depth maps against projected plane polygons");
  m.def("fused_packed_tsdf_fusion", &FusedPackedTSDFFusionCUDA,
        "Fuse dense TSDF chunks from packed camera depth maps");
  m.def("packed_label_components", &PackedLabelComponentsCPU,
        "Pack exact CPU label components and boundary neighbor ids",
        pybind11::call_guard<pybind11::gil_scoped_release>());
}
