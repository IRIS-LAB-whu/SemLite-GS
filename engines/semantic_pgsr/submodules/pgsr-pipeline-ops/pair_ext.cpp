#include <torch/extension.h>
#include <vector>
torch::Tensor RaggedSampleForward(torch::Tensor,torch::Tensor,torch::Tensor);
torch::Tensor RaggedSamplePixelBackward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> RaggedSampleBackward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> StableHomography(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool);
torch::Tensor StableProject(torch::Tensor,torch::Tensor,torch::Tensor,bool);
std::vector<torch::Tensor> PatchNCCForward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("patch_ncc_forward",&PatchNCCForward);m.def("project",&StableProject);m.def("sample_forward",&RaggedSampleForward);m.def("sample_backward",&RaggedSampleBackward);m.def("sample_pixel_backward",&RaggedSamplePixelBackward);m.def("homography",&StableHomography);}
