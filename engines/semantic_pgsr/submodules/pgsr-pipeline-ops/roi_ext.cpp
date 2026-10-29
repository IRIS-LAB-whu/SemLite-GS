#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> RaggedPhoto(std::vector<torch::Tensor>,std::vector<std::vector<int64_t>>,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,bool,bool,bool,torch::Tensor,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("photo",&RaggedPhoto);}
