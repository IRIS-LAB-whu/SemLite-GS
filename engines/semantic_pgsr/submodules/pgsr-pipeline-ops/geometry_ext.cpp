#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> RaggedGeometry(std::vector<torch::Tensor>,std::vector<std::vector<int64_t>>,torch::Tensor,bool);
std::vector<torch::Tensor> RaggedGather(std::vector<torch::Tensor>,std::vector<std::vector<int64_t>>,torch::Tensor,std::vector<torch::Tensor>,bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("geometry",&RaggedGeometry);m.def("gather",&RaggedGather);}
