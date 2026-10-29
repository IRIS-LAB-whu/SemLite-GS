#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdlib>

// Host-side launch arguments: no device metadata allocation or H2D tensor copy.
constexpr int MAX_TASKS=48;
struct Task {
  const float* src;
  float* dst;
  const int64_t* ids;
  int64_t count, cols, row_stride, col_stride;
};
struct Tasks { Task task[MAX_TASKS]; };
__global__ void gather(Tasks tasks) {
  const Task t=tasks.task[blockIdx.y];
  for(int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
      j<t.count;j+=int64_t(gridDim.x)*blockDim.x)
    t.dst[j]=t.src[t.ids[j/t.cols]*t.row_stride+(j%t.cols)*t.col_stride];
}
void check_tensor(const torch::Tensor& x, c10::Device dev, torch::ScalarType dtype) {
  TORCH_CHECK(x.device()==dev && x.scalar_type()==dtype && x.is_contiguous(),
              "Expected contiguous tensor with matching CUDA device/dtype");
}
std::vector<torch::Tensor> forward(std::vector<torch::Tensor> source,
    std::vector<torch::Tensor> indices, std::vector<torch::Tensor> active) {
  TORCH_CHECK(!source.empty() && source.size()==indices.size() && source.size()==active.size()
    && source.size()<=MAX_TASKS/2,"Expected 1..24 parameter tasks");
  TORCH_CHECK(source[0].is_cuda(),"CUDA required");
  const c10::cuda::CUDAGuard guard(source[0].device());
  Tasks tasks{};std::vector<torch::Tensor> result;int64_t largest=0;
  for(size_t k=0;k<source.size();++k) {
    auto s=source[k];check_tensor(s,source[0].device(),torch::kFloat32);
    check_tensor(indices[k],s.device(),torch::kInt64);check_tensor(active[k],s.device(),torch::kInt64);
    TORCH_CHECK(s.dim()>=2 && indices[k].dim()==1 && active[k].dim()==1,"Invalid shape");
    int64_t cols=1;for(int d=1;d<s.dim();++d)cols*=s.size(d);
    for(auto ids:{indices[k],active[k]}) {
      auto shape=s.sizes().vec();shape[0]=ids.numel();auto out=torch::empty(shape,s.options());
      tasks.task[result.size()]={s.data_ptr<float>(),out.data_ptr<float>(),ids.data_ptr<int64_t>(),out.numel(),cols,cols,1};
      largest=std::max(largest,out.numel());result.push_back(out);
    }
  }
  if(largest)gather<<<dim3(std::min<int64_t>((largest+255)/256,256),result.size()),256,0,at::cuda::getCurrentCUDAStream()>>>(tasks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
std::vector<torch::Tensor> backward(std::vector<torch::Tensor> grads, std::vector<torch::Tensor> local) {
  TORCH_CHECK(!grads.empty()&&grads.size()==local.size()&&grads.size()<=MAX_TASKS,"Invalid backward tasks");
  TORCH_CHECK(grads[0].is_cuda(),"CUDA required");
  const c10::cuda::CUDAGuard guard(grads[0].device());
  Tasks tasks{};std::vector<torch::Tensor> result;int64_t largest=0;
  for(size_t k=0;k<grads.size();++k) {
    auto s=grads[k];
    bool strided=std::getenv("PGSR_COMPACT_STRIDED") && std::string(std::getenv("PGSR_COMPACT_STRIDED"))=="1";
    bool linear=s.dim()>=2;int64_t expected=s.stride(s.dim()-1);
    for(int d=s.dim()-1;d>=1;--d){if(s.size(d)>1&&s.stride(d)!=expected)linear=false;expected*=s.size(d);}
    if(!strided||!linear){grads[k]=s.contiguous();s=grads[k];}
    TORCH_CHECK(s.device()==grads[0].device()&&s.scalar_type()==torch::kFloat32,"gradient device/dtype mismatch");
    check_tensor(local[k],s.device(),torch::kInt64);
    TORCH_CHECK(s.dim()>=2&&local[k].dim()==1,"Invalid gradient shape");
    int64_t cols=1;for(int d=1;d<s.dim();++d)cols*=s.size(d);
    auto shape=s.sizes().vec();shape[0]=local[k].numel();auto out=torch::empty(shape,s.options());
    tasks.task[k]={s.data_ptr<float>(),out.data_ptr<float>(),local[k].data_ptr<int64_t>(),out.numel(),cols,s.stride(0),s.stride(s.dim()-1)};
    largest=std::max(largest,out.numel());result.push_back(out);
  }
  if(largest)gather<<<dim3(std::min<int64_t>((largest+255)/256,256),result.size()),256,0,at::cuda::getCurrentCUDAStream()>>>(tasks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
std::vector<torch::Tensor> crop_backward(std::vector<torch::Tensor>,std::vector<std::vector<int64_t>>);
std::vector<torch::Tensor> maps_forward(std::vector<torch::Tensor>);
std::vector<torch::Tensor> maps_backward(std::vector<torch::Tensor>,std::vector<torch::Tensor>);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("crop_backward",&crop_backward);m.def("maps_forward",&maps_forward);m.def("maps_backward",&maps_backward);m.def("forward",&forward);m.def("backward",&backward);}
