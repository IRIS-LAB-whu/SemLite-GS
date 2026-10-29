#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void indexedAdam(float* p,float* m,float* v,const int64_t* ids,const float* grad,
 int64_t count,int64_t cols,float weight1,float beta2,float weight2,float correction,float step_size,float eps) {
 int64_t j=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(j>=count)return;
 int64_t i=ids[j/cols]*cols+j%cols;float g=grad[j];
 // Match lerp_ and mul_().addcmul_() rounding of the PyTorch reference.
 float mi=m[i]+(g-m[i])*weight1,vi=beta2*v[i];vi+= weight2*g*g;
 p[i]+=step_size*mi/(sqrtf(vi)/correction+eps);m[i]=mi;v[i]=vi;
}
void IndexedAdam(torch::Tensor p,torch::Tensor m,torch::Tensor v,torch::Tensor ids,torch::Tensor grad,
 double beta1,double beta2,double correction,double step_size,double eps) {
 TORCH_CHECK(p.is_cuda()&&p.scalar_type()==torch::kFloat32&&p.is_contiguous()&&p.dim()>=1,"p must be contiguous CUDA float32");
 for(auto t:{m,v,grad})TORCH_CHECK(t.device()==p.device()&&t.scalar_type()==p.scalar_type()&&t.is_contiguous(),"Adam float input mismatch");
 TORCH_CHECK(m.sizes()==p.sizes()&&v.sizes()==p.sizes(),"moment shape mismatch");
 TORCH_CHECK(ids.device()==p.device()&&ids.scalar_type()==torch::kInt64&&ids.is_contiguous()&&ids.dim()==1,"indices must be CUDA int64 vector");
 TORCH_CHECK(p.size(0)>0,"empty population");
 int64_t cols=p.numel()/p.size(0);TORCH_CHECK(grad.numel()==ids.numel()*cols,"gradient shape mismatch");
 const c10::cuda::CUDAGuard guard(p.device());
 if(grad.numel())indexedAdam<<<(grad.numel()+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),m.data_ptr<float>(),v.data_ptr<float>(),ids.data_ptr<int64_t>(),grad.data_ptr<float>(),grad.numel(),cols,1.-beta1,beta2,1.-beta2,correction,step_size,eps);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 // Native writes must invalidate position-dependent context selections.
 p.unsafeGetTensorImpl()->bump_version();
 m.unsafeGetTensorImpl()->bump_version();
 v.unsafeGetTensorImpl()->bump_version();
}
