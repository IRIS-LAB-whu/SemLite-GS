// PGSR packed pixel losses. Separate normalization preserves masked means.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void l1Forward(const float* p, const float* t, const float* w,
                         const float* denom, float* out, int C, int N) {
    int b=blockIdx.y, j=blockIdx.x*blockDim.x+threadIdx.x;
    float v=0;
    if(j<C*N) v=fabsf(p[b*C*N+j]-t[b*C*N+j])*w[b*N+j%N]/fmaxf(denom[b],1.f);
    for(int d=16;d;d/=2) v+=__shfl_down_sync(0xffffffff,v,d);
    __shared__ float sums[8];
    if((threadIdx.x&31)==0) sums[threadIdx.x/32]=v;
    __syncthreads();
    if(threadIdx.x<32){v=threadIdx.x<8?sums[threadIdx.x]:0;for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);if(threadIdx.x==0)atomicAdd(out+b,v);}
}
__global__ void l1Backward(const float* p,const float* t,const float* w,const float* denom,
                          const float* go,float* gp,int C,int N) {
    int b=blockIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=C*N)return;
    int i=b*C*N+j;float d=p[i]-t[i];
    gp[i]=((d>0)-(d<0))*w[b*N+j%N]*go[b]/fmaxf(denom[b],1.f);
}
void check(const torch::Tensor& p,const torch::Tensor& t,const torch::Tensor& w,const torch::Tensor& denom) {
    TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kFloat32 && p.dim()==3 && p.is_contiguous(),"pred must be contiguous CUDA float32 [B,C,N]");
    TORCH_CHECK(t.sizes()==p.sizes() && t.device()==p.device() && t.scalar_type()==p.scalar_type() && t.is_contiguous(),"target mismatch");
    TORCH_CHECK(w.device()==p.device() && w.scalar_type()==p.scalar_type() && w.is_contiguous() && w.dim()==2 && w.size(0)==p.size(0) && w.size(1)==p.size(2),"weight mismatch");
    TORCH_CHECK(denom.device()==p.device() && denom.scalar_type()==p.scalar_type() && denom.is_contiguous() && denom.numel()==p.size(0),"denominator mismatch");
    TORCH_CHECK(p.size(0)>0 && p.size(1)>0 && p.size(2)>0,"empty pixel batch");
}
torch::Tensor PackedL1Forward(torch::Tensor p,torch::Tensor t,torch::Tensor w,torch::Tensor denom) {
    check(p,t,w,denom);const c10::cuda::CUDAGuard guard(p.device());auto out=torch::zeros({p.size(0)},p.options());
    l1Forward<<<dim3((p.size(1)*p.size(2)+255)/256,p.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),t.data_ptr<float>(),w.data_ptr<float>(),denom.data_ptr<float>(),out.data_ptr<float>(),p.size(1),p.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor PackedL1Backward(torch::Tensor p,torch::Tensor t,torch::Tensor w,torch::Tensor denom,torch::Tensor go) {
    check(p,t,w,denom);TORCH_CHECK(go.is_contiguous() && go.device()==p.device() && go.scalar_type()==p.scalar_type() && go.numel()==p.size(0),"gradient mismatch");
    const c10::cuda::CUDAGuard guard(p.device());auto gp=torch::empty_like(p);
    l1Backward<<<dim3((p.size(1)*p.size(2)+255)/256,p.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),t.data_ptr<float>(),w.data_ptr<float>(),denom.data_ptr<float>(),go.data_ptr<float>(),gp.data_ptr<float>(),p.size(1),p.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();return gp;
}
