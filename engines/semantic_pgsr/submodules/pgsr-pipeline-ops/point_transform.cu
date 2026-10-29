// Three-coordinate transforms avoid padding large point queues into GEMM tiles.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
__global__ void pointMatmul(const float* points,const float* m,float* out,int n,int s0,int s1,int s2){
 int i=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(i>=n)return;
 float p[3]={points[(b*n+i)*3],points[(b*n+i)*3+1],points[(b*n+i)*3+2]};
 #pragma unroll
 for(int j=0;j<3;++j){float v=0;
  #pragma unroll
  for(int k=0;k<3;++k)v=fmaf(p[k],m[b*s0+k*s1+j*s2],v);
  out[(b*n+i)*3+j]=v;
 }
}
torch::Tensor PointMatmul(torch::Tensor points,torch::Tensor matrix,bool transpose){
 TORCH_CHECK(points.is_cuda()&&points.scalar_type()==torch::kFloat32&&points.is_contiguous()&&points.dim()==3&&points.size(2)==3,"points must be CUDA float32 B N 3");
 TORCH_CHECK(matrix.device()==points.device()&&matrix.scalar_type()==points.scalar_type()&&matrix.dim()==3&&matrix.size(0)==points.size(0)&&matrix.size(1)==3&&matrix.size(2)==3,"matrix must be matching B 3 3");
 const c10::cuda::CUDAGuard guard(points.device());auto out=torch::empty_like(points);
 if(points.numel())pointMatmul<<<dim3((points.size(1)+255)/256,points.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(points.data_ptr<float>(),matrix.data_ptr<float>(),out.data_ptr<float>(),points.size(1),matrix.stride(0),matrix.stride(transpose?2:1),matrix.stride(transpose?1:2));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

// Preserve the two original rotations and rounding order; fuse their launches
// and camera translations without pre-multiplying calibration matrices.
__global__ void rigidChain(const float* p,const float* a,const float* b,float* out,int n,bool backward) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,c=blockIdx.y;if(i>=n)return;
    a+=c*16;b+=c*16;p+=(c*n+i)*3;out+=(c*n+i)*3;
    float x[3],y[3];
    for(int k=0;k<3;++k)x[k]=backward?p[k]:__fsub_rn(p[k],a[12+k]);
    for(int j=0;j<3;++j){float v=0;for(int k=0;k<3;++k)v=fmaf(x[k],backward?b[j*4+k]:a[j*4+k],v);y[j]=v;}
    for(int j=0;j<3;++j){float v=0;for(int k=0;k<3;++k)v=fmaf(y[k],backward?a[k*4+j]:b[k*4+j],v);out[j]=backward?v:__fadd_rn(v,b[12+j]);}
}
torch::Tensor RigidChain(torch::Tensor points,torch::Tensor first,torch::Tensor second,bool backward) {
    TORCH_CHECK(points.is_cuda()&&points.scalar_type()==torch::kFloat32&&points.is_contiguous()&&points.dim()==3&&points.size(2)==3,"points must be CUDA float32 B N 3");
    for(const auto& m:{first,second})TORCH_CHECK(m.device()==points.device()&&m.scalar_type()==points.scalar_type()&&m.is_contiguous()&&m.dim()==3&&m.size(0)==points.size(0)&&m.size(1)==4&&m.size(2)==4,"poses must be matching contiguous B 4 4");
    const c10::cuda::CUDAGuard guard(points.device());auto out=torch::empty_like(points);
    if(points.numel())rigidChain<<<dim3((points.size(1)+255)/256,points.size(0)),256,0,at::cuda::getCurrentCUDAStream()>>>(points.data_ptr<float>(),first.data_ptr<float>(),second.data_ptr<float>(),out.data_ptr<float>(),points.size(1),backward);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
