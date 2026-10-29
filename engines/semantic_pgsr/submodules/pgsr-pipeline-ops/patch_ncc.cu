// Fused PGSR plane-patch NCC. One block per patch, including its H Jacobian.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
__device__ float fetch(const float* image,int h,int w,int x,int y) {
  return x>=0&&x<w&&y>=0&&y<h ? image[y*w+x] : 0.f;
}
__device__ float sample(const float* image,int h,int w,float x,float y,float& dx,float& dy) {
  // Match align_corners=True's pixel -> normalized -> pixel rounding.
  x=__fmul_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fdiv_rn(__fmul_rn(2.f,x),float(w-1)),-1.f),1.f),.5f),float(w-1));
  y=__fmul_rn(__fmul_rn(__fadd_rn(__fadd_rn(__fdiv_rn(__fmul_rn(2.f,y),float(h-1)),-1.f),1.f),.5f),float(h-1));
  int ix=floorf(x),iy=floorf(y);float u=x-ix,v=y-iy;
  float a=fetch(image,h,w,ix,iy),b=fetch(image,h,w,ix+1,iy);
  float c=fetch(image,h,w,ix,iy+1),d=fetch(image,h,w,ix+1,iy+1);
  dx=(1-v)*(b-a)+v*(d-c);dy=(1-u)*(c-a)+u*(d-b);
  return (1-u)*(1-v)*a+u*(1-v)*b+(1-u)*v*c+u*v*d;
}
__device__ float reduce(float value,float* shared) {
  int t=threadIdx.x;
  shared[t]=value;__syncthreads();
  if(t<32){
    // PyTorch's contiguous short-row reduction: lane-strided input, then warp.
    value=shared[t]+shared[t+32];value+=shared[t+64];value+=shared[t+96];
    for(int d=16;d;d>>=1)value+=__shfl_down_sync(0xffffffff,value,d);
    if(t==0)shared[0]=value;
  }
  __syncthreads();value=shared[0];__syncthreads();return value;
}
__global__ void patchNCC(const float* ref,const float* near,const float* centers,
 const float* homographies,const float* bounds,const bool* valid,float* out,bool* good,float* jac,bool* critical,
 int rh,int rw,int nh,int nw,int k,int radius) {
 int patch=blockIdx.x,b=patch/k,t=threadIdx.x,side=2*radius+1,n=side*side;
 if(!valid[patch]){if(t==0){out[patch]=0;good[patch]=false;critical[patch]=false;}if(t<9)jac[patch*9+t]=0;return;}
 const float* H=homographies+patch*9;const float* bb=bounds+b*4;
 float x=centers[patch*2]+t%side-radius,y=centers[patch*2+1]+t/side-radius;
 float a=0,c=0,dx=0,dy=0,u=0,v=0,z=1,rawz=1;
 bool inside=true;
 if(t<n){
  rawz=__fadd_rn(fmaf(H[7],y,__fmul_rn(H[6],x)),H[8]);z=fabsf(rawz)>1e-8f?rawz:1e-8f;
  u=__fdiv_rn(__fadd_rn(fmaf(H[1],y,__fmul_rn(H[0],x)),H[2]),z);v=__fdiv_rn(__fadd_rn(fmaf(H[4],y,__fmul_rn(H[3],x)),H[5]),z);
  inside=isfinite(u)&&isfinite(v)&&x>=0&&x<=bb[0]&&y>=0&&y<=bb[1]&&u>=0&&u<=bb[2]&&v>=0&&v<=bb[3];
 }
 __shared__ float buf[128];
 float invalid=reduce(t<n&&!inside?1.f:0.f,buf);
 bool pixel_edge=t<n&&isfinite(u)&&isfinite(v)&&(fabsf(u-roundf(u))<.001f||fabsf(v-roundf(v))<.001f);
 bool sensitive=reduce(pixel_edge?1.f:0.f,buf)>0;
 if(t==0)critical[patch]=sensitive;
 if(invalid>0){if(t==0){out[patch]=0;good[patch]=false;}if(t<9)jac[patch*9+t]=0;return;}
 if(t<n){float unused1,unused2;a=sample(ref+b*rh*rw,rh,rw,x,y,unused1,unused2);c=sample(near+b*nh*nw,nh,nw,u,v,dx,dy);}
 float sa=reduce(a,buf),sb=reduce(c,buf),saa=reduce(a*a,buf),sbb=reduce(c*c,buf),sab=reduce(a*c,buf);
 float cov=__fsub_rn(sab,__fdiv_rn(__fmul_rn(sa,sb),float(n))),va=__fsub_rn(saa,__fdiv_rn(__fmul_rn(sa,sa),float(n))),vb=__fsub_rn(sbb,__fdiv_rn(__fmul_rn(sb,sb),float(n))),den=__fadd_rn(__fmul_rn(va,vb),1e-8f);
 float raw=__fsub_rn(1.f,__fdiv_rn(__fmul_rn(cov,cov),den)),score=isnan(raw)?raw:fminf(2.f,fmaxf(0.f,raw));bool accepted=score<.9f;
 if(t==0){out[patch]=score;good[patch]=accepted;critical[patch]=sensitive||fabsf(score-.9f)<.01f||fminf(va,vb)<.001f;}
 float dn=0;
 if(t<n&&accepted&&raw>=0&&raw<=2)dn=-2*cov*(a-sa/n)/den+cov*cov*va*2*(c-sb/n)/(den*den);
 float du=dn*dx/z,dv=dn*dy/z,dz=fabsf(rawz)>1e-8f?-(u*du+v*dv):0;
 for(int j=0;j<9;++j){float component=j/3==0?du:(j/3==1?dv:dz);float p=j%3==0?x:(j%3==1?y:1.f);float sum=reduce(t<n?component*p:0.f,buf);if(t==0)jac[patch*9+j]=sum;}
}
}
std::vector<torch::Tensor> PatchNCCForward(torch::Tensor ref,torch::Tensor near,torch::Tensor centers,torch::Tensor H,torch::Tensor bounds,torch::Tensor valid,int64_t radius) {
 TORCH_CHECK(ref.is_cuda()&&ref.scalar_type()==torch::kFloat32&&ref.is_contiguous()&&ref.dim()==4&&ref.size(1)==1,"ref must be CUDA float32 B1HW");
 for(auto t:{near,centers,H,bounds})TORCH_CHECK(t.device()==ref.device()&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"float input mismatch");
 TORCH_CHECK(near.dim()==4&&near.size(0)==ref.size(0)&&near.size(1)==1,"near shape mismatch");
 TORCH_CHECK(centers.dim()==3&&centers.size(0)==ref.size(0)&&centers.size(2)==2,"centers shape mismatch");
 int b=centers.size(0),k=centers.size(1);
 TORCH_CHECK(H.sizes()==torch::IntArrayRef({b,k,3,3}),"H shape mismatch");
 TORCH_CHECK(bounds.sizes()==torch::IntArrayRef({b,4}),"bounds shape mismatch");
 TORCH_CHECK(valid.device()==ref.device()&&valid.scalar_type()==torch::kBool&&valid.is_contiguous()&&valid.sizes()==torch::IntArrayRef({b,k}),"valid mismatch");
 TORCH_CHECK(radius>=0&&radius<=5&&ref.size(2)>1&&ref.size(3)>1&&near.size(2)>1&&near.size(3)>1,"unsupported patch/image size");
 const c10::cuda::CUDAGuard guard(ref.device());
 auto out=torch::empty({b,k},ref.options());auto good=torch::empty_like(valid);auto jac=torch::empty_like(H);auto critical=torch::empty_like(valid);
 if(b*k)patchNCC<<<b*k,128,0,at::cuda::getCurrentCUDAStream()>>>(ref.data_ptr<float>(),near.data_ptr<float>(),centers.data_ptr<float>(),H.data_ptr<float>(),bounds.data_ptr<float>(),valid.data_ptr<bool>(),out.data_ptr<float>(),good.data_ptr<bool>(),jac.data_ptr<float>(),critical.data_ptr<bool>(),ref.size(2),ref.size(3),near.size(2),near.size(3),k,radius);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,good,jac,critical};
}
