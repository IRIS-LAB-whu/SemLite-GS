// Native 3D quaternion-axis selection and per-camera plane maps.
// Explicit rounding avoids contracting the reference pointwise operations.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
namespace {
__device__ __forceinline__ float add(float a,float b){return __fadd_rn(a,b);}
__device__ __forceinline__ float sub(float a,float b){return __fsub_rn(a,b);}
__device__ __forceinline__ float mul(float a,float b){return __fmul_rn(a,b);}
__device__ __forceinline__ float dot3(const float* a,const float* b){return add(add(mul(a[0],b[0]),mul(a[2],b[2])),mul(a[1],b[1]));}
__device__ void axis_terms(const float* q,int k,float* a){
 const float w=q[0],x=q[1],y=q[2],z=q[3];
 if(k==0){a[0]=-add(mul(y,y),mul(z,z));a[1]=add(mul(x,y),mul(z,w));a[2]=sub(mul(x,z),mul(y,w));}
 else if(k==1){a[0]=sub(mul(x,y),mul(z,w));a[1]=-add(mul(x,x),mul(z,z));a[2]=add(mul(y,z),mul(x,w));}
 else{a[0]=add(mul(x,z),mul(y,w));a[1]=sub(mul(y,z),mul(x,w));a[2]=-add(mul(x,x),mul(y,y));}
}
__device__ float norm2(const float* q){return add(add(mul(q[0],q[0]),mul(q[1],q[1])),add(mul(q[2],q[2]),mul(q[3],q[3])));}
__global__ void axisF(const float* q,const float* s,float* out,int64_t* ids,int64_t n){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;q+=i*4;s+=i*3;out+=i*3;
 int k=0;for(int j=1;j<3;++j)if(!isnan(s[k])&&(isnan(s[j])||s[j]<s[k]))k=j;
 ids[i]=k;float a[3];axis_terms(q,k,a);float t=mul(__frcp_rn(norm2(q)),2.f);
 for(int j=0;j<3;++j)out[j]=add(j==k?1.f:0.f,mul(t,a[j]));
}
__global__ void axisB(const float* q,const int64_t* ids,const float* grad,float* out,int64_t n){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;q+=i*4;grad+=i*3;out+=i*4;
 const float w=q[0],x=q[1],y=q[2],z=q[3],u=grad[0],v=grad[1],h=grad[2];int k=ids[i];
 float a[3],d[4];axis_terms(q,k,a);
 if(k==0){
  d[0]=sub(mul(v,z),mul(h,y));d[1]=add(mul(v,y),mul(h,z));
  d[2]=sub(add(mul(-2*u,y),mul(v,x)),mul(h,w));d[3]=add(add(mul(-2*u,z),mul(v,w)),mul(h,x));
 }else if(k==1){
  d[0]=add(mul(-u,z),mul(h,x));d[1]=add(add(mul(u,y),mul(-2*v,x)),mul(h,w));
  d[2]=add(mul(u,x),mul(h,z));d[3]=add(add(mul(-u,w),mul(-2*v,z)),mul(h,y));
 }else{
  d[0]=sub(mul(u,y),mul(v,x));d[1]=add(sub(mul(u,z),mul(v,w)),mul(-2*h,x));
  d[2]=add(add(mul(u,w),mul(v,z)),mul(-2*h,y));d[3]=add(mul(u,x),mul(v,y));
 }
 float inv=__frcp_rn(norm2(q)),t=mul(2.f,inv),dt=dot3(grad,a),radial=mul(mul(-2.f,t),mul(inv,dt));
 for(int j=0;j<4;++j)out[j]=add(mul(t,d[j]),mul(radial,q[j]));
}
__device__ void camera_values(const float* xyz,const float* n,const float* m,const float* center,float* nc,float* pc,float& orientation,float& signed_distance){
 float delta[3]={sub(center[0],xyz[0]),sub(center[1],xyz[1]),sub(center[2],xyz[2])};orientation=dot3(n,delta)<0?-1.f:1.f;
 for(int j=0;j<3;++j){float a=0,b=0;for(int k=0;k<3;++k){a=fmaf(mul(orientation,n[k]),m[k*4+j],a);b=fmaf(xyz[k],m[k*4+j],b);}nc[j]=a;pc[j]=add(b,m[12+j]);}
 signed_distance=dot3(nc,pc);
}
__global__ void mapsF(const float* xyz,const float* n,const float* m,const float* c,float* out,int64_t count){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
 float nc[3],pc[3],orientation,d;camera_values(xyz+i*3,n+i*3,m,c,nc,pc,orientation,d);
 for(int j=0;j<3;++j)out[i*5+j]=nc[j];out[i*5+3]=1.f;out[i*5+4]=fabsf(d);
}
__global__ void mapsB(const float* xyz,const float* n,const float* m,const float* c,const bool* mask,const float* grad,float* dx,float* dn,int64_t count){
 int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;dx+=i*3;dn+=i*3;
 if(mask&&!mask[i]){for(int j=0;j<3;++j)dx[j]=dn[j]=0.f;return;}
 float nc[3],pc[3],orientation,d;camera_values(xyz+i*3,n+i*3,m,c,nc,pc,orientation,d);grad+=i*5;
 float g=mul(grad[4],float((d>0)-(d<0))),gn[3],gp[3];
 for(int j=0;j<3;++j){gn[j]=add(grad[j],mul(g,pc[j]));gp[j]=mul(g,nc[j]);}
 for(int j=0;j<3;++j){float a=0,b=0;for(int k=0;k<3;++k){a=fmaf(gn[k],m[j*4+k],a);b=fmaf(gp[k],m[j*4+k],b);}dn[j]=mul(orientation,a);dx[j]=b;}
}
void rows(const torch::Tensor& x,int cols){TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat32&&x.is_contiguous()&&x.dim()==2&&x.size(1)==cols,"Expected contiguous CUDA float32 N x ",cols);}
void camera_check(const torch::Tensor& xyz,const torch::Tensor& n,const torch::Tensor& m,const torch::Tensor& c){
 rows(xyz,3);rows(n,3);TORCH_CHECK(n.device()==xyz.device()&&n.size(0)==xyz.size(0),"normal mismatch");
 TORCH_CHECK(m.device()==xyz.device()&&m.scalar_type()==xyz.scalar_type()&&m.is_contiguous()&&m.dim()==2&&m.size(0)==4&&m.size(1)==4,"view must be 4 x 4");
 TORCH_CHECK(c.device()==xyz.device()&&c.scalar_type()==xyz.scalar_type()&&c.is_contiguous()&&c.dim()==1&&c.numel()==3,"camera center must be length 3");
}
}
std::vector<torch::Tensor> AxisForward(torch::Tensor q,torch::Tensor s){
 rows(q,4);rows(s,3);TORCH_CHECK(s.device()==q.device()&&s.size(0)==q.size(0),"scales mismatch");const c10::cuda::CUDAGuard guard(q.device());
 auto out=torch::empty({q.size(0),3},q.options()),ids=torch::empty({q.size(0)},q.options().dtype(torch::kInt64));
 if(q.size(0))axisF<<<(q.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(),s.data_ptr<float>(),out.data_ptr<float>(),ids.data_ptr<int64_t>(),q.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,ids};
}
torch::Tensor AxisBackward(torch::Tensor q,torch::Tensor ids,torch::Tensor g){
 rows(q,4);rows(g,3);TORCH_CHECK(g.device()==q.device()&&g.size(0)==q.size(0)&&ids.device()==q.device()&&ids.is_contiguous()&&ids.scalar_type()==torch::kInt64&&ids.dim()==1&&ids.size(0)==q.size(0),"axis backward mismatch");const c10::cuda::CUDAGuard guard(q.device());auto out=torch::empty_like(q);
 if(q.size(0))axisB<<<(q.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(),ids.data_ptr<int64_t>(),g.data_ptr<float>(),out.data_ptr<float>(),q.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
torch::Tensor CameraMapsForward(torch::Tensor xyz,torch::Tensor n,torch::Tensor m,torch::Tensor c){
 camera_check(xyz,n,m,c);const c10::cuda::CUDAGuard guard(xyz.device());auto out=torch::empty({xyz.size(0),5},xyz.options());
 if(xyz.size(0))mapsF<<<(xyz.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(xyz.data_ptr<float>(),n.data_ptr<float>(),m.data_ptr<float>(),c.data_ptr<float>(),out.data_ptr<float>(),xyz.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> CameraMapsBackward(torch::Tensor xyz,torch::Tensor n,torch::Tensor m,torch::Tensor c,torch::Tensor mask,torch::Tensor g){
 camera_check(xyz,n,m,c);rows(g,5);TORCH_CHECK(g.device()==xyz.device()&&g.size(0)==xyz.size(0),"map gradient mismatch");
 TORCH_CHECK(mask.device()==xyz.device()&&mask.scalar_type()==torch::kBool&&mask.is_contiguous()&&mask.dim()==1&&(mask.numel()==0||mask.numel()==xyz.size(0)),"mask must be empty or N bools");const c10::cuda::CUDAGuard guard(xyz.device());auto dx=torch::empty_like(xyz),dn=torch::empty_like(n);
 if(xyz.size(0))mapsB<<<(xyz.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(xyz.data_ptr<float>(),n.data_ptr<float>(),m.data_ptr<float>(),c.data_ptr<float>(),mask.numel()?mask.data_ptr<bool>():nullptr,g.data_ptr<float>(),dx.data_ptr<float>(),dn.data_ptr<float>(),xyz.size(0));C10_CUDA_KERNEL_LAUNCH_CHECK();return {dx,dn};
}
