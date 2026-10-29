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
__device__ void camera_values(const float* xyz,const float* n,const float* m,const float* center,float* nc,float* pc,float& orientation,float& signed_distance){
 float delta[3]={sub(center[0],xyz[0]),sub(center[1],xyz[1]),sub(center[2],xyz[2])};orientation=dot3(n,delta)<0?-1.f:1.f;
 for(int j=0;j<3;++j){float a=0,b=0;for(int k=0;k<3;++k){a=fmaf(mul(orientation,n[k]),m[k*4+j],a);b=fmaf(xyz[k],m[k*4+j],b);}nc[j]=a;pc[j]=add(b,m[12+j]);}
 signed_distance=dot3(nc,pc);
}

struct MapTask { const float *xyz,*normal,*view,*center,*grad; const bool* mask; float *out,*dx,*dn; int64_t rows; };
struct MapTasks { MapTask items[8]; };
__global__ void batchMapsForward(MapTasks tasks){
 auto t=tasks.items[blockIdx.y];int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=t.rows)return;
 auto xyz=t.xyz,n=t.normal,m=t.view,c=t.center;auto out=t.out;
 float nc[3],pc[3],orientation,d;camera_values(xyz+i*3,n+i*3,m,c,nc,pc,orientation,d);
 for(int j=0;j<3;++j)out[i*5+j]=nc[j];out[i*5+3]=1.f;out[i*5+4]=fabsf(d);
}
__global__ void batchMapsBackward(MapTasks tasks){
 auto t=tasks.items[blockIdx.y];int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=t.rows)return;
 auto xyz=t.xyz,n=t.normal,m=t.view,c=t.center,grad=t.grad;auto mask=t.mask;auto dx=t.dx+i*3,dn=t.dn+i*3;
 if(mask&&!mask[i]){for(int j=0;j<3;++j)dx[j]=dn[j]=0.f;return;}
 float nc[3],pc[3],orientation,d;camera_values(xyz+i*3,n+i*3,m,c,nc,pc,orientation,d);grad+=i*5;
 float g=mul(grad[4],float((d>0)-(d<0))),gn[3],gp[3];
 for(int j=0;j<3;++j){gn[j]=add(grad[j],mul(g,pc[j]));gp[j]=mul(g,nc[j]);}
 for(int j=0;j<3;++j){float a=0,b=0;for(int k=0;k<3;++k){a=fmaf(gn[k],m[j*4+k],a);b=fmaf(gp[k],m[j*4+k],b);}dn[j]=mul(orientation,a);dx[j]=b;}
}
struct CropTask { const float* grad; float* out; int64_t C,H,W,y,x,h,w,s0,s1,s2; };
struct CropTasks {CropTask items[32];};
__global__ void batchCropBackward(CropTasks tasks){
 auto t=tasks.items[blockIdx.y];
 for(int64_t j=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;j<t.C*t.H*t.W;j+=int64_t(gridDim.x)*blockDim.x){
  int64_t x=j%t.W-t.x,y=(j/t.W)%t.H-t.y,c=j/(t.H*t.W);
  t.out[j]=(x>=0&&y>=0&&x<t.w&&y<t.h)?t.grad[c*t.s0+y*t.s1+x*t.s2]:0.f;
 }
}
void check(const torch::Tensor& t,c10::Device d,torch::ScalarType ty){TORCH_CHECK(t.device()==d&&t.scalar_type()==ty,"device/dtype mismatch");}
void mapCheck(const std::vector<torch::Tensor>& v,size_t k){
 const auto& x=v[k];TORCH_CHECK(x.is_cuda()&&x.dim()==2&&x.size(1)==3,"xyz shape");
 for(int j=0;j<4;++j){check(v[k+j],x.device(),torch::kFloat32);TORCH_CHECK(v[k+j].is_contiguous(),"map input contiguous required");}
 TORCH_CHECK(v[k+1].sizes()==x.sizes()&&v[k+2].dim()==2&&v[k+2].size(0)==4&&v[k+2].size(1)==4&&v[k+3].numel()==3,"map shape mismatch");
 check(v[k+4],x.device(),torch::kBool);TORCH_CHECK(v[k+4].is_contiguous()&&(v[k+4].numel()==0||v[k+4].numel()==x.size(0)),"map mask mismatch");
}
}
std::vector<torch::Tensor> maps_forward(std::vector<torch::Tensor> v){
 TORCH_CHECK(!v.empty()&&v.size()%5==0&&v.size()<=40,"1..8 views required");
 const c10::cuda::CUDAGuard guard(v[0].device());MapTasks tasks{};int64_t largest=0;std::vector<torch::Tensor> out;
 for(size_t k=0;k<v.size();k+=5){mapCheck(v,k);check(v[k],v[0].device(),torch::kFloat32);auto x=v[k];auto o=torch::empty({x.size(0),5},x.options());
  tasks.items[k/5]={x.data_ptr<float>(),v[k+1].data_ptr<float>(),v[k+2].data_ptr<float>(),v[k+3].data_ptr<float>(),nullptr,nullptr,o.data_ptr<float>(),nullptr,nullptr,x.size(0)};
  largest=std::max(largest,x.size(0));out.push_back(o);
 }
 if(largest)batchMapsForward<<<dim3((largest+255)/256,out.size()),256,0,at::cuda::getCurrentCUDAStream()>>>(tasks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> maps_backward(std::vector<torch::Tensor> v,std::vector<torch::Tensor> grads){
 TORCH_CHECK(!v.empty()&&v.size()==grads.size()*5&&grads.size()<=8,"backward view count mismatch");
 const c10::cuda::CUDAGuard guard(v[0].device());MapTasks tasks{};int64_t largest=0;std::vector<torch::Tensor> out;
 for(size_t k=0;k<v.size();k+=5){mapCheck(v,k);check(v[k],v[0].device(),torch::kFloat32);auto x=v[k];auto g=grads[k/5];
  check(g,x.device(),torch::kFloat32);TORCH_CHECK(g.is_contiguous()&&g.dim()==2&&g.size(0)==x.size(0)&&g.size(1)==5,"map grad mismatch");
  auto dx=torch::empty_like(x),dn=torch::empty_like(v[k+1]);
  tasks.items[k/5]={x.data_ptr<float>(),v[k+1].data_ptr<float>(),v[k+2].data_ptr<float>(),v[k+3].data_ptr<float>(),g.data_ptr<float>(),v[k+4].numel()?v[k+4].data_ptr<bool>():nullptr,nullptr,dx.data_ptr<float>(),dn.data_ptr<float>(),x.size(0)};
  largest=std::max(largest,x.size(0));out.push_back(dx);out.push_back(dn);
 }
 if(largest)batchMapsBackward<<<dim3((largest+255)/256,grads.size()),256,0,at::cuda::getCurrentCUDAStream()>>>(tasks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> crop_backward(std::vector<torch::Tensor> grads,std::vector<std::vector<int64_t>> sizes){
 TORCH_CHECK(!grads.empty()&&grads.size()==sizes.size()&&grads.size()<=32,"1..32 crop gradients required");
 const c10::cuda::CUDAGuard guard(grads[0].device());CropTasks tasks{};int64_t largest=0;std::vector<torch::Tensor> out;
 for(size_t k=0;k<grads.size();++k){auto g=grads[k];auto s=sizes[k];check(g,grads[0].device(),torch::kFloat32);
  TORCH_CHECK(s.size()==5&&g.dim()==3,"crop dimensions");int64_t C=s[0],H=s[1],W=s[2],y=s[3],x=s[4],h=g.size(1),w=g.size(2);
  TORCH_CHECK(g.size(0)==C&&y>=0&&x>=0&&y+h<=H&&x+w<=W,"crop bounds");auto o=torch::empty({C,H,W},g.options());
  tasks.items[k]={g.data_ptr<float>(),o.data_ptr<float>(),C,H,W,y,x,h,w,g.stride(0),g.stride(1),g.stride(2)};
  largest=std::max(largest,o.numel());out.push_back(o);
 }
 if(largest)batchCropBackward<<<dim3(std::min<int64_t>((largest+255)/256,256),out.size()),256,0,at::cuda::getCurrentCUDAStream()>>>(tasks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
