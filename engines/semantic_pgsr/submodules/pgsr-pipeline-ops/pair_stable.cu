// Layout-invariant FP32 sampling and 3x3 plane homographies.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
namespace {
__device__ float pixel_round(float p,int length) {
 // Reproduce scalar PyTorch normalization, using this image's extent.
 float s=float(max(length-1,1));
 float g=__fsub_rn(__fmul_rn(__fmul_rn(2.f,p),__fdiv_rn(1.f,s)),1.f);
 return __fmul_rn(__fmul_rn(__fadd_rn(g,1.f),.5f),float(length-1));
}
__device__ bool inside(int x,int y,int w,int h){return x>=0&&x<w&&y>=0&&y<h;}
template<bool ImageGradient>
__global__ void sample_kernel(const float* im,const float* pixels,const float* bounds,const float* grad,
 float* out,float* gi,float* gp,int channels,int height,int width,int count,bool backward) {
 int i=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(i>=count)return;
 int w=int(bounds[b*2])+1,h=int(bounds[b*2+1])+1;
 float x=pixel_round(pixels[(b*count+i)*2],w),y=pixel_round(pixels[(b*count+i)*2+1],h);
 if(!isfinite(x)||!isfinite(y)||fabsf(x)>2e9f||fabsf(y)>2e9f){x=-100.f;y=-100.f;}
 int ix=int(floorf(x)),iy=int(floorf(y));
 float dx[4]={ix+1-x,x-ix,ix+1-x,x-ix};
 float dy[4]={iy+1-y,iy+1-y,y-iy,y-iy};
 int xs[4]={ix,ix+1,ix,ix+1},ys[4]={iy,iy,iy+1,iy+1};
 float gx=0,gy=0;
 for(int c=0;c<channels;++c){
  float vals[4]={0,0,0,0},ws[4]={0,0,0,0};
  float value=0,g=backward?grad[(b*count+i)*channels+c]:0;
  for(int j=0;j<4;++j)if(inside(xs[j],ys[j],w,h)){
   long at=((long(b)*channels+c)*height+ys[j])*width+xs[j];
   float v=im[at],weight=__fmul_rn(dx[j],dy[j]);
   if(backward){
    if (ImageGradient) atomicAdd(gi+at,__fmul_rn(weight,g));
    // Match grid_sample's corner accumulation and signs.
    float vx=__fmul_rn(__fmul_rn(v,dy[j]),g),vy=__fmul_rn(__fmul_rn(v,dx[j]),g);
    gx=__fadd_rn(gx,(j%2)?vx:-vx);gy=__fadd_rn(gy,(j>=2)?vy:-vy);
   }else {vals[j]=v;ws[j]=weight;}
  }
  if(!backward){
   value=fmaf(vals[0],ws[0],__fmul_rn(vals[1],ws[1]));
   value=fmaf(vals[2],ws[2],value);
   out[(b*count+i)*channels+c]=__fadd_rn(value,__fmul_rn(vals[3],ws[3]));
  }
 }
 if(backward){
  // grid_sample unnormalize derivative, then reverse pixel normalization.
  gx=__fmul_rn(__fmul_rn(__fmul_rn(gx,float(w-1)*.5f),__fdiv_rn(1.f,float(max(w-1,1)))),2.f);
  gy=__fmul_rn(__fmul_rn(__fmul_rn(gy,float(h-1)*.5f),__fdiv_rn(1.f,float(max(h-1,1)))),2.f);
  gp[(b*count+i)*2]=gx;gp[(b*count+i)*2+1]=gy;
 }
}
__device__ void mm(const float* a,const float* b,float* c,bool ta=false,bool tb=false,bool fused=false){
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){float v=0;for(int k=0;k<3;++k)v=fused?fmaf(a[ta?k*3+i:i*3+k],b[tb?j*3+k:k*3+j],v):__fadd_rn(v,__fmul_rn(a[ta?k*3+i:i*3+k],b[tb?j*3+k:k*3+j]));c[i*3+j]=v;}
}
__global__ void homography_kernel(const float* normals,const float* distance,const float* views,const float* near,
 const float* kn,const float* ki,const float* grad,float* out,float* gn,float* gd,int count,bool backward){
 int p=blockIdx.x*blockDim.x+threadIdx.x,b=blockIdx.y;if(p>=count)return;
 int id=b*count+p;float r[9],t[3],q[9],temp[9],value[9];
 for(int j=0;j<9;++j)r[j]=views[b*9+j];for(int j=0;j<3;++j)t[j]=near[b*3+j];
 float d=fmaxf(distance[id],1e-8f);const float* n=normals+id*3;
 if(!backward){
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)q[i*3+j]=__fsub_rn(r[i*3+j],__fdiv_rn(__fmul_rn(t[i],n[j]),d));
  mm(kn+b*9,q,temp);mm(temp,ki+b*9,value,false,false,true);
  for(int i=0;i<9;++i)out[id*9+i]=value[i];
 }else{
  mm(grad+id*9,ki+b*9,temp,false,true);mm(kn+b*9,temp,q,true,false);
  float dd[9];
  for(int j=0;j<3;++j){float v=0;for(int i=0;i<3;++i){
   float g=__fdiv_rn(-q[i*3+j],d);v=__fadd_rn(v,__fmul_rn(g,t[i]));
   dd[i*3+j]=__fmul_rn(q[i*3+j],__fdiv_rn(__fdiv_rn(__fmul_rn(t[i],n[j]),d),d));
  }gn[id*3+j]=v;}
  // Fixed tree reduction, independent of B or padded K.
  dd[0]=__fadd_rn(dd[0],dd[8]);for(int stride=4;stride;stride/=2)for(int i=0;i<stride;++i)dd[i]=__fadd_rn(dd[i],dd[i+stride]);
  gd[id]=distance[id]>=1e-8f?dd[0]:0.f;
 }
}
void check(const torch::Tensor& x,const torch::Tensor& ref){TORCH_CHECK(x.is_cuda()&&x.device()==ref.device()&&x.scalar_type()==torch::kFloat32&&x.is_contiguous(),"expected contiguous CUDA float32 inputs on same device");}
}
torch::Tensor RaggedSampleForward(torch::Tensor im,torch::Tensor pixels,torch::Tensor bounds){
 check(im,im);check(pixels,im);check(bounds,im);
 TORCH_CHECK(im.dim()==4&&pixels.dim()==3&&pixels.size(0)==im.size(0)&&pixels.size(2)==2&&bounds.sizes()==torch::IntArrayRef({im.size(0),2}),"invalid sampling shapes");
 c10::cuda::CUDAGuard guard(im.device());int b=im.size(0),n=pixels.size(1);auto out=torch::empty({b,n,im.size(1)},im.options());
 if(b&&n)sample_kernel<true><<<dim3((n+255)/256,b),256,0,at::cuda::getCurrentCUDAStream()>>>(im.data_ptr<float>(),pixels.data_ptr<float>(),bounds.data_ptr<float>(),nullptr,out.data_ptr<float>(),nullptr,nullptr,im.size(1),im.size(2),im.size(3),n,false);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<torch::Tensor> RaggedSampleBackward(torch::Tensor im,torch::Tensor pixels,torch::Tensor bounds,torch::Tensor grad){
 check(im,im);check(pixels,im);check(bounds,im);check(grad,im);
 c10::cuda::CUDAGuard guard(im.device());auto gi=torch::zeros_like(im),gp=torch::empty_like(pixels);int b=im.size(0),n=pixels.size(1);
 TORCH_CHECK(grad.sizes()==torch::IntArrayRef({b,n,im.size(1)}),"invalid sampling gradient");
 if(b&&n)sample_kernel<true><<<dim3((n+255)/256,b),256,0,at::cuda::getCurrentCUDAStream()>>>(im.data_ptr<float>(),pixels.data_ptr<float>(),bounds.data_ptr<float>(),grad.data_ptr<float>(),nullptr,gi.data_ptr<float>(),gp.data_ptr<float>(),im.size(1),im.size(2),im.size(3),n,true);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {gi,gp};
}
torch::Tensor RaggedSamplePixelBackward(torch::Tensor im,torch::Tensor pixels,torch::Tensor bounds,torch::Tensor grad){
 check(im,im);check(pixels,im);check(bounds,im);check(grad,im);
 TORCH_CHECK(im.dim()==4&&pixels.dim()==3&&pixels.size(0)==im.size(0)&&pixels.size(2)==2&&bounds.sizes()==torch::IntArrayRef({im.size(0),2}),"invalid sampling shapes");
 c10::cuda::CUDAGuard guard(im.device());auto gp=torch::empty_like(pixels);int b=im.size(0),n=pixels.size(1);
 TORCH_CHECK(grad.sizes()==torch::IntArrayRef({b,n,im.size(1)}),"invalid sampling gradient");
 if(b&&n)sample_kernel<false><<<dim3((n+255)/256,b),256,0,at::cuda::getCurrentCUDAStream()>>>(im.data_ptr<float>(),pixels.data_ptr<float>(),bounds.data_ptr<float>(),grad.data_ptr<float>(),nullptr,nullptr,gp.data_ptr<float>(),im.size(1),im.size(2),im.size(3),n,true);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return gp;
}
std::vector<torch::Tensor> StableHomography(torch::Tensor normals,torch::Tensor distance,torch::Tensor view,torch::Tensor near,torch::Tensor kn,torch::Tensor ki,torch::Tensor grad,bool backward){
 for(auto t:{normals,distance,view,near,kn,ki})check(t,normals);
 int b=normals.size(0),n=normals.size(1);
 TORCH_CHECK(normals.dim()==3&&normals.size(2)==3&&distance.sizes()==torch::IntArrayRef({b,n})&&view.sizes()==torch::IntArrayRef({b,3,3})&&near.sizes()==torch::IntArrayRef({b,3})&&kn.sizes()==torch::IntArrayRef({b,3,3})&&ki.sizes()==kn.sizes(),"invalid homography inputs");
 c10::cuda::CUDAGuard guard(normals.device());auto out=torch::empty({b,n,3,3},normals.options());auto gn=torch::empty_like(normals),gd=torch::empty_like(distance);
 if(backward){check(grad,normals);TORCH_CHECK(grad.sizes()==out.sizes(),"invalid homography gradient");}
 if(b&&n)homography_kernel<<<dim3((n+127)/128,b),128,0,at::cuda::getCurrentCUDAStream()>>>(normals.data_ptr<float>(),distance.data_ptr<float>(),view.data_ptr<float>(),near.data_ptr<float>(),kn.data_ptr<float>(),ki.data_ptr<float>(),backward?grad.data_ptr<float>():nullptr,out.data_ptr<float>(),gn.data_ptr<float>(),gd.data_ptr<float>(),n,backward);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return backward?std::vector<torch::Tensor>{gn,gd}:std::vector<torch::Tensor>{out};
}

namespace {
__global__ void project_kernel(const float* pixels,const float* H,const float* grad,float* out,float* gh,int k,int count,bool backward){
 int patch=blockIdx.x,t=threadIdx.x;
 if(!backward){
  if(t>=count)return;float x=pixels[(patch*count+t)*2],y=pixels[(patch*count+t)*2+1];
  for(int j=0;j<3;++j)out[(patch*count+t)*3+j]=__fadd_rn(fmaf(H[patch*9+j*3+1],y,__fmul_rn(H[patch*9+j*3],x)),H[patch*9+j*3+2]);
 }else if(t<9){
  float value=0;int row=t/3,col=t%3;
  for(int i=0;i<count;++i){float x=col==2?1.f:pixels[(patch*count+i)*2+col];value=fmaf(grad[(patch*count+i)*3+row],x,value);}
  gh[patch*9+t]=value;
 }
}
}
torch::Tensor StableProject(torch::Tensor pixels,torch::Tensor H,torch::Tensor grad,bool backward){
 check(pixels,pixels);check(H,pixels);
 TORCH_CHECK(pixels.dim()==4&&pixels.size(3)==2&&H.sizes()==torch::IntArrayRef({pixels.size(0),pixels.size(1),3,3}),"invalid patch projection inputs");
 int b=pixels.size(0),k=pixels.size(1),p=pixels.size(2);TORCH_CHECK(p<=128,"patch too large");
 c10::cuda::CUDAGuard guard(pixels.device());auto out=torch::empty({b,k,p,3},pixels.options());auto gh=torch::empty_like(H);
 if(backward){check(grad,pixels);TORCH_CHECK(grad.sizes()==out.sizes(),"invalid projection gradient");}
 if(b*k)project_kernel<<<b*k,128,0,at::cuda::getCurrentCUDAStream()>>>(pixels.data_ptr<float>(),H.data_ptr<float>(),backward?grad.data_ptr<float>():nullptr,out.data_ptr<float>(),gh.data_ptr<float>(),k,p,backward);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return backward?gh:out;
}
