// Real ROI pointwise RGB, guard and normal losses. SSIM halo stays unchanged.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
namespace {
constexpr int MAX_VIEWS=8;
struct View {
 const float *rgb,*app,*dn,*normal,*gt,*mask,*weight,*guard,*guard_mask,*counts,*guard_counts,*ssim;
 float *grgb,*gapp,*gdn,*gn,*gguard;
 int w,h,x,y,rw,rh,pw,ph,start,end;
};
struct Batch {View v[MAX_VIEWS];int n;bool guard,normal,image_weight;};
__device__ float sign(float x){return float((x>0)-(x<0));}
__device__ void reduce(float a,float b,float c,float* out,int owner,int n){
 for(int d=16;d;d/=2){a+=__shfl_down_sync(0xffffffff,a,d);b+=__shfl_down_sync(0xffffffff,b,d);c+=__shfl_down_sync(0xffffffff,c,d);}
 __shared__ float sums[3][8];int lane=threadIdx.x&31,warp=threadIdx.x/32;
 if(lane==0){sums[0][warp]=a;sums[1][warp]=b;sums[2][warp]=c;}
 __syncthreads();
 if(threadIdx.x<32){a=lane<8?sums[0][lane]:0;b=lane<8?sums[1][lane]:0;c=lane<8?sums[2][lane]:0;
  for(int d=16;d;d/=2){a+=__shfl_down_sync(0xffffffff,a,d);b+=__shfl_down_sync(0xffffffff,b,d);c+=__shfl_down_sync(0xffffffff,c,d);}
  if(lane==0){atomicAdd(out+owner,a);atomicAdd(out+n+owner,b);atomicAdd(out+2*n+owner,c);}
 }
}
template<bool Backward>
__global__ void pixel_loss(Batch batch,const float* grad,float* out){
 int owner=0;while(owner+1<batch.n&&blockIdx.x>=batch.v[owner].end)owner++;
 View v=batch.v[owner];int i=(blockIdx.x-v.start)*blockDim.x+threadIdx.x,area=v.rw*v.rh;float l1=0,guard=0,normal=0;
 if(i<3*area){
  int p=i%area,c=i/area,xx=p%v.rw-v.x,yy=p/v.rw-v.y;
  bool inside=xx>=0&&yy>=0&&xx<v.w&&yy<v.h,use_app=v.app&&*v.ssim<.5f;
  float prediction=use_app?v.app[i]:v.rgb[i],d=prediction-v.gt[i];
  float denom=fmaxf(*v.counts*3.f,1.f),weight=v.mask[p];
  if(Backward){
   float g=sign(d)*weight*grad[owner]/denom,gg=0;
   if(batch.guard)gg=sign(v.rgb[i]-v.guard[i])*v.guard_mask[p]*grad[batch.n+owner]/fmaxf(*v.guard_counts*3.f,1.f);
   v.grgb[i]=use_app?0.f:g;if(v.gapp)v.gapp[i]=use_app?g:0.f;
   if(batch.guard)v.gguard[(c*v.ph+p/v.rw)*v.pw+p%v.rw]=gg;
  }else{
   l1=fabsf(d)*weight/denom;
   if(batch.guard)guard=fabsf(v.rgb[i]-v.guard[i])*v.guard_mask[p]/fmaxf(*v.guard_counts*3.f,1.f);
  }
  if(batch.normal&&inside){
   int q=(c*v.h+yy)*v.w+xx;float nd=v.dn[q]-v.normal[q],nw=batch.image_weight?weight*v.weight[p]:weight;
   if(Backward){float g=sign(nd)*nw*grad[2*batch.n+owner]/fmaxf(*v.counts,1.f);v.gdn[q]=g;v.gn[q]=-g;}
   else normal=fabsf(nd)*nw/fmaxf(*v.counts,1.f);
  }
 }
 if(!Backward)reduce(l1,guard,normal,out,owner,batch.n);
}
void check(torch::Tensor t,torch::Tensor ref){TORCH_CHECK(t.is_cuda()&&t.device()==ref.device()&&t.scalar_type()==torch::kFloat32&&t.is_contiguous(),"expected contiguous CUDA float32");}
}
std::vector<torch::Tensor> RaggedPhoto(std::vector<torch::Tensor> inputs,std::vector<std::vector<int64_t>> rois,torch::Tensor counts,torch::Tensor guard_counts,torch::Tensor ssim,torch::Tensor guard_prediction,bool guard_on,bool normal_on,bool image_weight,torch::Tensor grad,bool backward){
 int n=rois.size();TORCH_CHECK(n>0&&n<=MAX_VIEWS&&inputs.size()==n*9,"invalid photo batch");c10::cuda::CUDAGuard device(inputs[0].device());
 for(auto t:{counts,guard_counts,ssim}){check(t,inputs[0]);TORCH_CHECK(t.numel()==n,"invalid photo metadata");}
 if(backward){check(grad,inputs[0]);TORCH_CHECK(grad.numel()==n*3,"invalid loss gradient");}
 check(guard_prediction,inputs[0]);TORCH_CHECK(guard_prediction.dim()==4&&guard_prediction.size(0)==n&&guard_prediction.size(1)==3,"invalid guard render batch");
 torch::Tensor guard_gradient;if(backward&&guard_on)guard_gradient=torch::zeros_like(guard_prediction);
 Batch batch{};batch.n=n;batch.guard=guard_on;batch.normal=normal_on;batch.image_weight=image_weight;int blocks=0;
 std::vector<torch::Tensor> out;
 if(!backward)out.push_back(torch::zeros({3*n},inputs[0].options()));
 for(int i=0;i<n;i++){
  TORCH_CHECK(rois[i].size()==6,"invalid photo ROI");auto r=rois[i];View& v=batch.v[i];v.x=r[0];v.y=r[1];v.w=r[2];v.h=r[3];v.rw=r[4];v.rh=r[5];v.pw=guard_prediction.size(3);v.ph=guard_prediction.size(2);
  TORCH_CHECK(v.rw<=v.pw&&v.rh<=v.ph,"invalid guard canvas");
  TORCH_CHECK(v.x>=0&&v.y>=0&&v.w>0&&v.h>0&&v.x+v.w<=v.rw&&v.y+v.h<=v.rh,"photo ROI outside halo");
  int k=9*i,area=v.rw*v.rh;for(int j=0;j<9;j++)check(inputs[k+j],inputs[0]);
  TORCH_CHECK(inputs[k].numel()==3*area&&(inputs[k+1].numel()==0||inputs[k+1].numel()==3*area)&&(!normal_on||(inputs[k+2].numel()==3*v.w*v.h&&inputs[k+3].numel()==3*v.w*v.h)),"invalid photo predictions");
  TORCH_CHECK(inputs[k+4].numel()==3*area&&inputs[k+5].numel()==area&&inputs[k+6].numel()==area&&inputs[k+7].numel()==3*area&&inputs[k+8].numel()==area,"invalid photo supervision");
  v.rgb=inputs[k].data_ptr<float>();v.app=inputs[k+1].numel()?inputs[k+1].data_ptr<float>():nullptr;v.dn=inputs[k+2].data_ptr<float>();v.normal=inputs[k+3].data_ptr<float>();
  v.gt=inputs[k+4].data_ptr<float>();v.mask=inputs[k+5].data_ptr<float>();v.weight=inputs[k+6].data_ptr<float>();v.guard=inputs[k+7].data_ptr<float>();v.guard_mask=inputs[k+8].data_ptr<float>();
  v.counts=counts.data_ptr<float>()+i;v.guard_counts=guard_counts.data_ptr<float>()+i;v.ssim=ssim.data_ptr<float>()+i;
  v.start=blocks;blocks+=(3*area+255)/256;v.end=blocks;
  if(backward){for(int j=0;j<4;j++)out.push_back(torch::empty_like(inputs[k+j]));v.grgb=out[4*i].data_ptr<float>();v.gapp=inputs[k+1].numel()?out[4*i+1].data_ptr<float>():nullptr;v.gdn=out[4*i+2].data_ptr<float>();v.gn=out[4*i+3].data_ptr<float>();v.gguard=guard_on?guard_gradient.data_ptr<float>()+i*3*v.pw*v.ph:nullptr;}
 }
 if(backward)pixel_loss<true><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(batch,grad.data_ptr<float>(),nullptr);
 else pixel_loss<false><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(batch,nullptr,out[0].data_ptr<float>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();if(backward)out.push_back(guard_on?guard_gradient:torch::empty_like(guard_prediction));return out;
}
