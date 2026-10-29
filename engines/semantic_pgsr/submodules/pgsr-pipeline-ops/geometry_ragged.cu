// Experimental fused geometry: execute only real ROI pixels; preserve padded
// output layout solely for the existing reduction and NCC selection semantics.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>
namespace {
constexpr int MAX_VIEWS=8;
struct View {
 const float *depth,*rays,*near_depth,*near_mask,*view,*near_view,*rc,*nc;
 const bool *mask;
 float *gd,*gn;
 int n,w,h,x,y,nw,nh,nx,ny,block_start,block_end;
};
struct Batch {View views[MAX_VIEWS];int count,stride;};
__device__ void rigid(const float* p,const float* a,const float* b,float* out,bool back=false){
 float x[3],y[3];
 for(int k=0;k<3;k++)x[k]=back?p[k]:__fsub_rn(p[k],a[12+k]);
 for(int j=0;j<3;j++){float v=0;for(int k=0;k<3;k++)v=fmaf(x[k],back?b[j*4+k]:a[j*4+k],v);y[j]=v;}
 for(int j=0;j<3;j++){float v=0;for(int k=0;k<3;k++)v=fmaf(y[k],back?a[k*4+j]:b[k*4+j],v);out[j]=back?v:__fadd_rn(v,b[12+j]);}
}
__device__ float pixel(float p,int length){
 float s=float(max(length-1,1));
 float g=__fsub_rn(__fmul_rn(__fmul_rn(2.f,p),__fdiv_rn(1.f,s)),1.f);
 return __fmul_rn(__fmul_rn(__fadd_rn(g,1.f),.5f),float(length-1));
}
struct Sample {int at[4];float dx[4],dy[4],vals[4],weight[4];};
__device__ float sample(const float* image,float px,float py,int w,int h,Sample& s){
 float x=pixel(px,w),y=pixel(py,h);
 if(!isfinite(x)||!isfinite(y)||fabsf(x)>2e9f||fabsf(y)>2e9f){x=-100.f;y=-100.f;}
 int ix=int(floorf(x)),iy=int(floorf(y));
 for(int j=0;j<4;j++){
  int xx=ix+j%2,yy=iy+j/2;
  s.dx[j]=j%2?x-ix:ix+1-x;s.dy[j]=j>=2?y-iy:iy+1-y;
  s.at[j]=(xx>=0&&xx<w&&yy>=0&&yy<h)?yy*w+xx:-1;
  s.weight[j]=s.at[j]>=0?__fmul_rn(s.dx[j],s.dy[j]):0.f;
  s.vals[j]=s.at[j]>=0?image[s.at[j]]:0.f;
 }
 float value=fmaf(s.vals[0],s.weight[0],__fmul_rn(s.vals[1],s.weight[1]));
 value=fmaf(s.vals[2],s.weight[2],value);
 return __fadd_rn(value,__fmul_rn(s.vals[3],s.weight[3]));
}
__device__ void proj_backward(const float* p,const float* focal,const float* g,float* out){
 float z=fmaxf(p[2],1e-8f),gz[2];
 for(int j=0;j<2;j++){
  float u=__fmul_rn(g[j],focal[j]);out[j]=__fdiv_rn(u,z);
  gz[j]=__fmul_rn(-u,__fdiv_rn(__fdiv_rn(p[j],z),z));
 }
 out[2]=p[2]>=1e-8f?__fadd_rn(gz[0],gz[1]):0.f;
}
template<bool Backward>
__global__ void geometry(Batch batch,const float* grad,float* noise,bool* valid){
 int b=0;while(b+1<batch.count&&blockIdx.x>=batch.views[b].block_end)b++;
 View v=batch.views[b];int i=(blockIdx.x-v.block_start)*blockDim.x+threadIdx.x;if(i>=v.n)return;
 int id=b*batch.stride+i;float p[3],near[3],back[3],ref[3],local[2],delta[2];
 float ray[3];
 if(v.rays){for(int j=0;j<3;j++)ray[j]=v.rays[i*3+j];}
 else{
  ray[0]=__fmul_rn(__fsub_rn(float(v.x+i%v.w),v.rc[2]),__fdiv_rn(1.f,v.rc[0]));
  ray[1]=__fmul_rn(__fsub_rn(float(v.y+i/v.w),v.rc[3]),__fdiv_rn(1.f,v.rc[1]));ray[2]=1.f;
 }
 for(int j=0;j<3;j++)p[j]=__fmul_rn(ray[j],v.depth[i]);
 rigid(p,v.view,v.near_view,near);
 float nz=fmaxf(near[2],1e-8f);
 for(int j=0;j<2;j++)local[j]=__fsub_rn(__fadd_rn(__fmul_rn(__fdiv_rn(near[j],nz),v.nc[j]),v.nc[2+j]),float(j?v.ny:v.nx));
 Sample s,sm;float mapped=sample(v.near_depth,local[0],local[1],v.nw,v.nh,s);
 for(int j=0;j<3;j++)back[j]=__fmul_rn(__fdiv_rn(near[j],nz),mapped);
 rigid(back,v.near_view,v.view,ref);
 float rz=fmaxf(ref[2],1e-8f);
 for(int j=0;j<2;j++)delta[j]=__fsub_rn(__fadd_rn(__fmul_rn(__fdiv_rn(ref[j],rz),v.rc[j]),v.rc[2+j]),float(j?v.y+i/v.w:v.x+i%v.w));
 float norm=sqrtf(__fadd_rn(__fmul_rn(delta[0],delta[0]),__fmul_rn(delta[1],delta[1])));
 if(!Backward){
  noise[id]=norm;
  float maskvalue=v.near_mask?sample(v.near_mask,local[0],local[1],v.nw,v.nh,sm):sample(v.near_depth,local[0],local[1],v.nw,v.nh,sm);
  // A missing neighbor mask represents ones, including bilinear boundary weights.
  if(!v.near_mask){maskvalue=fmaf(1.f,sm.weight[0],sm.weight[1]);maskvalue=fmaf(1.f,sm.weight[2],maskvalue);maskvalue=__fadd_rn(maskvalue,sm.weight[3]);}
  valid[id]=local[0]>=0&&local[1]>=0&&local[0]<=v.nw-1&&local[1]<=v.nh-1&&near[2]>0&&mapped>0&&v.depth[i]>0&&v.mask[i]&&maskvalue>.999f&&ref[2]>0&&isfinite(norm);
 }else{
  float g=grad[id],gr[3],gb[3],gn[3],gp[3],gdelta[2];
  float divisor=norm==0?1.f:norm;
  for(int j=0;j<2;j++)gdelta[j]=__fmul_rn(__fdiv_rn(delta[j],divisor),g);
  proj_backward(ref,v.rc,gdelta,gr);rigid(gr,v.near_view,v.view,gb,true);
  float gm[3],gz[3];
  for(int j=0;j<3;j++){
   float q=__fdiv_rn(near[j],nz);gm[j]=__fmul_rn(gb[j],q);
   float t=__fmul_rn(gb[j],mapped);gn[j]=__fdiv_rn(t,nz);gz[j]=__fmul_rn(-t,__fdiv_rn(q,nz));
  }
  if(near[2]>=1e-8f)gn[2]=__fadd_rn(gn[2],__fadd_rn(__fadd_rn(gz[0],gz[2]),gz[1]));
  float mapped_grad=__fadd_rn(__fadd_rn(gm[0],gm[2]),gm[1]);
  float gx=0,gy=0;
  for(int j=0;j<4;j++)if(s.at[j]>=0){
   atomicAdd(v.gn+s.at[j],__fmul_rn(s.weight[j],mapped_grad));
   float vx=__fmul_rn(__fmul_rn(s.vals[j],s.dy[j]),mapped_grad),vy=__fmul_rn(__fmul_rn(s.vals[j],s.dx[j]),mapped_grad);
   gx=__fadd_rn(gx,j%2?vx:-vx);gy=__fadd_rn(gy,j>=2?vy:-vy);
  }
  gx=__fmul_rn(__fmul_rn(__fmul_rn(gx,float(v.nw-1)*.5f),__fdiv_rn(1.f,float(max(v.nw-1,1)))),2.f);
  gy=__fmul_rn(__fmul_rn(__fmul_rn(gy,float(v.nh-1)*.5f),__fdiv_rn(1.f,float(max(v.nh-1,1)))),2.f);
  float local_grad[2]={gx,gy},gn_sample[3];proj_backward(near,v.nc,local_grad,gn_sample);
  for(int j=0;j<3;j++)gn[j]=__fadd_rn(gn[j],gn_sample[j]);
  rigid(gn,v.view,v.near_view,gp,true);
  float gd[3];for(int j=0;j<3;j++)gd[j]=__fmul_rn(gp[j],ray[j]);
  v.gd[i]=__fadd_rn(__fadd_rn(gd[0],gd[2]),gd[1]);
 }
}
void check(const torch::Tensor& t,const torch::Tensor& ref,torch::ScalarType type=torch::kFloat32){
 TORCH_CHECK(t.is_cuda()&&t.device()==ref.device()&&t.scalar_type()==type&&t.is_contiguous(),"expected same-device contiguous CUDA inputs");
}
}
std::vector<torch::Tensor> RaggedGeometry(std::vector<torch::Tensor> inputs,std::vector<std::vector<int64_t>> rois,torch::Tensor grad,bool backward){
 // Inputs per view: depth, rays, selection, neighbor depth/mask, poses, intrinsics.
 int b=rois.size();TORCH_CHECK(b>0&&b<=MAX_VIEWS&&inputs.size()==b*9,"invalid geometry batch");
 c10::cuda::CUDAGuard guard(inputs[0].device());Batch batch{};batch.count=b;int blocks=0,n=0;
 std::vector<torch::Tensor> result;
 for(int j=0;j<b;j++){TORCH_CHECK(rois[j].size()==8,"expected reference and neighbor ROI");n=std::max(n,int(inputs[j*9].numel()));}
 batch.stride=n;
 if(backward){check(grad,inputs[0]);TORCH_CHECK(grad.sizes()==torch::IntArrayRef({b,n}),"invalid geometry gradient");}
 else{result={torch::zeros({b,n},inputs[0].options()),torch::zeros({b,n},inputs[0].options().dtype(torch::kBool))};}
 for(int j=0;j<b;j++){
  int k=j*9;for(int t=0;t<9;t++)check(inputs[k+t],inputs[0],t==2?torch::kBool:torch::kFloat32);
  auto r=rois[j];View& v=batch.views[j];v.x=r[0];v.y=r[1];v.w=r[2];v.h=r[3];v.nx=r[4];v.ny=r[5];v.nw=r[6];v.nh=r[7];v.n=v.w*v.h;
  TORCH_CHECK(v.w>0&&v.h>0&&v.nw>0&&v.nh>0&&inputs[k].numel()==v.n&&(inputs[k+1].numel()==v.n*3||inputs[k+1].numel()==0)&&inputs[k+2].numel()==v.n&&inputs[k+3].numel()==v.nw*v.nh,"invalid ROI tensor shapes");
  TORCH_CHECK((inputs[k+4].numel()==0||inputs[k+4].numel()==v.nw*v.nh)&&inputs[k+5].numel()==16&&inputs[k+6].numel()==16&&inputs[k+7].numel()>=4&&inputs[k+8].numel()>=4,"invalid camera/mask shapes");
  v.depth=inputs[k].data_ptr<float>();v.rays=inputs[k+1].numel()?inputs[k+1].data_ptr<float>():nullptr;v.mask=inputs[k+2].data_ptr<bool>();v.near_depth=inputs[k+3].data_ptr<float>();v.near_mask=inputs[k+4].numel()?inputs[k+4].data_ptr<float>():nullptr;
  v.view=inputs[k+5].data_ptr<float>();v.near_view=inputs[k+6].data_ptr<float>();v.rc=inputs[k+7].data_ptr<float>();v.nc=inputs[k+8].data_ptr<float>();
  v.block_start=blocks;blocks+=(v.n+255)/256;v.block_end=blocks;
  if(backward){result.push_back(torch::empty_like(inputs[k]));result.push_back(torch::zeros_like(inputs[k+3]));v.gd=result[2*j].data_ptr<float>();v.gn=result[2*j+1].data_ptr<float>();}
 }
 auto stream=at::cuda::getCurrentCUDAStream();
 if(backward)geometry<true><<<blocks,256,0,stream>>>(batch,grad.data_ptr<float>(),nullptr,nullptr);
 else geometry<false><<<blocks,256,0,stream>>>(batch,nullptr,result[0].data_ptr<float>(),result[1].data_ptr<bool>());
 C10_CUDA_KERNEL_LAUNCH_CHECK();return result;
}
namespace {
struct GatherView {const float *normals,*distance;float *gn,*gd;int x,y,w,h;};
struct GatherBatch {GatherView views[MAX_VIEWS];int count,k;};
template<bool Backward>
__global__ void gather_kernel(GatherBatch batch,const int64_t* ids,const float* gn,const float* gd,float* normal,float* distance,float* pixels){
 int b=blockIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=batch.k)return;
 GatherView v=batch.views[b];int id=b*batch.k+j;int64_t p=ids[id];bool valid=p>=0&&p<v.w*v.h;
 if(Backward){if(valid){for(int c=0;c<3;c++)atomicAdd(v.gn+c*v.w*v.h+p,gn[id*3+c]);atomicAdd(v.gd+p,gd[id]);}}
 else{
  for(int c=0;c<3;c++)normal[id*3+c]=valid?v.normals[c*v.w*v.h+p]:0.f;
  distance[id]=valid?v.distance[p]:0.f;
  pixels[id*2]=valid?float(v.x+p%v.w):0.f;pixels[id*2+1]=valid?float(v.y+p/v.w):0.f;
 }
}
}
std::vector<torch::Tensor> RaggedGather(std::vector<torch::Tensor> inputs,std::vector<std::vector<int64_t>> rois,torch::Tensor ids,std::vector<torch::Tensor> grads,bool backward){
 int b=rois.size();TORCH_CHECK(b>0&&b<=MAX_VIEWS&&inputs.size()==b*2,"invalid gather batch");
 c10::cuda::CUDAGuard guard(inputs[0].device());check(ids,inputs[0],torch::kInt64);TORCH_CHECK(ids.dim()==2&&ids.size(0)==b,"invalid gather indices");
 int k=ids.size(1);GatherBatch batch{};batch.count=b;batch.k=k;std::vector<torch::Tensor> out;
 if(backward){TORCH_CHECK(grads.size()==2,"invalid gather gradients");for(auto t:grads)check(t,inputs[0]);TORCH_CHECK(grads[0].sizes()==torch::IntArrayRef({b,k,3})&&grads[1].sizes()==ids.sizes(),"invalid gather gradient shapes");}
 else out={torch::empty({b,k,3},inputs[0].options()),torch::empty({b,k},inputs[0].options()),torch::empty({b,k,2},inputs[0].options())};
 for(int j=0;j<b;j++){
  TORCH_CHECK(rois[j].size()==4,"invalid gather ROI");auto r=rois[j];auto& v=batch.views[j];v.x=r[0];v.y=r[1];v.w=r[2];v.h=r[3];
  for(int t=0;t<2;t++)check(inputs[2*j+t],inputs[0]);TORCH_CHECK(v.w>0&&v.h>0&&inputs[2*j].sizes()==torch::IntArrayRef({3,v.h,v.w})&&inputs[2*j+1].numel()==v.w*v.h,"invalid gather input shapes");
  v.normals=inputs[2*j].data_ptr<float>();v.distance=inputs[2*j+1].data_ptr<float>();
  if(backward){out.push_back(torch::zeros_like(inputs[2*j]));out.push_back(torch::zeros_like(inputs[2*j+1]));v.gn=out[2*j].data_ptr<float>();v.gd=out[2*j+1].data_ptr<float>();}
 }
 if(k){auto stream=at::cuda::getCurrentCUDAStream();if(backward)gather_kernel<true><<<dim3((k+255)/256,b),256,0,stream>>>(batch,ids.data_ptr<int64_t>(),grads[0].data_ptr<float>(),grads[1].data_ptr<float>(),nullptr,nullptr,nullptr);
 else gather_kernel<false><<<dim3((k+255)/256,b),256,0,stream>>>(batch,ids.data_ptr<int64_t>(),nullptr,nullptr,out[0].data_ptr<float>(),out[1].data_ptr<float>(),out[2].data_ptr<float>());}
 C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
