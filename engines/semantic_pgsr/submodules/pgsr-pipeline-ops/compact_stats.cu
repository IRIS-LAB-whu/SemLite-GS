#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

namespace {
constexpr int MAX_GROUPS=16, MAX_VIEWS=32;
struct View {
  const bool* visible;
  const int* radii;
  const int* observed;
  const float* signed_grad;
  const float* absolute_grad;
};
struct Group {
  const int64_t* global;
  const int64_t* local;
  int64_t end;
  int first, count;
};
struct Batch { Group groups[MAX_GROUPS]; View views[MAX_VIEWS]; int count; };

__device__ float norm2(const float* p) {
  return sqrtf(__fadd_rn(__fmul_rn(p[0],p[0]),__fmul_rn(p[1],p[1])));
}
__global__ void accumulate(Batch batch, int64_t total, float scale,
                          float* radii, float* signed_sum, float* absolute_sum,
                          float* denom, float* denom_abs) {
  int64_t row=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(row>=total)return;
  int g=0;while(row>=batch.groups[g].end)++g;
  Group group=batch.groups[g];
  int64_t j=row-(g?batch.groups[g-1].end:0);
  int64_t dst=group.global[j],src=group.local[j];
  float r=radii[dst],s=signed_sum[dst],a=absolute_sum[dst],d=denom[dst],da=denom_abs[dst];
  // Each target row has exactly one thread. Views retain the Python reference
  // order, including reference/neighbor interleaving, without atomic additions.
  for(int v=group.first;v<group.first+group.count;++v) {
    View view=batch.views[v];
    if(!view.visible[src])continue;
    if(view.observed[src]>0) {
      float next=float(view.radii[src]);
      r=isnan(r)?r:fmaxf(r,next);
    }
    if(view.signed_grad && view.absolute_grad) {
      s=__fadd_rn(s,__fmul_rn(norm2(view.signed_grad+3*src),scale));
      a=__fadd_rn(a,__fmul_rn(norm2(view.absolute_grad+3*src),scale));
      d=__fadd_rn(d,1.f);da=__fadd_rn(da,1.f);
    }
  }
  radii[dst]=r;signed_sum[dst]=s;absolute_sum[dst]=a;denom[dst]=d;denom_abs[dst]=da;
}
}

// Called only with unique CompactModel target rows and disjoint owner groups.
// These invariants are established by ActiveSetCache/CompactModel, not inferred
// from arbitrary caller-supplied index tensors (which would require a readback).
void CompactStats(std::vector<torch::Tensor> outputs,
                  std::vector<torch::Tensor> global, std::vector<torch::Tensor> local,
                  std::vector<int64_t> counts,
                  std::vector<torch::Tensor> visible, std::vector<torch::Tensor> radii,
                  std::vector<torch::Tensor> observed,
                  std::vector<torch::Tensor> signed_grad, std::vector<torch::Tensor> absolute_grad,
                  std::vector<bool> has_grad, double scale) {
  TORCH_CHECK(outputs.size()==5,"five statistic buffers required");
  auto device=outputs[0].device();
  TORCH_CHECK(device.is_cuda(),"CUDA statistics required");
  int64_t population=outputs[0].numel();
  for(auto& t:outputs)TORCH_CHECK(t.device()==device && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.numel()==population,"statistic buffer mismatch");
  const int n=global.size(),nv=visible.size();
  TORCH_CHECK(n<=MAX_GROUPS && nv<=MAX_VIEWS,"batch exceeds compact stats capacity");
  TORCH_CHECK(local.size()==n && counts.size()==n,"group metadata mismatch");
  TORCH_CHECK(radii.size()==nv && observed.size()==nv && signed_grad.size()==nv && absolute_grad.size()==nv && has_grad.size()==nv,"view metadata mismatch");
  const c10::cuda::CUDAGuard guard(device);
  Batch batch{};batch.count=n;int first=0;int64_t total=0;
  for(int i=0;i<n;++i) {
    for(auto t:{global[i],local[i]})TORCH_CHECK(t.device()==device && t.scalar_type()==torch::kInt64 && t.is_contiguous() && t.dim()==1,"invalid compact indices");
    TORCH_CHECK(global[i].numel()==local[i].numel() && counts[i]>=0 && first+counts[i]<=nv,"invalid group size");
    total+=global[i].numel();
    batch.groups[i]={global[i].data_ptr<int64_t>(),local[i].data_ptr<int64_t>(),total,first,int(counts[i])};
    first+=counts[i];
  }
  TORCH_CHECK(first==nv,"unassigned views");
  for(int i=0;i<nv;++i) {
    auto v=visible[i];int64_t rows=v.numel();
    TORCH_CHECK(v.device()==device && v.scalar_type()==torch::kBool && v.is_contiguous() && v.dim()==1,"invalid visibility");
    for(auto t:{radii[i],observed[i]})TORCH_CHECK(t.device()==device && t.scalar_type()==torch::kInt32 && t.is_contiguous() && t.numel()==rows,"invalid radius/observation");
    if(has_grad[i])for(auto t:{signed_grad[i],absolute_grad[i]})TORCH_CHECK(t.device()==device && t.scalar_type()==torch::kFloat32 && t.is_contiguous() && t.dim()==2 && t.size(0)==rows && t.size(1)==3,"invalid screen gradient");
    batch.views[i]={v.data_ptr<bool>(),radii[i].data_ptr<int>(),observed[i].data_ptr<int>(),
      has_grad[i]?signed_grad[i].data_ptr<float>():nullptr,
      has_grad[i]?absolute_grad[i].data_ptr<float>():nullptr};
  }
  if(total) {
    accumulate<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(batch,total,float(scale),
      outputs[0].data_ptr<float>(),outputs[1].data_ptr<float>(),outputs[2].data_ptr<float>(),outputs[3].data_ptr<float>(),outputs[4].data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    for(auto& t:outputs)t.unsafeGetTensorImpl()->bump_version();
  }
}
