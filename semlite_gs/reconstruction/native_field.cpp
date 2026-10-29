#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <unordered_map>
#include <open3d/pipelines/integration/MarchingCubesConst.h>
#include <open3d/pipelines/integration/ScalableTSDFVolume.h>
#include <open3d/pipelines/integration/UniformTSDFVolume.h>
namespace py=pybind11;
namespace oi=open3d::pipelines::integration;
template<class T> T* pointer(py::handle obj,const char* name) {
    if (py::str(obj.get_type().attr("__name__")).cast<std::string>() != name)
        throw std::runtime_error("Unexpected Open3D object type");
    auto* instance=reinterpret_cast<py::detail::instance*>(obj.ptr());
    return reinterpret_cast<T*>(instance->get_value_and_holder().value_ptr());
}
py::dict export_field(py::handle obj) {
    auto* volume=pointer<oi::ScalableTSDFVolume>(obj,"ScalableTSDFVolume");
    const int r=volume->volume_unit_resolution_;
    if(r!=16 || volume->volume_unit_length_!=r*volume->voxel_length_)
        throw std::runtime_error("Legacy ABI or unit resolution mismatch");
    size_t count=0;
    for(auto &p:volume->volume_units_) if(p.second.volume_) {
        auto &u=*p.second.volume_;
        if(u.resolution_!=r || u.voxels_.size()!=size_t(r*r*r)) throw std::runtime_error("Invalid uniform unit layout");
        for(auto &v:u.voxels_) if(v.weight_>0) ++count;
    }
    py::array_t<int> coords({py::ssize_t(count),py::ssize_t(3)});
    py::array_t<double> phi(count); py::array_t<float> weight(count);
    auto c=coords.mutable_unchecked<2>();auto f=phi.mutable_unchecked<1>();auto w=weight.mutable_unchecked<1>();
    size_t k=0;
    for(auto &p:volume->volume_units_) if(p.second.volume_) {
        auto &u=*p.second.volume_;
        for(int x=0;x<r;++x) for(int y=0;y<r;++y) for(int z=0;z<r;++z) {
            auto &v=u.voxels_[u.IndexOf(x,y,z)];if(v.weight_<=0) continue;
            if(!std::isfinite(v.tsdf_) || !std::isfinite(v.weight_) || std::abs(v.tsdf_)>1.00001f)
                throw std::runtime_error("Invalid TSDF scalar layout");
            c(k,0)=p.first[0]*r+x;c(k,1)=p.first[1]*r+y;c(k,2)=p.first[2]*r+z;
            f(k)=v.tsdf_*volume->sdf_trunc_;w(k)=v.weight_;++k;
        }
    }
    py::dict out;out["coordinates"]=coords;out["phi"]=phi;out["weight"]=weight;
    out["voxel"]=volume->voxel_length_;out["sdf_trunc"]=volume->sdf_trunc_;
    out["lattice_offset"]=py::make_tuple(.5,.5,.5);out["units"]=volume->volume_units_.size();return out;
}
void attach_unit(py::handle target,py::handle unit,int x,int y,int z) {
    auto* v=pointer<oi::ScalableTSDFVolume>(target,"ScalableTSDFVolume");
    auto* u=pointer<oi::UniformTSDFVolume>(unit,"UniformTSDFVolume");
    if(v->volume_unit_resolution_!=16 || u->resolution_!=16) throw std::runtime_error("Unit mismatch");
    Eigen::Vector3i id(x,y,z);auto &entry=v->volume_units_[id];entry.index_=id;
    entry.volume_=std::shared_ptr<oi::UniformTSDFVolume>(u,[](oi::UniformTSDFVolume*){});
}
struct LatticeHash {
    template<size_t N> size_t operator()(const std::array<int,N>& v) const {
        size_t h=0;
        for(int x:v) h^=std::hash<int>{}(x)+0x9e3779b9+(h<<6)+(h>>2);
        return h;
    }
};
py::tuple zero_observations(py::array_t<int,py::array::c_style|py::array::forcecast> coordinates,
        py::array_t<double,py::array::c_style|py::array::forcecast> phi,
        py::array_t<float,py::array::c_style|py::array::forcecast> weight,
        double voxel,double trunc,py::array_t<double> offset,double minimum_weight) {
    auto c=coordinates.unchecked<2>();auto p=phi.unchecked<1>();auto w=weight.unchecked<1>();auto o=offset.unchecked<1>();
    if(c.shape(1)!=3 || p.shape(0)!=c.shape(0) || w.shape(0)!=c.shape(0) || o.shape(0)!=3)
        throw std::runtime_error("Invalid TSDF observation arrays");
    std::unordered_map<std::array<int,3>,size_t,LatticeHash> samples;
    samples.reserve(c.shape(0));
    for(py::ssize_t i=0;i<c.shape(0);++i) samples.emplace(std::array<int,3>{c(i,0),c(i,1),c(i,2)},i);
    std::unordered_map<std::array<int,4>,int,LatticeHash> edges;
    std::vector<std::array<double,3>> points;
    std::vector<std::array<int,3>> patches;
    {
    py::gil_scoped_release release;
    for(py::ssize_t i=0;i<c.shape(0);++i) {
        double values[8];int code=0;bool supported=true;
        for(int j=0;j<8;++j) {
            std::array<int,3> q{c(i,0)+shift[j][0],c(i,1)+shift[j][1],c(i,2)+shift[j][2]};
            auto it=samples.find(q);
            if(it==samples.end() || w(it->second)<=minimum_weight) {supported=false;break;}
            values[j]=static_cast<float>(p(it->second)/trunc);
            if(values[j]<0) code|=1<<j;
        }
        if(!supported || code==0 || code==255) continue;
        int ids[12];
        for(int e=0;e<12;++e) if(edge_table[code]&(1<<e)) {
            std::array<int,4> key{c(i,0)+edge_shift[e][0],c(i,1)+edge_shift[e][1],c(i,2)+edge_shift[e][2],edge_shift[e][3]};
            auto it=edges.find(key);
            if(it!=edges.end()) {ids[e]=it->second;continue;}
            int a=edge_to_vert[e][0],b=edge_to_vert[e][1];
            double f0=std::abs(values[a]),f1=std::abs(values[b]);
            std::array<double,3> position;
            for(int k=0;k<3;++k) position[k]=(key[k]+o(k))*voxel;
            position[key[3]]+=f0*voxel/(f0+f1);
            ids[e]=points.size();edges.emplace(key,ids[e]);points.push_back(position);
        }
        for(int j=0;tri_table[code][j]!=-1;j+=3)
            patches.push_back({ids[tri_table[code][j]],ids[tri_table[code][j+2]],ids[tri_table[code][j+1]]});
    }
    }
    py::array_t<double> v({py::ssize_t(points.size()),py::ssize_t(3)});
    py::array_t<int> f({py::ssize_t(patches.size()),py::ssize_t(3)});
    auto vv=v.mutable_unchecked<2>();auto ff=f.mutable_unchecked<2>();
    for(size_t i=0;i<points.size();++i) for(int k=0;k<3;++k) vv(i,k)=points[i][k];
    for(size_t i=0;i<patches.size();++i) for(int k=0;k<3;++k) ff(i,k)=patches[i][k];
    return py::make_tuple(v,f);
}

PYBIND11_MODULE(native_field,m){m.def("zero_observations",&zero_observations);m.def("export_field",&export_field);m.def("attach_unit",&attach_unit);}
