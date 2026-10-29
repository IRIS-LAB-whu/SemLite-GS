#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Edge_collapse_visitor_base.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/LindstromTurk_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Bounded_normal_change_filter.h>
#include <CGAL/intersections.h>
#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
namespace SMS=CGAL::Surface_mesh_simplification;
using K=CGAL::Exact_predicates_inexact_constructions_kernel;
using Mesh=CGAL::Surface_mesh<K::Point_3>;
using Tags=Mesh::Property_map<Mesh::Vertex_index,std::vector<int>>;
struct Plane {Eigen::Vector3d n;double d;};
std::vector<int> unite(const std::vector<int>& a,const std::vector<int>& b){
  std::set<int> s(a.begin(),a.end());s.insert(b.begin(),b.end());return {s.begin(),s.end()};
}
struct Placement {
  Tags tags;const std::vector<Plane>* planes;Eigen::Vector3d low,high;int mode;
  template<class P>boost::optional<typename P::Point> operator()(const P& p)const{
    boost::optional<typename P::Point> raw;
    if(mode==1)raw=p.p0();
    else if(mode==2)raw=p.p1();
    else if(mode==3)raw=CGAL::midpoint(p.p0(),p.p1());
    else raw=SMS::LindstromTurk_placement<Mesh>()(p);
    if(!raw)return boost::none;
    Eigen::Vector3d x(raw->x(),raw->y(),raw->z());
    auto ids=unite(tags[p.v0()],tags[p.v1()]);
    if(!ids.empty()){
      Eigen::Matrix3d A=Eigen::Matrix3d::Zero();Eigen::Vector3d b=Eigen::Vector3d::Zero();
      for(int id:ids){const auto& q=(*planes)[id];A+=q.n*q.n.transpose();b+=q.n*(-q.d-q.n.dot(x));}
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> e(A);
      auto y=e.eigenvectors().transpose()*b;Eigen::Vector3d delta=Eigen::Vector3d::Zero();
      for(int j=0;j<3;++j)if(e.eigenvalues()[j]>1e-9)delta+=e.eigenvectors().col(j)*(y[j]/e.eigenvalues()[j]);
      x+=delta;
      for(int id:ids)if(std::abs((*planes)[id].n.dot(x)+(*planes)[id].d)>1e-6)return boost::none;
    }
    if(!x.allFinite()||(x.array()<low.array()-1e-6).any()||(x.array()>high.array()+1e-6).any())return boost::none;
    return typename P::Point(x[0],x[1],x[2]);
  }
};
struct Visitor:SMS::Edge_collapse_visitor_base<Mesh>{
  Tags tags;std::vector<int> pending;explicit Visitor(Tags t):tags(t){}
  void OnCollapsing(const Profile& p,const boost::optional<Point>&){pending=unite(tags[p.v0()],tags[p.v1()]);}
  void OnCollapsed(const Profile&,const vertex_descriptor& v){tags[v]=pending;}
};
struct IntersectionFilter {
  using Points=std::array<K::Point_3,3>;using Ids=std::array<Mesh::Vertex_index,3>;
  static bool bad(const Points& a,const Ids& ai,const Points& b,const Ids& bi){
    K::Triangle_3 ta(a[0],a[1],a[2]),tb(b[0],b[1],b[2]);
    if(!CGAL::do_overlap(ta.bbox(),tb.bbox()))return false;
    std::vector<std::pair<int,int>> shared;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)if(ai[i]==bi[j])shared.push_back({i,j});
    if(shared.size()>=3)return true;
    if(shared.size()==2){
      int i=shared[0].first,j=shared[1].first,k=3-i-j;
      int l=3-shared[0].second-shared[1].second;
      if(!CGAL::coplanar(a[i],a[j],a[k],b[l]))return false;
      return CGAL::cross_product(a[j]-a[i],a[k]-a[i])*CGAL::cross_product(a[j]-a[i],b[l]-a[i])>0;
    }
    if(shared.size()==1){
      int i=shared[0].first,j=shared[0].second;
      return CGAL::do_intersect(ta,K::Segment_3(b[(j+1)%3],b[(j+2)%3]))||CGAL::do_intersect(tb,K::Segment_3(a[(i+1)%3],a[(i+2)%3]));
    }
    return CGAL::do_intersect(ta,tb);
  }
  template<class P>boost::optional<typename P::Point> operator()(const P& p,boost::optional<typename P::Point> op)const{
    if(!op)return op;
    const auto& m=p.surface_mesh();auto u=p.v0(),v=p.v1();
    const auto& local=p.triangles();auto it=local.begin();
    std::vector<Points> replacements;std::vector<Ids> identities;
    if(p.left_face_exists())++it;if(p.right_face_exists())++it;
    for(;it!=local.end();++it){
      auto a=it->v0,c=it->v2;auto pa=get(p.vertex_point_map(),a),pc=get(p.vertex_point_map(),c);
      if(CGAL::collinear(pa,*op,pc))return boost::none;
      Points replacement{pa,*op,pc};Ids ids{a,u,c};
      for(std::size_t j=0;j<replacements.size();++j)if(bad(replacement,ids,replacements[j],identities[j]))return boost::none;
      replacements.push_back(replacement);identities.push_back(ids);
      for(auto f:faces(m)){
        auto h=halfedge(f,m);auto x=source(h,m),y=target(h,m),z=target(next(h,m),m);
        if(x==u||x==v||y==u||y==v||z==u||z==v)continue;
        Points other{get(p.vertex_point_map(),x),get(p.vertex_point_map(),y),get(p.vertex_point_map(),z)};
        if(bad(replacement,ids,other,Ids{x,y,z}))return boost::none;
      }
    }
    return op;
  }
};
struct Stop {
  const Mesh* m;std::size_t target;std::chrono::steady_clock::time_point start;double seconds;
  template<class C,class P>bool operator()(const C&,const P&,std::size_t,std::size_t)const{
    return m->number_of_faces()<=target||std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>seconds;
  }
};
int main(int argc,char**argv){
  if(argc!=7&&argc!=8){std::cerr<<"input.off planes.txt target output.ply seconds bounds.txt [placement_mode]\n";return 2;}
  Mesh m;if(!CGAL::IO::read_polygon_mesh(argv[1],m)||!CGAL::is_triangle_mesh(m))return 3;
  std::ifstream in(argv[2]);std::size_t np,nv;in>>np>>nv;if(!in||nv!=m.number_of_vertices())return 4;
  std::vector<Plane> planes(np);for(auto& p:planes)in>>p.n[0]>>p.n[1]>>p.n[2]>>p.d;
  auto tags=m.add_property_map<Mesh::Vertex_index,std::vector<int>>("v:planes",{}).first;
  for(auto v:m.vertices()){
    double x,y,z;std::size_t n;in>>x>>y>>z>>n;
    if(CGAL::squared_distance(m.point(v),K::Point_3(x,y,z))>1e-12)return 5;
    for(std::size_t j=0;j<n;++j){int id;in>>id;if(id<0||std::size_t(id)>=np)return 6;tags[v].push_back(id);}
  }
  if(!in)return 7;
  Eigen::Vector3d low,high;std::ifstream bounds(argv[6]);bounds>>low[0]>>low[1]>>low[2]>>high[0]>>high[1]>>high[2];if(!bounds)return 8;
  auto start=std::chrono::steady_clock::now();Visitor visitor(tags);
  int mode=argc==8?std::stoi(argv[7]):0;std::size_t count=0;
  if(mode==0)count=SMS::edge_collapse(m,Stop{&m,std::stoul(argv[3]),start,std::stod(argv[5])},
    CGAL::parameters::get_cost(SMS::LindstromTurk_cost<Mesh>()).get_placement(Placement{tags,&planes,low,high,mode})
    .filter(SMS::Bounded_normal_change_filter<IntersectionFilter>(IntersectionFilter{})).visitor(visitor));
  else count=SMS::edge_collapse(m,Stop{&m,std::stoul(argv[3]),start,std::stod(argv[5])},
    CGAL::parameters::get_cost(SMS::Edge_length_cost<Mesh>()).get_placement(Placement{tags,&planes,low,high,mode})
    .filter(SMS::Bounded_normal_change_filter<IntersectionFilter>(IntersectionFilter{})).visitor(visitor));
  double residual=0;for(auto v:m.vertices())for(int id:tags[v]){
    auto p=m.point(v);residual=std::max(residual,std::abs(planes[id].n.dot(Eigen::Vector3d(p.x(),p.y(),p.z()))+planes[id].d));
  }
  m.collect_garbage();
  std::cout<<"{\"faces\":"<<m.number_of_faces()<<",\"removed_edges\":"<<count<<",\"plane_residual_m\":"<<residual<<"}\n";
  return CGAL::IO::write_polygon_mesh(argv[4],m,CGAL::parameters::stream_precision(17))?0:9;
}
