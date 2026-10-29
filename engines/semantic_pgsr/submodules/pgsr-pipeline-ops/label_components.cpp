#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

#include "pipeline_ops.h"

namespace {

class UnionFind {
 public:
  explicit UnionFind(int64_t size)
      : parent_(static_cast<size_t>(size)),
        rank_(static_cast<size_t>(size), 0) {
    std::iota(parent_.begin(), parent_.end(), int64_t{0});
  }

  int64_t Find(int64_t value) {
    int64_t root = value;
    while (parent_[static_cast<size_t>(root)] != root) {
      root = parent_[static_cast<size_t>(root)];
    }
    while (parent_[static_cast<size_t>(value)] != value) {
      const int64_t next = parent_[static_cast<size_t>(value)];
      parent_[static_cast<size_t>(value)] = root;
      value = next;
    }
    return root;
  }

  void Unite(int64_t first, int64_t second) {
    first = Find(first);
    second = Find(second);
    if (first == second) {
      return;
    }
    if (rank_[static_cast<size_t>(first)] <
        rank_[static_cast<size_t>(second)]) {
      std::swap(first, second);
    }
    parent_[static_cast<size_t>(second)] = first;
    if (rank_[static_cast<size_t>(first)] ==
        rank_[static_cast<size_t>(second)]) {
      ++rank_[static_cast<size_t>(first)];
    }
  }

 private:
  std::vector<int64_t> parent_;
  std::vector<uint8_t> rank_;
};

inline bool IsCandidate(int32_t label, int64_t target_mode) {
  return target_mode == 0 ? label == -1 : label >= 0;
}

}  // namespace

std::vector<torch::Tensor> PackedLabelComponentsCPU(
    const torch::Tensor& labels,
    const torch::Tensor& neighbors,
    const int64_t target_mode) {
  TORCH_CHECK(!labels.is_cuda(), "labels must be a CPU tensor");
  TORCH_CHECK(!neighbors.is_cuda(), "neighbors must be a CPU tensor");
  TORCH_CHECK(labels.scalar_type() == torch::kInt32,
              "labels must have dtype int32");
  TORCH_CHECK(neighbors.scalar_type() == torch::kInt32,
              "neighbors must have dtype int32");
  TORCH_CHECK(labels.is_contiguous(), "labels must be contiguous");
  TORCH_CHECK(neighbors.is_contiguous(), "neighbors must be contiguous");
  TORCH_CHECK(labels.dim() == 1, "labels must have shape [N]");
  TORCH_CHECK(neighbors.dim() == 2, "neighbors must have shape [N, K]");
  TORCH_CHECK(neighbors.size(0) == labels.size(0),
              "labels and neighbors must share N");
  TORCH_CHECK(target_mode == 0 || target_mode == 1,
              "target_mode must be 0 (-1) or 1 (nonnegative)");

  const int64_t face_count = labels.size(0);
  const int64_t neighbor_width = neighbors.size(1);
  TORCH_CHECK(
      face_count <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "face count exceeds the packed component id range");
  const int32_t* label_data = labels.data_ptr<int32_t>();
  const int32_t* neighbor_data = neighbors.data_ptr<int32_t>();

  // Match the legacy NumPy/SciPy path: any non-negative index is validated,
  // even when it belongs to a row that is not selected in this pass.
  const int64_t neighbor_count = neighbors.numel();
  for (int64_t index = 0; index < neighbor_count; ++index) {
    TORCH_CHECK(
        neighbor_data[index] < face_count,
        "neighbor index is out of range");
  }

  UnionFind union_find(face_count);
  for (int64_t face = 0; face < face_count; ++face) {
    const int32_t label = label_data[face];
    if (!IsCandidate(label, target_mode)) {
      continue;
    }
    const int64_t row_start = face * neighbor_width;
    for (int64_t slot = 0; slot < neighbor_width; ++slot) {
      const int64_t neighbor = neighbor_data[row_start + slot];
      if (neighbor < 0) {
        continue;
      }
      if (IsCandidate(label_data[neighbor], target_mode) &&
          label_data[neighbor] == label) {
        union_find.Unite(face, neighbor);
      }
    }
  }

  std::vector<int32_t> root_to_component(
      static_cast<size_t>(face_count), -1);
  std::vector<int64_t> component_counts;
  std::vector<int64_t> boundary_counts;
  std::vector<int32_t> component_label_values;
  for (int64_t face = 0; face < face_count; ++face) {
    const int32_t label = label_data[face];
    if (!IsCandidate(label, target_mode)) {
      continue;
    }
    const int64_t root = union_find.Find(face);
    int32_t component = root_to_component[static_cast<size_t>(root)];
    if (component < 0) {
      component = static_cast<int32_t>(component_counts.size());
      root_to_component[static_cast<size_t>(root)] = component;
      component_counts.push_back(0);
      boundary_counts.push_back(0);
      component_label_values.push_back(label);
    }
    ++component_counts[static_cast<size_t>(component)];
    const int64_t row_start = face * neighbor_width;
    for (int64_t slot = 0; slot < neighbor_width; ++slot) {
      const int64_t neighbor = neighbor_data[row_start + slot];
      if (neighbor >= 0) {
        ++boundary_counts[static_cast<size_t>(component)];
      }
    }
  }

  const int64_t component_count =
      static_cast<int64_t>(component_counts.size());
  const auto long_options = torch::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(torch::kCPU);
  const auto int_options = torch::TensorOptions()
                               .dtype(torch::kInt32)
                               .device(torch::kCPU);
  torch::Tensor component_offsets =
      torch::empty({component_count + 1}, long_options);
  torch::Tensor boundary_offsets =
      torch::empty({component_count + 1}, long_options);
  int64_t* component_offset_data = component_offsets.data_ptr<int64_t>();
  int64_t* boundary_offset_data = boundary_offsets.data_ptr<int64_t>();
  component_offset_data[0] = 0;
  boundary_offset_data[0] = 0;
  for (int64_t component = 0; component < component_count; ++component) {
    component_offset_data[component + 1] =
        component_offset_data[component] +
        component_counts[static_cast<size_t>(component)];
    boundary_offset_data[component + 1] =
        boundary_offset_data[component] +
        boundary_counts[static_cast<size_t>(component)];
  }

  torch::Tensor component_faces =
      torch::empty({component_offset_data[component_count]}, long_options);
  torch::Tensor boundary_ids =
      torch::empty({boundary_offset_data[component_count]}, long_options);
  torch::Tensor component_labels =
      torch::empty({component_count}, int_options);
  int64_t* component_face_data = component_faces.data_ptr<int64_t>();
  int64_t* boundary_id_data = boundary_ids.data_ptr<int64_t>();
  int32_t* component_label_data = component_labels.data_ptr<int32_t>();
  std::vector<int64_t> component_write(
      component_offset_data, component_offset_data + component_count);
  std::vector<int64_t> boundary_write(
      boundary_offset_data, boundary_offset_data + component_count);
  for (int64_t component = 0; component < component_count; ++component) {
    component_label_data[component] =
        component_label_values[static_cast<size_t>(component)];
  }

  for (int64_t face = 0; face < face_count; ++face) {
    const int32_t label = label_data[face];
    if (!IsCandidate(label, target_mode)) {
      continue;
    }
    const int64_t root = union_find.Find(face);
    const int32_t component =
        root_to_component[static_cast<size_t>(root)];
    component_face_data[
        component_write[static_cast<size_t>(component)]++] = face;
    const int64_t row_start = face * neighbor_width;
    for (int64_t slot = 0; slot < neighbor_width; ++slot) {
      const int64_t neighbor = neighbor_data[row_start + slot];
      if (neighbor >= 0) {
        boundary_id_data[
            boundary_write[static_cast<size_t>(component)]++] = neighbor;
      }
    }
  }

  return {
      component_offsets,
      component_faces,
      component_labels,
      boundary_offsets,
      boundary_ids,
  };
}
