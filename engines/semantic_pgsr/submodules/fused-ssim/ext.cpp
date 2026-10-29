#include <torch/extension.h>
#include <pybind11/stl.h>
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
  m.def("packed_masked_photo_forward", &packed_masked_photo_forward);
  m.def("packed_masked_photo_backward", &packed_masked_photo_backward);
}
