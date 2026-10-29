# Third-party sources

Original copyright notices and licenses are retained in the source directories.
Review these licenses before redistribution or use; this project is not a blanket
MIT release.

| Source | Included components | License location |
|---|---|---|
| FastGS / Gaussian Splatting | Initialization, rendering and native CUDA modules | `engines/fastgs/LICENSE`, `LICENSE_ORIGINAL.md` and submodule notices |
| PGSR | SemLite-GS semantic-selective refinement, rendering and native CUDA modules | `engines/semantic_pgsr/LICENSE.md` and submodule notices |
| SAM3 | Model source and BPE vocabulary; no model weights | `engines/sam3/LICENSE` |
| Open3D 0.17 | C++ headers and zero-crossing lookup tables for TSDF processing | `third_party/open3d/LICENSE` and header notices |
| GLM | Header-only math library for CUDA rasterizers | Each rasterizer's `third_party/glm/copying.txt` |
| PyTorch3D | Pure-PyTorch rotation conversion functions | `semlite_gs/vendor/PYTORCH3D_LICENSE` |

Open3D's bridge supports the 0.18/0.19 runtime layouts and performs an isolated
compatibility check during compilation. Unused upstream documentation, demos and test data are
omitted from the vendored dependencies. Required headers and attribution remain.

SAM3 checkpoints are downloaded separately under the model's own access terms.
