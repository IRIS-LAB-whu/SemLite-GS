# PGSR pipeline auxiliary operations

Native operations for masked refinement: packed plane-depth correction, TSDF
fusion, connected-label packing, indexed updates, compact statistics and batched
photometric/geometric losses. License terms are retained in `LICENSE.md`.

Build using the PGSR environment from the project root:

```bash
python -m semlite_gs.assets pgsr
```

For a standalone extension build, run `python setup.py build_ext --inplace` from
this directory with `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` set appropriately.
