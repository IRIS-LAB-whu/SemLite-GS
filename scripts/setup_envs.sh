#!/usr/bin/env bash
set -euo pipefail
semlite_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$semlite_root"
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 11.8 toolkit directory}"
export MAX_JOBS="${MAX_JOBS:-2}"
for name in fastgs pgsr mesh; do
  conda create -y -n "semlite-$name" python=3.8 pip
 done
conda create -y -n semlite-sam3 python=3.12 pip
conda run -n semlite-fastgs python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
for name in pgsr mesh; do
  conda run -n "semlite-$name" python -m pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
 done
for name in fastgs pgsr; do
  conda run -n "semlite-$name" python -m pip install -r requirements-engines.txt
  conda run --no-capture-output -n "semlite-$name" python -m semlite_gs.assets "$name"
 done
conda run -n semlite-mesh python -m pip install -r requirements-mesh.txt
conda run --no-capture-output -n semlite-mesh python -m semlite_gs.assets field
conda run -n semlite-sam3 python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
conda run -n semlite-sam3 python -m pip install -r requirements-sam3.txt
