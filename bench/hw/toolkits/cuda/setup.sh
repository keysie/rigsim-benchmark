#!/bin/sh
# Runs inside the session container (common image nvidia/cuda:<version>-devel-ubuntu24.04): compiler and cuBLAS
# are in the image, so no network is needed. sm_60 = P100, sm_61 = P40 / GTX 10, sm_70 = V100, sm_75 = Turing (the
# stick measures these after switching to the proprietary modules), sm_80 = A100, sm_86 = consumer Ampere (3060-3090),
# sm_89 = Ada, sm_90 = Hopper; PTX for anything newer. CUDA 12.x is the last toolkit for sm_60 / sm_61.
set -eu
cd /opt/rigsim-hwbench
nvcc -O3 -std=c++17 -o /usr/local/bin/probe src/probe.cu -lcublas -lpthread \
  -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
nvcc --version | tail -1
echo "probe ready"
