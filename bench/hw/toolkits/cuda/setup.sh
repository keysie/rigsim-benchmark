#!/bin/sh
# Runs inside the session container (common image nvidia/cuda:<version>-devel-ubuntu24.04): compiler and cuBLAS
# are in the image, so no network is needed. sm_86 = consumer Ampere (3060-3090), sm_89 = Ada, sm_80 = A100,
# sm_90 = Hopper; PTX for anything newer.
set -eu
cd /opt/rigsim-hwbench
nvcc -O3 -std=c++17 -o /usr/local/bin/probe src/probe.cu -lcublas -lpthread \
  -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
nvcc --version | tail -1
echo "probe ready"
