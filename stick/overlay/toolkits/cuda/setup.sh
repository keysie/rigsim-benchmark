#!/bin/sh
# Stick variant of the cuda toolkit setup: same probe, compiled for every architecture CUDA 12.8 supports that the
# stick's drivers can meet — Maxwell (sm_52) .. Blackwell (sm_120), PTX for anything newer. The AI box's variant
# builds sm_80+ only.
set -eu
cd /opt/rigsim-hwbench
nvcc -O3 -std=c++17 -Wno-deprecated-gpu-targets -o /usr/local/bin/probe src/probe.cu -lcublas -lpthread \
  -gencode arch=compute_52,code=sm_52 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_120,code=sm_120 \
  -gencode arch=compute_120,code=compute_120
nvcc --version | tail -1
echo "probe ready"
