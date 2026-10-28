#!/bin/sh
# Stick variant of the cpu toolkit setup: installs likwid, fio and gcc from the offline debs the stick put into
# ./debs (no network on the stick), then builds the pointer chase as the normal toolkit does, plus the CPU GEMM
# probe (OpenBLAS, oneDNN for int8).
set -eu
cd /opt/rigsim-hwbench
if ls debs/*.deb >/dev/null 2>&1; then
  dpkg -i debs/*.deb >/dev/null 2>&1 || dpkg -i debs/*.deb
else
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq --no-install-recommends likwid fio python3-minimal gcc libc6-dev libopenblas-dev libdnnl-dev >/dev/null
fi
gcc -O2 -o /usr/local/bin/chase src/chase.c
gcc -O2 -fopenmp -o /usr/local/bin/cgemm src/cgemm.c -lopenblas -ldnnl
install -m 0755 src/cpuprobe /usr/local/bin/cpuprobe
echo "cpu toolkit ready"
