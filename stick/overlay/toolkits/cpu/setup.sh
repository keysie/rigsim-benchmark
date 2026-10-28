#!/bin/sh
# Stick variant of the cpu toolkit setup: installs likwid, fio and gcc from the offline debs the stick put into
# ./debs (no network on the stick), then builds the pointer chase as the normal toolkit does, plus the CPU GEMM
# probe (OpenBLAS, oneDNN for int8).
set -eu
cd /opt/rigsim-hwbench
if ls debs/*.deb >/dev/null 2>&1; then
  # apt orders the local packages by their dependencies (dpkg -i in one go does not: python3 pre-depends on a
  # configured python3-minimal — first trial run on the stick: fio and likwid left unconfigured). dpkg as fallback,
  # configuring between passes.
  export DEBIAN_FRONTEND=noninteractive
  if ! apt-get install -y -qq --no-install-recommends --no-download $(printf './%s ' debs/*.deb) >/dev/null 2>&1; then
    for pass in 1 2 3; do
      dpkg -i debs/*.deb >/dev/null 2>&1 && break
      dpkg --configure -a >/dev/null 2>&1 || true
    done
    dpkg --configure -a
  fi
  for c in likwid-bench fio gcc; do
    command -v "$c" >/dev/null || { echo "offline packages incomplete: $c missing"; dpkg --audit; exit 1; }
  done
else
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq --no-install-recommends likwid fio python3-minimal gcc libc6-dev libopenblas-dev libdnnl-dev >/dev/null
fi
gcc -O2 -o /usr/local/bin/chase src/chase.c
gcc -O2 -fopenmp -o /usr/local/bin/cgemm src/cgemm.c -lopenblas -ldnnl
install -m 0755 src/cpuprobe /usr/local/bin/cpuprobe
echo "cpu toolkit ready"
