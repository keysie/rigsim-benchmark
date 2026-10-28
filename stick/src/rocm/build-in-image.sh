#!/bin/bash
# Runs inside the ROCm image at stick build time (build.sh): the CUDA probe hipified and built with hipBLAS, and
# llama.cpp's llama-bench with the HIP backend, both for every GPU target this ROCm's compiler accepts out of
# $AMDGPU_TARGETS. Everything lands in /rigsim; the container is then committed as the stick's ROCm image.
#   /src/probe.cu (read-only mount), $LLAMA_REF, $AMDGPU_TARGETS
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
command -v git >/dev/null && command -v cmake >/dev/null || { apt-get update -qq; apt-get install -y -qq git cmake >/dev/null; }
mkdir -p /rigsim && cd /rigsim
echo 'int main() { return 0; }' > /tmp/t.hip
T=()
for a in $AMDGPU_TARGETS; do
  if hipcc --offload-arch="$a" -c /tmp/t.hip -o /tmp/t.o 2>/dev/null; then T+=("$a"); else echo "target $a: not supported by this ROCm"; fi
done
(IFS=';'; echo "${T[*]}") > TARGETS
echo "targets: $(cat TARGETS)"
cat /opt/rocm/.info/version > ROCM_VERSION 2>/dev/null || hipconfig --version > ROCM_VERSION
# probe: the CUDA source, hipified; built or recorded as missing (llama-bench still runs)
hipify-perl /src/probe.cu > probe.hip.cpp
if hipcc -O3 -std=c++17 $(printf -- '--offload-arch=%s ' "${T[@]}") -o probe probe.hip.cpp -lhipblas -lpthread 2> probe-build.log; then
  echo "probe built"
else
  echo "probe NOT built (probe-build.log):"; tail -5 probe-build.log; rm -f probe
fi
git clone -q https://github.com/ggml-org/llama.cpp /tmp/llama && cd /tmp/llama && git checkout -q "$LLAMA_REF"
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -B b -DGGML_HIP=ON \
  -DGPU_TARGETS="$(cat /rigsim/TARGETS)" -DAMDGPU_TARGETS="$(cat /rigsim/TARGETS)" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF >/dev/null
cmake --build b --target llama-bench test-backend-ops -j"$(nproc)"
cp b/bin/llama-bench b/bin/test-backend-ops /rigsim/ && git rev-parse HEAD > /rigsim/LLAMA_COMMIT
cd / && rm -rf /tmp/llama
# Blender's HIP device runs in this image (the system image has no ROCm runtime): the X / GL libraries Blender
# loads at start, even in background mode
apt-get update -qq && apt-get install -y -qq --no-install-recommends libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 \
  libxxf86vm1 libxfixes3 libegl1 >/dev/null || echo "WARN: Blender libraries not installed (Blender HIP unavailable)"
rm -rf /var/lib/apt/lists/*
printf '#!/bin/bash\nexec "$@"\n' > /rigsim/run.sh && chmod +x /rigsim/run.sh
echo "rocm image ready"
