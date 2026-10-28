#!/bin/bash
# Runs inside the oneAPI image at stick build time (build.sh): the SYCL probe (oneMKL) and llama.cpp's llama-bench
# with the SYCL backend, plus Intel's GPU compute runtime (Level Zero) if the image lacks it. Lands in /rigsim; the
# container is then committed as the stick's oneAPI image.   /src/probe_sycl.cpp (read-only mount), $LLAMA_REF
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true
apt-get update -qq
apt-get install -y -qq git cmake >/dev/null
if ! ls /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so* >/dev/null 2>&1; then
  # Intel's GPU repository has the current compute runtime (Arc B-series, Lunar Lake); Ubuntu's is the fallback
  if curl -fsSL https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg; then
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" \
      > /etc/apt/sources.list.d/intel-gpu.list
    apt-get update -qq || true
  fi
  apt-get install -y -qq intel-level-zero-gpu level-zero intel-opencl-icd >/dev/null \
    || apt-get install -y -qq libze-intel-gpu1 libze1 intel-opencl-icd >/dev/null \
    || echo "WARN: no Level Zero GPU runtime in the image"
fi
mkdir -p /rigsim && cd /rigsim
icpx --version | head -1 > ONEAPI_VERSION
if icpx -fsycl -O3 -std=c++17 -DRIGSIM_MKL_INT8 -qmkl -o probe_sycl /src/probe_sycl.cpp 2> probe-build.log; then
  echo "probe_sycl built (with int8)"
elif icpx -fsycl -O3 -std=c++17 -qmkl -o probe_sycl /src/probe_sycl.cpp 2>> probe-build.log; then
  echo "probe_sycl built (no int8 gemm in this oneMKL)"
else
  echo "probe_sycl NOT built (probe-build.log):"; tail -5 probe-build.log; rm -f probe_sycl
fi
git clone -q https://github.com/ggml-org/llama.cpp /tmp/llama && cd /tmp/llama && git checkout -q "$LLAMA_REF"
cmake -B b -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF >/dev/null
cmake --build b --target llama-bench test-backend-ops -j"$(nproc)"
cp b/bin/llama-bench b/bin/test-backend-ops /rigsim/ && git rev-parse HEAD > /rigsim/LLAMA_COMMIT
cd / && rm -rf /tmp/llama && apt-get clean && rm -rf /var/lib/apt/lists/*
printf '#!/bin/bash\nsource /opt/intel/oneapi/setvars.sh >/dev/null 2>&1 || true\nexec "$@"\n' > /rigsim/run.sh && chmod +x /rigsim/run.sh
echo "oneapi image ready"
