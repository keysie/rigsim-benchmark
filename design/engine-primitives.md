# Engine primitives: low-precision GEMM, small transfers, storage — design

Sprint `sprint/engine-primitives`, started 2026-11-04 (user). Status: **cleared by the user (2026-11-04): steps 1–5 in
this sprint**, step 6 (collectives) next. Building step by step. Covers the three gaps
2–4 of the coverage estimate (agent-docs findings, "Coverage of what LLM engines depend on"):

2. matrix throughput in the formats engines actually run (bf16, fp8, fp4, int8, weight-only int4/int8),
3. latency and bandwidth of small transfers (host ↔ GPU, GPU ↔ GPU, between CPU cores),
4. storage read speed (model loading, experts read from SSD, mmap page faults).

Every measurement must serve two uses:

- **(a) the hardware model** (rigsim's four primitives: compute, storage, link, router): a value is useful when it
  identifies one parameter of one primitive — a peak, an efficiency, a fixed cost, a capacity of a shared link.
- **(b) the engine bound**: an upper limit on what an engine can reach on this machine, computed from the values without
  running the engine — decode ≤ bytes per token ÷ effective read rate, prefill ≤ FLOPs ÷ effective matrix rate,
  tensor parallel ≤ 1 ÷ (compute + 2 × layers × all-reduce time), experts from SSD ≤ bytes per token ÷ random-read rate.

Vendors: CPUs Intel and AMD (x86_64 only, as the stick); GPUs NVIDIA (CUDA), AMD (ROCm/HIP), Intel (oneAPI/SYCL,
Level Zero), plus Vulkan as the common path. Where a vendor's library has no such data type the step records
`unsupported` with the reason, not a failure (rule D of the stick sprint).

General rules for all new measurements:

- **Sweeps, not single points.** One size or shape cannot separate a fixed cost from a rate, nor a memory-bound from a
  compute-bound regime. Every new measurement is a sweep whose shape the model is fitted to (α + n/β for transfers,
  roofline for GEMM, Little's law for queues).
- **The conditions next to the value.** Clock, power and temperature of the device sampled during each point (the
  samplers exist for NVIDIA and DRM; ROCm needs `rocm-smi`/sysfs, Intel the sysfs frequency files), the ISA/kernel the
  library chose (oneDNN verbose, cuBLASLt heuristic name, rocBLAS solution), the buffer's NUMA node.
- **Budget.** All three items together add at most ~25 minutes on a typical one-GPU machine; every grid has a smoke
  variant.

---

## 2. Matrix throughput in the formats engines use

### What engines run

| Engine path | Data types in the hot loop | Who implements it |
|---|---|---|
| vLLM / SGLang, dense fp16/bf16 | bf16 or fp16 × same, fp32 accumulate | cuBLAS(Lt), hipBLASLt, oneDNN/oneMKL (XPU) |
| vLLM fp8 (Ada, Hopper, Blackwell, MI300, RDNA4) | e4m3 × e4m3, scaled, fp32 accumulate | cuBLASLt / hipBLASLt |
| vLLM fp4 (Blackwell) | nvfp4 / mxfp4 block-scaled | cuBLASLt |
| W8A8 int8 (SmoothQuant etc.) | int8 × int8 → int32 | cuBLASLt IMMA, hipBLASLt, oneDNN |
| Weight-only int4 / int8 (AWQ, GPTQ, Marlin, GGUF) | int4/int8 weights dequantised to fp16/bf16 in the kernel | engine kernels (Marlin, llama.cpp mmvq/mmq) |
| CPU offload (Strata experts, llama.cpp CPU) | int8 VNNI / AMX, bf16 AVX512_BF16 / AMX, quantised GGUF kernels | oneDNN, llama.cpp |

### Hardware-level questions

1. **Peak per data type and its clock dependence** — the matrix units' rate per SM / CU / Xe-core per clock, per type.
   Spec sheets give the theoretical figure; the measurement gives the share a library reaches at the measured clock.
2. **The shape curve** — rate versus M (tokens in flight) at fixed N×K typical for LLM layers. Small M is limited by
   reading the weights (memory-bound GEMV region), large M by the matrix units; the bend sits at the arithmetic
   intensity where they meet. This is exactly what bounds decode (M = batch) and prefill (M = chunk).
3. **Weight-only quantisation cost** — how much a dequantising kernel loses against the plain type at the same M
   (extra instructions per weight); vendor-neutral through llama.cpp's own kernels.

### Measurements

**GPU, vendor libraries** (CUDA probe `probe.cu`, its HIP build in the ROCm image, `probe_sycl` in the oneAPI image):

- Types: fp32, tf32, fp16 (fp16 acc), fp16_fp32acc, bf16, int8 (all exist in the probes already; gpu-gemm on CUDA
  only runs fp32/fp16/int8 today) **plus** fp8 e4m3 (cuBLASLt with scales: sm_89+; hipBLASLt: gfx942, gfx12) and
  nvfp4/mxfp4 (cuBLASLt block-scaled, sm_100/sm_120) — new code in the probes, `unsupported` elsewhere.
- Shapes: M ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192} × (N, K) ∈ {(4096, 4096),
  (14336, 4096), (4096, 14336)} — the projections of an 8B-class layer; plus one square 8192³ for the peak.
  14 × 3 + 1 = 43 shapes per type; at ~0.2 s each ~1.5 min per type and card. Smoke: M ∈ {1, 512}, one N×K.
- Per point: ops/s, achieved bytes/s of the weight read (2·N·K·size/time) in the small-M region, mean SM clock and
  power during the point, the kernel/heuristic name.
- Vulkan (vkprobe, all vendors): its matrix kernel is a plain shader (lower bound); keep as is, add fp16 and int8
  dot-product variants where the device reports them (`VK_KHR_cooperative_matrix` types listed in vulkaninfo).

**GPU, llama.cpp kernels** (vendor-neutral, weight-only quantisation): test-backend-ops' perf cases for `MUL_MAT`
already run n = 1, 2, 3, 4, 5, 8 and 512 at m×k = 4096×14336 for f32, f16, bf16, q4_0, q8_0, q4_K, q5_K, q6_K, iq4_xs,
mxfp4 on every backend (checked in run `…-03de`) — the decode region and one prefill point, no change needed. The
cases are fixed in the binary; n = 16 … 256 (batched serving) would need a patch to the llama.cpp build — later, if the
gap shows in the data. What is missing is the *extraction*: these values are not in `tools/measurements.py` yet.

**CPU** (cpu-gemm, oneDNN matmul, Intel and AMD):

- Types: fp32, bf16 (AVX512_BF16 on Cooper Lake / Sapphire Rapids+ / Zen 4+, AMX-BF16 on Sapphire Rapids+), fp16
  (AVX512-FP16 / AMX-FP16 on Granite Rapids), int8 u8×s8 (VNNI, AVX-VNNI, AMX-INT8) — today fp32, fp64, int8.
- The same M sweep at N×K = 4096×4096 and 14336×4096, one core and all cores.
- Record the ISA oneDNN chose (`ONEDNN_VERBOSE=1`: e.g. `brg:avx512_core_amx`) and the cpuinfo flags — on hybrid CPUs
  also which core types ran (pinning from the ladder sprint's open point).

### What it gives

- (a) Per device and type: peak share at the measured clock (library efficiency), the effective weight-read rate in
  the GEMV region (to compare with stream bandwidth: how much of VRAM/RAM bandwidth a GEMV reaches), the bend point.
  These become rates per (operation class, data type) in rigsim's `hw/rates.py`; engines then only state their
  data types and shapes.
- (b) Decode bound per type: weight bytes ÷ GEMV-region rate at M = batch; prefill bound: FLOPs ÷ rate at M = chunk;
  the gain of fp8 / fp4 / int4 against bf16 on *this* card, measured instead of assumed (2× / 4× on paper).

### Open questions

- fp8/fp4 need scale tensors and cuBLASLt matmul descriptors — real code; worth it only on cards that have them
  (Ada, Hopper, Blackwell, MI300, RDNA4). Proposal: implement for CUDA first, HIP second.
- Marlin/Machete (vLLM's int4 kernels) are not a library; llama.cpp's kernels stand in for "weight-only int4". The
  difference stays an engine factor in rigsim.

---

## 3. Small transfers

### Why

Engines move many small pieces: tensor parallel all-reduces one activation per layer per token (decode: hidden ×
batch × 2 bytes = 8 KiB … 1 MiB, twice per layer); Strata/llama.cpp offload sends activations between CPU and GPU per
layer and pages experts (1–50 MiB each) on demand; vLLM swaps KV blocks (16 tokens × layers). At these sizes the fixed
cost per transfer dominates and today's 512 MiB copies say nothing about it.

### Hardware-level questions

1. **α and β per path** — fixed cost and asymptotic rate of host→GPU, GPU→host, GPU→GPU (with and without peer
   access), in both directions and at the same time; the size where half the rate is reached (n½).
2. **Pinned versus pageable host memory** — engines use both; pageable goes through a staging buffer.
3. **Overlap** — does a copy run alongside a kernel at full speed (copy engines), and what does it cost the kernel.
4. **Shared links** — two copies through one root port / switch / chipset uplink at once (the router primitive).
5. **Collectives** — all-reduce and all-gather time versus message size over 2…N GPUs (ring/tree, PCIe vs NVLink,
   consumer cards without P2P go through host memory).
6. **CPU side** — core-to-core cache-line latency (same CCX/CCD, across CCDs on AMD, P- vs E-cores on Intel, across
   sockets) and the cost of a barrier over N threads: llama.cpp's CPU path synchronises all threads after every
   operation (hundreds per token), so this bounds small-model CPU decode.

### Measurements

**Host ↔ GPU** (CUDA probe, HIP probe, `probe_sycl`; one implementation per vendor API):

- Sizes 4 KiB … 256 MiB in ×4 steps (12 points), each: (i) one synchronous copy at a time — latency; (ii) a queue of
  copies with one sync at the end — rate; pinned and pageable; h2d, d2h, bidirectional; repeats until ≥ 0.2 s.
- Overlap: a fixed compute kernel (the probe's own) alone, then with a 256 MiB copy stream alongside — both rates.
- All cards at once (exists for 512 MiB) extended to 3 sizes (64 KiB, 4 MiB, 256 MiB).
- The host buffer's NUMA node and the card's node recorded (`numa_node` in sysfs); on two-socket machines the copy
  once from each node.
- Integrated GPUs (UMA): copies are memcpys in the same RAM; measure instead the GPU's read rate of a host-allocated
  (zero-copy / host-visible) buffer in vkprobe / `probe_sycl` — the path an iGPU engine uses.

**GPU ↔ GPU:**

- Pairwise copy sweep (same 12 sizes) with and without peer access, one direction and both (extends gpu-p2p).
- Collectives with the vendor library's own benchmark, built from source and pinned like llama.cpp:
  `nccl-tests` (all_reduce_perf, all_gather_perf; NVIDIA), `rccl-tests` (AMD) — sizes 8 B … 256 MiB, ×2 steps,
  2 … all GPUs. Intel multi-GPU (oneCCL) only if such a machine shows up. Output: algorithm/bus bandwidth and time per
  size; NCCL's chosen algorithm/protocol (`NCCL_DEBUG=INFO`) and topology (`NCCL_TOPO_DUMP_FILE`).

**CPU cores** (cpu toolkit, new C probe, Intel and AMD):

- Core-to-core latency matrix: cache-line ping-pong between every pair of physical cores (or a sample on > 32 cores),
  ~5 ms per pair.
- Barrier cost: spin barrier and futex barrier over 2 … all threads, with the threads on P-cores only / all cores /
  one CCD / across CCDs.

### What it gives

- (a) α/β per link and direction, the router capacity of shared uplinks, copy-engine overlap as a yes/no + loss —
  directly the link and router primitives of rigsim's machine graph; core-to-core latency per topology level for the
  CPU graph.
- (b) Tensor-parallel decode bound = per-layer compute + 2 × all-reduce(t(size)); expert paging bound = experts per
  token × t(expert size); CPU decode floor = ops per token × barrier time.

### Open questions

- nccl-tests/rccl-tests add build time and need MPI-free mode (`-g N` single process) — both support it.
- Pageable copies depend on the kernel's page cache and THP; record THP state.

---

## 4. Storage read speed

### Why

Model loading (tens to hundreds of GB), MoE experts streamed from NVMe when RAM is short (Strata/llama.cpp with mmap,
KTransformers-style offload), and the page-fault path when a model is mmapped and evicted. Today the stick measures no
drive at all; drives are the fourth primitive in rigsim and still unmodelled.

### Hardware-level questions

1. **Sequential read rate** per drive (large blocks, queue depth) and what limits it: the drive, its PCIe link (from
   `inventory/pcie.json`), or a shared uplink (chipset DMI/PCIe uplink shared with chipset-attached GPUs and USB).
2. **Random read rate versus block size and queue depth** — 4 KiB (page faults), 64 KiB–1 MiB (readahead), 4–16 MiB
   (expert-sized); latency per request at QD 1 (Little's law: rate = QD × size ÷ latency).
3. **Contention** — the drive read while a GPU on the same uplink copies host→GPU (chipset-attached cards: the
   3060 on the reference box sits behind the chipset).
4. **The buffered path** — rate through the page cache (readahead), which is what mmap-based loaders see.

### Measurements

- **Read-only, on the raw block device, never writing**: `fio` with `--readonly` (refuses any write job) and
  `direct=1`, on every internal NVMe/SATA drive ≥ 8 GB except the stick; libaio/io_uring; each job time-limited.
  Nothing read is kept — only timings.
- Jobs: sequential 1 MiB at QD 1, 8, 32; random 4 KiB at QD 1, 32, 128; random 128 KiB QD 8; random 4 MiB and 16 MiB
  at QD 1 and 4 — ~10 s each, spread over the device (`--size` = whole device, random offsets), ~2 min per drive.
- Buffered sequential read of 4 GiB (`direct=0`, caches dropped first) — the readahead path.
- Contention: sequential 1 MiB QD 32 alone, then together with `probe copy --dir h2d` on each GPU (5 s each).
- Recorded: see "Decided" below — kind, link, rounded capacity, model family, temperature before/after; an OPAL-locked
  drive reads as errors → `unsupported`.

### What it gives

- (a) The storage primitive: sequential capacity, per-request latency, the queue depth needed for full rate (Little's
  law), the share of a shared uplink — the same graph solver handles drive + GPU flows over the chipset link.
- (b) Load time = model bytes ÷ sequential rate (direct) or buffered rate (mmap); MoE-from-SSD bound = expert bytes per
  token ÷ random-read rate at the engine's queue depth and block size.

### Decided (user, 2026-11-04)

- **Read access to the machine's drives: yes, as long as no content and no identifying feature of a drive ends up in
  the report.** So the run records only timings and the drive's kind: interface (NVMe/SATA), PCIe link, capacity
  rounded to a decade step, vendor/model family, and NVMe temperature. Never: serial, WWN/EUI-64/NGUID, namespace or
  partition UUIDs, labels, partition tables, filesystem types, firmware revision, SMART counters beyond temperature,
  or any byte read. Nothing read is written anywhere. `rigsim.nodisks` on the kernel command line skips the question.
- Drives behind RAID/VMD (Intel VMD hides NVMe from Linux without `vmd`): record `unsupported` with the reason.

---

## Order and size of the work

| Step | What | Effort | Budget per run |
|---|---|---|---|
| 1 | GEMM: bf16/tf32/fp16_fp32acc on CUDA via the existing probe types, M sweep on all vendor probes; CPU bf16 / AMX via oneDNN; MUL_MAT n-sweep in test-backend-ops | small | +6–10 min |
| 2 | Transfer sweeps in the three probes (h2d/d2h/bidir, pinned/pageable, overlap), P2P sweep | medium | +3–5 min |
| 3 | Storage with fio, read-only | medium | +2 min per drive |
| 4 | CPU core-to-core + barrier probe | small–medium | +1–3 min |
| 5 | fp8 / fp4 in the CUDA probe (cuBLASLt), then HIP | medium–large | +2 min |
| 6 | nccl-tests / rccl-tests on multi-GPU machines | medium | +3–6 min |

Each step: new or extended questions (additive run format), new value keys in `tools/measurements.py` (VERSION bump),
the matching rigsim primitives afterwards (in rigsim, separately). Decided: steps 1–5 in this sprint, 6 next.
