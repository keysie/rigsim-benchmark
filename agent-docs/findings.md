# Findings — rigsim-benchmark (curated)

Current facts only. Stale or superseded items are corrected or removed here and recorded in the journal.
Tag: *docs* (read from a repo document or source) / *measured* / *inferred* / *decision*. Last curated: 2026-10-31.

## Data and format

- State 2026-10-31 (docs, counted): stick 0.13.2, format 1.4, 15 run folders, 12 machine hashes, predictions v0.1.0/v0.2.0/v0.3.0.
- Runs are never changed after publication; there is no list of known errors; judging a result against a tool bug is up
  to the user of the data, from what the run records (decision, README).
- One submission/PR may only add new run folders exactly as the stick wrote them; a `stopped` run keeps only finished
  steps; nothing identifying; credit sits in `results/attribution/<run>.json` and may be absent (page: "Submitted by someone").
- `<machine>` = salted hash of board identifiers: groups runs of the same machine without identifying it.
- Runs without `format` (stick builds before 0.9.0) are not accepted. The validator knows every format up to the newest
  released with it; a newer run is rejected until the repo is updated.
- Shared runs are CC BY 4.0; the submitter confirms the machine owners agreed.

## Stick decisions (docs)

- 2026-10-29 decided: llama.cpp is built from source, not taken from release archives, because the archives lack
  `test-backend-ops` on the GPU backends; archives would save only ~5–10 % of the llama.cpp build, which runs only when
  the version changes. All backends come from one commit; rule, tag and commit are recorded.
- `LLAMA_REF` default `release` = newest patch of the previous release line; `latest-release` = newest release.
- Build is reproducible by recorded versions (`tools/manifest.py`): packages, image digests, llama.cpp commits and
  binary hashes, models' SHA-256, MLC and Blender versions; each run adds what actually ran.
- Screens are switched off while measuring (scanout costs integrated graphics a few percent of RAM bandwidth).
- Blender runs by the official offline route (build default script 3.1.1; the 0.13.1 runs used script 3.1.2 — check which is real), one device type at a time.
- Format 1.1–1.4 added, in order: GPU identity per result (`pci`, `uuid`), `pcie.json`, `operator_skips`, `powercap.json`.

## Interface with rigsim (docs: predictions/README.md, tools/predict.py)

- Predictions are keyed by value key, in the measured unit; only measured and supported keys are predicted; the rest is
  `unsupported` with a reason. Failed predictions are stored with `error` and retried only for the next tag.
- When `tools/measurements.py` `VERSION` changes, the newest tag predicts every run again.
- Timeout per prediction run in `predict.py`: 600 s.
- rigsim's `predict.py` (2026-10-31) supports GPUs by nvidia-smi name and memory from its catalogue, power limits, PCIe
  link from `pcie.json`, CPU memory values (MLC idle latency, all-core read bandwidth) for Intel Core 6th–13th gen and AMD
  Ryzen 1000–7000; no prediction is marked `calibration` (no run went into a constant).
- Pitfall: Windows checkouts convert line endings to CRLF and break the build scripts; `.gitattributes` forces LF but a
  wrong checkout keeps its endings (repair: `git rm -rq --cached . && git reset -q --hard` from WSL).

## New CPU questions: first results (stick 0.13.1, two runs, 2026-10-31; measured)

- `cpu-gemm-ladder`: the fixed grid (1…64 threads) fails every step above the machine's logical CPU count (rc 1), so
  every run ends `done with failures` (7600U: 8 steps, 13700K: 3). Harmless but noisy.
- `cpu-power-time`: works; `package_w` is −1 in both runs (energy counter unreadable inside the container), so only
  throughput, mean clock and temperature are recorded over time. The idle-start reading is taken at the start of the
  idle period, when a laptop can still be warm (73 °C).
- Hybrid/SMT parts: OpenBLAS static partition makes the ladder beyond the physical cores measure the slowest thread
  (see rigsim findings, same date).
- `tools/measurements.py` does not yet expose gemm, ladder or power-time values, so rigsim predicts none of them.

## Audit of all 15 runs (measured; 2026-10-31, by reading `run.json` of every run; unit tests: 133 pass)

Works in the runs: inventory, cpu-mem-latency, cpu-mem-ladder, cpu-gemm, NVIDIA questions, vkprobe, llama.cpp (CPU, CUDA,
Vulkan), tool versions and `stick_check`. Does **not** deliver data:
- **Blender: 0 of 15 runs have a successful result.** 8 runs: "not in the seed" (stick builds without it); 5 runs (0.11.0 ..
  0.13.1, both newest): rc 127 — the Blender binary cannot start, `libXfixes.so.3` missing in the run environment; the
  first two runs (0.9.0, 0.11.0): rc 1, cause not checked.
  Fix written: PR #19 (libraries installed in the system image, `ldd` check at build); not yet verified on a booted stick.
- **oneAPI:** on 9 of 15 runs "the runtime sees no GPU" (older Intel iGPUs; likely unsupported by Level Zero — *inferred*), recorded
  as rc 1 instead of skipped. On the 13700K (UHD 770) it works, but 5 of 92 steps abort (rc 134, the SYCL llama runs).
- **Vulkan on older Intel iGPUs:** test-backend-ops `mul_mat_id` and `flash_attn_ext` abort (rc −6) in all 8 runs on
  such machines — driver or ggml limit, not investigated.
- **Seed content differs between builds:** MLC missing in 2 runs ("not in the seed"), Blender in 8. A stick built without
  a component silently produces incomplete runs; the run records it, the data page does not warn.
- **cpu-gemm-ladder grid** fails above the CPU's logical count; **package energy** reads −1 (see above).
- Consequence: nearly every run ends `done with failures`, so the status carries no information about real problems.
- The unit tests cannot catch any of this: they check the stick's logic, not the contents of the seed or the run environment.

## Seed self-test (stick 0.13.5, decision 2026-10-31; read from source, not yet run in a real build)

- User decision: a missing part **stops the build** — it becomes an issue to fix instead of runs without the expected
  data. `ALLOW_INCOMPLETE=1` builds a test stick anyway. Parts left out by an option (empty `MLC_URL` /
  `BLENDER_LAUNCHER_URL`, `SKIP_MODELS`, `SKIP_ROCM`, `SKIP_ONEAPI`) count as "left out", not as failures.
- Checked: both question images and the offline debs; vkprobe, llama.cpp Vulkan, Blender and MLC with `ldd` in the system
  image (chroot, binary bind-mounted); llama.cpp CUDA with `ldd` inside the CUDA image (where it runs); 2 models; Blender's
  version, executable, three scenes and script, found by the stick's own `blender_offline` / `benchmark_script`; vendor
  image archives. Result in `seed/SELFTEST.json`. Replaces #19's warning-only Blender check.
- NVIDIA driver libraries (`libcuda.so*`, `libnvidia-*`) are never in the CUDA image: the container toolkit hands them in
  from the system image at `docker run --gpus`. They are checked in the system image (ldconfig in the chroot) instead.
  Found by the first real build with the self-test (2026-10-31), which flagged `libcuda.so.1` for llama.cpp CUDA.
- Not checked: that a binary actually runs (no GPU in the build); the ROCm/oneAPI images' contents.

## Boot and disk layout (stick 0.13.5, 2026-10-31; QEMU-tested, not yet on the affected machines)

- Up to 0.13.4 the image was an MBR-only hybrid: partition 1 (ISO) from sector 0, ESP as MBR type 0xEF, no boot flag.
  Some UEFI firmwares saw such a stick but offered no boot option (user report). Since 0.13.5: GPT with ISO partition
  from sector 64, a real GPT ESP, a gap partition, a protective MBR with the boot flag, GRUB hybrid MBR for BIOS;
  the data partition is GPT partition 4 (type Linux filesystem) behind the image, the backup GPT at the stick's end.
- The BIOS GRUB core lacked `configfile` (it was only in `--install-modules`): in QEMU legacy boot stopped at `grub>`
  with "configfile.mod not found", with the old and the new layout alike. Whether real machines booted legacy at all
  before is unknown (*inferred*: with CSM on, they may have used the UEFI route).
- Sticks flashed before 0.13.5 have their data partition running to the disk's last sector, where the GPT backup
  belongs: `flash.sh` does not offer [u]pdate for them (nor when the new image does not fit in front of the data
  partition); it says why and offers [s]eed only or [e]rase with backup.
- Test: `sudo bash stick/tests/boot/boot-test.sh` (root, QEMU, OVMF, SeaBIOS; ~1–2 min without KVM). QEMU's small
  `bios.bin` has no xHCI support; the test uses the default (256k) SeaBIOS.

## First hardware run of stick 0.13.5 (2026-10-31; 13700K + RTX 3080 + UHD 770, Secure Boot off; measured)

- Build with the seed self-test passed (after the libcuda fix); the run took 2 h 26 min.
- Blender: CPU, OptiX and CUDA each render all three scenes (~32–41 s per scene) — the first Blender data from the stick.
  oneAPI (UHD 770): rc 0 after 0–1 s per scene, i.e. no real render; logs not yet checked.
- `cpu-gemm-ladder`: 9 steps (1 … 24 threads), none failing — the cap works.
- gpu-p2p on a one-GPU machine: 2 items "failed", question rc 0.
- oneAPI on the UHD 770: 5 of 92 steps abort with rc 134 (SYCL llama runs, mul_mat, mul_mat_id, flash_attn_ext); the
  only cause of `done with failures` in this run. 45 min of the run go to oneAPI, ~47 min to Vulkan on the iGPU.
- Blender's benchmark script raises `ValueError: No matching device of type OneAPI found` when the device type is missing,
  and Blender still exits 0 ("Blender quit") — rc alone cannot tell a render from a non-render (run `…-9c93`).
- gpu-p2p on one card: the probe says "d2d needs two visible cards" (rc 1 per item).
- SYCL aborts on the UHD 770 (run `…-9c93`; read from the logs): llama-bench and test-backend-ops die in
  `ggml_sycl_op_mul_mat` (ggml-sycl.cpp:3743) with "could not create a memory object" — the text of oneDNN's
  `dnnl::error` (*inferred*). The single-token mul_mat cases pass (10–108 GFLOPS); it fails at the next batched case,
  where llama.cpp's SYCL backend hands GEMMs to oneDNN (`GGML_SYCL_ENABLE_DNN=1`, `GGML_SYCL_FA_ONEDNN=1` in its
  environment dump) — which would also explain mul_mat_id and flash_attn_ext. Hypothesis to test: the same run with
  `GGML_SYCL_ENABLE_DNN=0` (and `GGML_SYCL_FA_ONEDNN=0`). Not "unsupported hardware": a backend path failing.
- Manual test on the UHD 770 (2026-11-01, user, seeded oneAPI image): llama-bench TinyLlama Q4_0 aborts with oneDNN on,
  runs with `GGML_SYCL_ENABLE_DNN=0 GGML_SYCL_FA_ONEDNN=0`: pp512 220.4 tok/s, tg128 38.2 tok/s (3 repeats, spread
  < 0.2 %). Hypothesis confirmed for llama-bench; test-backend-ops MUL_MAT / FLASH_ATTN_EXT without oneDNN still open
  (the manual command lacked the stick's type filter, so MUL_MAT ran the full set and was interrupted).
  MUL_MAT (with the stick's type filter) without oneDNN: "2/2 backends passed, OK" (2026-11-01). FLASH_ATTN_EXT (stick filter) and MUL_MAT_ID
  without oneDNN: both "OK" too. **Hypothesis confirmed for all four aborted steps** (measured by the user).

## Second hardware run (stick 0.13.7, 2026-11-01, run `…-c72d`, same 13700K machine; measured)

- D and E work: gpu-p2p skipped with the reason; Blender oneAPI recorded as not available; the oneDNN retry rescued
  both llama-bench runs (tinyllama 44 s, llama-2-7b 146 s incl. the failed first attempt), MUL_MAT and MUL_MAT_ID.
- Remaining failure: FLASH_ATTN_EXT on the UHD 770 hit the 30-min test-backend-ops timeout (rc 127, 1800 s) with oneDNN
  on — a hang or a crawl instead of the abort of run `…-9c93`, so the retry (which keys on the error text) did not fire.
  Without oneDNN it passed in the manual test. Cost of the failed first attempts: ~30 min timeout + a few min each.
- Testing one stick step on a normal Linux host (2026-11-01, procedure given to the user): import `stick.py` from a
  checkout and call `run_vendor("oneapi", data, rundir, log)` with a scratch `data` dir whose `seed` links to the
  stick's seed (data partition mounted read-only) — never the stick's own data dir, or `load_image` writes its
  "loaded" marker there and the stick later skips loading the image into its own store. `vendor_plan` and
  `BACKEND_OPS` can be narrowed for a short test. Caveat: the host's kernel and GPU driver differ from the stick's.

## Older NVIDIA GPUs and P100 forks (2026-10-31; read from GitHub and the repo)

- **qr0n/p100-llama-cpp** (MIT, 25 stars, created 2026-09-07): upstream llama.cpp `b10660` + 18 patches, with a
  verify script that re-derives the tree from upstream; CUDA 12.x; claims e.g. Llama-3.1-8B Q4_K_M tg 39.3 → 79.6 on
  one P100 (fp16 matrix-vector path), tensor-split gains on 2+ cards; patch 0010 carries Kmic-68's code. Patches are not
  submitted upstream, by the author's decision.
- **Kmic-68/llama.cpp** (fork of ggml-org/llama.cpp; default branch `p100-optimizations`, pushed 2026-10-25): tuned for
  2× P100; cuBLAS-GEMM flash attention for long prefill and GQA-6 tile kernels.
- **qr0n/p100-strata** (MIT): Strata v0.2.2 + 6 patches for P100 (MXFP4 expert kernels, …). Relevant to rigsim's Strata
  engine, not to the stick (Strata is not on the stick).
- CUDA 13 dropped Pascal (stated in qr0n/p100-strata); the stick's CUDA image is 12.8.1.
- Space (2026-10-31, Ubuntu noble archive, driver 580.178.04, HWE kernel 7.0.0-42): the open modules package is 8.3 MB
  (nvidia.ko 20.7 MB uncompressed); the proprietary set comes as `linux-objects-nvidia-580-<kver>` (98 MB download, 194 MB
  installed, nv-kernel.o alone 113 MB) plus Canonical's detached signatures, linked at install time. Estimate for the old-GPU
  sprint: system image +30–100 MB (proprietary set, more if the objects stay), llama.cpp CUDA +~1/9 per added
  architecture, the two forks built for sm_60/61 only, firmware and user-space unchanged. To be measured on a build.
- Build facts (2026-10-31): Ubuntu's proprietary NVIDIA module package for a kernel is a stub that depends on
  `linux-objects-nvidia-580-<kver>` + `linux-signatures-nvidia-<kver>`; its postinst links the .ko files from `bits/`
  (late-link, debconf default true). Both flavours install to their own directory with the same module names and do
  not conflict as packages, so the swap flavour is installed, copied out and purged before the boot flavour.
- The CUDA probe used to be compiled for sm_80+ only: Turing, Volta and Pascal could not run it even with a driver.
- Existing 15 runs extract identical values with measurements VERSION 3 (checked 2026-10-31).
