# Open points — rigsim-benchmark

Roughly by priority (top = most important). Tags: **[user]** = stated by the user or in a repo document,
**[agent]** = derived by an agent from reading, not yet confirmed. Closed items move to the journal, not here.
Last revised: 2026-10-31.

## High

1. **Old GPUs: CUDA on Pascal / Volta / Turing** [user, 2026-10-31; sprint `sprint/old-gpus`; design proposed, not
   cleared]. Facts (read from source and runs): the default stick loads NVIDIA's open kernel modules (Turing and newer
   only); Pascal needs the proprietary modules (`NVIDIA_FLAVOUR=legacy`, a separate ISO today); runs already contain
   Quadro P400/P1000 and GTX 970 machines with no CUDA data at all. The CUDA probe (`bench/hw/toolkits/cuda/setup.sh`) is
   compiled for sm_80/86/89/90 + compute_90 PTX only, so it cannot run on Turing, Volta or Pascal even with a driver;
   llama.cpp CUDA is built for 52;61;70;75;… — sm_60 (P100) only via JIT from compute_52 PTX. CUDA image 12.8.1 still
   supports Pascal; CUDA 13 dropped it. Proposal: (a) both module flavours of driver branch 580 on one stick, chosen at
   boot from the PCI IDs (any Blackwell → open, Pascal/Volta skipped and recorded; else any pre-Turing → proprietary;
   else open); same user-space libraries; (b) CUDA probe and llama.cpp built for 60;61;70;75 too; keep a CUDA 12.x
   build for pre-Turing when the main build moves to CUDA 13, stick picks by compute capability; (c) P100 forks as extra,
   separately keyed llama-bench runs, only on sm_60/61, pinned and built from source; (d) driver branch 580 is the last
   for Pascal — a later move of the main stick to a newer branch would need two user-space stacks.
   **Revised 2026-10-31 (user: mixed hardware should get every card measured; fallentree ignored):** switch modules
   *during* the run instead of choosing once at boot. Phase 1 boots with the open modules and measures every card they
   drive (Turing and newer, Blackwell included); if pre-Turing cards are present, phase 2 stops every GPU user (Docker
   containers, nvidia-persistenced, nvidia-drm's console binding), unloads nvidia_uvm / nvidia_drm / nvidia_modeset /
   nvidia, loads the proprietary set from its own directory and measures only those cards. Both flavours must be the
   exact same driver version (one user-space stack). Cross-phase multi-GPU tests (P2P, tensor split across old and new
   cards) are impossible and recorded as such; results are keyed by PCI address, which survives the reload. If the unload
   fails, the old cards are recorded as skipped with the reason. *Inferred, to verify on hardware:* the open module
   ignores pre-Turing GPUs and the proprietary one ignores Blackwell while driving the rest; unload works headless.

2. **Breadth of the data** [agent]: 15 runs from 12 machines, one machine has 4. rigsim's roadmap wants low-level
   measurements to calibrate and validate the hardware model (rigsim `engine.md` §11 step 2), which needs a spread of
   CPUs, GPU generations and memory configurations. Ask the user which machines are available next and which gaps matter
   (AMD/Intel GPUs, older NVIDIA, DDR5, dual-socket).
3. **Measurement questions aimed at single primitives** [user, via rigsim open points]: core-type compute per operation
   class, single-core bandwidth, GPU clock vs power limit under compute- and memory-heavy load, host-device copies,
   reference GEMM per precision. Check which of these the ten questions in `stick/questions/` already answer, and which
   need new ones (new question = new stick minor version, additive format change, new value keys).
4. **Per-kernel timings as the source of engine kernel factors** [user, via rigsim]: rigsim's kernel factors are
   unmeasured (1.0) and must come from per-kernel measurements, not llama-bench. `test-backend-ops` data exists on
   the stick; confirm with the user that `tools/measurements.py` should expose it as values (new keys, `VERSION` bump).
5. **CPU power over time** [user, via rigsim `cpu-compute.md`]: real CPUs run PL2 for up to tau, then PL1. The stick has
   `cpu-power-time` and `powercap.json` (RAPL limits, format 1.4); whether they cover tau and temperature is not verified.

## Medium

6. **UEFI boot fails on some machines without CSM** [user, 2026-10-31; symptom: the firmware sees the stick but offers
   no boot option]. **Fix written** on `sprint/stick-data` (stick 0.13.5): GPT hybrid layout like current Ubuntu ISOs
   (`stick/tools/layout.sh`, shared by build and flash), FAT16 ESP, BIOS core with `configfile`. QEMU boot test passes
   (UEFI, UEFI + Secure Boot, BIOS; CI "Boot test"). Open: confirm on one of the affected machines; old-layout sticks
   cannot be updated in place (flash offers seed-only or erase).
7. **Make sure the stick collects the right data** [user, 2026-10-31: first priority]. Findings: see findings "Audit of all
   15 runs". Plan cleared by the user. (A) build-time seed self-test — **written** on `sprint/stick-data` (stick 0.13.5,
   `stick/tools/selftest.py`): stops the build when a part asked for is missing or a binary lacks libraries;
   not yet run in a real build; (B) Blender's missing libraries (PR #19, merged) and the ladder grid (PR #22, merged) — both not verified on hardware; (C) make package energy readable or record why not; (D) record unsupported hardware as skipped with
   the reason, not as a failure; (E) triage the SYCL and Vulkan aborts; (F) a run-completeness audit (expected vs
   present data per machine class) as a tool and on the data page; (G) an end-to-end test (VM) if the user wants one.

8. **Fix the new CPU questions** [agent, 2026-10-31]: (a) derive the ladder grid from the CPU count so runs stop ending
   `done with failures`; (b) make package energy readable (find out why `energy_uj` is −1 in the container: permissions,
   Secure Boot lockdown or sysfs masking — *unknown*); (c) expose cpu-gemm, ladder and power-time as value keys
   (`measurements.py` `VERSION` bump); (d) consider per-thread rates or pinning so hybrid/SMT ladders are interpretable;
   (e) take the idle-start reading after a cool-down or record the starting state. Each is a stick or tools change; propose first.

9. **Value coverage in `tools/measurements.py`** [agent]: rigsim predicts only keys that were measured and supported;
   check the share of `unsupported` in `predictions/v0.3.0/` per key family and report which GPUs or CPUs are missing
   from rigsim's catalogue.
10. **Tests in CI** [agent]: CI runs `stick/tests` and `tests` only; there are no hardware-free tests of a full build or
   flash. Whether that is acceptable is a user decision.

## Low

11. **Blender**: runs by the official offline route with script version 3.1.1 (default); revisit when the script or scenes
   change [user: stick/README].
12. **llama.cpp release archives** are not used because they lack `test-backend-ops`; revisit if that changes [user:
    stick/README, decided 2026-10-29].
13. **Known stick limits** [user: stick/README]: no CUDA below Turing in the default image (use `NVIDIA_FLAVOUR=legacy`;
    Kepler and older not at all), MLC cannot switch prefetchers off under Secure Boot, ROCm only officially on some AMD
    generations, vkprobe's matrix GEMM is a plain kernel (lower bound), RAM timings as the BIOS set them are often
    invisible from Linux.
