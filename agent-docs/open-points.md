# Open points — rigsim-benchmark

Roughly by priority (top = most important). Tags: **[user]** = stated by the user or in a repo document,
**[agent]** = derived by an agent from reading, not yet confirmed. Closed items move to the journal, not here.
Last revised: 2026-10-31.

## High

1. **Old GPUs: verify on hardware** [user, 2026-10-31; sprint `sprint/old-gpus`; **built**, stick 0.14.0, format 1.7].
   Built: the other module flavour harvested in `chroot-setup.sh` (`/usr/lib/rigsim/nvidia-swap`, same version checked
   at build and by the self-test); `stick.py` `nvidia_phase2` after every step (rmmod, measured cards kept off with
   `driver_override`, insmod nvidia / nvidia-uvm / nvidia-modeset, CUDA questions + llama-bench CUDA into `nvidia-swap/`);
   CUDA probe for sm_60/61/70/75, llama.cpp + sm_60; forks qr0n (`dc5c644c`) and Kmic-68 (`7461904d`) for sm_60/61 as
   backend `cuda+<fork>`; validator and measurements (VERSION 4: second-phase cards continue gpuN). To verify: the
   build (fork builds in the CUDA 12.8 image, proprietary late-link in the chroot), the unload on a real machine,
   which cards each flavour actually drives, cuBLAS precisions a Pascal card does not have (tf32, bf16) — probably
   failures in gpu-gemm that should be recorded as unsupported. Not covered: Vulkan / OpenCL / Blender for the
   second-phase cards. rigsim's `predict.py` does not know `cuda+<fork>` (returns unsupported) — a rigsim topic.



2. **Engine primitives** [user, 2026-11-04; sprint `sprint/engine-primitives`; design proposed]: low-precision GEMM with
   M sweeps (all GPU vendors, CPU oneDNN incl. AMX/bf16, llama.cpp MUL_MAT n-sweep), small-transfer sweeps (α/β,
   pinned/pageable, overlap, P2P, collectives via nccl-/rccl-tests, CPU core-to-core + barriers), storage via read-only
   fio. Details and order: `design/engine-primitives.md`.

3. **Breadth of the data** [agent]: 15 runs from 12 machines, one machine has 4. rigsim's roadmap wants low-level
   measurements to calibrate and validate the hardware model (rigsim `engine.md` §11 step 2), which needs a spread of
   CPUs, GPU generations and memory configurations. Ask the user which machines are available next and which gaps matter
   (AMD/Intel GPUs, older NVIDIA, DDR5, dual-socket).
4. **Measurement questions aimed at single primitives** [user, via rigsim open points]: core-type compute per operation
   class, single-core bandwidth, GPU clock vs power limit under compute- and memory-heavy load, host-device copies,
   reference GEMM per precision. Check which of these the ten questions in `stick/questions/` already answer, and which
   need new ones (new question = new stick minor version, additive format change, new value keys).
5. **Per-kernel timings as the source of engine kernel factors** [user, via rigsim]: rigsim's kernel factors are
   unmeasured (1.0) and must come from per-kernel measurements, not llama-bench. `test-backend-ops` data exists on
   the stick; confirm with the user that `tools/measurements.py` should expose it as values (new keys, `VERSION` bump).
6. **CPU power over time** [user, via rigsim `cpu-compute.md`]: real CPUs run PL2 for up to tau, then PL1. The stick has
   `cpu-power-time` and `powercap.json` (RAPL limits, format 1.4); whether they cover tau and temperature is not verified.

## Medium

7. **DDR timing improvements** [user, 2026-10-31, from a side chat; sprint `ddr-timing-improvements`; checked,
   **parked** by the user until the other branches are done; proposed order when resumed: 2 + 3 + lockdown state, 1 as
   a spike, 4 parked, 5 as a rigsim topic]. Five items; assessment (measured = from the 15 runs): (1) programmed timings from controller registers —
   10 of 15 runs had Secure Boot on, where kernel lockdown (*inferred*: Ubuntu enables it with Secure Boot) blocks
   /dev/mem, PCI config writes and debugfs, i.e. Intel MCHBAR (MMIO) and AMD SMN (config index/data) reads; register
   maps differ per generation — high effort, only for Secure-Boot-off machines; propose as a spike, and record the
   lockdown state (`/sys/kernel/security/lockdown`) first. (2) SPD status in `run.json` — cheap; 7 of 15 runs have "No
   EEPROM found", visible only in `inventory/spd.txt`; also worth finding out why (DDR5 / spd5118, i801). (3) latency
   probe — pin to one core and record it and its core type (hybrid CPUs: P vs E) and frequency (the runner records the
   governor, not the frequency); a 1 MiB point (grid is 64 / 1024 / 4096 MiB); p95/p99 as proposed do not work: chase
   averages 20M loads per pass and the median is over 5 pass averages — a tail needs per-access timing (rdtscp), a
   different probe. (4) row-hit vs row-miss — research-grade (DRAMA-style bank mapping), park. (5) predictions — the
   side chat's premise does not hold: the predicted value is `mlc/idle_latency`, and rigsim is too *high* already
   without CAS: 13700K DDR5-6200 measured 64.5 ns vs predicted 76.9 (no SPD, fixed term only), i7-9700 DDR4-2666 58.1
   vs 90.6 (with CAS). Adding programmed CAS raises predictions; the error is the fixed term (75.35 ns, from the X299
   mesh machine) — a per-platform term in rigsim, not timings.

8. **Second-phase values of the new GPU questions** [agent, 2026-11-04]: after the NVIDIA module switch the stick also
   runs `gpu-gemm-shapes`, `pcie-xfer`, `gpu-p2p-xfer`, `gpu-overlap` into `nvidia-swap/q/`, but `measurements.py`
   extracts only llama-bench and gpu-vram from there (with gpuN continued). Extend the gemm/xfer/overlap extractors the same way.

9. **Storage + GPU copy contention** [agent, 2026-11-04; design item 4.3]: drive reads while a chipset-attached GPU copies
   host→card — shows a shared chipset uplink. Needs fio and the CUDA probe at once (host fio + probe container).

10. **MLC on long runs of small machines** [user decision 2026-11-04: left as it is]: an hour into the run the kernel
   granted only 750 of 1000 hugepages on the 8 GB laptop and MLC stopped (rc 22). Options if revisited: drop caches +
   compact memory before the request and retry; reserve the hugepages at boot; skip MLC with the reason when short.

11. **Stick: what is left after `sprint/stick-data`** [user plan of 2026-10-31; A, B, D, E and the UEFI boot are done and
   verified on hardware, see journal]. (C) package energy: `energy_uj` reads −1 in the container — permissions, Secure
   Boot lockdown or sysfs masking, *unknown*; (F) run-completeness audit (expected vs present data per machine class) as
   a tool and on the data page; (G) an end-to-end test in a VM, if wanted; copy `seed/SELFTEST.json` into each run
   (additive format change). Propose first.
12. **CPU questions, rest** [agent, 2026-10-31; ladder grid done]: (a) expose cpu-gemm, ladder and power-time as value keys
   (`measurements.py` VERSION bump); (b) per-thread rates or pinning so hybrid / SMT ladders are interpretable; (c) take
   the idle-start reading after a cool-down or record the starting state. Propose first.

13. **Value coverage in `tools/measurements.py`** [agent]: rigsim predicts only keys that were measured and supported;
   check the share of `unsupported` in `predictions/v0.3.0/` per key family and report which GPUs or CPUs are missing
   from rigsim's catalogue.
14. **Tests in CI** [agent]: CI runs `stick/tests` and `tests` only; there are no hardware-free tests of a full build or
   flash. Whether that is acceptable is a user decision.

## Low

15. **Blender oneAPI** [agent, 2026-10-31]: Blender's own oneAPI device needs Intel's Level Zero runtime and GPU
   compute runtime in the system image (Blender runs there, not in the oneAPI image); today Blender finds no oneAPI
   device (run `…-9c93`). Adding them = more system image size; decide with the user.

16. **Blender**: runs by the official offline route with script version 3.1.1 (default); revisit when the script or scenes
   change [user: stick/README].
17. **llama.cpp release archives** are not used because they lack `test-backend-ops`; revisit if that changes [user:
    stick/README, decided 2026-10-29].
18. **Known stick limits** [user: stick/README]: no CUDA below Turing in the default image (use `NVIDIA_FLAVOUR=legacy`;
    Kepler and older not at all), MLC cannot switch prefetchers off under Secure Boot, ROCm only officially on some AMD
    generations, vkprobe's matrix GEMM is a plain kernel (lower bound), RAM timings as the BIOS set them are often
    invisible from Linux.
