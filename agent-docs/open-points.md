# Open points — rigsim-benchmark

Roughly by priority (top = most important). Tags: **[user]** = stated by the user or in a repo document,
**[agent]** = derived by an agent from reading, not yet confirmed. Closed items move to the journal, not here.
Last revised: 2026-10-31.

## High

1. **Breadth of the data** [agent]: 15 runs from 12 machines, one machine has 4. rigsim's roadmap wants low-level
   measurements to calibrate and validate the hardware model (rigsim `engine.md` §11 step 2), which needs a spread of
   CPUs, GPU generations and memory configurations. Ask the user which machines are available next and which gaps matter
   (AMD/Intel GPUs, older NVIDIA, DDR5, dual-socket).
2. **Measurement questions aimed at single primitives** [user, via rigsim open points]: core-type compute per operation
   class, single-core bandwidth, GPU clock vs power limit under compute- and memory-heavy load, host-device copies,
   reference GEMM per precision. Check which of these the ten questions in `stick/questions/` already answer, and which
   need new ones (new question = new stick minor version, additive format change, new value keys).
3. **Per-kernel timings as the source of engine kernel factors** [user, via rigsim]: rigsim's kernel factors are
   unmeasured (1.0) and must come from per-kernel measurements, not llama-bench. `test-backend-ops` data exists on
   the stick; confirm with the user that `tools/measurements.py` should expose it as values (new keys, `VERSION` bump).
4. **CPU power over time** [user, via rigsim `cpu-compute.md`]: real CPUs run PL2 for up to tau, then PL1. The stick has
   `cpu-power-time` and `powercap.json` (RAPL limits, format 1.4); whether they cover tau and temperature is not verified.

## Medium

5. **MLC on long runs of small machines** [user decision 2026-11-04: left as it is]: an hour into the run the kernel
   granted only 750 of 1000 hugepages on the 8 GB laptop and MLC stopped (rc 22). Options if revisited: drop caches +
   compact memory before the request and retry; reserve the hugepages at boot; skip MLC with the reason when short.

6. **Stick: what is left after `sprint/stick-data`** [user plan of 2026-10-31; A, B, D, E and the UEFI boot are done and
   verified on hardware, see journal]. (C) package energy: `energy_uj` reads −1 in the container — permissions, Secure
   Boot lockdown or sysfs masking, *unknown*; (F) run-completeness audit (expected vs present data per machine class) as
   a tool and on the data page; (G) an end-to-end test in a VM, if wanted; copy `seed/SELFTEST.json` into each run
   (additive format change). Propose first.
7. **CPU questions, rest** [agent, 2026-10-31; ladder grid done]: (a) expose cpu-gemm, ladder and power-time as value keys
   (`measurements.py` VERSION bump); (b) per-thread rates or pinning so hybrid / SMT ladders are interpretable; (c) take
   the idle-start reading after a cool-down or record the starting state. Propose first.

8. **Value coverage in `tools/measurements.py`** [agent]: rigsim predicts only keys that were measured and supported;
   check the share of `unsupported` in `predictions/v0.3.0/` per key family and report which GPUs or CPUs are missing
   from rigsim's catalogue.
9. **Tests in CI** [agent]: CI runs `stick/tests` and `tests` only; there are no hardware-free tests of a full build or
   flash. Whether that is acceptable is a user decision.

## Low

10. **Blender oneAPI** [agent, 2026-10-31]: Blender's own oneAPI device needs Intel's Level Zero runtime and GPU
   compute runtime in the system image (Blender runs there, not in the oneAPI image); today Blender finds no oneAPI
   device (run `…-9c93`). Adding them = more system image size; decide with the user.

11. **Blender**: runs by the official offline route with script version 3.1.1 (default); revisit when the script or scenes
   change [user: stick/README].
12. **llama.cpp release archives** are not used because they lack `test-backend-ops`; revisit if that changes [user:
    stick/README, decided 2026-10-29].
13. **Known stick limits** [user: stick/README]: no CUDA below Turing in the default image (use `NVIDIA_FLAVOUR=legacy`;
    Kepler and older not at all), MLC cannot switch prefetchers off under Secure Boot, ROCm only officially on some AMD
    generations, vkprobe's matrix GEMM is a plain kernel (lower bound), RAM timings as the BIOS set them are often
    invisible from Linux.
