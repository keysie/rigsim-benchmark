# Run format

Each run folder is one boot of the stick on one machine. The format has a name and a version, recorded in
`run.json` → `format`: `{"name": "rigsim-stick-run", "version": "1.0"}`.

## Compatibility rules

- **Minor versions only add.** A new minor version (1.1, 1.2, …) may add fields to `run.json` or `manifest.json`,
  add files, or add folders for new tools. Nothing is ever removed, renamed, moved, or given another type or
  meaning. Every reader of 1.x reads every 1.y run; readers ignore fields and files they do not know.
- **A change that cannot be additive is a new major version** (2.0) — and is avoided: a run once published stays
  readable by every tool that read it before.
- The validator (`tools/validate.py`) knows every version up to the newest it was released with; a run in a newer
  version than the repository knows is rejected until the repository is updated.
- Runs without `format` (stick builds before 0.9.0) are not accepted.

## Versions of everything

Every result can be traced to the exact software that produced it:

- **`run.json` → `stick`**: the stick's release (`version`, semver, from `stick/VERSION`), the build (`build`:
  short commit, `-dirty` if built from uncommitted changes), the full `commit`, `dirty`, and `scripts_sha256` — one
  hash over every script, question and toolkit file on the stick.
- **`run.json` → `questions` → `<step>` → `tools`**: for each step, the versions its results came from — container
  images by digest with the package versions inside them, llama.cpp commit and binary hashes per backend, the
  vendor images' ROCm / oneAPI versions, the models' SHA-256, MLC and Blender versions, the NVIDIA driver, Vulkan
  driver versions per device, the kernel, and the stick itself.
- **`manifest.json`**: everything at once — `system` (the system image's build manifest: packages, the stick's
  files), `seed` (the seed's build manifest: images, llama.cpp builds, vendor images, models, tools), `runtime`
  (what actually ran: kernel, NVIDIA driver, Docker, Vulkan drivers). System image and seed may come from different
  builds (`flash.sh` can update one without the other); both are recorded.

## Version 1.0 (2026-10-28)

`run.json` fields: `format`, `run` (= folder name), `stick`, `seed`, `fingerprint` (= middle part of the folder
name), `fingerprint_stable`, `start_utc`, `end_utc`, `status` (`done` / `done with failures` = complete; `stopped`,
`error: …` = not), `hardware` (`cpu`, `board`, `ram_gib`, `gpus`, …), `questions` (one entry per step: `rc` or
`skipped`, plus `tools`), `stick_check`, `stick_read`, `docker`, `displays`, `screens_blanked`.

Files: see [README.md](README.md) and the layout in `tools/validate.py` (the exact set of files a run may contain).

## Changelog

| Version | Stick | Change |
|---|---|---|
| 1.0 | 0.9.0 | first versioned format: `format`, `stick` as an object, `tools` per step, `manifest.json` |
| 1.7 | 0.15.0 | `swap_was_on` (swap areas that were on at the start, by type; the stick switches them off); `hardware.spd` (`status`: decoded / raw / unavailable, `reason`, `spd_drivers`, `smbus_adapters`, `raw_files`) and `hardware.lockdown` (the kernel's lockdown mode); `cpu-mem-latency` results carry `cpu`, `core_type`, `mhz_mean`, `mhz_max` (one line per core it ran on). Step `storage` (read-only drive timings with fio): `storage/drives.json` (per internal drive: `drive` = kernel name, `kind`, `capacity_class_gb`, `rotational`, `model`, `logical_block`, NVMe `link`, `temp_c_before`/`after`, `zero_share` = share of 256 evenly sampled 1 MiB blocks that read as zeros — nothing that identifies the drive, no content); entries of test-backend-ops ops may carry `time_limit` (stopped at the limit; the cases measured before are kept in the output) and `storage/results.json` (per drive and job: `rc`, `bytes_per_s`, `iops`, `lat_mean_us`, `lat_p50_us`, `lat_p99_us`, `ios`); new questions `cpu-gemm-shapes`, `gpu-gemm-shapes`, `pcie-xfer`, `gpu-p2p-xfer`, `gpu-overlap`; `cgemm` results carry `m`, `nn`, `k`, `impl`. Also: NVIDIA cards the boot modules did not drive (older than Turing with the open modules) are measured after a module switch: `nvidia_swap` in `run.json` (`flavour`, `version`, `pci` = the cards it was for, `kept_off`, `gpus` = nvidia-smi lines as in `hardware.gpus`, `not_driven`, or `failed` / `skipped` with the reason) and the folder `nvidia-swap/` with `q/<question>/…`, `llama/…` and `samples-nvidia.csv` as in the run's own top level; its steps are in `questions` as `nvidia-swap/<step>`. llama.cpp forks for Pascal run as their own backend: `llama/cuda+<fork>-gpu<N>-<model>.jsonl` (and `.log`) |
| 1.6 | 0.13.9 | Every measurement step records `memory` (`available_gib` at its start, `limit_gib` = the memory limit it ran under, `cgroup`, `oom_kills` if the kernel ended processes at the limit) and then `out_of_memory` with the reason; such a step counts as done for the run's status. llama-bench entries may be `skipped` with a reason when a model does not fit in RAM (CPU, integrated GPUs) and carry `out_of_memory: true` when ended at the limit, and `device_lost` (the driver's reason and how many cases were measured before) when the GPU was reset; all three count as done. Vulkan models are also skipped when they do not fit the GPU memory llama.cpp reports. `mlc` records `buffer_kib` when the stick set MLC's buffer (machines under 12 GiB) and `hugepages` (`before`, `requested`, `allocated`, or `error`): the stick reserves 1000 per NUMA node for MLC and gives them back. (The old-GPU sprint's 1.6 becomes 1.7.) |
| 1.5 | 0.13.7 (onednn_off: 0.13.8) | A oneAPI step that failed on oneDNN's allocation error ("could not create a memory object") runs once more without oneDNN: its entry in `questions.oneapi.runs` gets `retry` (`reason`, `env` = the variables set, `first_rc`, `first_output`), the first attempt's output is kept as `oneapi/<step>.onednn-failed.txt`, `rc` is the retry's. After that, every further llama.cpp step on the same device runs without oneDNN right away and records `onednn_off` (`after` = the step that failed, `env`). (Number to coordinate with the old-GPU sprint, which also adds 1.5.) |
| 1.4 | 0.13.0 | `inventory/powercap.json`: the package power limits (RAPL: PL1 'long_term', PL2 'short_term', limits in W, time windows in s) as the board configured them, where Linux exposes them |
| 1.3 | 0.12.0 | `operator_skips` in `run.json`: steps the person at the machine skipped (Enter twice); each such step has `skipped` = "skipped by the operator …" |
| 1.2 | 0.11.0 | `inventory/pcie.json`: every PCI device with its parent, path from the root complex, PCIe port type (root port, switch upstream / downstream port, endpoint, …), current and maximum link width and speed, names, driver, NUMA node — the route of each card through CPU or chipset ports and switches |
| 1.1 | 0.10.0 | the card behind every GPU result: `pci` and `uuid` in every vkprobe line, `pci` in the SYCL probe's device list and in `questions` → `rocm` / `oneapi` → `devices`; `oneapi/sycl-ls.txt`; Blender's offline route (`route`, `types_unavailable`, `hip_runtime`) |
