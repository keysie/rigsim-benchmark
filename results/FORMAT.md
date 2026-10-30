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
| 1.4 | 0.13.0 | `inventory/powercap.json`: the package power limits (RAPL: PL1 'long_term', PL2 'short_term', limits in W, time windows in s) as the board configured them, where Linux exposes them |
| 1.3 | 0.12.0 | `operator_skips` in `run.json`: steps the person at the machine skipped (Enter twice); each such step has `skipped` = "skipped by the operator …" |
| 1.2 | 0.11.0 | `inventory/pcie.json`: every PCI device with its parent, path from the root complex, PCIe port type (root port, switch upstream / downstream port, endpoint, …), current and maximum link width and speed, names, driver, NUMA node — the route of each card through CPU or chipset ports and switches |
| 1.1 | 0.10.0 | the card behind every GPU result: `pci` and `uuid` in every vkprobe line, `pci` in the SYCL probe's device list and in `questions` → `rocm` / `oneapi` → `devices`; `oneapi/sycl-ls.txt`; Blender's offline route (`route`, `types_unavailable`, `hip_runtime`) |
