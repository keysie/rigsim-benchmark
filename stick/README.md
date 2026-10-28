# The measurement stick

A bootable USB stick that measures any x86_64 machine unattended: boot from it, walk away, it powers the machine
off when done (one and a half to three hours). One stick collects runs from as many machines as you like.

## What it measures

- **Inventory:** CPU, board, memory (incl. SPD timings), GPUs, PCIe links, connected screens — serials removed.
- **CPU and memory:** latency and bandwidth by core count, Intel MLC, matrix speed (fp32, fp64, int8 via VNNI/AMX).
- **NVIDIA:** cuBLAS GEMM over precisions and shapes, VRAM bandwidth, PCIe and card-to-card copies, kernel launch
  cost, clocks and power while measuring.
- **Any GPU (Vulkan):** bandwidth, dispatch cost, vector and matrix-unit speed; on integrated graphics the RAM
  bandwidth CPU and GPU share. OpenCL peaks via clpeak.
- **AMD (ROCm) and Intel (oneAPI):** the vendor libraries' GEMM over the same grid, bandwidth, launch cost.
- **llama.cpp:** llama-bench (TinyLlama 1.1B, Llama 2 7B) and every kernel timed on its own (test-backend-ops), on
  CUDA, Vulkan, ROCm, SYCL and the CPU — all backends from one llama.cpp commit.
- **Blender Benchmark** on the CPU and every GPU device type.

Screens are switched off while measuring (a screen's scanout costs integrated graphics a few percent of RAM
bandwidth) and on again at the end.

## Requirements

- **Build:** any x86_64 Linux with Docker and internet. Everything else runs in a build container.
- **Flash:** Linux with util-linux, e2fsprogs and rsync (most distributions have them).
- **Stick:** 128 GB (64 GB without the AMD and Intel images, see `SKIP_ROCM` / `SKIP_ONEAPI`).
- **Target machines:** x86_64, UEFI or legacy boot, Secure Boot may stay on.

## Build

```bash
git clone https://github.com/keysie/rigsim-benchmark && cd rigsim-benchmark
MLC_URL='<Intel MLC download link>' \
BLENDER_LAUNCHER_URL='<Blender benchmark launcher, Linux CLI>' \
stick/build.sh            # -> stick/out/rigsim-stick.iso + stick/out/seed/
```

- **Intel MLC:** download link from Intel's Memory Latency Checker page (Intel's licence; fetched at build time,
  never part of this repository). **Blender launcher:** the Linux command-line launcher from
  opendata.blender.org. Without either, that step is recorded as skipped.
- The first build takes a few hours (the ROCm image alone one to two). Later builds reuse everything whose inputs
  did not change and take minutes plus the rootfs.
- Each step shows one live line with progress; `stick/build.sh -v` prints everything instead. Full logs:
  `stick/out/logs/<time>/`.

Useful variants:

```bash
SKIP_ROCM=1 SKIP_ONEAPI=1 stick/build.sh          # quick test build without the AMD / Intel images
NVIDIA_FLAVOUR=legacy stick/build.sh              # for Maxwell (GTX 900) .. Ada; default image: Turing .. Blackwell
MIRROR=http://ch.archive.ubuntu.com/ubuntu stick/build.sh   # a nearby Ubuntu mirror
```

| Variable | Default | Meaning |
|---|---|---|
| `NVIDIA_FLAVOUR` | `open` | `legacy`: proprietary modules for pre-Turing cards (no Blackwell) |
| `NVIDIA_BRANCH` | `580` | NVIDIA driver branch (580 is the last one for Maxwell, Pascal, Volta) |
| `SUITE` / `KERNEL` | `noble` / `hwe` | Ubuntu release / `ga` for its release kernel instead of the newer HWE kernel |
| `LLAMA_REF` | `master` | llama.cpp branch, tag or commit (resolved to one commit for all backends) |
| `ROCM_IMAGE` / `AMDGPU_TARGETS` | see `build.sh` | ROCm base image / gfx targets to compile; check against current releases |
| `ONEAPI_IMAGE` | see `build.sh` | oneAPI base image |
| `SKIP_ROCM`, `SKIP_ONEAPI`, `SKIP_MODELS` | — | leave that part off the stick |
| `MIRROR` | archive.ubuntu.com | Ubuntu mirror |
| `BUILD_IN_CONTAINER` | `1` | `0`: build on a Debian/Ubuntu host directly |

To force a full rebuild: `rm -rf stick/out/cache stick/out/seed`.

## Flash

```bash
lsblk -o NAME,SIZE,MODEL,RM,TRAN       # find the stick: RM 1, TRAN usb
sudo stick/flash.sh /dev/sdX
```

- **New stick:** asks you to type `yes`, writes the system, creates the data partition `RIGSIM` (ext4) and copies the
  seed (images, models, tools) — expect 30 minutes or more for the copy.
- **Existing rigsim stick:** shows its version and runs, then offers `[u]pdate` (new system, results kept),
  `[s]eed only`, `[e]rase` or `[a]bort`. Update and erase first copy all runs to `~/rigsim-stick-backup-<time>`.

## Use

1. Plug the stick into the machine, boot from it (boot menu: usually F8, F11 or F12).
2. A notice shows for 30 seconds: any key aborts, nothing is measured.
3. Walk away. The first boot of a stick loads its container images (a few minutes more). At the end the machine
   powers off.

While it runs: `Ctrl+Alt+F2` opens a shell; `systemctl stop rigsim-stick` stops cleanly. To keep the machine on
after the run, add `rigsim.nopoweroff` to the kernel line in the boot menu (`e` to edit), or put a file
`NO_POWEROFF` on the data partition.

**On someone else's machine:** nothing is written to its disks, no power limits are changed, the CPU clock is
restored after each step. Tell the owner the results may be published (CC BY 4.0, no serials); the start screen
says so too.

## Results

Each boot adds one folder `runs/<UTC time>-<machine>-<random>/` on the data partition; nothing is ever
overwritten. To share them in this repository:

```bash
tools/contribute.sh /run/media/$USER/RIGSIM  # new complete runs -> a branch and a pull request
```

Details, the upload alternative and the licence: [Contributing runs](../README.md#contributing-runs). Just copying
them somewhere else: `stick/tools/collect.sh /run/media/$USER/RIGSIM ~/stick-results`.

## Troubleshooting

| Symptom | Look at |
|---|---|
| Build step fails | the last lines shown, then `stick/out/logs/<time>/NN-<step>.log`; rerun with `-v` |
| Stick not in the boot menu | `sudo fdisk -l /dev/sdX`: three partitions (ISO, EFI, `RIGSIM`) |
| Secure Boot error at start | try once with Secure Boot off; report the exact message |
| Ends in an `(initramfs)` prompt | the live system did not find itself on the stick: the lines before the prompt |
| Every step fails within a second | `run.json` → `docker`: the container check (storage, error) |
| A step failed | `run.json` → its entry; the step's folder; `stick.log` |

## Limits

- Kepler and older NVIDIA cards: no CUDA; Vulkan / OpenCL only where a driver exists.
- Under Secure Boot the kernel runs in lockdown: Intel MLC cannot switch prefetchers off (recorded as
  `prefetchers_left_on`).
- ROCm officially supports only some consumer AMD generations; unsupported cards run with the closest target
  (`HSA_OVERRIDE_GFX_VERSION`, recorded) or get Vulkan and OpenCL only.
- vkprobe's matrix GEMM is a plain kernel, not a tuned library: a lower bound for that path.
- RAM timings: the modules' SPD is read (decoded where `decode-dimms` can, always as raw bytes with the serial
  number zeroed); what the BIOS actually set is often invisible from Linux.
- **Blender (open):** the benchmark launcher asked opendata.blender.org for metadata on the offline stick and listed
  no device in the first trial run; it is recorded as a failure until an offline way is found.

## Versions

`stick/VERSION` is the stick's release number (semver): patch for fixes, minor for new measurements or additive
format changes, major only for a run format that is not backwards compatible (see
[`results/FORMAT.md`](../results/FORMAT.md)). Bump it in the commit that changes what the stick measures or
writes.

Every build records the exact versions of what it puts on the stick (`tools/manifest.py`: packages of the system
image, container images by digest and the packages inside them, llama.cpp commits and binary hashes, vendor images,
models, MLC, Blender, a hash over all stick scripts). Every run copies this, adds what actually ran (kernel, NVIDIA
driver, Docker, Vulkan drivers) into `manifest.json`, and attaches to each step the versions its results came
from. The data page shows them per run.

## Development

```bash
python3 -m pytest -q stick/tests            # orchestrator, build and flash logic (no hardware needed)
```

How it works inside: `build.sh` (image and seed), `ubuntu/` (rootfs setup, GRUB), `payload/stick.py` (the
orchestrator on the stick), `questions/` (bench runner questions), `src/` (vkprobe, the SYCL probe, the ROCm and
oneAPI image builds), `flash.sh`.
