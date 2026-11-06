"""The stick's orchestrator logic (no root, no hardware): run directories never reused, fingerprints salted and
stable, inventory sanitised, the profile valid for the bench runner, GPU questions skipped with a reason, the index
accumulates, power-off rules. Run: python3 -m pytest -q stick/tests"""
import datetime as dt
import os
import subprocess
import json
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stick", HERE / "payload" / "stick.py")
stick = importlib.util.module_from_spec(spec); spec.loader.exec_module(stick)
sys.path.insert(0, str(HERE.parent / "bench" / "hw" / "runner"))
import hwq  # noqa: E402

NOW = dt.datetime(2026, 10, 27, 15, 0, 0, tzinfo=dt.timezone.utc)


def dmi(tmp_path, **kv):
    d = tmp_path / "dmi"; d.mkdir(exist_ok=True)
    for k, v in kv.items():
        (d / k).write_text(v + "\n")
    return d


def test_run_directories_are_never_reused(tmp_path):
    runs = tmp_path / "runs"
    dirs = {stick.new_run_dir(runs, "abc123", NOW) for _ in range(50)}     # same second, same machine
    assert len(dirs) == 50 and all(d.is_dir() for d in dirs)
    (runs / "keep.txt").write_text("x")
    stick.new_run_dir(runs, "abc123", NOW)
    assert (runs / "keep.txt").read_text() == "x"


def test_fingerprint_is_salted_stable_and_not_an_identifier(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    d = dmi(tmp_path, product_uuid="4C4C4544-0042-1234-8057-B4C04F4B3332", board_serial="XYZ123",
            board_vendor="ASUSTeK", board_name="PRIME X299-A")
    a, stable = stick.fingerprint(data, d)
    b, _ = stick.fingerprint(data, d)
    assert a == b and stable and len(a) == 12
    salt = (data / ".salt").read_bytes()
    assert stick.fingerprint(data, d)[0] == a and (data / ".salt").read_bytes() == salt   # salt made once
    other = tmp_path / "other"; other.mkdir()
    assert stick.fingerprint(other, d)[0] != a                                            # another stick
    assert "XYZ123" not in a and "4C4C" not in a


def test_placeholder_ids_give_an_unstable_fingerprint(tmp_path):
    data = tmp_path / "data"; data.mkdir()
    d = dmi(tmp_path, product_uuid="", board_serial="Default string", product_serial="To Be Filled By O.E.M.")
    a, stable = stick.fingerprint(data, d)
    assert not stable and a != stick.fingerprint(data, d)[0]


def test_sanitize_removes_serials_uuids_macs_and_addresses():
    text = """Base Board Information
	Manufacturer: ASUSTeK COMPUTER INC.
	Serial Number: 170512345600123
	UUID: 4C4C4544-0042-1234-8057-B4C04F4B3332
	Asset Tag: Default string
	Part Number: CMK32GX4M2B3200C16
link/ether 00:1a:2b:3c:4d:5e inet 192.168.1.23 fe80:0:0:0:21a:2bff:fe3c:4d5e
	Capabilities: [100 v1] Device Serial Number 00-11-22-ff-ff-33-44-55"""
    out = stick.sanitize(text)
    for bad in ("170512345600123", "4C4C4544", "Asset Tag", "00:1a:2b", "192.168.1.23", "fe80:0", "Device Serial"):
        assert bad not in out, bad
    assert "ASUSTeK" in out and "CMK32GX4M2B3200C16" in out          # part numbers are useful and not personal


def test_dimms_from_dmidecode():
    dmi_text = """Handle 0x0040, DMI type 17, 92 bytes
Memory Device
	Size: 32 GB
	Locator: DIMM_A1
	Type: DDR4
	Speed: 3200 MT/s
	Configured Memory Speed: 3200 MT/s
	Rank: 2
	Manufacturer: Corsair
	Part Number: CMK64GX4M2E3200C16

Handle 0x0041, DMI type 17, 92 bytes
Memory Device
	Size: No Module Installed
	Locator: DIMM_A2
"""
    d = stick.parse_dimms(dmi_text)
    assert len(d) == 1 and d[0]["Type"] == "DDR4" and d[0]["Configured Memory Speed"] == "3200 MT/s"


def test_profile_is_valid_for_the_bench_runner():
    for q in stick.PROFILE:
        f = HERE / "questions" / f"{q}.toml"
        p = hwq.load_plan(str(f))
        items = hwq.expand(p, False, [dict(name="gpu0"), dict(name="gpu1")])
        assert items and hwq.command_for(items[0])
        assert "power_w" not in p["grid"], f"{q}: no power-limit sweeps on other people's machines"
        assert not p.get("testdir"), f"{q}: nothing written to the machine's disks"
        assert (HERE.parent / "bench" / "hw" / "toolkits" / p["toolkit"]).is_dir()


def test_gpu_questions_are_skipped_with_the_reason():
    pl = stick.plan(HERE / "questions", False, "NVIDIA module not loaded: Secure Boot is on")
    by = {q: skip for q, _, skip in pl}
    assert by["cpu-mem-latency"] is None and by["cpu-mem-ladder"] is None
    assert by["gpu-gemm"] == by["pcie-copy"] == by["gpu-vram"] == "NVIDIA module not loaded: Secure Boot is on"
    assert all(skip is None for _, _, skip in stick.plan(HERE / "questions", True, ""))


def test_index_accumulates(tmp_path):
    for i in range(3):
        stick.append_index(tmp_path, dict(run=f"r{i}", cpu="AMD Ryzen 9\t7950X", status="done"))
    lines = (tmp_path / "index.tsv").read_text().splitlines()
    assert lines[0] == stick.INDEX_HEADER.strip() and len(lines) == 4
    assert all(len(l.split("\t")) == 9 for l in lines)


def test_power_off_rules(tmp_path):
    assert stick.should_power_off(tmp_path, "boot=live quiet")
    assert not stick.should_power_off(tmp_path, "boot=live rigsim.nopoweroff")
    (tmp_path / "NO_POWEROFF").write_text("")
    assert not stick.should_power_off(tmp_path, "boot=live quiet")


def test_overlay_only_adds_offline_setup_cpu_gemm_and_wider_cuda_archs():
    s = (HERE / "overlay" / "toolkits" / "cpu" / "setup.sh").read_text()
    assert "dpkg -i debs/*.deb" in s and "libopenblas-dev" in s and "cgemm.c -lopenblas" in s
    assert sorted(p.relative_to(HERE / "overlay") for p in (HERE / "overlay").rglob("*") if p.is_file()) == \
        sorted([Path("toolkits/cpu/setup.sh"), Path("toolkits/cpu/src/cgemm.c"), Path("toolkits/cpu/src/c2c.c"),
                Path("toolkits/cuda/setup.sh")])
    cu = (HERE / "overlay" / "toolkits" / "cuda" / "setup.sh").read_text()
    for sm in ("sm_52", "sm_61", "sm_75", "sm_86", "sm_89", "sm_120"):
        assert sm in cu
    # the CUDA probe the stick compiles is the bench's own (d2d copies included), not a copy
    assert "d2d" in (HERE.parent / "bench" / "hw" / "toolkits" / "cuda" / "src" / "probe.cu").read_text()


def test_nothing_outside_stick_is_changed_by_the_build():
    b = (HERE / "build.sh").read_text()
    assert "rm -rf \"$W\"" in b and "$OUT/work" in b and "cp -r \"$REPO/bench/hw/runner\"" in b
    assert not any(tok in b for tok in ("> \"$REPO", ">> \"$REPO", "rm -rf \"$REPO"))


def test_secure_boot_chain_uses_only_ubuntu_signed_binaries_and_no_dkms():
    b = (HERE / "build.sh").read_text()
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert "shimx64.efi.signed" in b and "gcdx64.efi.signed" in b and "mmx64.efi" in b
    assert "grub-mkimage --format=x86_64-efi" not in b                 # an own EFI GRUB would be unsigned
    assert 'NVMOD="linux-modules-nvidia-${NVIDIA_BRANCH}${MODSUFFIX}-$KFL"' in c and "no-dkms" in c   # signed, kernel series
    assert "nvidia-dkms" in c and "exit 1" in c                         # the build fails if DKMS slips in
    assert "clpeak" in c and "intel-opencl-icd" in c and "mesa-opencl-icd" in c


CLINFO_LIST = """Platform #0: NVIDIA CUDA
 +-- Device #0: NVIDIA GeForce RTX 3090
 `-- Device #1: NVIDIA GeForce RTX 3060
Platform #1: Intel(R) OpenCL Graphics
 `-- Device #0: Intel(R) UHD Graphics 770
Platform #2: rusticl
 `-- Device #0: AMD Radeon RX 7900 XTX (radeonsi, navi31, LLVM 17.0.6, DRM 3.57, 6.8.0-45-generic)
Platform #3: Portable Computing Language
 `-- Device #0: cpu-haswell-AMD Ryzen 9 7950X 16-Core Processor
Platform #4: rusticl
 `-- Device #0: llvmpipe (LLVM 17.0.6, 256 bits)
"""

CLPEAK = """Platform: NVIDIA CUDA
  Device: NVIDIA GeForce RTX 3090
    Driver version  : 580.95.05 (Linux x64)
    Compute units   : 82
    Clock frequency : 1695 MHz

    Global memory bandwidth (GBPS)
      float   : 836.41
      float4  : 870.12

    Single-precision compute (GFLOPS)
      float   : 35512.30

    Transfer bandwidth (GBPS)
      enqueueWriteBuffer              : 12.34
      enqueueReadBuffer               : 13.10

    Kernel launch latency : 5.67 us
"""


def test_opencl_devices_all_vendors_without_software_renderers():
    d = stick.opencl_devices(CLINFO_LIST)
    assert [(p, i) for p, i, _, _ in d] == [(0, 0), (0, 1), (1, 0), (2, 0)]
    assert "UHD Graphics 770" in d[2][3] and "7900 XTX" in d[3][3]


def test_clpeak_output_parsed_with_units():
    r = stick.parse_clpeak(CLPEAK)
    assert r["Global memory bandwidth (GBPS)"]["float4"] == 870.12
    assert r["Single-precision compute (GFLOPS)"]["float"] == 35512.30
    assert r["Transfer bandwidth (GBPS)"]["enqueueReadBuffer"] == 13.10
    assert r["info"]["Compute units"] == 82 and r["info"]["Clock frequency (MHz)"] == 1695
    assert r["Transfer bandwidth (GBPS)"]["Kernel launch latency (us)"] == 5.67


def sysfs(tmp_path):
    drm = tmp_path / "drm"
    pci = tmp_path / "devices"
    def card(name, bdf, vendor, files=None):
        dev = pci / bdf; dev.mkdir(parents=True)
        (dev / "vendor").write_text(vendor + "\n")
        for k, v in (files or {}).items():
            (dev / k).parent.mkdir(parents=True, exist_ok=True); (dev / k).write_text(v)
        c = drm / name; c.mkdir(parents=True); (c / "device").symlink_to(dev)
        return c
    card("card0", "0000:00:02.0", "0x8086")
    (drm / "card0" / "gt_act_freq_mhz").write_text("1550\n")
    card("card1", "0000:03:00.0", "0x1002", {"pp_dpm_sclk": "0: 500Mhz\n1: 2400Mhz *\n", "pp_dpm_mclk": "0: 96Mhz\n3: 1249Mhz *\n",
         "gpu_busy_percent": "97\n", "hwmon/hwmon3/power1_average": "212000000\n"})
    card("card2", "0000:01:00.0", "0x10de")
    for con, status, modes in (("card0-HDMI-A-1", "connected", "1920x1080\n1280x720\n"), ("card1-DP-1", "connected",
                               "3840x2160\n"), ("card1-DP-2", "disconnected", "")):
        d = drm / con; d.mkdir(); (d / "status").write_text(status + "\n"); (d / "modes").write_text(modes)
        (d / "enabled").write_text("enabled\n"); (d / "dpms").write_text("On\n")
    return drm


def test_displays_name_the_gpu_and_mode(tmp_path):
    d = stick.displays(sysfs(tmp_path))
    assert [(x["card"], x["connector"], x["pci"], x["mode"]) for x in d] == [
        ("card0", "HDMI-A-1", "0000:00:02.0", "1920x1080"), ("card1", "DP-1", "0000:03:00.0", "3840x2160")]


def test_screens_powered_down_and_back(tmp_path):
    fb = tmp_path / "graphics"
    for n in ("fb0", "fb1"):
        (fb / n).mkdir(parents=True); (fb / n / "blank").write_text("0")
    assert stick.blank_screens(True, fb) == ["fb0", "fb1"] and (fb / "fb0" / "blank").read_text() == "4"
    stick.blank_screens(False, fb)
    assert (fb / "fb1" / "blank").read_text() == "0"


def test_amd_and_intel_clocks_sampled_nvidia_left_to_the_runner(tmp_path):
    rows = {r["vendor"]: r for r in stick.drm_sample(sysfs(tmp_path))}
    assert set(rows) == {"amd", "intel"}
    assert rows["amd"]["sclk_mhz"] == 2400 and rows["amd"]["mclk_mhz"] == 1249
    assert rows["amd"]["busy_pct"] == 97 and rows["amd"]["power_w"] == 212.0
    assert rows["intel"]["gt_mhz"] == 1550


def test_sampler_writes_timestamped_rows(tmp_path):
    out = tmp_path / "s.jsonl"
    with stick.DrmSampler(out, period=0.01, drm=sysfs(tmp_path)):
        import time; time.sleep(0.08)
    import json
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) >= 4 and all("t" in r for r in rows)


MLC_IDLE = """Intel(R) Memory Latency Checker - v3.11b
Command line parameters: --idle_latency

Using buffer size of 1800.000MiB
*** Unable to modify prefetchers (try executing 'modprobe msr')
Each iteration took 245.3 base frequency clocks (	81.6	ns)
"""
MLC_BW = """Measuring Maximum Memory Bandwidths for the system
Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
Using all the threads from each core if Hyper-threading is enabled
Using traffic with the following read-write ratios
ALL Reads        :	61834.21
3:1 Reads-Writes :	57321.10
Stream-triad like:	58012.44
"""


def test_mlc_values_parsed():
    v = stick.parse_mlc(MLC_BW)
    assert v["ALL Reads"] == 61834.21 and v["3:1 Reads-Writes"] == 57321.10 and v["Stream-triad like"] == 58012.44


VULKANINFO = """Devices:
========
GPU0:
	apiVersion         = 1.4.303
	deviceType         = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
	deviceName         = NVIDIA GeForce RTX 3090
GPU1:
	deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
	deviceName         = Intel(R) UHD Graphics 770 (ADL-S GT1)
GPU2:
	deviceType         = PHYSICAL_DEVICE_TYPE_CPU
	deviceName         = llvmpipe (LLVM 19.1.1, 256 bits)
"""


def test_vulkan_devices_without_software_renderer():
    assert stick.vulkan_devices(VULKANINFO) == [(0, "NVIDIA GeForce RTX 3090"), (1, "Intel(R) UHD Graphics 770 (ADL-S GT1)")]


def test_blender_device_types_cpu_first_each_once():
    out = """AMD Ryzen 9 7950X 16-Core Processor    CPU
NVIDIA GeForce RTX 3090    CUDA
NVIDIA GeForce RTX 3090    OPTIX
NVIDIA GeForce RTX 3060    OPTIX
"""
    assert stick.blender_device_types(out) == ["CPU", "OPTIX", "CUDA"]


def test_missing_seed_parts_are_skipped_with_a_reason(tmp_path):
    log = tmp_path / "log"
    assert "MLC" in stick.run_mlc(tmp_path, tmp_path / "r", log)["skipped"]
    assert "seed" in stick.run_llama(tmp_path, tmp_path / "r", "img", True, log)["skipped"]
    assert "Blender" in stick.run_blender(tmp_path, tmp_path / "r", log)["skipped"]


def test_build_fetches_mlc_from_intel_and_never_commits_it():
    b = (HERE / "build.sh").read_text()
    assert "MLC_URL" in b and "$OUT/seed/tools/mlc" in b
    assert not list(HERE.rglob("mlc")) or all("out" in p.parts for p in HERE.rglob("mlc"))
    assert "out/" in (HERE / ".gitignore").read_text()


def test_flash_offers_update_backs_up_first_and_keeps_the_data_partition():
    f = (HERE / "flash.sh").read_text()
    assert "[u]pdate (keep results), [s]eed only, [e]rase everything, [a]bort?" in f
    upd = f.split("update() {", 1)[1].split("\n}\n", 1)[0]
    assert upd.index("backup_runs") < upd.index("dd if=")                 # backup before anything is written
    assert 'add_data_part "$DEV" "$start" "$sectors"' in upd and "mkfs" not in upd  # same sectors, no new filesystem
    # an MBR-layout stick (data partition to the very end) is not updated in place to the GPT layout, and [u]pdate is
    # not offered then (before, it was offered again after every refusal — user report 2026-10-31)
    assert "disk_sectors - 34" in f[f.index("update_blocker() {"):f.index("update() {")]
    assert 'blocker="$(update_blocker "$part")"' in f and 'prompt="[s]eed only, [e]rase everything, [a]bort? "' in f
    assert '[[ -z "$blocker" ]] && { update' in f
    assert "does not fit in front of the data partition" in f.split("update_blocker() {", 1)[1] and "e2fsck -n" in upd
    e = f.split("\n    e) ", 1)[1].split(";;", 1)[0]
    assert e.index("backup_runs") < e.index("fresh")
    assert "mount -o ro" in f.split("backup_runs() {", 1)[1].split("\n}\n", 1)[0]
    assert "RESERVE_MIB" in f.split("fresh() {", 1)[1]


# --- second round: launch cost, CPU int8, Vulkan probe, AMD / Intel vendor steps, stick read speed ----------------
def test_new_questions_launch_cost_int8_and_2048_shapes():
    import tomllib
    assert "gpu-launch" in stick.PROFILE
    for q in stick.PROFILE:
        hwq.load_plan(HERE / "questions" / f"{q}.toml")
    cg = tomllib.loads((HERE / "questions" / "cpu-gemm.toml").read_text())
    assert "int8" in cg["grid"]["prec"]
    gg = tomllib.loads((HERE / "questions" / "gpu-gemm.toml").read_text())
    assert {"8x2048x2048", "2048x2048x2048"} <= set(gg["grid"]["shape"])
    probe = (HERE.parent / "bench" / "hw" / "toolkits" / "cuda" / "src" / "probe.cu").read_text()
    assert 'mode == "launch"' in probe and "cudaGraphLaunch" in probe and 'mode == "info"' in probe
    setup = (HERE / "overlay" / "toolkits" / "cpu" / "setup.sh").read_text()
    assert "-ldnnl" in setup and "libdnnl-dev" in (HERE / "build.sh").read_text()     # offline debs carry it too


def test_vk_plan_skips_software_and_adds_shared_for_integrated():
    devs = [dict(index=0, device="llvmpipe (LLVM 20)", device_type="cpu"),
            dict(index=1, device="AMD Radeon RX 7900 XTX (RADV NAVI31)", device_type="discrete"),
            dict(index=2, device="Intel(R) Graphics (ARL)", device_type="integrated")]
    keys = [k for k, _ in stick.vk_plan(devs)]
    assert not any(k.startswith("dev0-") for k in keys)
    assert "dev1-launch" in keys and "dev1-gemm-i8-4096x4096x4096" in keys and "dev1-shared" not in keys
    assert "dev2-shared" in keys and "dev2-stream-copy" in keys


def test_vulkan_probe_runs_through_a_fake_probe(tmp_path):
    fake = tmp_path / "bin-vkprobe"
    fake.write_text('#!/bin/sh\nif [ "$1" = list ]; then echo \'{"mode":"list","index":0,"device":"Fake GPU","device_type":"discrete"}\'; '
                    'else echo \'{"mode":"\'$1\'"}\'; fi\n')
    fake.chmod(0o755)
    log = tmp_path / "log"
    r = stick.run_vulkan_probe(tmp_path, log, probe=fake)
    assert r["rc"] == 0 and len(r["runs"]) == len(stick.vk_plan([dict(index=0, device="Fake GPU", device_type="discrete")]))
    assert (tmp_path / "vkprobe" / "dev0-launch.jsonl").read_text().strip() == '{"mode":"launch"}'


def test_hsa_override_only_for_targets_not_built():
    targets = ["gfx1030", "gfx1100", "gfx1201"]
    assert stick.hsa_override("gfx1100", targets) is None
    assert stick.hsa_override("gfx1031", targets) == "10.3.0"
    assert stick.hsa_override("gfx1103:xnack-", targets) == "11.0.0"
    assert stick.hsa_override("gfx803", targets) is None             # no known family: run without, record failure


def test_vendor_plan_full_grid_and_vendor_images(tmp_path):
    p = stick.vendor_plan("rocm")
    gemms = [k for k, _ in p if k.startswith("gemm-")]
    assert len(gemms) == len(stick.VENDOR_GEMM_PRECS) * len(stick.VENDOR_GEMM_SHAPES)
    assert {"launch", "stream-read", "copy-bidir"} <= {k for k, _ in p}
    assert all(cmd[0] == "/rigsim/probe_sycl" for _, cmd in stick.vendor_plan("oneapi"))
    img = tmp_path / "seed" / "images"; img.mkdir(parents=True)
    (img / "VENDOR.tsv").write_text("rocm\trigsim/rocm:abc\trocm/dev\noneapi\trigsim/oneapi:abc\tintel/x\n")
    (img / "vendor-rocm.tar").write_text("x")
    assert stick.vendor_images(tmp_path) == {"rocm": "rigsim/rocm:abc"}       # oneapi tar missing: not offered


def test_vendor_steps_skip_without_that_vendor(tmp_path, monkeypatch):
    monkeypatch.setattr(stick, "gpu_vendors", lambda drm=None: {"intel"})
    assert "no AMD GPU" in stick.run_vendor("rocm", tmp_path, tmp_path, tmp_path / "log")["skipped"]
    assert "oneapi image not in the seed" in stick.run_vendor("oneapi", tmp_path, tmp_path, tmp_path / "log")["skipped"]


def test_gpu_vendors_from_sysfs(tmp_path):
    for card, vendor in (("card0", "0x10de"), ("card1", "0x1002"), ("card2", "0x8086")):
        (tmp_path / card / "device").mkdir(parents=True)
        (tmp_path / card / "device" / "vendor").write_text(vendor + "\n")
    (tmp_path / "card1-DP-1").mkdir()
    assert stick.gpu_vendors(tmp_path) == {"amd", "intel"}


def test_vendor_images_load_only_where_needed_and_stick_speed_is_logged(tmp_path):
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'if not img.name.startswith("vendor-")' in src
    (tmp_path / "seed" / "models").mkdir(parents=True)
    (tmp_path / "seed" / "models" / "m.gguf").write_bytes(b"x" * (3 << 20))
    r = stick.stick_read_speed(tmp_path, mib=2)
    assert r["file"] == "m.gguf" and r["mib"] == 2 and r["mib_per_s"] > 0
    assert stick.stick_read_speed(tmp_path / "nothing")["skipped"]


def test_build_wires_vkprobe_vendor_images_and_hwe_kernel():
    b = (HERE / "build.sh").read_text()
    assert 'install -m 0755 "$C/vkprobe/vkprobe" "$P/bin/vkprobe"' in b
    assert "vendor rocm" in b and "vendor oneapi" in b and "VENDOR.tsv" in b and "SKIP_ROCM" in b
    assert '-e LLAMA_REF="$LLAMA_COMMIT"' in b and '-e REF="$LLAMA_COMMIT"' in b   # every backend, one llama.cpp commit
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert "generic-hwe-" in c and '"linux-modules-extra-$KVER"' in c and '"linux-modules-extra-$KFL"' not in c
    for f in ("src/rocm/build-in-image.sh", "src/oneapi/build-in-image.sh", "src/vkprobe/build.sh",
              "src/vkprobe/vkprobe.c", "src/syclprobe/probe_sycl.cpp"):
        assert (HERE / f).exists(), f
    assert "hipify-perl /src/probe.cu" in (HERE / "src/rocm/build-in-image.sh").read_text()


def test_backend_ops_per_device_with_fallbacks(tmp_path):
    calls = []

    def run(cmd, timeout):
        calls.append(cmd)
        if "--output" in cmd:
            return 1, "unknown argument: --output"
        return 0, "MUL_MAT(type_a=q4_0,...): 1234 runs - 12.3 us/run - 2.1 GFLOP/run - 170 TFLOPS"

    r = stick.run_backend_ops(run, tmp_path, "cuda-gpu0", "/llama/test-backend-ops", "CUDA0", tmp_path / "log")
    assert len(r) == len(stick.BACKEND_OPS) and all(v["rc"] == 0 for v in r.values())
    flt = stick.BACKEND_OPS_FILTER["MUL_MAT"]
    assert calls[0][-2:] == ["--output", "sql"]                                  # the format that carries timings
    assert calls[1] == ["/llama/test-backend-ops", "perf", "-o", "MUL_MAT", "-p", flt, "-b", "CUDA0"]   # output dropped once
    assert (tmp_path / "cuda-gpu0-ops-mul_mat.txt").read_text().startswith(f"/llama/test-backend-ops perf -o MUL_MAT -p {flt} -b CUDA0\n")
    for f in ("build.sh", "src/rocm/build-in-image.sh", "src/oneapi/build-in-image.sh"):
        assert "test-backend-ops" in (HERE / f).read_text() and "LLAMA_BUILD_TESTS=ON" in (HERE / f).read_text()
    src = (HERE / "payload" / "stick.py").read_text()
    for name in ('"CUDA0"', '"Vulkan0"', '"CPU"', '"ROCm0"', '"SYCL0"'):
        assert name in src


def test_scripts_install_missing_packages_and_ask_first():
    deps = (HERE / "tools" / "deps.sh").read_text()
    assert all(" -y" not in l for l in deps.splitlines() if "apt-get install" in l and not l.lstrip().startswith("#"))
    assert "apt-get install --no-install-recommends" in deps
    for f in ("build.sh", "flash.sh"):
        s = (HERE / f).read_text()
        assert 'source "$S/tools/deps.sh"' in s and "\nneed " in s
        assert s.index("need ") < s.index('DEV="${1') if f == "flash.sh" else True       # before anything else


def test_build_shows_a_ticker_per_step_and_keeps_full_logs(tmp_path):
    import subprocess
    b = (HERE / "build.sh").read_text()
    assert 'source "$S/tools/ticker.sh"' in b and b.count('step "') >= 14 and "sudo -v" in b
    assert 'cmake --build b --target llama-bench test-backend-ops -j"$(nproc)"\n' in b      # make's [ NN%] reaches the log
    script = tmp_path / "t.sh"
    script.write_text(f'set -euo pipefail\nLOGS={tmp_path}/logs; source {HERE}/tools/ticker.sh\n'
                      'ok() { echo "[ 50%] x"; }\nbad() { false; echo never; }\nstep "fine" ok\nstep "broken" bad\necho after\n')
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert r.returncode == 1 and "after" not in r.stdout and "full log:" in r.stdout
    logs = sorted((tmp_path / "logs").iterdir())
    assert [p.name for p in logs] == ["01-fine.log", "02-broken.log"] and "never" not in logs[1].read_text()


def test_build_reruns_itself_in_the_build_container(tmp_path):
    """With a fake docker on PATH: the wrapper builds the builder image, mounts the repository at the same path and
    the docker socket, passes the build settings through, hands out/ back to the user, keeps the exit code."""
    import os
    import subprocess
    sock = tmp_path / "docker.sock"
    import socket
    srv = socket.socket(socket.AF_UNIX); srv.bind(str(sock))
    calls = tmp_path / "calls"
    fake = tmp_path / "bin" / "docker"; fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n'
                    'case "$1" in image) exit 1;; run) case "$*" in *build.sh*) exit 3;; esac;; esac\nexit 0\n')
    fake.chmod(0o755)
    env = dict(os.environ, PATH=f"{fake.parent}:{os.environ['PATH']}", DOCKER_SOCK=str(sock), SKIP_ROCM="1",
               MIRROR="http://ch.archive.ubuntu.com/ubuntu")
    env.pop("RIGSIM_IN_CONTAINER", None); env.pop("BUILD_IN_CONTAINER", None)
    r = subprocess.run(["bash", str(HERE / "build.sh")], env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    srv.close()
    log = calls.read_text().splitlines()
    repo = str(HERE.parent)
    assert r.returncode == 3                                                     # the inner build's exit code
    assert any(l.startswith("build -q -t rigsim/stick-builder:") for l in log)
    run = next(l for l in log if "build.sh" in l)
    assert "--privileged" in run and f"-v {repo}:{repo}" in run and f"-v {sock}:/var/run/docker.sock" in run
    assert "-e RIGSIM_IN_CONTAINER=1" in run and "-e SKIP_ROCM=1" in run and "-e MIRROR=http://ch.archive.ubuntu.com/ubuntu" in run
    b = (HERE / "build.sh").read_text()
    inner = [l for l in b.splitlines() if "docker run" in l and "BUILDER" not in l and not l.lstrip().startswith("#")]
    assert inner and all("label=disable" in l for l in inner)                   # SELinux hosts need no relabel


def test_ticker_in_a_terminal_survives_output_without_percentage(tmp_path):
    """Regression: in a terminal, a step whose output had no "NN%" yet ended the whole build (grep under pipefail)
    before the step wrote anything — "fails with an empty log". Runs in a pseudo-terminal via script(1)."""
    import shutil
    import subprocess
    if not shutil.which("script"):
        import pytest; pytest.skip("script(1) not available")
    for verbose in ("0", "1"):
        t = tmp_path / f"t{verbose}.sh"
        t.write_text(f'set -euo pipefail\nVERBOSE={verbose}; LOGS={tmp_path}/logs{verbose}; source {HERE}/tools/ticker.sh\n'
                     'slow() { sleep 0.6; echo "plain output"; }\nstep "quiet" slow\necho AFTER-OK\n')
        r = subprocess.run(["script", "-qec", f"bash {t}", "/dev/null"], capture_output=True, text=True)
        assert r.returncode == 0 and "AFTER-OK" in r.stdout
        assert (tmp_path / f"logs{verbose}" / "01-quiet.log").read_text().strip() == "plain output"
        assert ("plain output" in r.stdout) == (verbose == "1")                 # -v: live on the terminal too
    assert '[[ "${1:-}" == -v' in (HERE / "build.sh").read_text()


def test_build_reads_rootfs_files_only_through_rootfile():
    """Regression (vmlinuz, shimx64.efi.signed): files of the rootfs read from outside went through absolute links
    and resolved against the build host. Every read now goes through rootfile(), which resolves inside the chroot."""
    import re
    b = (HERE / "build.sh").read_text()
    assert 'readlink -e "$1"' in b and "rootfile() {" in b
    reads = [l for l in b.splitlines() if re.search(r'(cp|cat)\b[^|]*"\$ROOT/(usr|boot|tmp/core)', l)]
    assert reads == [], reads
    for f in ("/boot/vmlinuz-$KV", "/boot/initrd.img-$KV", "shimx64.efi.signed", "mmx64.efi", "gcdx64.efi.signed",
              "cdboot.img", "boot_hybrid.img", "/tmp/core.img"):
        assert re.search(r'rootfile "?[^)]*' + re.escape(f), b), f
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert 'ln -sf "$(basename' in c                                          # relative links inside the rootfs


def test_build_caches_every_expensive_part():
    b = (HERE / "build.sh").read_text()
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    # apt packages kept between builds: mmdebstrap syncs its cache in and out, the chroot step gets it bind-mounted
    assert "sync-in $APT_CACHE /var/cache/apt/archives" in b and "sync-out /var/cache/apt/archives $APT_CACHE" in b
    assert 'mount --bind "$APT_CACHE" "$ROOT/var/cache/apt/archives"' in b and "KEEP_APT_CACHE=1" in b
    assert '[ "${KEEP_APT_CACHE:-0}" = 1 ] || apt-get clean' in c
    # llama.cpp keyed by the resolved commit, image and build script; vendor images by base digest + commit + sources
    assert "LLAMA_URL=https://github.com/ggml-org/llama.cpp" in b and 'git ls-remote "$LLAMA_URL"' in b
    assert 'cached_step "$C/llama-cuda.key" "$(key "$LLAMA_COMMIT" "$(image_id "$CUDA_IMAGE")" "$LLAMA_CUDA")"' in b
    assert 'tag="rigsim/$name:$k"' in b and 'docker image inspect "$tag"' in b and 'cache_ok "$tar.key" "$k"' in b
    assert 'cached_step "$tar.key" "$(image_id "$img")" "save $img"' in b
    for marker in ('"$C/vkprobe/.key"', '"$OUT/seed/debs/.key"', '"$OUT/seed/blender/.key"', '"$OUT/seed/tools/mlc.key"'):
        assert f"cached_step {marker}" in b, marker


def test_cached_step_skips_on_same_key_and_records_only_success(tmp_path):
    import subprocess
    src = (HERE / "build.sh").read_text()
    helpers = src[src.index("key() {"):src.index('APT_CACHE="$C/apt-$SUITE"')]
    t = tmp_path / "t.sh"
    t.write_text(f"set -euo pipefail\nLOGS={tmp_path}/logs; source {HERE}/tools/ticker.sh\n{helpers}\n"
                 f"work() {{ echo ran >> {tmp_path}/ran; }}\nbad() {{ false; }}\n"
                 f'cached_step {tmp_path}/m.key k1 "first" work\ncached_step {tmp_path}/m.key k1 "again" work\n'
                 f'cached_step {tmp_path}/m.key k2 "changed" work\ncached_step {tmp_path}/b.key k1 "broken" bad\n')
    r = subprocess.run(["bash", str(t)], capture_output=True, text=True)
    assert (tmp_path / "ran").read_text().split() == ["ran", "ran"]             # first and changed; "again" cached
    assert "again  (cached)" in r.stdout and r.returncode != 0 and not (tmp_path / "b.key").exists()


def test_flash_rereads_without_partprobe():
    """Regression: partprobe (libparted) refuses a hybrid ISO's table — partition 1 starts at sector 0 and contains
    the table ("recursive partition"), which stopped flash.sh before the data partition was formatted."""
    f = (HERE / "flash.sh").read_text()
    code = "\n".join(l for l in f.splitlines() if not l.lstrip().startswith("#"))
    assert "partprobe" not in code and "blockdev --rereadpt" in code and "udevadm settle" in code
    assert code.count("  reread\n") == 2 and "does not show the new data partition" in code


def test_stick_checks_itself_before_measuring(tmp_path):
    b = (HERE / "build.sh").read_text()
    assert "md5sum > md5sum.txt" in b and b.index("md5sum.txt") < b.index('step "ISO (BIOS + UEFI, GPT hybrid)"')
    assert "! -path ./boot/grub/bios.img" in b                       # xorriso patches it: never matches
    assert b.index('step "ISO checksums') > b.index('step "ISO (BIOS + UEFI, GPT hybrid)"') and "md5sum --quiet -c md5sum.txt" in b
    unit = (HERE / "ubuntu/rootfs/etc/systemd/system/rigsim-stick.service").read_text()
    assert "casper-md5check.service" in unit.split("After=")[1].splitlines()[0]
    f = tmp_path / "r.json"
    f.write_text('{"result": "fail", "checksum_missmatch": ["./casper/filesystem.squashfs"]}')
    assert stick.stick_check(f) == dict(result="fail", mismatched=["./casper/filesystem.squashfs"])
    assert stick.stick_check(tmp_path / "none")["result"].startswith("unknown")


def test_containerd_store_on_the_data_partition_and_loaded_images_verified(tmp_path, monkeypatch):
    """Regression (first real boot): containerd's snapshots lived on the live system's overlay root ("not supported
    as upperdir"), so every container question failed in a second, and the images were loaded into RAM while the
    markers on the stick said 'loaded'."""
    import io
    import json as js
    import tarfile
    src = (HERE / "payload" / "stick.py").read_text()
    body = src[src.index("def start_docker("):src.index("def image_tags(")]
    assert body.index('"mount", "--bind", str(croot), str(cl)') < body.index('"systemctl", "start", "containerd"')
    assert '"docker", "run", "--rm", "--network", "none", tags[0], "true"' in body
    assert 'sh(["umount", "/var/lib/containerd"]' in src
    img = tmp_path / "seed" / "images" / "debian_trixie.tar"; img.parent.mkdir(parents=True)
    with tarfile.open(img, "w") as t:
        b = js.dumps([{"RepoTags": ["debian:trixie"]}]).encode()
        i = tarfile.TarInfo("manifest.json"); i.size = len(b); t.addfile(i, io.BytesIO(b))
    assert stick.image_tags(img) == ["debian:trixie"]
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / f".loaded-{img.name}").write_text(str(img.stat().st_size))
    monkeypatch.setattr(stick, "sh", lambda cmd, timeout=0: (1, "No such image") if cmd[:3] == ["docker", "image", "inspect"] else (0, ""))
    assert stick._loaded(img, tmp_path) is False                       # marker there, image gone: load again
    monkeypatch.setattr(stick, "sh", lambda cmd, timeout=0: (0, ""))
    assert stick._loaded(img, tmp_path) is True


# --- findings of the first trial run (i7-13700K, RTX 3080, UHD 770; stick 1543d5a) -----------------------------------
def test_trial_run_fixes_inventory_and_tools(tmp_path):
    import re
    inv = dict(stick.INVENTORY)
    assert inv["dmidecode.txt"][1:] == ["-t", "bios", "-t", "system", "-t", "baseboard", "-t", "processor", "-t", "memory",
                                        "-t", "cache", "-t", "slot"]
    assert "vulkan-loader.txt" in inv and "gpu-driver.txt" in inv
    # clpeak: only the options this clpeak knows (Ubuntu's has no --compute-hp)
    assert "--compute-hp" not in stick.clpeak_tests("--global-bandwidth --compute-sp --compute-integer --kernel-latency")
    assert stick.clpeak_tests("nothing known") == ["--all-tests"]
    # test-backend-ops: sql (timings), filtered cases
    plan = dict(stick.backend_ops_plan("t", "Vulkan0"))
    assert plan["ops-mul_mat"][-2:] == ["--output", "sql"] and "-p" in plan["ops-flash_attn_ext"] and "-p" not in plan["ops-add"]
    assert re.search(stick.BACKEND_OPS_FILTER["FLASH_ATTN_EXT"], "hsk=128,hsv=128,nh=8,nr23=[4,1],kv=8192,nb=1,mask=1,"
                     "sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,")
    assert not re.search(stick.BACKEND_OPS_FILTER["FLASH_ATTN_EXT"], "hsk=576,hsv=512,kv=8192,type_K=f16,")
    assert not re.search(stick.BACKEND_OPS_FILTER["MUL_MAT"], "type_a=iq2_xxs,type_b=f32,m=4096,n=1")
    # tool outputs: local IPs and MACs masked (the Blender launcher printed the LAN resolver)
    assert stick.scrub("lookup on 192.168.130.5:53 via 100.101.1.2, mac aa:bb:cc:dd:ee:ff, driver 580.178.04, 8.8.8.8") == \
        "lookup on x.x.x.x:53 via x.x.x.x, mac <mac>, driver 580.178.04, 8.8.8.8"
    src = (HERE / "payload" / "stick.py").read_text()
    assert src.count("write_text(scrub(") >= 15 and "NvidiaSampler(rundir / \"samples-nvidia.csv\", nvidia)" in src
    assert "the launcher failed before listing devices" in src


def test_spd_dump_zeroes_serial_bytes(tmp_path):
    i2c = tmp_path / "i2c"
    for name, drv, size, serial in (("0-0050", "ee1004", 512, range(325, 329)), ("1-0051", "spd5118", 1024, range(517, 521))):
        dev = i2c / name; dev.mkdir(parents=True)
        (tmp_path / "drivers" / drv).mkdir(parents=True, exist_ok=True)
        (dev / "driver").symlink_to(tmp_path / "drivers" / drv)
        (dev / "eeprom").write_bytes(bytes([0xAB]) * size)
    (i2c / "0-0060").mkdir()                                                        # not an SPD: ignored
    dst = tmp_path / "inv"; dst.mkdir()
    assert stick.spd_raw(dst, i2c) == ["spd-0-0050.hex", "spd-1-0051.hex"]
    for name, serial in (("spd-0-0050.hex", range(325, 329)), ("spd-1-0051.hex", range(517, 521))):
        b = bytes.fromhex("".join(dst.joinpath(name).read_text().splitlines()[1:]).replace(" ", ""))
        assert all(b[i] == 0 for i in serial) and b[serial.start - 1] == 0xAB and b[serial.stop] == 0xAB
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert "nvidia-modprobe" in c and "the NVIDIA Vulkan driver misses libraries" in c


# --- versions: format, manifest, tools per step ------------------------------------------------------------------
def test_tools_for_each_step_come_from_the_manifest():
    man = dict(system=dict(stick=dict(version="0.9.0", commit="a" * 40), packages={"clpeak": "1.1.2-1", "libvulkan1": "1.3.275",
                                                                              "mesa-vulkan-drivers": "25.0.7", "docker.io": "26"},
                           files={"bin/vkprobe": dict(sha256="c" * 64)}),
               seed={"images": {"nvidia/cuda:12.8.1-devel-ubuntu24.04": dict(digest="nvidia/cuda@sha256:1"),
                                "debian:trixie": dict(digest="debian@sha256:2")},
                     "llama.cpp": {"cuda": dict(commit="d" * 40)}, "models": {"m.gguf": dict(sha256="e" * 64)},
                     "vendor": {"rocm": dict(rocm_version="6.4.1")}, "tools": {"mlc": dict(sha256="f" * 64)}},
               runtime=dict(kernel="7.0.0", nvidia_driver="580.178.04", vulkan_devices=[dict(deviceName="RTX 3080")]))
    t = stick.tools_for
    assert t("cpu-gemm", man)["images"] == {"debian:trixie": dict(digest="debian@sha256:2")}
    assert set(t("gpu-gemm", man)["images"]) == {"nvidia/cuda:12.8.1-devel-ubuntu24.04"} and t("gpu-gemm", man)["nvidia_driver"]
    assert t("opencl", man)["packages"] == {"clpeak": "1.1.2-1"}
    assert t("vkprobe", man)["vkprobe"]["sha256"] == "c" * 64 and t("vkprobe", man)["vulkan_devices"]
    assert t("rocm", man)["vendor"] == dict(rocm_version="6.4.1")
    assert t("llama-bench", man)["llama.cpp"]["cuda"]["commit"] == "d" * 40 and t("llama-bench", man)["models"]
    assert t("mlc", man, dict(mlc_version="v3.11b"))["mlc_version"] == "v3.11b"
    for step in ("cpu-gemm", "blender", "mlc", "anything"):
        assert t(step, man)["stick"]["version"] == "0.9.0" and t(step, man)["stick"]["format"] == stick.FORMAT["version"]


def test_mlc_version_from_its_output(tmp_path):
    (tmp_path / "mlc").mkdir()
    (tmp_path / "mlc" / "idle_latency.txt").write_text("Intel(R) Memory Latency Checker - v3.11b\nCommand line parameters\n")
    assert stick.mlc_version(tmp_path) == "v3.11b"


def test_build_writes_both_manifests_and_the_stick_version():
    b = (HERE / "build.sh").read_text()
    assert 'STICK_VERSION="$(cat "$S/VERSION")"' in b and '"$P/STICK_VERSION"' in b
    assert 'manifest.py" rootfs "$ROOT" "$P"' in b and 'manifest.py" seed "$OUT/seed" "$OUT/seed/MANIFEST.json"' in b
    assert b.index('manifest.py" rootfs') < b.index('step "squashfs"')        # inside the image, before it is packed
    import re
    assert re.fullmatch(r"\d+\.\d+\.\d+", (HERE / "VERSION").read_text().strip())
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'run["questions"][q]["tools"] = tools_for(q, man)' in src and '(rundir / "manifest.json").write_text' in src


def test_manifest_tool(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("manifest", HERE / "tools" / "manifest.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    assert M.dpkg_list("nvidia-utils-580\t580.178.04\nlibfoo\t1\nmesa-vulkan-drivers\t25.0.7\n", M.ROOTFS_PACKAGES) == \
        {"mesa-vulkan-drivers": "25.0.7", "nvidia-utils-580": "580.178.04"}
    p = tmp_path / "payload"; (p / "runner").mkdir(parents=True)
    (p / "stick.py").write_text("a"); (p / "runner" / "hwq.py").write_text("b")
    h1 = M.tree_sha256(p); (p / "stick.py").write_text("c")
    assert M.tree_sha256(p) != h1                                         # any script change changes the fingerprint
    seed = tmp_path / "seed"
    (seed / "llama" / "cuda").mkdir(parents=True); (seed / "llama" / "cuda" / "COMMIT").write_text("d" * 40 + "\n")
    (seed / "llama" / "cuda" / "llama-bench").write_bytes(b"x")
    (seed / "models").mkdir(); (seed / "models" / "SHA256SUMS").write_text("e" * 64 + "  m.gguf\n")
    (seed / "debs").mkdir(); (seed / "debs" / "libdnnl3_3.5.3-1_amd64.deb").write_bytes(b"")
    (seed / "images").mkdir()
    monkeypatch.setattr(M, "sh", lambda cmd: "")
    monkeypatch.setenv("IMAGES", "debian:trixie")
    m = M.seed(seed, tmp_path / "out.json")
    assert m["llama.cpp"]["cuda"]["commit"] == "d" * 40 and "llama-bench" in m["llama.cpp"]["cuda"]["files"]
    assert m["models"] == {"m.gguf": dict(sha256="e" * 64)}
    assert m["images"]["debian:trixie"]["offline_packages"] == {"libdnnl3": "3.5.3-1"}


def test_blender_types_guessed_when_the_launcher_cannot_list_devices(tmp_path):
    for card, vendor in (("card1", "0x8086"), ("card2", "0x10de")):
        (tmp_path / card / "device").mkdir(parents=True); (tmp_path / card / "device" / "vendor").write_text(vendor)
    assert stick.guessed_blender_types(tmp_path) == ["CPU", "OPTIX", "CUDA", "ONEAPI"]


def test_blender_offline_route(tmp_path, monkeypatch):
    """Official offline route: Blender from the launcher's cache + the benchmark script + a scene, per device type;
    a guessed GPU type the first scene fails on is 'unavailable', not a failure."""
    b = tmp_path / "seed" / "blender"
    exe = b / "home" / ".cache" / "blender-benchmark-launcher" / "blender" / "4.2.0" / "blender-4.2.0-linux-x64" / "blender"
    exe.parent.mkdir(parents=True); exe.write_text("#!/bin/sh\n"); exe.chmod(0o755)
    for sc in stick.BLENDER_SCENES:
        f = b / "home" / ".cache" / "blender-benchmark-launcher" / "scenes" / sc / "main.blend"
        f.parent.mkdir(parents=True); f.write_bytes(b"x" * 10)
    (b / "script").mkdir(); (b / "script" / "main.py").write_text("")
    (b / "BLENDER_VERSION").write_text("4.2.0\n")
    found_exe, scenes = stick.blender_offline(b, "4.2.0")
    assert found_exe == exe and set(scenes) == set(stick.BLENDER_SCENES)
    monkeypatch.setattr(stick, "guessed_blender_types", lambda drm=None: ["CPU", "OPTIX", "ONEAPI"])
    calls = []

    class R:
        def __init__(self, rc): self.returncode, self.stdout, self.stderr = rc, '{"stats": {}}', ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        assert kw["cwd"] == str(b / "script") and cmd[-2:][0] == "--device-type"
        return R(0 if cmd[-1] in ("CPU", "OPTIX") else 1)
    monkeypatch.setattr(stick, "run_cmd", fake_run)
    r = stick.run_blender(tmp_path, tmp_path / "run", tmp_path / "log")
    assert r["route"] == "offline" and r["rc"] == 0 and r["types_unavailable"] == ["ONEAPI"]
    assert set(r["runs"]) == {f"{t}-{s}" for t in ("cpu", "optix") for s in stick.BLENDER_SCENES}
    assert calls[0][:2] == [str(exe), "--background"] and "--python" in calls[0]
    b_sh = (HERE / "build.sh").read_text()
    assert "blender-benchmark-script-3.1.1.tar.gz" in b_sh and "BLENDER_SCRIPT_URL" in b_sh


def test_blender_hip_runs_inside_the_rocm_image(tmp_path, monkeypatch):
    b = tmp_path / "seed" / "blender"
    exe = b / "home" / "cache" / "4.2.0" / "blender"; exe.parent.mkdir(parents=True); exe.write_text(""); exe.chmod(0o755)
    f = b / "home" / "cache" / "scenes" / "monster" / "main.blend"; f.parent.mkdir(parents=True); f.write_bytes(b"x")
    (b / "script").mkdir(); (b / "script" / "main.py").write_text(""); (b / "BLENDER_VERSION").write_text("4.2.0")
    monkeypatch.setattr(stick, "BLENDER_SCENES", ["monster"])
    monkeypatch.setattr(stick, "guessed_blender_types", lambda drm=None: ["CPU", "HIP"])
    monkeypatch.setattr(stick, "vendor_images", lambda data: {"rocm": "rigsim/rocm:k1"})
    monkeypatch.setattr(stick, "load_image", lambda *a, **k: dict(rc=0, cached=True))
    calls = []

    class R:
        returncode, stdout, stderr = 0, "{}", ""
    monkeypatch.setattr(stick, "run_cmd", lambda cmd, **kw: calls.append(cmd) or R())
    r = stick.run_blender(tmp_path, tmp_path / "run", tmp_path / "log")
    hip = next(c for c in calls if c[-1] == "HIP")
    assert hip[:2] == ["docker", "run"] and "/dev/kfd" in hip and f"{b}:{b}:ro" in hip and "rigsim/rocm:k1" in hip
    assert hip[hip.index("rigsim/rocm:k1") + 1:][:2] == ["/rigsim/run.sh", str(exe)]
    assert r["rc"] == 0 and "hip-monster" in r["runs"] and r["hip_runtime"] == "ROCm image rigsim/rocm:k1"
    monkeypatch.setattr(stick, "vendor_images", lambda data: {})                     # no ROCm image: unavailable
    assert stick.run_blender(tmp_path, tmp_path / "run2", tmp_path / "log")["types_unavailable"] == ["HIP"]
    assert "libxkbcommon0" in (HERE / "src" / "rocm" / "build-in-image.sh").read_text()


def test_image_load_reports_progress(tmp_path, monkeypatch, capsys):
    img = tmp_path / "seed" / "images" / "nvidia_cuda.tar"; img.parent.mkdir(parents=True); img.write_bytes(b"x" * (4 << 20))
    (tmp_path / "docker").mkdir()
    positions = iter([1 << 20, 3 << 20, 4 << 20])
    monkeypatch.setattr(stick, "archive_position", lambda pid, path: next(positions))
    monkeypatch.setattr(stick, "_loaded", lambda img, data: False)

    class P:
        n = 0
        returncode = None
        pid = 1
        def __init__(self, *a, **k): pass
        def wait(self, timeout=None):
            P.n += 1
            if P.n < 4:
                raise stick.subprocess.TimeoutExpired("docker", timeout)
            self.returncode = 0
            return 0
    monkeypatch.setattr(stick.subprocess, "Popen", P)
    r = stick.load_image(img, tmp_path, None, every=0.01)
    out = capsys.readouterr().out
    assert r["rc"] == 0 and "(25%)" in out and "(75%)" in out and "archive read, unpacking" in out and "-" not in out.split("GiB")[0][-6:]
    assert (tmp_path / "docker" / ".loaded-nvidia_cuda.tar").read_text() == str(4 << 20)


def test_archive_position_from_proc(tmp_path):
    img = tmp_path / "a.tar"; img.write_bytes(b"x")
    pr = tmp_path / "proc" / "42"; (pr / "fd").mkdir(parents=True); (pr / "fdinfo").mkdir()
    (pr / "fd" / "7").symlink_to(img); (pr / "fdinfo" / "7").write_text("pos:\t123456\nflags:\t0100000\n")
    assert stick.archive_position(42, img, tmp_path / "proc") == 123456
    assert stick.archive_position(43, img, tmp_path / "proc") is None

def test_flash_preloads_images_into_a_store_file_and_copies_it_once():
    """Unpacking the ROCm image onto a USB stick took hours (tens of thousands of small files, read and write on the
    same stick): the store is built in one file on the flashing machine and copied to the stick in one go."""
    f = (HERE / "flash.sh").read_text()
    seed = f.split("seed_stick() {", 1)[1].split("\n}\n", 1)[0]
    assert seed.rstrip().endswith('preload_images "$1"')                 # every mode that seeds also preloads
    b = f.split("build_store() {", 1)[1].split("\n}\n", 1)[0]
    ns = f.split("STORE_IN_NAMESPACE='", 1)[1].split("exit $rc'", 1)[0]
    assert 'truncate -s "${STORE_GB}G" "$STORE"' in b and "mkfs.ext4" in b
    # every chroot mount in a private mount namespace: nothing propagates to the host (Nobara lost /dev/pts)
    assert 'unshare --mount --propagation private bash -c "$STORE_IN_NAMESPACE"' in b and "mount --rbind" not in b
    assert ns.index("mount --make-rprivate /") < ns.index("mount --rbind /dev")
    assert 'mount -o loop "$STORE" "$t/store"' in ns and 'mount --bind -o ro "$OUT/seed" "$r/mnt/rigsim/seed"' in ns
    assert 'mount --bind "$t/store/containerd" "$r/var/lib/containerd"' in ns and 'chroot "$r"' in ns
    assert "still has mounts; left in place" in b
    outside = f.replace(f.split("STORE_IN_NAMESPACE='", 1)[1].split("exit $rc'", 1)[0], "")
    assert "--rbind" not in outside                                                # no recursive bind outside it
    pre = f.split("preload_images() {", 1)[1].split("\n}\n", 1)[0]
    assert 'dd if="$STORE" of="$MNT/store.img" bs=16M conv=sparse' in pre and "NO_PRELOAD" in pre
    assert "not copied" in pre and pre.rstrip().endswith("return 0")              # never fails the flash
    assert "dockerd --data-root /mnt/rigsim/docker --containerd=/run/containerd/containerd.sock" in f
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'sys.argv[1] == "--preload"' in src and "def mount_store(" in src
    ms = src.split("def mount_store(", 1)[1].split("\ndef ", 1)[0]
    assert '"e2fsck", "-p"' in ms and '"mount", "-o", "loop"' in ms and 'return "partition"' in ms
    assert src.index("store = mount_store(data, log)") < src.index('cl = Path("/var/lib/containerd")')

def _rocm_loop(tmp_path, targets: str, fail: str) -> subprocess.CompletedProcess:
    """The ROCm image's llama.cpp build loop, run with a pretend cmake: `fail` = the targets that do not compile
    ("*" = an error not tied to a target)."""
    s = (HERE / "src" / "rocm" / "build-in-image.sh").read_text()
    loop = s[s.index(": > /rigsim/TARGETS_DROPPED"):s.index("\ndone\n", s.index("for attempt in")) + 6]
    root = tmp_path / "rigsim"; root.mkdir(); (root / "TARGETS").write_text(targets + "\n")
    bin_ = tmp_path / "bin"; bin_.mkdir()
    (bin_ / "hipconfig").write_text("#!/bin/sh\necho /opt\n")
    (bin_ / "cmake").write_text(f"""#!/bin/bash
[ "$1" = --build ] || exit 0
[ "{fail}" = "*" ] && {{ echo "error: something else"; exit 1; }}
for t in $(tr ';' ' ' < {root}/TARGETS); do
  case " {fail} " in *" $t "*) echo "14 errors generated when compiling for $t."; exit 1;; esac
done
""")
    for f in bin_.iterdir():
        f.chmod(0o755)
    script = "set -euo pipefail\n" + loop.replace("/tmp/", "@TMP@/").replace("/rigsim/", f"{root}/").replace("@TMP@", str(tmp_path))
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=tmp_path,
                          env=dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}"))


def test_rocm_build_drops_targets_llama_cpp_cannot_compile(tmp_path):
    r = _rocm_loop(tmp_path, "gfx900;gfx1031;gfx1100;gfx1201", "gfx1031 gfx1201")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / "rigsim" / "TARGETS").read_text().strip() == "gfx900;gfx1100"
    assert (tmp_path / "rigsim" / "TARGETS_DROPPED").read_text().split() == ["gfx1031", "gfx1201"]
    assert "TARGETS_DROPPED" in (HERE / "tools" / "manifest.py").read_text()


def test_rocm_build_says_when_no_target_compiles(tmp_path):
    """Build of 2026-11-06 (ROCm 6.4.1, llama.cpp v0.8.0): every target failed in turn; the old loop gave up after four
    attempts with "not a per-target error"."""
    r = _rocm_loop(tmp_path, "gfx900;gfx1031;gfx1100;gfx1201;gfx942", "gfx900 gfx1031 gfx1100 gfx1201 gfx942")
    assert r.returncode == 1 and "for every GPU target" in r.stdout
    r = _rocm_loop(tmp_path / "x" if (tmp_path / "x").mkdir() is None else None, "gfx900;gfx1100", "*")
    assert r.returncode == 1 and "not for one GPU target" in r.stdout


def test_llama_cpp_release_choice(tmp_path):
    """Releases only; default 'release' = newest patch of the previous release line (time for hotfixes)."""
    import subprocess
    s = (HERE / "build.sh").read_text()
    fn = s[s.index("pick_release() {"):s.index("\n}\n", s.index("pick_release() {")) + 3]
    git = tmp_path / "git"
    git.write_text("#!/bin/sh\nfor t in v0.4.0 v0.4.1 v0.5.0 v0.9.0 v0.9.2 v0.9.1 v0.10.0 b10549 v0.10.0-rc1; do "
                   "printf 'x\\trefs/tags/%s\\n' $t; done\n")
    git.chmod(0o755)
    run = lambda mode: subprocess.run(["bash", "-c", f"set -euo pipefail\nLLAMA_URL=x\n{fn}pick_release {mode}"],
                                      capture_output=True, text=True, env=dict(PATH=f"{tmp_path}:/usr/bin:/bin")).stdout.strip()
    assert run("release") == "v0.9.2" and run("latest-release") == "v0.10.0"
    assert 'LLAMA_REF="${LLAMA_REF:-release}"' in s and 'export LLAMA_SELECT="$LLAMA_REF"' in s
    assert "LLAMA_SELECT" in (HERE / "tools" / "manifest.py").read_text()


def test_rocm_llama_build_is_position_independent():
    s = (HERE / "src" / "rocm" / "build-in-image.sh").read_text()
    assert "-DCMAKE_POSITION_INDEPENDENT_CODE=ON" in s and "-DCMAKE_HIP_FLAGS=-fPIC" in s


def test_rootfs_cached_for_a_week_unless_forced():
    b = (HERE / "build.sh").read_text()
    assert 'MAX_DAYS="${ROOTFS_MAX_AGE_DAYS:-7}"' in b and '"${REBUILD_ROOTFS:-0}" != 1' in b
    assert 'srckey "$S/ubuntu/chroot-setup.sh" "$S/ubuntu/rootfs"' in b                  # its inputs, not grub.cfg
    cached, built = b.split('if [[ "${REBUILD_ROOTFS:-0}" != 1', 1)[1].split("\nelse\n", 1)
    assert 'cp -a "$RCACHE/." "$ROOT/"' in cached and "rootfs_mounts" in cached and "mmdebstrap" not in cached
    assert built.index("chroot-setup.sh") < built.index('step "rootfs cache') and 'cp -ax "$ROOT/."' in built
    assert b.index('export ROOTFS_BUILT') < b.index('manifest.py" rootfs')
    assert "ROOTFS_BUILT" in (HERE / "tools" / "manifest.py").read_text()
    assert "REBUILD_ROOTFS ROOTFS_MAX_AGE_DAYS" in b                                       # passed into the build container


def test_second_trial_run_fixes():
    setup = (HERE / "overlay" / "toolkits" / "cpu" / "setup.sh").read_text()
    assert "apt-get install -y -qq --no-install-recommends --no-download" in setup and "dpkg --configure -a" in setup
    assert "offline packages incomplete" in setup
    probe = (HERE.parent / "bench" / "hw" / "toolkits" / "cuda" / "src" / "probe.cu").read_text()
    assert 'atoi(arg(argc, argv, "--count", "2000").c_str())' in probe
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert "libEGL_nvidia.so.0/" in c and "s/libGLX_nvidia" in c
    src = (HERE / "payload" / "stick.py").read_text()
    assert '"ONEAPI_DEVICE_SELECTOR=level_zero:gpu"' in src and '"sycl-ls", "--verbose"' in src


def test_identical_gpus_stay_apart():
    """Two cards of the same model: every GPU result can be traced to its card's PCI address."""
    vk = (HERE / "src" / "vkprobe" / "vkprobe.c").read_text()
    assert "VkPhysicalDevicePCIBusInfoPropertiesEXT" in vk and '\\"pci\\":\\"%s\\",\\"uuid\\"' in vk
    assert "ext_intel_pci_address" in (HERE / "src" / "syclprobe" / "probe_sycl.cpp").read_text()
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'pci=dev.get("pci", "")' in src and '"CUDA_DEVICE_ORDER=PCI_BUS_ID"' in src
    assert tuple(map(int, stick.FORMAT["version"].split("."))) >= (1, 1)


def _cfg(port_type: int) -> bytes:
    c = bytearray(256)
    c[6] = 0x10                     # capabilities list present
    c[0x34] = 0x40                  # first capability: power management, then PCIe
    c[0x40], c[0x41] = 0x01, 0x60
    c[0x60], c[0x61] = 0x10, 0x00
    c[0x62:0x64] = ((port_type << 4) | 2).to_bytes(2, "little")
    return bytes(c)


def test_pcie_topology_from_sysfs(tmp_path):
    """A GPU behind a switch on a chipset root port, another on a CPU root port: parents, port types, links, paths."""
    root = tmp_path / "devices" / "pci0000:00"
    tree = {"0000:00:01.0": (root, 4, "16.0 GT/s PCIe", "16", "0x060400"),
            "0000:01:00.0": (root / "0000:00:01.0", 0, "16.0 GT/s PCIe", "16", "0x030000"),
            "0000:00:1c.0": (root, 4, "8.0 GT/s PCIe", "4", "0x060400"),
            "0000:02:00.0": (root / "0000:00:1c.0", 5, "8.0 GT/s PCIe", "4", "0x060400"),
            "0000:03:00.0": (root / "0000:00:1c.0" / "0000:02:00.0", 6, "8.0 GT/s PCIe", "4", "0x060400"),
            "0000:04:00.0": (root / "0000:00:1c.0" / "0000:02:00.0" / "0000:03:00.0", 0, "8.0 GT/s PCIe", "4", "0x030000")}
    bus = tmp_path / "bus"; bus.mkdir()
    for bdf, (parent, pt, speed, width, cls) in tree.items():
        d = parent / bdf; d.mkdir(parents=True, exist_ok=True)
        for f, v in (("vendor", "0x10de"), ("device", "0x2204"), ("class", cls), ("current_link_speed", speed),
                     ("current_link_width", width), ("max_link_speed", "16.0 GT/s PCIe"), ("max_link_width", "16")):
            (d / f).write_text(v + "\n")
        (d / "config").write_bytes(_cfg(pt))
        (bus / bdf).symlink_to(d)
    t = stick.pcie_topology(bus, names={"0000:00:1c.0": dict(name="Raptor Lake PCH PCI Express Root Port")})
    d = t["devices"]
    assert sorted(t["gpus"]) == ["0000:01:00.0", "0000:04:00.0"]
    assert d["0000:04:00.0"]["path"] == ["0000:00:1c.0", "0000:02:00.0", "0000:03:00.0", "0000:04:00.0"]
    assert [d[h]["port_type"] for h in d["0000:04:00.0"]["path"]] == ["root port", "switch upstream port",
                                                                     "switch downstream port", "endpoint"]
    assert d["0000:04:00.0"]["link"] == dict(speed="8.0 GT/s PCIe", width="4", max_speed="16.0 GT/s PCIe", max_width="16")
    assert d["0000:00:1c.0"]["name"] == "Raptor Lake PCH PCI Express Root Port" and d["0000:01:00.0"]["parent"] == "0000:00:01.0"
    assert d["0000:01:00.0"]["host_bridge"] == "pci0000:00"
    assert stick.pcie_port_type(b"") == "" and "pcie.json" in stick.INVENTORY_EXTRA and tuple(map(int, stick.FORMAT["version"].split("."))) >= (1, 2)


def test_status_line(tmp_path, monkeypatch):
    import json
    st = stick.Status()
    st.start(total_steps=14)
    st.begin("cpu-gemm"); st.set("7/24 items")
    line = st.line()
    assert line[0] in st.SPIN and "[1/14 cpu-gemm]" in line and "7/24 items" in line and "(run 0:00)" in line
    st.begin("vkprobe")
    assert "[2/14 vkprobe]" in st.line() and "7/24" not in st.line()                  # detail resets per step
    st.stop()
    d = tmp_path / "q"; d.mkdir()
    assert stick.question_progress(d) == "preparing"
    (d / "state.json").write_text(json.dumps({"a": {"status": "done"}, "b": {"status": "failed"}, "c": {"status": "pending"}}))
    assert stick.question_progress(d) == "2/3 items"
    src = (HERE / "payload" / "stick.py").read_text()
    for hook in ('STATUS.set(f"{name} {k}/{len(MLC_RUNS)}")', 'STATUS.set(f"{key} {k}/{len(vplan)}")',
                 'STATUS.set(f"test-backend-ops {prefix}', 'STATUS.set(question_progress(d))', 'STATUS.begin(name, skippable=True)'):
        assert hook in src, hook


def test_unattended_shutdown():
    """Third trial run: systemd's SIGTERM during 'systemctl poweroff' became a traceback, and casper stopped at
    'Please remove the installation medium, then press ENTER' — no power-off without someone there."""
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'signal.signal(signal.SIGTERM, signal.SIG_IGN)' in src and '"systemctl", "--no-block", "poweroff"' in src
    assert src.count("power_off()") >= 3                                    # definition + both call sites
    g = (HERE / "ubuntu" / "grub.cfg").read_text()
    assert all("noprompt noeject" in l for l in g.splitlines() if l.strip().startswith("linux "))
    b = (HERE / "build.sh").read_text()
    assert "xargs -0 -r -I{} tar -xf {}" in b and "unpack-1" in b and "offline_route_missing" in src


def test_blender_uses_the_launchers_own_script_and_skips_resource_forks(tmp_path):
    """The launcher's cache as found on the user's build: Blender, scenes and its script already unpacked."""
    b = tmp_path / "blender"; c = b / "home" / ".cache" / "blender-benchmark-launcher"
    exe = c / "blender-versions" / "4f4f" / "blender-4.2.0-linux-x64" / "blender"; exe.parent.mkdir(parents=True)
    exe.write_text(""); exe.chmod(0o755)
    for v in ("3.1.2", "3.0.0"):
        m = c / "benchmark-scripts" / f"h{v}" / f"blender_benchmark_script-{v}" / "main.py"; m.parent.mkdir(parents=True); m.write_text("")
    for sc in stick.BLENDER_SCENES:
        f = c / "scenes" / f"h-{sc}" / sc / "main.blend"; f.parent.mkdir(parents=True); f.write_bytes(b"x" * 100)
    junk = c / "scenes" / "h-classroom" / "._classroom" / "main.blend"; junk.parent.mkdir(parents=True); junk.write_bytes(b"x" * 999)
    assert stick.benchmark_script(b).parent.name == "blender_benchmark_script-3.1.2"
    found, scenes = stick.blender_offline(b, "4.2.0")
    assert found == exe and "._classroom" not in str(scenes["classroom"]) and set(scenes) == set(stick.BLENDER_SCENES)


def test_flash_verifies_everything_it_wrote(tmp_path):
    import subprocess
    f = (HERE / "flash.sh").read_text()
    for mode in ('verify_stick "$part" 1\n  echo "stick ready', 'verify_stick "$part" 1\n  echo "stick updated',
                 's) seed_stick "$part"; verify_stick "$part" 0'):
        assert mode in f, mode
    v = f.split("verify_stick() {", 1)[1].split("\n}\n", 1)[0]
    assert 'mount -o ro "$1"' in v and "--tree \"$OUT/seed\"" in v and '--file "$STORE"' in v and "exit 5" in v
    # the verifier itself: identical passes (the MBR table flash.sh rewrites is ignored), a changed byte fails
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "images").mkdir(parents=True); (dst / "images").mkdir(parents=True)
    (src / "images" / "a.tar").write_bytes(bytes(range(256)) * 4000); (dst / "images" / "a.tar").write_bytes(bytes(range(256)) * 4000)
    iso, dev = tmp_path / "iso", tmp_path / "dev"
    iso.write_bytes(b"\x01" * 100000); d = bytearray(iso.read_bytes()); d[450:460] = b"\xff" * 10; dev.write_bytes(bytes(d) + b"rest")
    run = lambda *a: subprocess.run(["python3", str(HERE / "tools" / "verify.py"), *a], capture_output=True, text=True)
    r = run("--iso", str(iso), str(dev), "--tree", str(src), str(dst))
    assert r.returncode == 0 and "verification passed: 2 item(s)" in r.stdout
    d[5000] = 0; dev.write_bytes(bytes(d))
    (dst / "images" / "b.tar").write_bytes(b"")
    (src / "images" / "c.tar").write_bytes(b"x")
    r = run("--iso", str(iso), str(dev), "--tree", str(src), str(dst))
    assert r.returncode == 1 and "system image: content differs" in r.stdout and "images/c.tar: missing on the stick" in r.stdout


def test_operator_can_skip_a_step(tmp_path, monkeypatch):
    import threading as th
    # a running command is stopped when the step is skipped, and the skip surfaces as StepSkipped
    stick.SKIP.clear()
    th.Timer(0.6, stick.SKIP.set).start()
    import time as tm
    t0 = tm.time()
    try:
        stick.run_cmd(["sleep", "30"])
        raised = False
    except stick.StepSkipped:
        raised = True
    assert raised and tm.time() - t0 < 5
    stick.SKIP.clear()
    assert stick.run_cmd(["echo", "hi"]).stdout.strip() == "hi"
    # Enter twice within the window arms and skips; only while a skippable step runs
    st = stick.STATUS
    st.begin("vkprobe", skippable=True)
    assert "Enter twice: skip step" in st.line() and not stick.SKIP.is_set()
    st.begin("docker")
    assert "Enter twice" not in st.line()
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'run["questions"][q] = dict(skipped=OPERATOR_SKIP)' in src and 'run.setdefault("operator_skips", [])' in src
    assert '"hwq.py"), "restore", "--dir", str(d)]' in src                         # a skipped question still restores
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert "41apt_cdrom" in c and c.index("30accessibility") < c.index("update-initramfs -u -k all")


def test_laptop_run_fixes(tmp_path):
    """A laptop without NVIDIA card (i7-7600U, HD 620): '16/15' steps, NVRM messages, CDI unit failures."""
    src = (HERE / "payload" / "stick.py").read_text()
    assert "STATUS.n = 2 + len(qplan) + 8" in src
    begins = src.count('STATUS.begin("inventory")') + src.count('STATUS.begin("docker")') + 1     # + the question loop
    assert begins == 3 and src.count("STATUS.begin(name, skippable=True)") == 2   # + nvidia_phase2 (counts its own)
    named = src.split('for name, step in (("mlc"', 1)[1].split("STATUS.begin(name", 1)[0]
    assert named.count("lambda:") == 8                                       # the eight named steps counted
    for vendor, cls, want in (("0x8086", "0x030000", False), ("0x10de", "0x030000", True), ("0x10de", "0x040300", False)):
        d = tmp_path / vendor / cls / "0000:01:00.0"; d.mkdir(parents=True)
        (d / "vendor").write_text(vendor + "\n"); (d / "class").write_text(cls + "\n")
        assert stick.has_nvidia_gpu(d.parent) is want                        # an NVIDIA audio function is no GPU
    c = (HERE / "ubuntu" / "chroot-setup.sh").read_text()
    assert "systemctl mask nvidia-cdi-refresh.service" in c
    assert all("loglevel=3" in l for l in (HERE / "ubuntu" / "grub.cfg").read_text().splitlines() if l.strip().startswith("linux "))


def test_flash_finds_the_stick_without_udev(tmp_path):
    """WSL has no udev: lsblk shows no labels. The RIGSIM partition is found with blkid on the partition itself."""
    import subprocess
    f = (HERE / "flash.sh").read_text()
    fn = f.split("data_part() {", 1)[1].split("\n}\n", 1)[0]
    bin_ = tmp_path / "bin"; bin_.mkdir()
    (bin_ / "lsblk").write_text("#!/bin/sh\nprintf '/dev/sdx disk\\n/dev/sdx1 part\\n/dev/sdx2 part\\n/dev/sdx3 part\\n'\n")
    (bin_ / "blkid").write_text('#!/bin/sh\n[ "$5" = /dev/sdx3 ] && echo RIGSIM; exit 0\n')
    for b in bin_.iterdir():
        b.chmod(0o755)
    script = f'DEV=/dev/sdx\ndata_part() {{{fn}\n}}\ndata_part'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": f"{bin_}:/usr/bin:/bin"})
    assert r.stdout.strip() == "/dev/sdx3"
    assert "NAME,LABEL" not in f and f.count('part="$(data_part || true)"') >= 2
    assert '"$(lsblk -dno TRAN "$DEV")" == usb' in f


def test_powercap_limits(tmp_path):
    z = tmp_path / "intel-rapl:0"; z.mkdir()
    (z / "name").write_text("package-0\n"); (z / "enabled").write_text("1\n")
    for i, (n, lim, win) in enumerate((("long_term", 65000000, 28000000), ("short_term", 90000000, 2440))):
        (z / f"constraint_{i}_name").write_text(n + "\n")
        (z / f"constraint_{i}_power_limit_uw").write_text(f"{lim}\n"); (z / f"constraint_{i}_time_window_us").write_text(f"{win}\n")
    (tmp_path / "intel-rapl:0:0").mkdir(); (tmp_path / "intel-rapl:0:0" / "name").write_text("core\n")
    p = stick.powercap(tmp_path)
    assert list(p["zones"]) == ["intel-rapl:0"]
    assert p["zones"]["intel-rapl:0"]["constraints"][0] == dict(name="long_term", limit_w=65.0, window_s=28.0)
    assert "powercap.json" in stick.INVENTORY_EXTRA and stick.FORMAT["version"] >= "1.4"
def test_lscpu_cache_ids_are_not_masked_as_ipv6():
    line = "  0    0      0    0 0:0:0:0          yes 4954.5649 567.0890 3842.5410"
    assert "<ip6>" in stick.sanitize(line)                                  # the general scrub would mask it
    assert "0:0:0:0" in stick.sanitize(line, ipv6=False) and "lscpu-e.txt" in stick.NO_IPV6_SCRUB
def test_cpu_power_questions():
    """The thread ladder (where the power budget starts to bind) and steady load after idle (its time behaviour)."""
    import tomllib
    lad = tomllib.loads((HERE / "questions" / "cpu-gemm-ladder.toml").read_text())
    assert lad["grid"]["threads"][0] == 1 and len(lad["grid"]["threads"]) >= 8 and "--n" in lad["args"]["cmd"]
    pt = tomllib.loads((HERE / "questions" / "cpu-power-time.toml").read_text())
    cmd = " ".join(pt["args"]["cmd"])
    assert "--sustain 120" in cmd and "--idle 30" in cmd and pt["timeout_s"] >= 200
    src = (HERE / "overlay" / "toolkits" / "cpu" / "src" / "cgemm.c").read_text()
    assert "nanosleep" in src and "scaling_cur_freq" in src and "energy_uj" in src and "more threads than logical CPUs" in src
    assert '"Package id 0"' in src and '"Tctl"' in src and "temp_c" in src


def test_every_question_file_is_in_the_profile():
    """A question file the profile does not list never runs (cpu-gemm-ladder / cpu-power-time in stick 0.13.0)."""
    files = {f.stem for f in (HERE / "questions").glob("*.toml")}
    assert files == set(stick.PROFILE), files ^ set(stick.PROFILE)


def test_runner_lists_every_item_up_front():
    """The stick's counter 'done/total' reads the runner's state.json: all planned items must be in it from the
    start, or the total grows with the done count ('13/13', '14/14')."""
    src = (HERE.parent / "bench" / "hw" / "runner" / "hwq.py").read_text()
    i = src.index('state = json.loads(sf.read_text()) if sf.exists() else {}')
    assert src.index('state.setdefault(it["id"], {"status": "pending"})', i) < src.index("def on_signal", i)


# --- seed self-test (stick/tools/selftest.py) ------------------------------------------------------------------------
_st_spec = importlib.util.spec_from_file_location("selftest", HERE / "tools" / "selftest.py")
selftest = importlib.util.module_from_spec(_st_spec); _st_spec.loader.exec_module(selftest)
FULL_ENV = dict(MLC_URL="x", BLENDER_LAUNCHER_URL="x", LLAMA_FORKS="qr0n=u@c:llama.cpp kmic68=u@c")


def _exe(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text("#!/bin/sh\n"); p.chmod(0o755); return p


def _full_seed(tmp: Path) -> tuple[Path, Path]:
    s, pay = tmp / "seed", tmp / "rootfs" / "opt" / "rigsim"
    for img in ("nvidia/cuda:12", "debian:trixie"):
        f = selftest.image_tar(s, img); f.parent.mkdir(parents=True, exist_ok=True); f.write_text("x")
    (s / "debs").mkdir(parents=True); (s / "debs" / "a.deb").write_text("x")
    _exe(pay / "bin" / "vkprobe")
    for be in ("vulkan", "cuda"):
        for b in selftest.LLAMA_BINS:
            _exe(s / "llama" / be / b)
    (s / "models").mkdir()
    for m in ("a", "b"):
        (s / "models" / f"{m}.gguf").write_text("x")
    bl = s / "blender"; bl.mkdir(); (bl / "BLENDER_VERSION").write_text("4.2.0")
    _exe(bl / "unpacked" / "blender" / "blender-4.2.0-linux-x64" / "blender")
    for n in stick.BLENDER_SCENES:
        f = bl / "unpacked" / "scenes" / n / f"{n}.blend"; f.parent.mkdir(parents=True); f.write_text("x")
    (bl / "script").mkdir(); (bl / "script" / "main.py").write_text("x")
    _exe(s / "tools" / "mlc" / "mlc")
    for v in ("rocm", "oneapi"):
        (s / "images" / f"vendor-{v}.tar").write_text("x")
    for f in ("qr0n", "kmic68"):
        _exe(s / "llama" / "forks" / f / "llama-bench"); (s / "llama" / "forks" / f / "COMMIT").write_text("abc")
    sw = tmp / "rootfs" / selftest.SWAP; sw.mkdir(parents=True)
    for m in ("nvidia", "nvidia-uvm", "nvidia-modeset", "nvidia-drm"):
        (sw / f"{m}.ko").write_text("x")
    (sw / "VERSION").write_text("580.178.04\n"); (sw / "BOOT_VERSION").write_text("580.178.04\n")
    return s, pay


def _run(s, pay, env=FULL_ENV, ldd_root=lambda p: [], ldd_image=lambda i, p: [], root_has=lambda lib: True):
    return selftest.SelfTest(s, pay, env, "nvidia/cuda:12", ["nvidia/cuda:12", "debian:trixie"],
                             ldd_root, ldd_image, stick, root_has).run()


def test_selftest_passes_a_complete_seed(tmp_path):
    s, pay = _full_seed(tmp_path)
    r = _run(s, pay)
    assert all(x["status"] == "ok" for x in r.values()), r


def test_selftest_finds_the_audit_failures(tmp_path):
    """The audit of 2026-10-31: Blender without libXfixes, MLC and a vendor image not in the seed."""
    s, pay = _full_seed(tmp_path)
    (s / "tools" / "mlc" / "mlc").unlink(); (s / "images" / "vendor-oneapi.tar").unlink()
    r = _run(s, pay, ldd_root=lambda p: ["libXfixes.so.3"] if p.name == "blender" else [])
    assert {k for k, v in r.items() if v["status"] == "missing"} == {"blender", "mlc", "oneapi"}, r
    assert "libXfixes.so.3" in r["blender"]["problems"][0]


def test_selftest_checks_cuda_binaries_in_the_cuda_image(tmp_path):
    s, pay = _full_seed(tmp_path)
    seen = []
    r = _run(s, pay, ldd_image=lambda i, p: seen.append((i, p.name)) or ["libcublas.so.12"])
    assert r["llama.cpp cuda"]["status"] == "missing" and r["llama.cpp vulkan"]["status"] == "ok"
    assert ("nvidia/cuda:12", "llama-bench") in seen


def test_selftest_driver_libraries_come_from_the_system_image(tmp_path):
    """First real build (2026-10-31): libcuda.so.1 is never in the CUDA image — the container toolkit brings it in from
    the system image. Fine when the system image has it; missing there, it is a real gap."""
    s, pay = _full_seed(tmp_path)
    img = lambda i, p: ["libcuda.so.1"]
    assert _run(s, pay, ldd_image=img)["llama.cpp cuda"]["status"] == "ok"
    r = _run(s, pay, ldd_image=img, root_has=lambda lib: False)["llama.cpp cuda"]
    assert r["status"] == "missing" and "driver library" in r["problems"][0]
    assert selftest.is_driver_lib("libnvidia-ml.so.1") and not selftest.is_driver_lib("libcublas.so.12")


def test_selftest_parts_left_out_on_purpose_are_not_failures(tmp_path):
    s, pay = _full_seed(tmp_path)
    for p in (s / "models").glob("*"):
        p.unlink()
    (s / "images" / "vendor-rocm.tar").unlink()
    r = _run(s, pay, env=dict(SKIP_MODELS="1", SKIP_ROCM="1", MLC_URL="", BLENDER_LAUNCHER_URL="", LLAMA_FORKS=""))
    assert {k for k, v in r.items() if v["status"] == "left out"} == {"models", "rocm", "mlc", "blender", "llama.cpp forks"}
    assert not [k for k, v in r.items() if v["status"] == "missing"]


def test_selftest_exit_code_and_allow_incomplete(tmp_path, monkeypatch):
    s, pay = _full_seed(tmp_path)
    (s / "debs" / "a.deb").unlink()
    monkeypatch.setattr(selftest, "ldd_in_rootfs", lambda root, p: [])
    monkeypatch.setattr(selftest, "ldd_in_image", lambda i, p: [])
    monkeypatch.setattr(selftest, "rootfs_has_lib", lambda root, lib: True)
    monkeypatch.setattr(selftest, "load_stick", lambda: stick)
    for k, v in FULL_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("ALLOW_INCOMPLETE", raising=False)
    args = ["--seed", str(s), "--rootfs", str(tmp_path / "rootfs"), "--payload", str(pay), "--cuda-image",
            "nvidia/cuda:12", "--image", "nvidia/cuda:12", "--image", "debian:trixie", "--out", str(tmp_path / "o.json")]
    assert selftest.main(args) == 1
    assert json.loads((tmp_path / "o.json").read_text())["parts"]["debs"]["status"] == "missing"
    monkeypatch.setenv("ALLOW_INCOMPLETE", "1")
    assert selftest.main(args) == 0


def test_missing_libs_parses_ldd():
    out = "\tlinux-vdso.so.1 (0x...)\n\tlibXfixes.so.3 => not found\n\tlibc.so.6 => /lib/libc.so.6\n"
    assert selftest.missing_libs(out) == ["libXfixes.so.3"]


def test_selftest_checks_forks_and_the_module_swap_set(tmp_path):
    """Old-GPU sprint: the P100 forks and the other NVIDIA module flavour (same driver version) must be on the stick."""
    s, pay = _full_seed(tmp_path)
    (s / "llama" / "forks" / "kmic68" / "llama-bench").unlink()
    sw = tmp_path / "rootfs" / selftest.SWAP
    (sw / "nvidia-uvm.ko").unlink(); (sw / "BOOT_VERSION").write_text("580.95.05")
    r = _run(s, pay)
    assert r["llama.cpp forks"]["status"] == "missing" and "llama-bench" in r["llama.cpp forks"]["problems"][0]
    p = r["nvidia module swap"]["problems"]
    assert r["nvidia module swap"]["status"] == "missing" and any("nvidia-uvm" in x for x in p) and any("differ" in x for x in p)


# --- old-GPU sprint: NVIDIA module switch, Pascal builds and forks ---------------------------------------------------
def _pci(root: Path, addr: str, vendor: str, cls: str) -> None:
    d = root / addr; d.mkdir(parents=True)
    (d / "vendor").write_text(vendor + "\n"); (d / "class").write_text(cls + "\n")


def test_nvidia_pci_devices_and_norm_pci(tmp_path):
    _pci(tmp_path, "0000:01:00.0", "0x10de", "0x030000"); _pci(tmp_path, "0000:02:00.0", "0x10de", "0x030200")
    _pci(tmp_path, "0000:01:00.1", "0x10de", "0x040300"); _pci(tmp_path, "0000:00:02.0", "0x8086", "0x030000")
    assert stick.nvidia_pci_devices(tmp_path) == ["0000:01:00.0", "0000:02:00.0"]
    assert stick.norm_pci("00000000:01:00.0") == "0000:01:00.0" == stick.norm_pci("01:00.0")


def _swap_set(tmp: Path) -> Path:
    sw = tmp / "swap"; sw.mkdir()
    for m in ("nvidia", "nvidia-uvm", "nvidia-modeset", "nvidia-drm"):
        (sw / f"{m}.ko").write_text("x")
    (sw / "FLAVOUR").write_text("proprietary\n"); (sw / "VERSION").write_text("580.178.04\n")
    return sw


def test_swap_modules_unloads_keeps_measured_cards_off_and_loads_the_other_flavour(tmp_path, monkeypatch):
    monkeypatch.setattr(stick, "_kill_containers", lambda: None)
    sysfs = tmp_path / "sys"
    for m in ("nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia"):
        (sysfs / "module" / m).mkdir(parents=True)
    (sysfs / "bus" / "pci" / "devices" / "0000:01:00.0").mkdir(parents=True)
    cmds = []
    def run(cmd):
        cmds.append(cmd)
        return (0, "drm" if cmd[:3] == ["modinfo", "-F", "depends"] and "nvidia-modeset" in cmd[-1] else "")
    r = stick.swap_modules(["0000:01:00.0"], None, _swap_set(tmp_path), sysfs, run)
    assert "failed" not in r and r["flavour"] == "proprietary" and r["version"] == "580.178.04"
    rm = [c[1] for c in cmds if c[0] == "rmmod"]
    assert rm == ["nvidia_drm", "nvidia_modeset", "nvidia_uvm", "nvidia"]                # dependants first
    ins = [Path(c[1]).name for c in cmds if c[0] == "insmod"]
    assert ins == ["nvidia.ko", "nvidia-uvm.ko", "nvidia-modeset.ko"]                   # no nvidia-drm: headless
    ins_modeset = next(i for i, c in enumerate(cmds) if c[0] == "insmod" and c[1].endswith("nvidia-modeset.ko"))
    assert cmds.index(["modprobe", "drm"]) < ins_modeset                              # a module's dependencies first
    assert (sysfs / "bus" / "pci" / "devices" / "0000:01:00.0" / "driver_override").read_text() == "none"


def test_swap_modules_reports_why_it_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(stick, "_kill_containers", lambda: None)
    sysfs = tmp_path / "sys"; (sysfs / "module" / "nvidia").mkdir(parents=True)
    vt = sysfs / "class" / "vtconsole" / "vtcon1"; vt.mkdir(parents=True)
    (vt / "name").write_text("(M) frame buffer device\n"); (vt / "bind").write_text("1")
    r = stick.swap_modules([], None, _swap_set(tmp_path), sysfs,
                           lambda cmd: (1, "Module nvidia is in use") if cmd[0] == "rmmod" else (0, ""))
    assert "could not unload nvidia" in r["failed"] and (vt / "bind").read_text() == "0"     # tried the console first
    (tmp_path / "swap" / "FLAVOUR").unlink()
    assert "not on this stick" in stick.swap_modules([], None, tmp_path / "swap", sysfs, lambda c: (0, ""))["failed"]


def test_nvidia_phase2_only_when_a_card_is_left_undriven(tmp_path, monkeypatch):
    monkeypatch.setattr(stick, "nvidia_driven", lambda: {"0000:01:00.0": "RTX 3080, 10240 MiB, 00000000:01:00.0"})
    monkeypatch.setattr(stick, "nvidia_pci_devices", lambda: ["0000:01:00.0"])
    assert stick.nvidia_phase2(tmp_path, tmp_path, tmp_path, "img", {}, {}, None) is None


def test_run_llama_runs_the_pascal_forks_only_on_their_architectures(tmp_path, monkeypatch):
    data = tmp_path / "data"; seed = data / "seed" / "llama"
    for p in (seed / "cuda" / "llama-bench", seed / "forks" / "qr0n" / "llama-bench"):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text("x")
    (seed / "forks" / "qr0n" / "ARCHS").write_text("60 61\n")
    (data / "seed" / "models").mkdir(parents=True); (data / "seed" / "models" / "tiny.gguf").write_text("x")
    monkeypatch.setattr(stick, "sh", lambda cmd, timeout=120: (0, "0, 6.0\n1, 8.6") if "--query-gpu=index,compute_cap" in cmd
                        else (0, "0\n1"))
    seen = []
    class R:
        returncode, stdout, stderr = 0, "", ""
    monkeypatch.setattr(stick, "run_cmd", lambda cmd, **kw: seen.append(cmd) or R())
    r = stick.run_llama(data, tmp_path / "run", "img", True, tmp_path / "log", cuda_only=True)
    keys = set(r.get("runs", r))
    assert {"cuda-gpu0-tiny", "cuda-gpu1-tiny", "cuda+qr0n-gpu0-tiny"} <= keys and "cuda+qr0n-gpu1-tiny" not in keys
    assert not any(k.startswith(("vulkan", "cpu")) for k in keys)                       # second phase: CUDA only


# --- D: what a machine cannot do is skipped with the reason, not recorded as a failure (run d-9c93, 2026-10-31) ----
def test_plan_skips_questions_that_need_more_cards(tmp_path):
    for q in stick.PROFILE:
        src = HERE / "questions" / f"{q}.toml"
        (tmp_path / f"{q}.toml").write_text(src.read_text())
    one = {q: why for q, f, why in stick.plan(tmp_path, True, "", 1)}
    assert one["gpu-p2p"] == "needs 2 NVIDIA cards, this machine has 1"
    assert one["gpu-vram"] is None and one["cpu-gemm"] is None
    assert {q: why for q, f, why in stick.plan(tmp_path, True, "", 2)}["gpu-p2p"] is None
    assert {q: why for q, f, why in stick.plan(tmp_path, True, "")}["gpu-p2p"] is None        # count unknown: run it
    assert stick.question_min_gpus(HERE / "questions" / "gpu-p2p.toml") == 2


def test_blender_device_type_not_found_is_unavailable_and_python_errors_fail(tmp_path, monkeypatch):
    b = tmp_path / "data" / "seed" / "blender"; b.mkdir(parents=True); (b / "BLENDER_VERSION").write_text("4.2.0")
    script = b / "script" / "main.py"; script.parent.mkdir(); script.write_text("")
    scenes = {n: tmp_path / f"{n}.blend" for n in ("monster", "junkshop", "classroom")}
    monkeypatch.setattr(stick, "blender_offline", lambda b, v: (tmp_path / "blender", scenes))
    monkeypatch.setattr(stick, "benchmark_script", lambda b: script)
    monkeypatch.setattr(stick, "guessed_blender_types", lambda: ["CPU", "CUDA", "ONEAPI"])
    monkeypatch.setattr(stick, "vendor_images", lambda d: {})
    tb = "Traceback (most recent call last):\n  File ...\n"
    def fake(cmd, **kw):
        t, scene = cmd[-1], Path(cmd[cmd.index("CYCLES") + 1]).stem
        class R: returncode, stdout, stderr = 0, "Blender quit\n", ""
        if t == "ONEAPI":
            R.stderr = tb + "ValueError: No matching device of type OneAPI found. Options are: []\n"
        if t == "CUDA" and scene == "junkshop":
            R.stderr = tb + "RuntimeError: out of memory\n"
        return R()
    monkeypatch.setattr(stick, "run_cmd", fake)
    r = stick.run_blender(tmp_path / "data", tmp_path / "run", tmp_path / "log")
    assert r["types_unavailable"] == ["ONEAPI"] and not any(k.startswith("oneapi") for k in r["runs"])
    assert r["runs"]["cuda-junkshop"]["rc"] == 1 and r["runs"]["cuda-monster"]["rc"] == 0 and r["rc"] == 1


# --- E: oneAPI steps that fail on oneDNN run once more without it (run d-9c93, manual test 2026-11-01) ----------------
ONEDNN_OUT = "could not create a memory object\nException caught at file:/tmp/llama/ggml/src/ggml-sycl/ggml-sycl.cpp\n"


def test_onednn_retry_only_on_the_onednn_error(tmp_path):
    calls = []
    again = lambda env: calls.append(env) or (0, '{"avg_ts": 220.4}')
    rc, out, retry = stick.onednn_retry(134, ONEDNN_OUT, tmp_path / "x.onednn-failed.txt", again)
    assert (rc, out) == (0, '{"avg_ts": 220.4}') and calls == [stick.NO_ONEDNN]
    assert retry == dict(reason="oneDNN: could not create a memory object", env=stick.NO_ONEDNN, first_rc=134,
                         first_output="x.onednn-failed.txt")
    assert ONEDNN_ERROR_TEXT in (tmp_path / "x.onednn-failed.txt").read_text()
    for rc0, out0 in ((134, "Segmentation fault"), (0, ONEDNN_OUT)):            # other aborts, or no failure: as is
        assert stick.onednn_retry(rc0, out0, tmp_path / "y.txt", again) == (rc0, out0, None)
    assert len(calls) == 1 and not (tmp_path / "y.txt").exists()


ONEDNN_ERROR_TEXT = "could not create a memory object"


def test_backend_ops_retry_without_onednn_before_the_other_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(stick, "backend_ops_plan", lambda binary, backend: [("ops-mul_mat", ["tbo", "perf", "-o", "MUL_MAT",
                                                                                          "-b", "SYCL0", "--output", "x"])])
    seen = []
    def run(cmd, t, env=None):
        seen.append((list(cmd), env))
        return (0, "OK") if env == stick.NO_ONEDNN else (134, ONEDNN_OUT)
    r = stick.run_backend_ops(run, tmp_path, "dev0", "tbo", "SYCL0", tmp_path / "log", onednn=stick.OneDnnGuard())
    assert r["dev0-ops-mul_mat"]["rc"] == 0 and r["dev0-ops-mul_mat"]["retry"]["first_rc"] == 134
    assert [e for _, e in seen] == [None, stick.NO_ONEDNN]                       # no --output / -b fallbacks needed
    assert (tmp_path / "dev0-ops-mul_mat.onednn-failed.txt").exists() and (tmp_path / "dev0-ops-mul_mat.txt").exists()
    seen.clear()
    r = stick.run_backend_ops(run, tmp_path, "dev1", "tbo", "Vulkan0", tmp_path / "log")   # not SYCL: no retry
    assert "retry" not in r["dev1-ops-mul_mat"] and all(e is None for _, e in seen)


def test_notice_header_shows_the_stick_version(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stick, "OPT", tmp_path)
    (tmp_path / "STICK_VERSION").write_text("0.13.7\n")
    assert stick.stick_version() == "0.13.7"
    monkeypatch.setattr(stick.sys, "stdin", None)             # no terminal: the notice prints and goes on
    stick.notice(0)
    assert "rigsim hardware stick 0.13.7" in capsys.readouterr().out
    (tmp_path / "STICK_VERSION").unlink()
    assert stick.stick_version() == "unknown"


def test_onednn_stays_off_on_a_device_after_its_first_failure(tmp_path):
    """Run c72d: FLASH_ATTN_EXT hung for 30 min with oneDNN after MUL_MAT had already failed on it. Once a device
    failed on oneDNN, its further steps run without it straight away and say since when."""
    g, seen = stick.OneDnnGuard(), []
    def attempt(env):
        seen.append(env)
        return (0, "ok") if env else (134, ONEDNN_OUT)
    rc, _, extra = g.run("dev0-llama-tiny", tmp_path / "a.txt", attempt)
    assert rc == 0 and "retry" in extra and seen == [None, stick.NO_ONEDNN]
    seen.clear()
    rc, _, extra = g.run("dev0-ops-flash_attn_ext", tmp_path / "b.txt", attempt)
    assert rc == 0 and seen == [stick.NO_ONEDNN] and not (tmp_path / "b.txt").exists()
    assert extra == dict(onednn_off=dict(after="dev0-llama-tiny", env=stick.NO_ONEDNN))
    g2 = stick.OneDnnGuard()                                                     # a device where oneDNN works
    assert g2.run("dev1-x", tmp_path / "c.txt", lambda env: (0, "ok")) == (0, "ok", {}) and not g2.off_after


# --- low memory (run 05ea, 2026-11-01: an 8 GB laptop without swap hung) ---------------------------------------------
def _mem(tmp_path, monkeypatch, avail_gib=6.0, total_gib=7.3, kills=(0,)):
    k = iter(kills)
    last = [0]
    def ok():
        try:
            last[0] = next(k)
        except StopIteration:
            pass
        return last[0]
    monkeypatch.setattr(stick, "meminfo", lambda path=None: {"MemAvailable": int(avail_gib * stick.GIB),
                                                              "MemTotal": int(total_gib * stick.GIB)})
    monkeypatch.setattr(stick, "oom_kills", lambda path=None: ok())
    cg = tmp_path / "cgroup"; cg.mkdir()
    return cg


def test_step_memory_sets_a_limit_for_processes_and_containers(tmp_path, monkeypatch):
    root = _mem(tmp_path, monkeypatch, avail_gib=6.0, kills=(3, 3))
    with stick.StepMemory("blender", None, root) as m:
        lim = int((root / "rigsim-step" / "memory.max").read_text())
        assert lim == int(6.0 * stick.GIB) - stick.MEM_RESERVE
        assert stick.docker_mem() == ["--memory", str(lim), "--memory-swap", str(lim)]
        assert stick.os.environ["RIGSIM_MEMORY_BYTES"] == str(lim)
        m.join(4242)
        assert (root / "rigsim-step" / "cgroup.procs").read_text() == "4242"
    assert stick.docker_mem() == [] and "RIGSIM_MEMORY_BYTES" not in stick.os.environ
    assert m.record() == dict(available_gib=6.0, limit_gib=round(lim / stick.GIB, 2), cgroup=True)


def test_out_of_memory_is_recorded_and_counts_as_done(tmp_path, monkeypatch):
    root = _mem(tmp_path, monkeypatch, kills=(5, 6))
    with stick.StepMemory("blender", None, root) as m:
        pass
    entry = dict(rc=1)
    stick.record_memory(entry, m, None)
    assert entry["memory"]["oom_kills"] == 1 and "ended 1 process(es)" in entry["out_of_memory"]
    assert stick.step_ok(entry) and not stick.step_ok(dict(rc=1)) and stick.step_ok(dict(skipped="x"))


def test_no_cgroup_v2_still_runs(tmp_path, monkeypatch):
    _mem(tmp_path, monkeypatch)
    with stick.StepMemory("mlc", None, tmp_path / "missing" / "root") as m:
        assert not m.cgroup and stick.docker_mem()                 # containers still limited
    assert m.record()["cgroup"] is False


def test_models_that_do_not_fit_in_ram_are_skipped(tmp_path, monkeypatch):
    _mem(tmp_path, monkeypatch, avail_gib=5.0)
    small, big = tmp_path / "tiny.gguf", tmp_path / "big.gguf"
    with open(small, "wb") as f: f.truncate(600 << 20)
    with open(big, "wb") as f: f.truncate(3800 << 20)
    assert stick.model_fits(small) == "" and "needs about 6.3 GiB of RAM, 5.0 GiB available" == stick.model_fits(big)
    summary = """GPU0:
\tapiVersion = 1.3
\tdeviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
\tdeviceName = NVIDIA GeForce RTX 3080
GPU1:
\tdeviceType = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
\tdeviceName = AMD Radeon Vega 8 Graphics (RADV RAVEN)
GPU2:
\tdeviceType = PHYSICAL_DEVICE_TYPE_CPU
\tdeviceName = llvmpipe (LLVM 19.1.1, 256 bits)
"""
    assert stick.vulkan_integrated(summary) == {1}


def test_mlc_gets_a_small_buffer_on_small_machines(tmp_path, monkeypatch):
    _mem(tmp_path, monkeypatch, total_gib=7.3)
    data = tmp_path / "data"; mlc = data / "seed" / "tools" / "mlc" / "mlc"; mlc.parent.mkdir(parents=True); mlc.write_text("")
    seen = []
    monkeypatch.setattr(stick, "sh", lambda cmd, timeout=120: seen.append(cmd) or (0, ""))
    monkeypatch.setattr(stick, "secure_boot", lambda: False)
    monkeypatch.setattr(stick, "reserve_hugepages", lambda: dict(before=0, requested=1000, allocated=1000))
    monkeypatch.setattr(stick, "release_hugepages", lambda h: None)
    r = stick.run_mlc(data, tmp_path / "run", tmp_path / "log")
    assert r["buffer_kib"] == 200 * 1024 and all("-b204800" in c for c in seen if str(mlc) in c[0])
    seen.clear(); _mem(tmp_path / "x", monkeypatch, total_gib=31.1) if (tmp_path / "x").mkdir() is None else None
    r = stick.run_mlc(data, tmp_path / "run", tmp_path / "log")
    assert "buffer_kib" not in r and not any("-b204800" in c for c in seen)


# --- flash.sh faults seen while flashing 0.13.9 (2026-11-03) ----------------------------------------------------------
_vspec = importlib.util.spec_from_file_location("verify", HERE / "tools" / "verify.py")
verify = importlib.util.module_from_spec(_vspec); _vspec.loader.exec_module(verify)


def test_erase_does_not_stop_when_the_stick_has_no_store_yet(tmp_path):
    """Issue #24: on a fresh partition `du store.img` failed and set -e + pipefail ended flash.sh silently."""
    f = (HERE / "flash.sh").read_text()
    line = next(l for l in f.splitlines() if 'old="$(du -B1 "$MNT/store.img"' in l).strip()
    script = f'set -euo pipefail\nMNT={tmp_path}\nf() {{ {line}; echo "went on: [$old]"; }}\nf\n'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "went on: []" in r.stdout


def test_verify_leaves_the_gpt_out_of_the_image_but_nothing_else(tmp_path):
    img = bytearray(64 * 512); img[512:520] = b"EFI PART"; img[40 * 512:40 * 512 + 4] = b"data"
    iso = tmp_path / "x.iso"; iso.write_bytes(bytes(img))
    dev = bytearray(img) + bytes(1 << 20)
    dev[450] = 7; dev[512 + 24] = 9; dev[3 * 512] = 1                  # MBR entry, GPT header, entry array
    (tmp_path / "dev").write_bytes(bytes(dev))
    run = lambda: verify.main(["--iso", str(iso), str(tmp_path / "dev")])
    assert verify.table_skip(str(iso)) == verify.GPT_TABLE and run() == 0
    dev[40 * 512] = ord("D"); (tmp_path / "dev").write_bytes(bytes(dev))
    assert run() == 1                                                   # the image itself still checked
    mbr = bytearray(img); mbr[512:520] = bytes(8); (tmp_path / "m.iso").write_bytes(bytes(mbr))
    assert verify.table_skip(str(tmp_path / "m.iso")) == verify.MBR_TABLE


def test_verify_hashes_holes_as_zeros_and_counts_what_is_read(tmp_path):
    dense, sparse = tmp_path / "d", tmp_path / "s"
    blob = os.urandom(1 << 20)
    with open(dense, "wb") as f:
        f.write(blob); f.write(bytes(64 << 20)); f.write(blob)
    with open(sparse, "wb") as f:
        f.write(blob); f.seek((1 << 20) + (64 << 20)); f.write(blob)
    n = dense.stat().st_size
    assert sparse.stat().st_size == n
    assert verify.digest(str(dense), n, None, "") == verify.digest(str(sparse), n, None, "")
    assert verify.allocated(str(sparse), n) < n // 4                    # holes not counted (where the FS reports them)
    assert verify.main(["--file", str(dense), str(sparse)]) == 0


def test_hugepages_reserved_for_mlc_and_given_back(tmp_path, monkeypatch):
    """Ryzen 5 2500U: MLC stopped with rc 22 without hugepages and ran with 1000 of them (2026-11-04)."""
    nr = tmp_path / "nr_hugepages"; nr.write_text("0\n")
    nodes = tmp_path / "node"; (nodes / "node0").mkdir(parents=True)
    _mem(tmp_path, monkeypatch, avail_gib=6.6)
    h = stick.reserve_hugepages(1000, nr, nodes)
    assert h == dict(before=0, requested=1000, allocated=1000) and nr.read_text() == "1000"
    stick.release_hugepages(h, nr)
    assert nr.read_text() == "0"
    _mem(tmp_path / "x" if (tmp_path / "x").mkdir() is None else None, monkeypatch, avail_gib=2.5)
    assert stick.reserve_hugepages(1000, nr, nodes)["requested"] == 256      # leaves 2 GiB available
    (nodes / "node1").mkdir()
    _mem(tmp_path / "y" if (tmp_path / "y").mkdir() is None else None, monkeypatch, avail_gib=30)
    assert stick.reserve_hugepages(1000, nr, nodes)["requested"] == 2000     # per NUMA node
    assert "error" in stick.reserve_hugepages(1000, tmp_path / "missing", nodes)


def test_gpu_memory_check_and_device_lost(tmp_path, monkeypatch):
    """Run cfed (Vega 8, 3996 MB GPU memory): 7B Q4_0 ran out of GPU memory; FLASH_ATTN_EXT then reset the GPU."""
    big, small = tmp_path / "7b.gguf", tmp_path / "tiny.gguf"
    with open(big, "wb") as f: f.truncate(3825065984)
    with open(small, "wb") as f: f.truncate(637 << 20)
    assert "needs about 4.8 GiB of GPU memory, the device has 3.9 GiB free" == stick.model_fits_gpu(big, 3960)
    assert stick.model_fits_gpu(small, 3960) == "" and stick.model_fits_gpu(big, None) == ""
    class R: stdout, stderr = "Backend 1/2: Vulkan0\n  Device memory: 3996 MB (3960 MB free)\n", ""
    tbo = tmp_path / "test-backend-ops"; tbo.write_text("")
    seen = []
    monkeypatch.setattr(stick, "run_cmd", lambda cmd, **kw: seen.append((cmd, kw.get("env", {}).get("GGML_VK_VISIBLE_DEVICES"))) or R())
    assert stick.vulkan_memory(tbo, 1) == 3960 and seen[0][1] == "1" and "RIGSIM_NONE" in seen[0][0]
    guilty = "  FLASH_ATTN_EXT(...): 8 runs - 1 us/run\n  FLASH_ATTN_EXT(...): 2 runs - 2 us/run\n" \
             "radv/amdgpu: The CS has been cancelled because the context is lost. This context is guilty of a hard recovery.\n" \
             "ggml_vulkan: device lost on Vulkan0\nwhat():  vk::Queue::submit: ErrorDeviceLost\n"
    assert stick.device_lost(guilty) == "the GPU was reset while running this (a hang or the driver's job timeout); 2 case(s) measured before"
    assert stick.device_lost("radv/amdgpu: Not enough memory for command submission.\nErrorDeviceLost").startswith("out of GPU memory")
    assert "after a reset caused elsewhere" in stick.device_lost("This context is innocent.\nErrorDeviceLost")
    assert stick.device_lost("Segmentation fault") == ""
    assert stick.item_ok(dict(rc=-6, device_lost="x")) and stick.item_ok(dict(skipped="y")) and not stick.item_ok(dict(rc=-6))


# --- engine primitives, step 1: matrix throughput over shapes and formats (design/engine-primitives.md) --------------
def test_gemm_shape_questions_expand():
    g = hwq.load_plan(str(HERE / "questions" / "gpu-gemm-shapes.toml"))
    items = hwq.expand(g, False, [dict(name="gpu0", index=0)])
    assert len(items) == 9 * 16 and {i["params"]["prec"] for i in items} >= {"bf16", "tf32", "fp16_fp32acc", "fp8", "nvfp4", "mxfp4"}
    assert {i["params"]["shape"].split("x")[0] for i in items} == {"1", "4", "16", "64", "256", "1024", "4096", "8192"}
    assert len(hwq.expand(g, True, [dict(name="gpu0", index=0)])) == 2
    c = hwq.load_plan(str(HERE / "questions" / "cpu-gemm-shapes.toml"))
    assert len(hwq.expand(c, False)) == 4 * 14 and len(hwq.expand(c, True)) == 4
    assert stick.PROFILE.index("cpu-gemm-shapes") == stick.PROFILE.index("cpu-gemm") + 1
    assert stick.PROFILE.index("gpu-gemm-shapes") == stick.PROFILE.index("gpu-gemm") + 1
    assert "1x4096x4096" in stick.VENDOR_GEMM_SHAPES and "8192x14336x4096" in stick.VENDOR_GEMM_SHAPES


def test_transfer_questions_and_probe_modes():
    x = hwq.load_plan(str(HERE / "questions" / "pcie-xfer.toml"))
    assert len(hwq.expand(x, False, [dict(name="gpu0", index=0)])) == 6 and len(hwq.expand(x, True, [dict(name="gpu0", index=0)])) == 1
    p = hwq.load_plan(str(HERE / "questions" / "gpu-p2p-xfer.toml"))
    assert stick.question_min_gpus(HERE / "questions" / "gpu-p2p-xfer.toml") == 2
    for q in ("pcie-xfer", "gpu-p2p-xfer", "gpu-overlap"):
        assert q in stick.PROFILE
    src = (HERE.parent / "bench" / "hw" / "toolkits" / "cuda" / "src" / "probe.cu").read_text()
    assert 'mode == "xfer"' in src and 'mode == "overlap"' in src and "cudaDeviceDisablePeerAccess" in src
    sy = (HERE / "src" / "syclprobe" / "probe_sycl.cpp").read_text()
    assert 'mode == "xfer"' in sy and "malloc_host" in sy
    keys = [k for k, _ in stick.vendor_plan("rocm")]
    assert "xfer-bidir-pageable" in keys and "overlap" in keys and "overlap" not in [k for k, _ in stick.vendor_plan("oneapi")]


def _blk(root, name, size_gb, model="Samsung SSD 980 PRO 1TB", removable="0", rot="0", devpath=None):
    d = root / name; (d / "queue").mkdir(parents=True)
    (d / "size").write_text(str(int(size_gb * 1e9 / 512))); (d / "removable").write_text(removable)
    (d / "queue" / "rotational").write_text(rot); (d / "queue" / "logical_block_size").write_text("512")
    target = root.parent / "devices" / (devpath or name); target.mkdir(parents=True)
    (target / "model").write_text(model + "\n"); (target / "serial").write_text("S5GXNF0R123456\n")
    (d / "device").symlink_to(target)


def test_internal_drives_without_identifying_features(tmp_path):
    root = tmp_path / "block"; root.mkdir()
    _blk(root, "nvme0n1", 1000.2); _blk(root, "sda", 512.1, model="ST2000DM008\x01", rot="1", devpath="pci0/ata1/sda")
    _blk(root, "sdb", 128, removable="1"); _blk(root, "sdc", 64, devpath="pci0/usb2/sdc"); _blk(root, "loop0", 100)
    _blk(root, "nvme1n1", 4)                                                     # too small
    d = stick.internal_drives({"sdb"}, root)
    assert [x["drive"] for x in d] == ["nvme0n1", "sda"]
    assert d[0] == dict(drive="nvme0n1", kind="nvme", capacity_class_gb=1000, rotational=False,
                        model="Samsung SSD 980 PRO 1TB", logical_block=512)
    assert d[1]["kind"] == "sata" and d[1]["rotational"] and d[1]["model"] == "ST2000DM008"
    assert "serial" not in json.dumps(d) and "S5GXNF" not in json.dumps(d)
    assert stick.capacity_class(960) == 1000 and stick.capacity_class(480) == 512


def test_storage_jobs_are_read_only_and_skippable(tmp_path):
    cmd = stick.fio_cmd("nvme0n1", "randread", "4k", 32, 8)
    assert cmd[:2] == ["fio", "--readonly"] and "--direct=1" in cmd and "--filename=/dev/nvme0n1" in cmd
    assert "--invalidate=1" in stick.fio_cmd("nvme0n1", "read", "1M", 8, 8, direct=False)
    assert all(rw in ("read", "randread") for _, rw, _, _ in stick.STORAGE_JOBS)
    assert stick.run_storage(tmp_path, tmp_path / "log", "quiet rigsim.nodisks splash")["skipped"].startswith("rigsim.nodisks")
    assert stick.parse_fio("garbage") == {}
    sample = json.dumps(dict(jobs=[dict(read=dict(bw_bytes=6.8e9, iops=6485.0, total_ios=50000, lat_ns=dict(mean=4.9e6),
                                                   clat_ns=dict(percentile={"50.000000": 4.8e6, "99.000000": 6.0e6})))]))
    assert stick.parse_fio("fio: note\n" + sample) == dict(bytes_per_s=6.8e9, iops=6485.0, lat_mean_us=4900.0,
                                                            lat_p50_us=4800.0, lat_p99_us=6000.0, ios=50000)
    assert "fio" in (HERE / "ubuntu" / "chroot-setup.sh").read_text()


# --- DDR items (folded into this sprint 2026-11-04): SPD status, lockdown, pinned latency probe ---------------------
def test_spd_status_says_why(tmp_path):
    inv = tmp_path / "inv"; inv.mkdir()
    i2c = tmp_path / "i2c"; ad = tmp_path / "ad"; (ad / "i2c-0").mkdir(parents=True)
    (ad / "i2c-0" / "name").write_text("SMBus I801 adapter at efa0\n")
    (inv / "spd.txt").write_text("No EEPROM found, the kernel probably does not support your hardware.\n")
    s = stick.spd_status(inv, [], i2c, ad)
    assert s["status"] == "unavailable" and "No EEPROM found" in s["reason"] and "SMBus I801 adapter" in s["reason"]
    assert "SPD devices bound: none" in s["reason"] and s["smbus_adapters"] == ["SMBus I801 adapter"]
    assert stick.spd_status(inv, ["spd-0-0050.hex"], i2c, ad)["status"] == "raw"
    (inv / "spd.txt").write_text("# decode-dimms\nDecoding EEPROM: /sys/bus/i2c/drivers/ee1004/0-0050\n")
    assert stick.spd_status(inv, [], i2c, ad)["status"] == "decoded"


def test_lockdown_mode(tmp_path):
    f = tmp_path / "lockdown"; f.write_text("none [integrity] confidentiality\n")
    assert stick.lockdown(f) == "integrity" and stick.lockdown(tmp_path / "missing") == ""




# --- behaviour instead of source text (bug hunt 2026-11-04) -----------------------------------------------------------
import re  # noqa: E402
import shutil as _shutil  # noqa: E402

CPU_SRC = HERE / "overlay" / "toolkits" / "cpu" / "src"
CHASE_SRC = HERE.parent / "bench" / "hw" / "toolkits" / "cpu" / "src" / "chase.c"


def _cc(tmp_path, src, out, *libs):
    """Compile a probe the way the toolkit does; skip where the compiler or a library is missing."""
    if not _shutil.which("gcc"):
        pytest.skip("no gcc")
    r = subprocess.run(["gcc", "-O2", "-o", str(tmp_path / out), str(src), *libs], capture_output=True, text=True)
    if r.returncode != 0:
        if any(l in r.stderr for l in ("dnnl", "cblas", "openblas")):
            pytest.skip("oneDNN / OpenBLAS headers or libraries not installed")
        raise AssertionError(r.stderr)
    return tmp_path / out


def _lines(out: str) -> list[dict]:
    return [json.loads(l) for l in out.splitlines() if l.startswith("{")]


def test_c2c_measures_pairs_and_barriers(tmp_path):
    exe = _cc(tmp_path, CPU_SRC / "c2c.c", "c2c", "-lpthread")
    pairs = _lines(subprocess.run([str(exe), "pairs", "--max-cores", "4", "--ms", "1"], capture_output=True,
                                  text=True, timeout=120).stdout)
    if len(os.sched_getaffinity(0)) < 2:                 # a one-CPU sandbox: the probe says so instead of hanging
        assert pairs and "error" in pairs[0]
        return
    assert pairs and all(p["relation"] in ("smt", "same-l3", "other-l3", "other-package") and p["ns"] > 0 for p in pairs)
    assert all(p["a"] != p["b"] for p in pairs)
    bars = _lines(subprocess.run([str(exe), "barrier", "--iters", "500"], capture_output=True, text=True,
                                 timeout=120).stdout)
    assert {b["kind"] for b in bars if "kind" in b} <= {"spin", "futex"}
    assert all(b["threads"] >= 2 and b["ns"] > 0 for b in bars if "kind" in b)


def test_chase_runs_pinned_and_reports_the_core(tmp_path):
    exe = _cc(tmp_path, CHASE_SRC, "chase")
    allowed = os.sched_getaffinity(0)
    lines = _lines(subprocess.run([str(exe), "1", "2"], capture_output=True, text=True, timeout=120).stdout)
    assert lines and all(l["cpu"] in allowed and l["core_type"] in "PE-" and l["ns_median"] > 0 for l in lines)
    want = min(allowed)
    lines = _lines(subprocess.run([str(exe), "1", "2", str(want)], capture_output=True, text=True, timeout=120).stdout)
    assert [l["cpu"] for l in lines] == [want]                             # an explicit core: only that one


def test_cgemm_shapes_formats_and_unsupported(tmp_path):
    exe = _cc(tmp_path, CPU_SRC / "cgemm.c", "cgemm", "-fopenmp", "-lopenblas", "-ldnnl")
    run = lambda *a: _lines(subprocess.run([str(exe), *a, "--threads", "1", "--reps", "2"], capture_output=True,
                                           text=True, timeout=300).stdout)
    f = run("--prec", "fp32", "--shape", "8x256x512")[0]
    assert (f["m"], f["nn"], f["k"]) == (8, 256, 512) and f["ops"] == 2 * 8 * 256 * 512 and f["ops_per_s"] > 0
    for prec in ("int8", "bf16", "fp16"):
        r = run("--prec", prec, "--shape", "16x512x512")[0]
        assert r.get("unsupported") or (r["ops_per_s"] > 0 and r["lib"] == "onednn"), r
        if prec != "int8" and not r.get("unsupported"):
            assert r["impl"] and not r["impl"].startswith("ref")             # never timing oneDNN's reference code
    r = subprocess.run([str(exe), "--prec", "fp7", "--n", "64"], capture_output=True, text=True)
    assert r.returncode == 1 and "unknown prec" in r.stdout


def test_fio_jobs_parse_and_refuse_writes(tmp_path):
    if not _shutil.which("fio"):
        pytest.skip("no fio")
    f = tmp_path / "disk.img"
    with open(f, "wb") as h:
        h.write(os.urandom(1 << 20)); h.truncate(64 << 20)
    swap = lambda cmd: [c if not c.startswith("--filename=") else f"--filename={f}" for c in cmd]
    r = subprocess.run(swap(stick.fio_cmd("x", "randread", "4k", 4, 1, direct=False)), capture_output=True, text=True)
    res = stick.parse_fio(r.stdout)
    assert r.returncode == 0 and res["bytes_per_s"] > 0 and res["iops"] > 0 and res["lat_p99_us"] >= res["lat_p50_us"] > 0
    w = subprocess.run(["fio", "--readonly", "--name=w", f"--filename={f}", "--rw=write", "--bs=4k", "--size=4k"],
                       capture_output=True, text=True)
    assert w.returncode != 0 and f.stat().st_size == 64 << 20                  # fio refused, nothing written


def _fake_vendor(tmp_path, monkeypatch, info: str):
    """run_vendor against a pretend docker: oneDNN fails the first llama.cpp step until it is switched off."""
    data = tmp_path / "data"; (data / "seed" / "models").mkdir(parents=True)
    (data / "seed" / "models" / "tiny.gguf").write_text("x"); (data / "seed" / "models" / "big.gguf").write_text("x")
    monkeypatch.setattr(stick, "gpu_vendors", lambda: {"intel"})
    monkeypatch.setattr(stick, "vendor_images", lambda d: {"oneapi": "img"})
    monkeypatch.setattr(stick, "load_image", lambda *a, **k: dict(rc=0))
    monkeypatch.setattr(stick, "vendor_plan", lambda kind: [("gemm-fp16-1x4096x4096", ["/rigsim/probe_sycl", "gemm"])])
    monkeypatch.setattr(stick, "BACKEND_OPS", ["MUL_MAT"])
    seen = []
    def sh(cmd, timeout=120):
        seen.append(cmd)
        if cmd[-1] == "info":
            return 0, info
        if "ls /rigsim" in cmd[-1]:
            return 0, "probe_sycl\nllama-bench\ntest-backend-ops\n"
        if cmd[:2] == ["docker", "exec"] and "/rigsim/probe_sycl" not in cmd:
            return (0, "ok") if "GGML_SYCL_ENABLE_DNN=0" in cmd else (134, ONEDNN_OUT)
        return 0, ""
    monkeypatch.setattr(stick, "sh", sh)
    return data, seen


def test_vendor_runtime_without_a_gpu_is_a_skip(tmp_path, monkeypatch):
    data, _ = _fake_vendor(tmp_path, monkeypatch, info="")
    r = stick.run_vendor("oneapi", data, tmp_path / "run", tmp_path / "log")
    assert "sees no GPU" in r["skipped"] and "rc" not in r


def test_vendor_onednn_retry_then_off_for_the_device_and_the_memory_limit(tmp_path, monkeypatch):
    data, seen = _fake_vendor(tmp_path, monkeypatch, info='{"mode":"info","index":0,"name":"UHD 770"}\n')
    _mem(tmp_path, monkeypatch, avail_gib=6.0)
    with stick.StepMemory("oneapi", None, tmp_path / "cgroup") as m:
        r = stick.run_vendor("oneapi", data, tmp_path / "run", tmp_path / "log")
    runs = r["runs"]
    llama = [k for k in runs if k.startswith("dev0-llama-")]
    first = [k for k in llama if "retry" in runs[k]]
    assert len(first) == 1 and runs[first[0]]["rc"] == 0                                 # the first llama step: retried
    assert all(runs[k]["onednn_off"]["after"] == first[0] for k in llama if k != first[0])   # then off for the device
    assert runs["dev0-ops-mul_mat"]["onednn_off"]["after"] == first[0]
    probe = [c for c in seen if "/rigsim/probe_sycl" in c and c[:2] == ["docker", "exec"]]
    assert probe and not any("GGML_SYCL_ENABLE_DNN=0" in c for c in probe)               # the probe has no oneDNN
    started = [c for c in seen if c[:3] == ["docker", "run", "-d"]]
    assert started and "--memory" in started[0] and str(m.limit) in started[0]
    assert r["rc"] == 0


def test_every_question_command_exists_in_its_probe():
    """Cross-file consistency: a question naming a probe mode or option the probe does not know fails only at run
    time on the stick — catch it here."""
    probe = (HERE.parent / "bench" / "hw" / "toolkits" / "cuda" / "src" / "probe.cu").read_text()
    modes = set(re.findall(r'mode == "(\w+)"', probe))
    cpu_bins = set(re.findall(r"-o /usr/local/bin/(\w+)", (HERE / "overlay" / "toolkits" / "cpu" / "setup.sh").read_text())) | {"cpuprobe"}
    for f in sorted((HERE / "questions").glob("*.toml")):
        q = hwq.load_plan(str(f))
        cmd = q["args"]["cmd"]
        if q["toolkit"] == "cuda":
            assert cmd[0] == "probe" and cmd[1] in modes, f.name
            for opt in (c for c in cmd[2:] if c.startswith("--")):
                assert f'"{opt}"' in probe, (f.name, opt)
        elif q["toolkit"] == "cpu":
            assert cmd[0] in cpu_bins, (f.name, cmd[0])
