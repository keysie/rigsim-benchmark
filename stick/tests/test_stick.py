"""The stick's orchestrator logic (no root, no hardware): run directories never reused, fingerprints salted and
stable, inventory sanitised, the profile valid for the bench runner, GPU questions skipped with a reason, the index
accumulates, power-off rules. Run: python3 -m pytest -q stick/tests"""
import datetime as dt
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
        sorted([Path("toolkits/cpu/setup.sh"), Path("toolkits/cpu/src/cgemm.c"), Path("toolkits/cuda/setup.sh")])
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
    assert "start=${start}, size=${sectors}" in upd and "mkfs" not in upd  # same sectors, no new filesystem
    assert "does not fit in front of the data partition" in upd and "e2fsck -n" in upd
    e = f.split("e) ", 1)[1].split(";;", 1)[0]
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
    assert "md5sum > md5sum.txt" in b and b.index("md5sum.txt") < b.index('step "ISO (BIOS + UEFI)"')
    assert "! -path ./boot/grub/bios.img" in b                       # xorriso patches it: never matches
    assert b.index('step "ISO checksums') > b.index('step "ISO (BIOS + UEFI)"') and "md5sum --quiet -c md5sum.txt" in b
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
    monkeypatch.setattr(stick.subprocess, "run", fake_run)
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
    monkeypatch.setattr(stick.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or R())
    r = stick.run_blender(tmp_path, tmp_path / "run", tmp_path / "log")
    hip = next(c for c in calls if c[-1] == "HIP")
    assert hip[:2] == ["docker", "run"] and "/dev/kfd" in hip and f"{b}:{b}:ro" in hip and "rigsim/rocm:k1" in hip
    assert hip[hip.index("rigsim/rocm:k1") + 1:][:2] == ["/rigsim/run.sh", str(exe)]
    assert r["rc"] == 0 and "hip-monster" in r["runs"] and r["hip_runtime"] == "ROCm image rigsim/rocm:k1"
    monkeypatch.setattr(stick, "vendor_images", lambda data: {})                     # no ROCm image: unavailable
    assert stick.run_blender(tmp_path, tmp_path / "run2", tmp_path / "log")["types_unavailable"] == ["HIP"]
    assert "libxkbcommon0" in (HERE / "src" / "rocm" / "build-in-image.sh").read_text()


def test_image_load_reports_progress(tmp_path, monkeypatch, capsys):
    img = tmp_path / "seed" / "images" / "nvidia_cuda.tar"; img.parent.mkdir(parents=True); img.write_bytes(b"x" * 1024)
    (tmp_path / "docker").mkdir()
    sizes = iter([0, 3 * 2**30, 6 * 2**30])
    monkeypatch.setattr(stick, "_du", lambda p: next(sizes))
    monkeypatch.setattr(stick, "_loaded", lambda img, data: False)

    class P:
        n = 0
        returncode = None
        def __init__(self, *a, **k): pass
        def wait(self, timeout=None):
            P.n += 1
            if P.n < 3:
                raise stick.subprocess.TimeoutExpired("docker", timeout)
            self.returncode = 0
            return 0
    monkeypatch.setattr(stick.subprocess, "Popen", P)
    r = stick.load_image(img, tmp_path, None, every=0.01)
    out = capsys.readouterr().out
    assert r["rc"] == 0 and "3.0 GiB written" in out and "6.0 GiB written" in out and "MB/s" in out
    assert (tmp_path / "docker" / ".loaded-nvidia_cuda.tar").read_text() == "1024"


def test_flash_preloads_images_with_the_sticks_own_docker():
    f = (HERE / "flash.sh").read_text()
    seed = f.split("seed_stick() {", 1)[1].split("\n}\n", 1)[0]
    assert seed.rstrip().endswith('preload_images "$1"')                 # every mode that seeds also preloads
    pre = f.split("preload_images() {", 1)[1].split("\n}\n", 1)[0]
    assert 'casper/filesystem.squashfs' in pre and 'chroot "$r"' in pre and "NO_PRELOAD" in pre
    assert pre.index('pm "$1" "$r/mnt/rigsim"') < pre.index('pm --bind "$r/mnt/rigsim/containerd" "$r/var/lib/containerd"')
    assert 'umount -l "$m"' in pre and "still has mounts; left in place" in pre and "return 0" in pre   # never fails flashing
    assert "dockerd --data-root /mnt/rigsim/docker --containerd=/run/containerd/containerd.sock" in f
    assert "python3 /opt/rigsim/stick.py --preload /mnt/rigsim" in f
    src = (HERE / "payload" / "stick.py").read_text()
    assert 'sys.argv[1] == "--preload"' in src and "def preload(data: Path) -> int:" in src


def test_rocm_build_drops_targets_llama_cpp_cannot_compile():
    s = (HERE / "src" / "rocm" / "build-in-image.sh").read_text()
    assert "errors? generated when compiling for gfx[0-9a-f]+" in s and "/rigsim/TARGETS_DROPPED" in s
    assert "for attempt in 1 2 3 4" in s and "not a per-target error" in s
    assert "TARGETS_DROPPED" in (HERE / "tools" / "manifest.py").read_text()


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
    assert stick.FORMAT["version"] == "1.1"
