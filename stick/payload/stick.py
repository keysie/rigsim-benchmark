#!/usr/bin/env python3
"""rigsim hardware stick — the unattended run on every boot (systemd unit rigsim-stick, tty1).

notice (30 s, any key aborts) -> mount the data partition RIGSIM -> new run directory (exclusive, never reused) ->
inventory (serials, UUIDs, MACs, addresses removed) -> Docker on the data partition, seed images loaded once ->
the stick profile through the bench runner (bench/hw/runner/hwq.py: baseline, items, restore) -> Intel MLC ->
clpeak on every OpenCL GPU -> llama-bench (CUDA; Vulkan on every vendor; CPU) -> Blender benchmark -> run.json + a line
in index.tsv -> unmount -> power off (unless NO_POWEROFF on the partition or rigsim.nopoweroff on the cmdline).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import secrets
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

OPT = Path("/opt/rigsim")
MNT = Path("/mnt/rigsim")
RUN = Path("/run/rigsim")
LABEL = "RIGSIM"
NOTICE_S = 30
QUESTION_TIMEOUT_S = 3 * 3600
OPENCL_TIMEOUT_S = 20 * 60
CLPEAK_TESTS = ["--global-bandwidth", "--compute-sp", "--compute-hp", "--compute-integer", "--transfer-bandwidth",
                "--kernel-latency"]
SOFTWARE_DEVICES = re.compile(r"(?i)llvmpipe|softpipe|pthread|\bcpu\b|pocl")
PROFILE = ["cpu-mem-latency", "cpu-mem-ladder", "cpu-gemm", "pcie-copy", "gpu-p2p", "gpu-vram", "gpu-gemm", "gpu-launch"]
DRM = Path("/sys/class/drm")
FBDEV = Path("/sys/class/graphics")
SAMPLE_S = 0.2
MLC_RUNS = [("idle_latency", ["--idle_latency"]), ("max_bandwidth", ["--max_bandwidth"]),
            ("loaded_latency", ["--loaded_latency"]), ("latency_matrix", ["--latency_matrix"]),
            ("bandwidth_matrix", ["--bandwidth_matrix"])]
MLC_TIMEOUT_S = 15 * 60
LLAMA_ARGS = ["-p", "512", "-n", "128", "-r", "3", "-o", "jsonl"]
LLAMA_TIMEOUT_S = 30 * 60
# llama.cpp's own per-kernel timing (test-backend-ops perf): the engine's kernel factors at the bottom, llama-bench
# validates on top. One call per operation, on one backend device.
BACKEND_OPS = ["MUL_MAT", "MUL_MAT_ID", "FLASH_ATTN_EXT", "RMS_NORM", "ADD", "MUL", "ROPE", "GLU", "SET_ROWS",
               "CPY", "SOFT_MAX", "GET_ROWS"]
# The perf suite runs every case for at least a second: the full set took 1 h 50 min on an iGPU + CPU (first trial
# run; flash attention alone timed out after 30 min). Keep the cases models actually use: common weight types,
# head sizes 64/128/256, an f16 or q8_0 cache up to 32K.
BACKEND_OPS_FILTER = {
    "MUL_MAT": r"type_a=(f16|bf16|f32|q4_0|q8_0|q4_K|q5_K|q6_K|iq4_xs|mxfp4),",
    "MUL_MAT_ID": r"type_a=(f16|q4_0|q8_0|q4_K|q6_K|mxfp4),",
    "FLASH_ATTN_EXT": r"hsk=(64|128|256),.*kv=(4096|8192|16384|32768),.*type_K=(f16|q8_0),",
}
BACKEND_OPS_TIMEOUT_S = 30 * 60
BLENDER_SCENES = ["monster", "junkshop", "classroom"]
BLENDER_TIMEOUT_S = 90 * 60
# Vulkan probe (every GPU vendor) and the vendor-native steps (AMD ROCm, Intel oneAPI)
VKPROBE = OPT / "bin" / "vkprobe"
VK_GEMM_TYPES = ["f16", "f16f32", "i8"]
VK_GEMM_SHAPES = ["512x4096x4096", "2048x2048x2048", "4096x4096x4096"]
VENDOR_GEMM_PRECS = ["fp32", "tf32", "fp16_fp32acc", "fp16", "bf16", "int8"]
VENDOR_GEMM_SHAPES = ["1x4096x4096", "8x2048x2048", "8x4096x4096", "32x4096x4096", "128x4096x4096", "512x2048x2048",
                      "2048x2048x2048", "512x8192x8192", "2048x8192x8192", "4096x4096x4096", "8192x8192x8192",
                      "16384x16384x4096"]
PROBE_TIMEOUT_S = 10 * 60
AMD_DEVICES = ["--device", "/dev/kfd", "--device", "/dev/dri", "--security-opt", "seccomp=unconfined"]
INTEL_DEVICES = ["--device", "/dev/dri"]
# HIP has no code for a gfx target it was not built for; the usual stand-in is the family's base target
GFX_FAMILY = [("gfx103", "10.3.0"), ("gfx110", "11.0.0"), ("gfx115", "11.5.0"), ("gfx120", "12.0.0")]
STICK_READ_MIB = 1024
INDEX_HEADER = "run\tstart_utc\tfingerprint\tcpu\tboard\tgpus\tram\tstatus\tstick\n"
DMI = Path("/sys/class/dmi/id")
SECURE_BOOT_VAR = Path("/sys/firmware/efi/efivars/SecureBoot-8be4df61-93ca-11d2-aa0d-00e098032b8c")


import threading  # noqa: E402


class Stopped(Exception):
    pass


class Status:
    """One live line at the bottom of the console, redrawn four times a second: a spinner (so a long step visibly
    lives), the step ('5/14 vkprobe'), what runs inside it ('dev0-gemm-f16 7/42'), its time and the run's. say()
    prints above it. Only on a terminal; in logs and pipes nothing changes."""
    SPIN = "|/-\\"

    def __init__(self):
        self.lock = threading.Lock()
        self.on = sys.stdout.isatty()
        self.step = self.detail = ""
        self.i = self.n = 0
        self.t_run = self.t_step = time.time()
        self.k = 0
        self._stop = threading.Event()
        self._thread = None

    def start(self, total_steps: int = 0) -> None:
        self.n, self.t_run = total_steps, time.time()
        if self.on and not self._thread:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def begin(self, step: str) -> None:
        with self.lock:
            self.i += 1
            self.step, self.detail, self.t_step = step, "", time.time()

    def set(self, detail: str) -> None:
        with self.lock:
            self.detail = detail

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.clear()

    def clear(self) -> None:
        if self.on:
            sys.stdout.write("\r\033[K"); sys.stdout.flush()

    @staticmethod
    def _t(sec: float) -> str:
        sec = int(sec)
        return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"

    def line(self) -> str:
        now = time.time()
        step = f"{self.i}/{self.n} {self.step}" if self.n else self.step
        parts = [self.SPIN[self.k % 4], f"[{step}]" if self.step else "", self.detail,
                 self._t(now - self.t_step) if self.step else "", f"(run {self._t(now - self.t_run)})"]
        return " ".join(x for x in parts if x)

    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            with self.lock:
                self.k += 1
                width = max(20, shutil.get_terminal_size((100, 24)).columns - 1)
                sys.stdout.write("\r\033[K" + self.line()[:width]); sys.stdout.flush()


STATUS = Status()


def say(msg: str, log: Path | None = None) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    with STATUS.lock:
        STATUS.clear()
        print(line, flush=True)
    if log:
        with open(log, "a") as f:
            f.write(line + "\n")


def sh(cmd: list[str], timeout: float = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.stderr.strip() else "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"


# --- identity, never an identifier itself ---------------------------------------------------------------------------
def salt(data: Path) -> bytes:
    """Created once per stick; makes the fingerprint useless outside this stick's data."""
    f = data / ".salt"
    if not f.exists():
        f.write_bytes(secrets.token_bytes(32))
    return f.read_bytes()


def fingerprint(data: Path, dmi: Path = DMI) -> tuple[str, bool]:
    """Salted hash of the board's identifiers: (12 hex chars, stable). Not stable if the firmware gives none."""
    parts = []
    for k in ("product_uuid", "board_serial", "product_serial", "board_vendor", "board_name", "product_name"):
        try:
            v = (dmi / k).read_text().strip()
        except OSError:
            v = ""
        if v and v.lower() not in ("to be filled by o.e.m.", "default string", "none", "0", "not specified"):
            parts.append(v)
    stable = any(p for p in parts[:3])
    raw = "|".join(parts) if stable else secrets.token_hex(8)
    return hashlib.sha256(salt(data) + raw.encode()).hexdigest()[:12], stable


def new_run_dir(runs: Path, fp: str, now: dt.datetime | None = None) -> Path:
    """Exclusive creation: an existing directory is never reused, whatever the clock says."""
    runs.mkdir(parents=True, exist_ok=True)
    now = now or dt.datetime.now(dt.timezone.utc)
    for _ in range(100):
        d = runs / f"{now:%Y%m%dT%H%M%SZ}-{fp}-{secrets.token_hex(2)}"
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError("could not create a unique run directory")


# --- inventory ---------------------------------------------------------------------------------------------------------
SENSITIVE = re.compile(r"(?i)(serial|uuid|asset tag|\bsn\b|chassis id|system id)")
# private and carrier-grade-NAT IPv4 (local resolver, LAN, tailnet) — never in a published file
PRIVATE_IPV4 = re.compile(r"\b(?:10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])|192\.168|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7]))"
                          r"\.\d{1,3}\.\d{1,3}\b")
MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}[:-]){5}[0-9a-fA-F]{2}\b")
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IPV6 = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b")


def scrub(text: str) -> str:
    """For every tool output: MAC and local IP addresses masked (a tool may print its resolver or a peer)."""
    return PRIVATE_IPV4.sub("x.x.x.x", MAC.sub("<mac>", text))


def sanitize(text: str) -> str:
    """Drops lines naming serials, UUIDs or asset tags; masks MAC and IP addresses."""
    out = [l for l in text.splitlines() if not SENSITIVE.search(l)]
    t = "\n".join(out)
    t = MAC.sub("<mac>", t)
    t = IPV6.sub("<ip6>", t)
    return IPV4.sub("x.x.x.x", t) + "\n"


INVENTORY = [
    ("lscpu.txt", ["lscpu"]), ("lscpu-e.txt", ["lscpu", "-e"]), ("cpuinfo.txt", ["cat", "/proc/cpuinfo"]),
    ("numa.txt", ["numactl", "-H"]), ("topology.txt", ["lstopo-no-graphics", "--of", "console"]),
    # one -t per type: dmidecode does not take a comma list (first trial run: rc=2, no DIMMs)
    ("dmidecode.txt", ["dmidecode", *[x for t in ("bios", "system", "baseboard", "processor", "memory", "cache", "slot")
                                      for x in ("-t", t)]]),
    ("lshw.json", ["lshw", "-sanitize", "-json"]),
    ("lspci-tree.txt", ["lspci", "-tvnn"]), ("lspci.txt", ["lspci", "-vvvnn"]),
    ("spd.txt", ["decode-dimms"]),
    ("nvidia-smi.txt", ["nvidia-smi", "-q"]), ("nvidia-topo.txt", ["nvidia-smi", "topo", "-m"]),
    ("clinfo.txt", ["clinfo"]), ("vulkaninfo.txt", ["vulkaninfo", "--summary"]),
    ("lsblk.txt", ["lsblk", "-o", "NAME,MODEL,SIZE,ROTA,TRAN,TYPE"]),
    ("sensors.txt", ["sensors"]), ("meminfo.txt", ["cat", "/proc/meminfo"]),
    ("kernel.txt", ["uname", "-srvm"]), ("cmdline.txt", ["cat", "/proc/cmdline"]),
    # why a GPU is (not) usable through Vulkan: the loader's own account, the GPU driver modules and device nodes
    ("vulkan-loader.txt", ["sh", "-c", "VK_LOADER_DEBUG=error,warn,driver vulkaninfo --summary 2>&1 | head -400"]),
    ("gpu-driver.txt", ["sh", "-c", "lsmod | grep -iE '^(nvidia|amdgpu|i915|xe) '; ls -l /dev/nvidia* /dev/dri /dev/kfd 2>&1; "
                       "for f in /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0; do [ -e $f ] && ldd $f | grep -i 'not found'; done; true"]),
]


def secure_boot() -> bool | None:
    try:
        return SECURE_BOOT_VAR.read_bytes()[-1] == 1
    except OSError:
        return None


def cpufreq() -> dict:
    d = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    out = {}
    for k in ("cpuinfo_min_freq", "cpuinfo_max_freq", "scaling_driver", "scaling_governor", "base_frequency"):
        try:
            out[k] = (d / k).read_text().strip()
        except OSError:
            pass
    return out


def parse_dimms(dmidecode: str) -> list[dict]:
    """Installed modules from dmidecode's 'Memory Device' sections (size, type, speeds, part number)."""
    dimms = []
    for block in dmidecode.split("\n\n"):
        if "Memory Device" not in block:
            continue
        f = dict(re.findall(r"^\s*([^:\n]+):\s*(.*)$", block, re.M))
        if not f.get("Size") or "No Module" in f.get("Size", ""):
            continue
        dimms.append({k: f.get(k) for k in ("Locator", "Size", "Type", "Speed", "Configured Memory Speed", "Rank",
                                             "Manufacturer", "Part Number")})
    return dimms


def collect_inventory(dst: Path, log: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    for mod in ("i2c-i801", "i2c-piix4", "ee1004", "spd5118", "at24", "i2c-dev", "msr",
                "nvidia_uvm", "nvidia_modeset", "nvidia_drm"):
        sh(["modprobe", mod], timeout=20)
    if shutil.which("nvidia-modprobe"):                  # device nodes the Vulkan driver opens (uvm, modeset)
        sh(["nvidia-modprobe", "-u", "-m", "-c", "0"], timeout=20)
    for k, (name, cmd) in enumerate(INVENTORY, 1):
        STATUS.set(f"{name} {k}/{len(INVENTORY)}")
        rc, out = sh(cmd, timeout=180)
        (dst / name).write_text(sanitize(out) if not name.endswith(".json") else MAC.sub("<mac>", out))
        if rc != 0:
            say(f"inventory: {name} rc={rc}", log)
    spd_raw(dst)
    (dst / "pcie.json").write_text(json.dumps(pcie_topology(names=lspci_names()), indent=1))
    cpu = next(iter(re.findall(r"^model name\s*:\s*(.*)$", (dst / "cpuinfo.txt").read_text(), re.M)), "unknown")
    board = " ".join(_dmi(k) for k in ("board_vendor", "board_name")).strip() or "unknown"
    rc, gpus = sh(["nvidia-smi", "--query-gpu=name,memory.total,pci.bus_id", "--format=csv,noheader"], timeout=60)
    gpu_list = [g.strip() for g in gpus.splitlines() if g.strip() and "---" not in g and "," in g] if rc == 0 else []
    # every other display device too (integrated graphics, AMD, Intel); NVIDIA cards are already listed above
    _, vga = sh(["sh", "-c", "lspci -nn | grep -Ei 'vga|3d controller|display'"], timeout=30)
    gpu_list += [l.split(": ", 1)[-1] for l in vga.splitlines()
                 if l.strip() and "--- stderr" not in l and not (gpu_list and "[10de:" in l)]
    mem_kb = int(next(iter(re.findall(r"MemTotal:\s*(\d+)", (dst / "meminfo.txt").read_text())), "0"))
    dimms = parse_dimms((dst / "dmidecode.txt").read_text())
    s = dict(cpu=cpu, board=board, bios=_dmi("bios_version"), gpus=gpu_list, ram_gib=round(mem_kb / 2**20, 1),
             dimms=dimms, cpufreq=cpufreq(), secure_boot=secure_boot(), kernel=sh(["uname", "-r"])[1].strip())
    (dst / "summary.json").write_text(json.dumps(s, indent=1))
    return s


# --- PCIe topology ----------------------------------------------------------------------------------------------
# Files written by the inventory besides the INVENTORY commands (the validator allows exactly these).
INVENTORY_EXTRA = ["summary.json", "pcie.json"]
PCI_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
PORT_TYPES = {0: "endpoint", 1: "legacy endpoint", 4: "root port", 5: "switch upstream port",
              6: "switch downstream port", 7: "PCIe to PCI bridge", 8: "PCI to PCIe bridge",
              9: "root complex integrated endpoint", 10: "root complex event collector"}


def pcie_port_type(config: bytes) -> str:
    """The PCI Express capability's device/port type (PCIe base spec, capability ID 0x10, bits 7:4)."""
    if len(config) < 0x40 or not (config[6] & 0x10):            # status: capabilities list present
        return ""
    ptr, seen = config[0x34] & 0xFC, set()
    while ptr and ptr not in seen and ptr + 4 <= len(config):
        seen.add(ptr)
        if config[ptr] == 0x10:
            return PORT_TYPES.get((int.from_bytes(config[ptr + 2:ptr + 4], "little") >> 4) & 0xF, "other")
        ptr = config[ptr + 1] & 0xFC
    return ""


def pcie_topology(devices: Path = Path("/sys/bus/pci/devices"), names: dict | None = None) -> dict:
    """Every PCI device with its parent, port type, and current and maximum link, read from sysfs: the path of each
    card from the root complex — CPU or chipset root port, switches, risers' bridges — as facts, not a judgement."""
    out = {}
    for link in sorted(devices.glob("*")):
        try:
            real = link.resolve()
        except OSError:
            continue
        chain = [part for part in real.parts if PCI_BDF.match(part)]
        root = next((part for part in real.parts if part.startswith("pci") and ":" in part), "")
        r = lambda f: _read(real / f)
        try:
            cfg = (real / "config").read_bytes()[:256]
        except OSError:
            cfg = b""
        n = (names or {}).get(link.name, {})
        drv = (real / "driver").resolve().name if (real / "driver").exists() else ""
        out[link.name] = dict(parent=chain[-2] if len(chain) >= 2 else "", host_bridge=root,
                              vendor=r("vendor"), device=r("device"), class_=r("class"),
                              subsystem=f"{r('subsystem_vendor')}:{r('subsystem_device')}".strip(":"),
                              name=n.get("name", ""), vendor_name=n.get("vendor", ""), kind=n.get("class", ""),
                              port_type=pcie_port_type(cfg), driver=drv, numa_node=r("numa_node"),
                              link=dict(speed=r("current_link_speed"), width=r("current_link_width"),
                                        max_speed=r("max_link_speed"), max_width=r("max_link_width")))
    paths = {}
    for bdf in out:
        path, cur, guard = [], bdf, 0
        while cur and guard < 16:
            path.insert(0, cur); cur = out.get(cur, {}).get("parent", ""); guard += 1
        paths[bdf] = path
    for bdf, d in out.items():
        d["path"] = paths[bdf]
    return dict(devices=out, gpus=[b for b, d in out.items() if d["class_"].startswith("0x03")])


def lspci_names() -> dict:
    """bdf -> class, vendor and device names (lspci -mm), for pcie.json."""
    rc, out = sh(["lspci", "-D", "-mm"], timeout=60)
    names = {}
    for line in out.splitlines():
        parts = re.findall(r'"([^"]*)"', line)
        bdf = line.split(" ", 1)[0]
        if PCI_BDF.match(bdf) and len(parts) >= 3:
            names[bdf] = dict(**{"class": parts[0]}, vendor=parts[1], name=parts[2])
    return names


# SPD serial-number bytes, zeroed before a dump is stored (JEDEC: DDR4 bytes 325-328, DDR5 bytes 517-520)
SPD_SERIAL = {"ee1004": range(325, 329), "spd5118": range(517, 521)}


def spd_raw(dst: Path, i2c: Path = Path("/sys/bus/i2c/devices")) -> list[str]:
    """The modules' SPD bytes as hex (decode-dimms cannot read DDR5 on every release): inventory/spd-<bus-addr>.hex,
    serial-number bytes zeroed."""
    out = []
    for dev in sorted(i2c.glob("*-00[5-9a-f]*")):
        drv = (dev / "driver").resolve().name if (dev / "driver").exists() else ""
        if drv not in SPD_SERIAL or not (dev / "eeprom").exists():
            continue
        try:
            b = bytearray((dev / "eeprom").read_bytes())
        except OSError:
            continue
        for i in SPD_SERIAL[drv]:
            if i < len(b):
                b[i] = 0
        name = f"spd-{dev.name}.hex"
        (dst / name).write_text(f"# {drv}, {len(b)} bytes, serial bytes zeroed\n" +
                                "\n".join(b[i:i + 16].hex(" ") for i in range(0, len(b), 16)) + "\n")
        out.append(name)
    return out


def _dmi(k: str) -> str:
    try:
        return (DMI / k).read_text().strip()
    except OSError:
        return ""


# --- run plan --------------------------------------------------------------------------------------------------------
def question_gpus(qfile: Path) -> str:
    m = re.search(r'(?m)^gpus\s*=\s*"(\w+)"', qfile.read_text())
    return m.group(1) if m else "none"


def plan(qdir: Path, nvidia: bool, nvidia_reason: str) -> list[tuple[str, Path, str | None]]:
    """(question, file, skip reason or None), in profile order."""
    out = []
    for q in PROFILE:
        f = qdir / f"{q}.toml"
        if not f.exists():
            out.append((q, f, "question file missing"))
        elif question_gpus(f) != "none" and not nvidia:
            out.append((q, f, nvidia_reason))
        else:
            out.append((q, f, None))
    return out


def nvidia_state() -> tuple[bool, str]:
    rc, out = sh(["nvidia-smi", "-L"], timeout=60)
    if rc == 0 and "GPU" in out:
        return True, ""
    flavour = "legacy" if "legacy" in (OPT / "VERSION").read_text() else "open" if (OPT / "VERSION").exists() else "?"
    if flavour == "open" and _nvidia_pre_turing():
        return False, "NVIDIA GPU older than Turing: needs the legacy image (open modules support Turing and newer)"
    rc2, lspci = sh(["sh", "-c", "lspci -nn | grep -i nvidia"], timeout=30)
    if not lspci.strip():
        return False, "no NVIDIA GPU"
    return False, "NVIDIA GPU present but the driver did not load (card unsupported by the driver?)"


def _nvidia_pre_turing() -> bool:
    """Maxwell / Pascal / Volta by name (the open modules refuse them)."""
    _, out = sh(["sh", "-c", "lspci -nn | grep -i nvidia | grep -Ei 'vga|3d'"], timeout=30)
    return bool(re.search(r"(?i)GM\d{3}|GP\d{3}|GV\d{3}|GTX (9\d\d|10\d\d)|Quadro [MP]\d|Titan (X|Xp|V)\b", out))


# --- vendor-neutral GPU measurement (OpenCL, clpeak) ---------------------------------------------------------------------
def opencl_devices(clinfo_list: str) -> list[tuple[int, int, str, str]]:
    """(platform, device, platform name, device name) from `clinfo --list`; software devices left out."""
    out, plat, pname = [], None, ""
    for line in clinfo_list.splitlines():
        m = re.match(r"\s*Platform #(\d+):\s*(.*)", line)
        if m:
            plat, pname = int(m.group(1)), m.group(2).strip(); continue
        m = re.match(r"\s*[`|+\\-]*\s*Device #(\d+):\s*(.*)", line)
        if m and plat is not None:
            name = m.group(2).strip()
            if not SOFTWARE_DEVICES.search(name) and not SOFTWARE_DEVICES.search(pname):
                out.append((plat, int(m.group(1)), pname, name))
    return out


def parse_clpeak(text: str) -> dict:
    """{section: {key: value}} from clpeak's text output (units are in the section names)."""
    res, sec = {}, "info"
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith(("Platform:", "Device:")):
            continue
        m = re.match(r"(.+?)\s*:\s*([-+]?\d+(?:\.\d+)?)\s*(\w+)?\s*$", t)
        if m:
            key = m.group(1).strip() + (f" ({m.group(3)})" if m.group(3) else "")
            res.setdefault(sec, {})[key] = float(m.group(2))
        elif ":" in t:
            k, v = t.split(":", 1)
            res.setdefault("info", {})[k.strip()] = v.strip()
        else:
            sec = t
    return res


def clpeak_tests(help_text: str | None = None) -> list[str]:
    """The wanted tests this clpeak knows (Ubuntu's has no --compute-hp: one unknown option ends it before any test)."""
    if help_text is None:
        help_text = subprocess.run(["clpeak", "--help"], capture_output=True, text=True).stdout
    return [t for t in CLPEAK_TESTS if t in help_text] or ["--all-tests"]


def run_opencl(rundir: Path, log: Path) -> dict:
    d = rundir / "opencl"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, RUSTICL_ENABLE="radeonsi")       # AMD through Mesa rusticl; Intel through compute-runtime
    r = subprocess.run(["clinfo", "--list"], capture_output=True, text=True, env=env, timeout=120)
    (d / "clinfo-list.txt").write_text(scrub(r.stdout + r.stderr))
    devices = opencl_devices(r.stdout)
    if not devices:
        say("opencl: no GPU devices", log)
        return dict(skipped="no OpenCL GPU devices")
    results = {}
    for k, (plat, dev, pname, name) in enumerate(devices, 1):
        STATUS.set(f"clpeak {name[:30]} {k}/{len(devices)}")
        key = f"p{plat}d{dev}"
        say(f"opencl {key}: {name} ({pname})", log)
        t0 = time.time()
        try:
            r = subprocess.run(["clpeak", "-p", str(plat), "-d", str(dev), *clpeak_tests()], capture_output=True,
                               text=True, env=env, timeout=OPENCL_TIMEOUT_S)
            rc, out = r.returncode, r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            rc, out = -9, f"timeout after {OPENCL_TIMEOUT_S} s"
        (d / f"clpeak-{key}.txt").write_text(scrub(out))
        parsed = parse_clpeak(out)
        (d / f"clpeak-{key}.json").write_text(json.dumps(dict(platform=pname, device=name, rc=rc, results=parsed), indent=1))
        results[key] = dict(platform=pname, device=name, rc=rc, seconds=round(time.time() - t0))
        say(f"opencl {key}: rc={rc} ({time.time() - t0:.0f} s)", log)
    return dict(rc=0 if all(v["rc"] == 0 for v in results.values()) else 1, devices=results)


# --- displays: what scans out from which GPU; screens off while measuring ---------------------------------------------
def displays(drm: Path = DRM) -> list[dict]:
    """Connected outputs: connector, the GPU's PCI address, preferred mode (first listed), DPMS state."""
    out = []
    for c in sorted(drm.glob("card*-*")):
        try:
            status = (c / "status").read_text().strip()
        except OSError:
            continue
        if status != "connected":
            continue
        card = c.name.split("-", 1)[0]
        dev = drm / card / "device"
        modes = (c / "modes").read_text().split() if (c / "modes").exists() else []
        out.append(dict(connector=c.name.split("-", 1)[1], card=card,
                        pci=dev.resolve().name if dev.exists() else None, mode=modes[0] if modes else None,
                        enabled=_read(c / "enabled"), dpms=_read(c / "dpms")))
    return out


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def blank_screens(off: bool, fb: Path = FBDEV) -> list[str]:
    """Powers the console's screens down (scanout stops: on integrated graphics it reads system RAM) or back on.
    Returns the framebuffers it changed."""
    done = []
    for f in sorted(fb.glob("fb*")):
        try:
            (f / "blank").write_text("4" if off else "0")
            done.append(f.name)
        except OSError:
            pass
    return done


# --- AMD / Intel GPU clocks, busy share and power from sysfs (NVIDIA is sampled by the bench runner) ------------------
def _star(text: str | None) -> float | None:
    """The active line of pp_dpm_* ('1: 2400Mhz *')."""
    for line in (text or "").splitlines():
        if line.rstrip().endswith("*"):
            m = re.search(r"(\d+)\s*Mhz", line, re.I)
            return float(m.group(1)) if m else None
    return None


def drm_sample(drm: Path = DRM) -> list[dict]:
    rows = []
    for card in sorted(drm.glob("card[0-9]*")):
        if "-" in card.name:
            continue
        dev = card / "device"
        vendor = _read(dev / "vendor")
        if vendor not in ("0x1002", "0x8086"):
            continue
        r = dict(card=card.name, pci=dev.resolve().name, vendor="amd" if vendor == "0x1002" else "intel")
        if vendor == "0x1002":
            r["sclk_mhz"] = _star(_read(dev / "pp_dpm_sclk"))
            r["mclk_mhz"] = _star(_read(dev / "pp_dpm_mclk"))
            busy = _read(dev / "gpu_busy_percent")
            r["busy_pct"] = float(busy) if busy and busy.isdigit() else None
            for h in sorted((dev / "hwmon").glob("hwmon*")):
                for k in ("power1_average", "power1_input"):
                    v = _read(h / k)
                    if v and v.isdigit():
                        r["power_w"] = int(v) / 1e6
                        break
        else:
            for f in (card / "gt_act_freq_mhz", card / "gt" / "gt0" / "rps_act_freq_mhz",
                      dev / "tile0" / "gt0" / "freq0" / "act_freq"):
                v = _read(f)
                if v and v.isdigit():
                    r["gt_mhz"] = float(v)
                    break
        rows.append(r)
    return rows


class DrmSampler:
    """Background sampler for the whole measurement phase; rows carry wall-clock time for alignment."""

    def __init__(self, path: Path, period: float = SAMPLE_S, drm: Path = DRM):
        self.path, self.period, self.drm = path, period, drm
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        with open(self.path, "a") as f:
            while not self.stop_evt.is_set():
                t = time.time()
                for r in drm_sample(self.drm):
                    f.write(json.dumps(dict(t=round(t, 3), **r)) + "\n")
                f.flush()
                self.stop_evt.wait(self.period)

    def __enter__(self):
        if drm_sample(self.drm):
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_evt.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5)


class NvidiaSampler:
    """NVIDIA clocks, power, utilisation every 500 ms for the whole measurement phase (the bench runner samples only
    inside its own questions; llama.cpp, Blender and the probes run outside it)."""
    FIELDS = "timestamp,index,pci.bus_id,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu,pstate"

    def __init__(self, path: Path, enabled: bool):
        self.path, self.enabled, self.proc = path, enabled, None

    def __enter__(self):
        if self.enabled and shutil.which("nvidia-smi"):
            self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={self.FIELDS}", "--format=csv,nounits",
                                          "-lms", "500", "-f", str(self.path)], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --- Intel Memory Latency Checker (downloaded from Intel at build time, not redistributed by this repository) --------
def parse_mlc(text: str) -> dict:
    """Numbers MLC prints as 'label : value' or 'label value' rows (bandwidth in MB/s, latency in ns)."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9][\w :\-/()]*?)\s*:?\s+([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if m and not m.group(1).strip().isdigit():
            out[m.group(1).strip().rstrip(":").strip()] = float(m.group(2))
    return out


def run_mlc(data: Path, rundir: Path, log: Path) -> dict:
    mlc = data / "seed" / "tools" / "mlc" / "mlc"
    if not mlc.exists():
        return dict(skipped="Intel MLC not in the seed (build without MLC_URL or download failed)")
    d = rundir / "mlc"
    d.mkdir(parents=True, exist_ok=True)
    sh(["modprobe", "msr"], timeout=20)
    res = {}
    for k, (name, args) in enumerate(MLC_RUNS, 1):
        STATUS.set(f"{name} {k}/{len(MLC_RUNS)}")
        t0 = time.time()
        rc, out = sh([str(mlc), *args], timeout=MLC_TIMEOUT_S)
        (d / f"{name}.txt").write_text(scrub(out))
        res[name] = dict(rc=rc, seconds=round(time.time() - t0), values=parse_mlc(out))
        say(f"mlc {name}: rc={rc} ({time.time() - t0:.0f} s)", log)
    prefetch = "unable" in " ".join((d / f).read_text().lower() for f in ("idle_latency.txt",) if (d / f).exists())
    return dict(rc=0 if all(v["rc"] == 0 for v in res.values()) else 1, runs={k: dict(rc=v["rc"], seconds=v["seconds"])
                for k, v in res.items()}, prefetchers_left_on=prefetch or bool(secure_boot()))


# --- llama-bench: CUDA (in the CUDA image) and Vulkan (on the host, every vendor) -----------------------------------------
def vulkan_devices(summary: str) -> list[tuple[int, str]]:
    """(index, name) from `vulkaninfo --summary`; software devices (llvmpipe) left out."""
    out, idx = [], None
    for line in summary.splitlines():
        m = re.match(r"\s*GPU(\d+):", line)
        if m:
            idx = int(m.group(1)); continue
        m = re.match(r"\s*deviceName\s*=\s*(.*)", line)
        if m and idx is not None:
            name = m.group(1).strip()
            if not SOFTWARE_DEVICES.search(name):
                out.append((idx, name))
            idx = None
    return out


def llama_models(data: Path) -> list[Path]:
    return sorted((data / "seed" / "models").glob("*.gguf"), key=lambda p: p.stat().st_size)


def _run_env(cmd: list[str], timeout: float, env: dict) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.stderr.strip() else "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"


def backend_ops_plan(binary: str, backend: str) -> list[tuple[str, list[str]]]:
    """(key, command) per operation; perf mode with sql output — the format that carries the timings (csv has only
    the support columns); the caller retries without --output on builds that lack it."""
    out = []
    for op in BACKEND_OPS:
        flt = ["-p", BACKEND_OPS_FILTER[op]] if op in BACKEND_OPS_FILTER else []
        out.append((f"ops-{op.lower()}", [binary, "perf", "-o", op, *flt, "-b", backend, "--output", "sql"]))
    return out


def run_backend_ops(run, d: Path, prefix: str, binary: str, backend: str, log: Path) -> dict:
    """run(cmd, timeout) -> (rc, output). Older builds lack --output: then plain console output. A build that does
    not know the backend name: once more without -b (every backend it sees, CPU included)."""
    runs = {}
    ops = backend_ops_plan(binary, backend)
    for k, (key, cmd) in enumerate(ops, 1):
        STATUS.set(f"test-backend-ops {prefix} {key[4:]} {k}/{len(ops)}")
        t0 = time.time()
        rc, out = run(cmd, BACKEND_OPS_TIMEOUT_S)
        if rc != 0 and rc != 127:                             # 127: timeout or not runnable, no point retrying
            cmd = cmd[:-2]; rc, out = run(cmd, BACKEND_OPS_TIMEOUT_S)
        if rc != 0 and rc != 127:
            i = cmd.index("-b"); cmd = cmd[:i] + cmd[i + 2:]; rc, out = run(cmd, BACKEND_OPS_TIMEOUT_S)
        (d / f"{prefix}-{key}.txt").write_text(scrub(" ".join(cmd) + "\n" + out))
        runs[f"{prefix}-{key}"] = dict(rc=rc, seconds=round(time.time() - t0, 1))
    say(f"test-backend-ops {prefix}: {sum(1 for v in runs.values() if v['rc'] == 0)}/{len(runs)} ops", log)
    return runs


def run_llama(data: Path, rundir: Path, cfg_image: str, nvidia: bool, log: Path) -> dict:
    seed = data / "seed" / "llama"
    models = llama_models(data)
    if not models or not seed.is_dir():
        return dict(skipped="llama.cpp or models not in the seed")
    d = rundir / "llama"
    d.mkdir(parents=True, exist_ok=True)
    runs = {}

    def one(key: str, cmd: list[str], env: dict | None = None):
        STATUS.set(f"llama-bench {key}")
        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=LLAMA_TIMEOUT_S,
                               env=dict(os.environ, **(env or {})))
            rc, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = -9, "", f"timeout after {LLAMA_TIMEOUT_S} s"
        (d / f"{key}.jsonl").write_text(scrub(out))
        (d / f"{key}.log").write_text(scrub(err))
        runs[key] = dict(rc=rc, seconds=round(time.time() - t0))
        say(f"llama {key}: rc={rc} ({time.time() - t0:.0f} s)", log)

    if nvidia and (seed / "cuda" / "llama-bench").exists():
        rc, out = sh(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], timeout=60)
        for gi in [l.strip() for l in out.splitlines() if l.strip().isdigit()]:
            for m in models:
                one(f"cuda-gpu{gi}-{m.stem}", ["docker", "run", "--rm", "--gpus", "all", "--network", "none",
                    "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "-e", f"CUDA_VISIBLE_DEVICES={gi}",
                    "-v", f"{seed / 'cuda'}:/llama:ro", "-v", f"{m.parent}:/models:ro", cfg_image,
                    "/llama/llama-bench", "-m", f"/models/{m.name}", "-ngl", "99", *LLAMA_ARGS])
            if (seed / "cuda" / "test-backend-ops").exists():
                docker = ["docker", "run", "--rm", "--gpus", "all", "--network", "none", "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                          "-e", f"CUDA_VISIBLE_DEVICES={gi}", "-v", f"{seed / 'cuda'}:/llama:ro", cfg_image]
                runs.update(run_backend_ops(lambda c, t: sh(docker + c, timeout=t), d, f"cuda-gpu{gi}",
                                            "/llama/test-backend-ops", "CUDA0", log))
    vk = seed / "vulkan" / "llama-bench"
    if vk.exists():
        _, summary = sh(["vulkaninfo", "--summary"], timeout=60)
        (d / "vulkaninfo-summary.txt").write_text(scrub(summary))
        tbo = seed / "vulkan" / "test-backend-ops"
        for vi, name in vulkan_devices(summary):
            for m in models:
                one(f"vulkan-dev{vi}-{m.stem}", [str(vk), "-m", str(m), "-ngl", "99", *LLAMA_ARGS],
                    env={"GGML_VK_VISIBLE_DEVICES": str(vi)})
            if tbo.exists():
                env = dict(os.environ, GGML_VK_VISIBLE_DEVICES=str(vi))
                runs.update(run_backend_ops(lambda c, t, env=env: _run_env(c, t, env), d, f"vulkan-dev{vi}",
                                            str(tbo), "Vulkan0", log))
        if tbo.exists():                                      # the CPU backend: layers / experts offloaded to the CPU
            runs.update(run_backend_ops(lambda c, t: _run_env(c, t, dict(os.environ, GGML_VK_VISIBLE_DEVICES="")),
                                        d, "cpu", str(tbo), "CPU", log))
        for m in models[:1]:                                  # CPU only: no GPU device, no offload
            one(f"cpu-{m.stem}", [str(vk), "-m", str(m), "-ngl", "0", *LLAMA_ARGS], env={"GGML_VK_VISIBLE_DEVICES": ""})
    if not runs:
        return dict(skipped="no usable llama.cpp backend")
    return dict(rc=0 if all(v["rc"] == 0 for v in runs.values()) else 1, runs=runs)


# --- Vulkan probe: every GPU, every vendor (bandwidth, dispatch cost, vector FMA, matrix units; iGPU shared RAM) -----
def parse_lines(text: str) -> list[dict]:
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def vk_plan(devices: list[dict]) -> list[tuple[str, list[str]]]:
    """(key, vkprobe args) per hardware device; software devices (llvmpipe) left out."""
    out = []
    for d in devices:
        if d.get("device_type") == "cpu" or SOFTWARE_DEVICES.search(d.get("device", "")):
            continue
        i = str(d["index"]); dev = ["--device", i]
        for op in ("read", "write", "copy"):
            out.append((f"dev{i}-stream-{op}", ["stream", *dev, "--op", op, "--mib", "1024", "--reps", "20"]))
        out.append((f"dev{i}-launch", ["launch", *dev, "--count", "2000"]))
        for prec in ("fp32", "fp16"):
            out.append((f"dev{i}-fma-{prec}", ["fma", *dev, "--prec", prec]))
        for t in VK_GEMM_TYPES:
            for shape in VK_GEMM_SHAPES:
                out.append((f"dev{i}-gemm-{t}-{shape}", ["gemm", *dev, "--type", t, "--shape", shape, "--reps", "10"]))
        if d.get("device_type") == "integrated":
            out.append((f"dev{i}-shared", ["shared", *dev, "--seconds", "8", "--threads", "0"]))
    return out


def run_vulkan_probe(rundir: Path, log: Path, probe: Path = VKPROBE) -> dict:
    if not probe.exists():
        return dict(skipped="vkprobe not in the image")
    d = rundir / "vkprobe"
    d.mkdir(parents=True, exist_ok=True)
    rc, out = sh([str(probe), "list"], timeout=120)
    (d / "list.jsonl").write_text(scrub(out))
    devices = [r for r in parse_lines(out) if r.get("mode") == "list"]
    runs = {}
    vplan = vk_plan(devices)
    for k, (key, args) in enumerate(vplan, 1):
        STATUS.set(f"{key} {k}/{len(vplan)}")
        t0 = time.time()
        rc, out = sh([str(probe), *args], timeout=PROBE_TIMEOUT_S)
        (d / f"{key}.jsonl").write_text(scrub(out))
        runs[key] = dict(rc=rc, seconds=round(time.time() - t0, 1))
    if not runs:
        return dict(skipped="no hardware Vulkan device", devices=len(devices))
    say(f"vkprobe: {len(runs)} runs on {len({k.split('-')[0] for k in runs})} device(s)", log)
    return dict(rc=0 if all(v["rc"] == 0 for v in runs.values()) else 1, runs=runs)


# --- vendor-native: AMD (ROCm: hipBLAS probe + llama.cpp HIP), Intel (oneAPI: oneMKL probe + llama.cpp SYCL) ---------
def gpu_vendors(drm: Path = DRM) -> set[str]:
    out = set()
    for card in drm.glob("card[0-9]*"):
        if "-" in card.name:
            continue
        v = _read(card / "device" / "vendor")
        out |= {"amd"} if v == "0x1002" else {"intel"} if v == "0x8086" else set()
    return out


def vendor_images(data: Path) -> dict[str, str]:
    """name -> image tag from the seed's VENDOR.tsv (only those whose tar is present)."""
    f = data / "seed" / "images" / "VENDOR.tsv"
    out = {}
    for line in (f.read_text().splitlines() if f.exists() else []):
        parts = line.split("\t")
        if len(parts) >= 2 and (data / "seed" / "images" / f"vendor-{parts[0]}.tar").exists():
            out[parts[0]] = parts[1]
    return out


def hsa_override(arch: str, targets: list[str]) -> str | None:
    """HSA_OVERRIDE_GFX_VERSION for a card whose gfx target was not compiled (None: built, or no known family)."""
    base = arch.split(":")[0]
    if base in targets:
        return None
    for prefix, ver in GFX_FAMILY:
        if base.startswith(prefix):
            return ver
    return None


def vendor_plan(kind: str) -> list[tuple[str, list[str]]]:
    """(key, command inside the container) per device, without the device selection (that is the container's env)."""
    probe = "/rigsim/probe" if kind == "rocm" else "/rigsim/probe_sycl"
    out = [(f"stream-{op}", [probe, "stream", "--op", op, "--mib", "1024", "--reps", "20"]) for op in ("read", "write", "copy")]
    out.append(("launch", [probe, "launch", "--count", "2000"]))
    for d in ("h2d", "d2h") + (("bidir",) if kind == "rocm" else ()):
        out.append((f"copy-{d}", [probe, "copy", "--dir", d, "--mib", "512", "--reps", "10"]))
    for prec in VENDOR_GEMM_PRECS:
        for shape in VENDOR_GEMM_SHAPES:
            out.append((f"gemm-{prec}-{shape}", [probe, "gemm", "--prec", prec, "--shape", shape, "--reps", "10"]))
    return out


def run_vendor(kind: str, data: Path, rundir: Path, log: Path) -> dict:
    vendor = "amd" if kind == "rocm" else "intel"
    if vendor not in gpu_vendors():
        return dict(skipped=f"no {vendor.upper()} GPU")
    if kind == "rocm" and not Path("/dev/kfd").exists():
        return dict(skipped="no /dev/kfd (amdgpu compute not available on this kernel)")
    tag = vendor_images(data).get(kind)
    if not tag:
        return dict(skipped=f"{kind} image not in the seed")
    res = dict(load=load_image(data / "seed" / "images" / f"vendor-{kind}.tar", data, log))
    if res["load"].get("rc"):
        return dict(res, rc=1, error="image did not load")
    devs = AMD_DEVICES if kind == "rocm" else INTEL_DEVICES
    models = llama_models(data)
    d = rundir / kind
    d.mkdir(parents=True, exist_ok=True)
    name = f"rigsim-{kind}"
    sh(["docker", "rm", "-f", name], timeout=60)
    base = ["docker", "run", "--rm", "--network", "none", *devs, "-v", f"{data / 'seed' / 'models'}:/models:ro", tag]
    probe = "/rigsim/probe" if kind == "rocm" else "/rigsim/probe_sycl"
    # oneAPI's device discovery hung past 300 s on an i7-13700K's UHD 770 (second trial run): only Level Zero GPUs,
    # and sycl-ls's own account first (bounded), so the next run shows where it stops
    if kind == "oneapi":
        base = [*base[:-1], "-e", "ONEAPI_DEVICE_SELECTOR=level_zero:gpu", base[-1]]
        _, ls = sh([*base, "/rigsim/run.sh", "timeout", "180", "sycl-ls", "--verbose"], timeout=240)
        (d / "sycl-ls.txt").write_text(scrub(ls))
    _, info = sh([*base, "/rigsim/run.sh", probe, "info"], timeout=300)
    (d / "info.jsonl").write_text(scrub(info))
    _, meta = sh([*base, "sh", "-c", "cat /rigsim/TARGETS /rigsim/ROCM_VERSION /rigsim/ONEAPI_VERSION /rigsim/LLAMA_COMMIT 2>/dev/null; "
                  "ls /rigsim"], timeout=120)
    (d / "image.txt").write_text(scrub(meta))
    targets = meta.splitlines()[0].split(";") if kind == "rocm" and meta else []
    have_probe = (Path(probe).name in meta.split())
    devices = [r for r in parse_lines(info) if r.get("mode") == "info"]
    if not devices:
        return dict(res, rc=1, error="the runtime sees no GPU (info.jsonl)")
    res["devices"], runs = [], {}
    for dev in devices:
        i = str(dev["index"])
        env = {"HIP_VISIBLE_DEVICES": i} if kind == "rocm" else {"ONEAPI_DEVICE_SELECTOR": f"level_zero:{i}"}
        ov = hsa_override(dev.get("arch", ""), targets) if kind == "rocm" else None
        if ov:
            env["HSA_OVERRIDE_GFX_VERSION"] = ov
        res["devices"].append(dict(index=dev["index"], pci=dev.get("pci", ""), name=dev.get("name"), arch=dev.get("arch"),
                                   hsa_override=ov))
        c = f"{name}-{i}"
        sh(["docker", "rm", "-f", c], timeout=60)
        envargs = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
        rc, _ = sh(["docker", "run", "-d", "--name", c, "--network", "none", *devs, *envargs,
                    "-v", f"{data / 'seed' / 'models'}:/models:ro", tag, "sleep", "infinity"], timeout=300)
        if rc != 0:
            runs[f"dev{i}"] = dict(rc=rc, error="container did not start"); continue
        plan = vendor_plan(kind) if have_probe else []
        plan += [(f"llama-{m.stem}", ["/rigsim/llama-bench", "-m", f"/models/{m.name}", "-ngl", "99", *LLAMA_ARGS]) for m in models]
        for k, (key, cmd) in enumerate(plan, 1):
            STATUS.set(f"dev{i} {key} {k}/{len(plan)}")
            t0 = time.time()
            rc, out = sh(["docker", "exec", c, "/rigsim/run.sh", *cmd], timeout=LLAMA_TIMEOUT_S if key.startswith("llama") else PROBE_TIMEOUT_S)
            (d / f"dev{i}-{key}.jsonl").write_text(scrub(out))
            runs[f"dev{i}-{key}"] = dict(rc=rc, seconds=round(time.time() - t0, 1))
        if "test-backend-ops" in meta.split():
            runs.update(run_backend_ops(lambda cmd, t, c=c: sh(["docker", "exec", c, "/rigsim/run.sh", *cmd], timeout=t),
                                        d, f"dev{i}", "/rigsim/test-backend-ops", "ROCm0" if kind == "rocm" else "SYCL0", log))
        sh(["docker", "rm", "-f", c], timeout=120)
        say(f"{kind} device {i} ({dev.get('name')}): {sum(1 for k in runs if k.startswith(f'dev{i}-'))} runs"
            + (f", HSA_OVERRIDE_GFX_VERSION={ov}" if ov else "") + ("" if have_probe else ", probe not in the image"), log)
    return dict(res, rc=0 if runs and all(v["rc"] == 0 for v in runs.values()) else 1, runs=runs)


# --- Blender Benchmark (official launcher CLI, Blender and scenes pre-downloaded into the seed) ---------------------------
def blender_device_types(devices_out: str) -> list[str]:
    """Device types the launcher offers ('NVIDIA GeForce RTX 3090    OPTIX'), CPU first, each type once."""
    found = []
    for line in devices_out.splitlines():
        parts = line.strip().rsplit(None, 1)
        if len(parts) == 2 and parts[1] in ("CPU", "CUDA", "OPTIX", "HIP", "ONEAPI") and parts[1] not in found:
            found.append(parts[1])
    return sorted(found, key=lambda t: ("CPU", "OPTIX", "CUDA", "HIP", "ONEAPI").index(t))


def guessed_blender_types(drm: Path = DRM) -> list[str]:
    """When the launcher cannot list devices (it asks opendata.blender.org first, the stick is offline): the types
    this machine's GPUs would offer. Tried one by one; whatever the launcher refuses is recorded as failed."""
    vendors = {_read(c / "device" / "vendor") for c in drm.glob("card[0-9]*") if "-" not in c.name}
    t = ["CPU"]
    if "0x10de" in vendors:
        t += ["OPTIX", "CUDA"]
    if "0x1002" in vendors:
        t.append("HIP")
    if "0x8086" in vendors:
        t.append("ONEAPI")
    return t


def blender_offline(b: Path, version: str) -> tuple[Path | None, dict[str, Path]]:
    """The pieces of the official offline route, from the launcher's cache in the seed: Blender's executable for
    this version and each scene's .blend (the largest in the scene's folder)."""
    cache = b                                        # the launcher's cache and what the build unpacked from it
    exes = [p for p in sorted(cache.rglob("blender")) if p.is_file() and os.access(p, os.X_OK)]
    exe = next((p for p in exes if version in str(p)), exes[0] if exes else None)
    scenes = {}
    for name in BLENDER_SCENES:
        blends = [p for p in cache.rglob("*.blend") if name in p.relative_to(cache).parts
                  and not any(x.startswith("._") for x in p.relative_to(cache).parts)]
        if blends:
            scenes[name] = max(blends, key=lambda p: p.stat().st_size)
    return exe, scenes


def benchmark_script(b: Path) -> Path:
    """The benchmark script's main.py: the launcher's own copy in its cache (benchmark-scripts/<hash>/
    blender_benchmark_script-<version>/, the version the launcher itself runs, newest first), else the one the
    build fetched (script/). macOS resource forks ('._*') are not scripts."""
    found = [p for p in (b / "home").rglob("main.py") if "benchmark-scripts" in p.parts and not p.name.startswith("._")]
    ver = lambda p: tuple(int(x) for x in re.findall(r"\d+", p.parent.name)[-3:])
    return max(found, key=ver) if found else b / "script" / "main.py"


def run_blender(data: Path, rundir: Path, log: Path) -> dict:
    """Blender Benchmark by the official offline route (opendata.blender.org/about, "Run Offline"): Blender, the
    benchmark script and a scene, one device type at a time — the same script the launcher runs, without the
    launcher, which needs opendata.blender.org even to list devices. The launcher is the fallback."""
    b = data / "seed" / "blender"
    ver = _read(b / "BLENDER_VERSION")
    d = rundir / "blender"
    exe, scenes = blender_offline(b, ver) if ver else (None, {})
    script = benchmark_script(b)
    if exe and scenes and script.exists():
        d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, HOME=str(b / "home"))
        rocm_tag = vendor_images(data).get("rocm")
        runs, unavailable = {}, []
        for t in guessed_blender_types():
            for i, (name, blend) in enumerate(scenes.items()):
                STATUS.set(f"{t} {name}")
                key = f"{t.lower()}-{name}"
                t0 = time.time()
                cmd = [str(exe), "--background", "--factory-startup", "-noaudio", "--debug-cycles", "--enable-autoexec",
                       "--engine", "CYCLES", str(blend), "--python", str(script), "--", "--device-type", t]
                if t == "HIP":
                    # HIP needs AMD's runtime, which only the ROCm image has: the same Blender, script and scene,
                    # mounted read-only at the same paths, run inside it
                    if not rocm_tag or load_image(data / "seed" / "images" / "vendor-rocm.tar", data, log).get("rc"):
                        unavailable.append(t)
                        break
                    cmd = ["docker", "run", "--rm", "--network", "none", *AMD_DEVICES, "-v", f"{b}:{b}:ro",
                           "-w", str(script.parent), "-e", "HOME=/tmp", rocm_tag, "/rigsim/run.sh", *cmd]
                try:
                    rr = subprocess.run(cmd, capture_output=True, text=True, cwd=str(script.parent), env=env,
                                        timeout=BLENDER_TIMEOUT_S)
                    rc, out, err = rr.returncode, rr.stdout, rr.stderr
                except subprocess.TimeoutExpired:
                    rc, out, err = -9, "", f"timeout after {BLENDER_TIMEOUT_S} s"
                (d / f"{key}.log").write_text(scrub(out + ("\n--- stderr ---\n" + err if err.strip() else "")))
                say(f"blender {key}: rc={rc} ({time.time() - t0:.0f} s)", log)
                if rc != 0 and i == 0 and t != "CPU":
                    unavailable.append(t)                       # guessed type this GPU / build does not offer
                    break
                runs[key] = dict(rc=rc, seconds=round(time.time() - t0))
        return dict(rc=0 if runs and all(v["rc"] == 0 for v in runs.values()) else 1, route="offline", blender=ver,
                    script=str(script.relative_to(b)),
                    runs=runs, types_unavailable=unavailable,
                    hip_runtime=f"ROCm image {rocm_tag}" if rocm_tag and any(k.startswith("hip-") for k in runs) else "")
    missing = [x for x, ok in (("Blender executable", exe), ("scenes", scenes), ("benchmark script", script.exists()))
               if not ok]
    say(f"blender: offline route not possible ({', '.join(missing) or 'no version'} missing in the seed); "
        "trying the launcher", log)
    r = run_blender_launcher(b, ver, d, log)
    return dict(r, offline_route_missing=missing)


def run_blender_launcher(b: Path, ver: str, d: Path, log: Path) -> dict:
    cli = b / "benchmark-launcher-cli"
    if not cli.exists() or not ver:
        return dict(skipped="Blender benchmark not in the seed")
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HOME=str(b / "home"), XDG_CACHE_HOME=str(b / "home" / ".cache"))
    r = subprocess.run([str(cli), "devices", "--blender-version", ver], capture_output=True, text=True, env=env,
                       timeout=300)
    (d / "devices.txt").write_text(scrub(r.stdout + r.stderr))
    runs = {}
    types, guessed = blender_device_types(r.stdout), False
    if not types and "error" in (r.stdout + r.stderr).lower():
        types, guessed = guessed_blender_types(), True
        say(f"blender: launcher could not list devices (offline?); trying {types}", log)
    for t in types:
        t0 = time.time()
        try:
            rr = subprocess.run([str(cli), "benchmark", "--blender-version", ver, "--device-type", t, "--json",
                                 *BLENDER_SCENES], capture_output=True, text=True, env=env, timeout=BLENDER_TIMEOUT_S)
            rc, out, err = rr.returncode, rr.stdout, rr.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = -9, "", f"timeout after {BLENDER_TIMEOUT_S} s"
        (d / f"{t.lower()}.json").write_text(scrub(out))
        (d / f"{t.lower()}.log").write_text(scrub(err))
        runs[t] = dict(rc=rc, seconds=round(time.time() - t0))
        say(f"blender {t}: rc={rc} ({time.time() - t0:.0f} s)", log)
    if not runs:
        if "error" in (r.stdout + r.stderr).lower():
            return dict(rc=1, error="the launcher failed before listing devices (blender/devices.txt)")
        return dict(skipped="the launcher found no device")
    return dict(rc=0 if all(v["rc"] == 0 for v in runs.values()) else 1, route="launcher", blender=ver, runs=runs,
                device_types_guessed=guessed)


def should_power_off(data: Path, cmdline: str) -> bool:
    return not (data / "NO_POWEROFF").exists() and "rigsim.nopoweroff" not in cmdline.split()


def append_index(data: Path, row: dict) -> None:
    f = data / "index.tsv"
    if not f.exists():
        f.write_text(INDEX_HEADER)
    clean = lambda v: str(v).replace("\t", " ").replace("\n", " ")
    with open(f, "a") as fh:
        fh.write("\t".join(clean(row.get(k, "")) for k in
                           ("run", "start_utc", "fingerprint", "cpu", "board", "gpus", "ram", "status", "stick")) + "\n")


# --- host actions ------------------------------------------------------------------------------------------------------
def notice(seconds: int = NOTICE_S) -> bool:
    """True = go on; False = a key was pressed (abort, nothing measured)."""
    print("\n" * 2 + "=" * 78)
    print(" rigsim hardware stick — measures this machine's CPU, memory and GPUs, then powers it off.")
    print(" Nothing is written to this machine's disks. Results stay on the stick (no serial numbers) and may be")
    print(" published as open data (CC BY 4.0) with the owner's consent: github.com/keysie/rigsim-benchmark")
    print(f" Press any key within {seconds} s to ABORT (a shell is on Ctrl+Alt+F2).")
    print("=" * 78, flush=True)
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    except Exception:                                   # noqa: BLE001 - no tty: no way to abort, go on
        old = None
    try:
        end = time.time() + seconds
        while time.time() < end:
            left = int(end - time.time())
            print(f"\r starting in {left:2d} s ", end="", flush=True)
            if old is not None and select.select([sys.stdin], [], [], 1.0)[0]:
                sys.stdin.read(1)
                print("\n aborted — nothing measured.", flush=True)
                return False
            elif old is None:
                time.sleep(1)
        print()
        return True
    finally:
        if old is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def mount_data() -> Path | None:
    rc, dev = sh(["blkid", "-L", LABEL], timeout=30)
    if rc != 0 or not dev.strip():
        return None
    MNT.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["mountpoint", "-q", str(MNT)]).returncode != 0:
        if sh(["mount", "-o", "noatime", dev.strip().splitlines()[0], str(MNT)])[0] != 0:
            return None
    return MNT


def start_docker(data: Path, log: Path) -> dict:
    """Docker and containerd keep everything on the data partition. The live system's root is an overlay; an
    overlay snapshot cannot sit on it ("not supported as upperdir"), so both stores move to the stick's ext4:
    Docker by data-root, containerd (its image store and snapshots, used by Docker's containerd image store) by a
    bind mount of /var/lib/containerd. A container is started once to prove it works."""
    sh(["systemctl", "stop", "docker", "docker.socket", "containerd"], timeout=120)
    root = data / "docker"
    croot = data / "containerd"
    root.mkdir(exist_ok=True); croot.mkdir(exist_ok=True)
    cl = Path("/var/lib/containerd"); cl.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["mountpoint", "-q", str(cl)]).returncode != 0:
        sh(["mount", "--bind", str(croot), str(cl)])
    cfg = Path("/etc/docker/daemon.json")
    d = json.loads(cfg.read_text()) if cfg.exists() else {}
    d["data-root"] = str(root)
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(json.dumps(d, indent=1))
    sh(["systemctl", "start", "containerd", "docker"], timeout=300)
    for img in sorted((data / "seed" / "images").glob("*.tar")):
        if not img.name.startswith("vendor-"):       # vendor images load only on a machine that has that vendor's GPU
            load_image(img, data, log)
    # proof: one container from the smallest seeded image (a broken storage setup fails every question in a second)
    tags = [t for img in sorted((data / "seed" / "images").glob("*.tar"), key=lambda p: p.stat().st_size)
            if not img.name.startswith("vendor-") for t in image_tags(img)]
    if not tags:
        return dict(ok=False, why="no seeded image")
    rc, out = sh(["docker", "run", "--rm", "--network", "none", tags[0], "true"], timeout=300)
    _, driver = sh(["docker", "info", "--format", "{{.Driver}} / {{.DriverStatus}}"], timeout=60)
    res = dict(ok=rc == 0, image=tags[0], storage=driver.strip()[:200])
    if rc != 0:
        res["error"] = out.strip()[-400:]
    say(f"docker: {'works' if rc == 0 else 'BROKEN — container questions will fail'} ({res['storage'][:60]})", log)
    return res


def image_tags(img: Path) -> list[str]:
    """Image names inside a docker-save archive (manifest.json RepoTags; OCI index annotations as fallback)."""
    import tarfile
    try:
        with tarfile.open(img) as t:
            for name in ("manifest.json", "index.json"):
                try:
                    m = json.load(t.extractfile(name))
                except KeyError:
                    continue
                if name == "manifest.json":
                    return [tag for e in m for tag in (e.get("RepoTags") or [])]
                return [x.get("annotations", {}).get("io.containerd.image.name") for x in m.get("manifests", [])
                        if x.get("annotations", {}).get("io.containerd.image.name")]
    except (OSError, tarfile.TarError, ValueError):
        pass
    return []


def archive_position(pid: int, img: Path, proc: Path = Path("/proc")) -> int | None:
    """How far process `pid` has read file `img`: the position of its descriptor for it (fdinfo `pos`)."""
    try:
        target = img.resolve()
        for fd in (proc / str(pid) / "fd").iterdir():
            try:
                if Path(os.readlink(fd)) == target:
                    for line in (proc / str(pid) / "fdinfo" / fd.name).read_text().splitlines():
                        if line.startswith("pos:"):
                            return int(line.split()[1])
            except OSError:
                continue
    except OSError:
        pass
    return None


def _du(p: Path) -> int:
    rc, out = sh(["du", "-sb", str(p)], timeout=300)
    try:
        return int(out.split()[0]) if rc == 0 else 0
    except (ValueError, IndexError):
        return 0


def load_image(img: Path, data: Path, log: Path | None, every: float = 30.0) -> dict:
    """docker load once per stick (marked on the data partition). Unpacking onto a USB stick can take long: a line
    every `every` seconds shows how much has been written and how fast (a slow stick shows here, not as a hang)."""
    if _loaded(img, data):
        return dict(rc=0, cached=True)
    mib = img.stat().st_size / 2**20
    say(f"docker: loading {img.name} ({mib:.0f} MiB) onto the stick", log)
    t0 = time.time()
    size_b = img.stat().st_size
    last, t_last = 0, t0
    p = subprocess.Popen(["docker", "load", "-q", "-i", str(img)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True)
    while True:
        try:
            p.wait(timeout=every)
            break
        except subprocess.TimeoutExpired:
            if time.time() - t0 > 3 * 3600:
                p.kill(); p.wait(); break
            # progress = how far docker has read the archive (its file position): exact and monotonic, unlike the
            # store's size, which also shrinks while containerd removes temporary and unreferenced data
            now, pos = time.time(), archive_position(p.pid, img)
            if pos is None:
                say(f"docker: {img.name}: loading, {now - t0:.0f} s", log)
            elif pos >= size_b:
                STATUS.set(f"loading {img.name}: unpacking")
                say(f"docker: {img.name}: archive read, unpacking onto the stick, {now - t0:.0f} s", log)
            else:
                STATUS.set(f"loading {img.name}: {pos / size_b:.0%}")
                say(f"docker: {img.name}: {pos / 2**30:.1f} of {size_b / 2**30:.1f} GiB ({pos / size_b:.0%}), "
                    f"{(pos - last) / 2**20 / max(now - t_last, 1e-6):.0f} MB/s, {now - t0:.0f} s", log)
                last, t_last = pos, now
    rc = p.returncode if p.returncode is not None else -9
    sec = time.time() - t0
    if rc == 0:
        (data / "docker" / f".loaded-{img.name}").write_text(str(img.stat().st_size))
    mib = img.stat().st_size / 2**20
    say(f"docker: {'loaded' if rc == 0 else 'FAILED to load'} {img.name} ({mib:.0f} MiB in {sec:.0f} s)", log)
    return dict(rc=rc, mib=round(mib), seconds=round(sec, 1))


def stick_check(path: Path = Path("/run/casper-md5check.json")) -> dict:
    """casper-md5check's verdict on the stick's files (md5sum.txt on the ISO): a corrupted stick is recorded."""
    try:
        r = json.loads(path.read_text())
    except (OSError, ValueError):
        return dict(result="unknown (no casper-md5check result)")
    bad = r.get("checksum_missmatch") or r.get("checksum_mismatch") or []
    return dict(result=r.get("result", "unknown"), mismatched=bad[:20])


def stick_read_speed(data: Path, mib: int = STICK_READ_MIB) -> dict:
    """Sequential read from the stick, page cache dropped for that file first: explains slow runs afterwards."""
    files = sorted((p for p in (data / "seed").rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    if not files:
        return dict(skipped="seed empty")
    f = files[-1]
    try:
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        t0, got = time.time(), 0
        while got < mib * 2**20:
            b = os.read(fd, min(8 << 20, mib * 2**20 - got))
            if not b:
                break
            got += len(b)
        sec = time.time() - t0
        os.close(fd)
    except OSError as e:
        return dict(error=str(e))
    return dict(file=f.name, mib=round(got / 2**20), seconds=round(sec, 2), mib_per_s=round(got / 2**20 / max(sec, 1e-6), 1))


def _loaded(img: Path, data: Path) -> bool:
    """Loaded before AND still there: the marker alone is not enough (a store that was lost, e.g. one that lived in
    RAM on an older stick version, leaves markers behind)."""
    mark = data / "docker" / f".loaded-{img.name}"
    if not (mark.exists() and mark.read_text().strip() == str(img.stat().st_size)):
        return False
    tags = image_tags(img)
    return bool(tags) and all(sh(["docker", "image", "inspect", t], timeout=60)[0] == 0 for t in tags)


def prepare_toolkits(data: Path) -> Path:
    tk = RUN / "toolkits"
    if tk.exists():
        shutil.rmtree(tk)
    shutil.copytree(OPT / "toolkits", tk)
    debs = data / "seed" / "debs"
    if debs.is_dir():
        shutil.copytree(debs, tk / "cpu" / "debs")
    return tk


def question_progress(d: Path) -> str:
    """'7/24 items' from the runner's state.json while a question runs."""
    try:
        st = json.loads((d / "state.json").read_text())
    except (OSError, ValueError):
        return "preparing"
    done = sum(1 for v in st.values() if v.get("status") not in ("pending", "running"))
    return f"{done}/{len(st)} items"


def run_question(q: str, qfile: Path, rundir: Path, cfg: Path, log: Path) -> dict:
    d = rundir / "q" / q
    say(f"{q}: start", log)
    t0 = time.time()
    pr = subprocess.Popen([sys.executable, str(OPT / "runner" / "hwq.py"), "run", str(qfile), "--dir", str(d),
                           "--config", str(cfg)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_parts: list[str] = []
    reader = threading.Thread(target=lambda: out_parts.append(pr.stdout.read()), daemon=True)
    reader.start()
    while True:
        try:
            pr.wait(timeout=2)
            break
        except subprocess.TimeoutExpired:
            if time.time() - t0 > QUESTION_TIMEOUT_S:
                pr.kill(); pr.wait(); out_parts.append(f"\ntimeout after {QUESTION_TIMEOUT_S} s"); break
            STATUS.set(question_progress(d))
    reader.join(timeout=10)
    rc, out = (pr.returncode if pr.returncode is not None else -9), "".join(out_parts)
    d.mkdir(parents=True, exist_ok=True)
    (d / "runner.log").write_text(scrub(out))
    counts = json.loads(sh([sys.executable, str(OPT / "runner" / "hwq.py"), "status", "--dir", str(d)])[1] or "{}")
    say(f"{q}: rc={rc} {counts} ({time.time() - t0:.0f} s)", log)
    return dict(rc=rc, items=counts, seconds=round(time.time() - t0))


# --- format and versions ---------------------------------------------------------------------------------------------
# The run folder's format (results/FORMAT.md). Minor versions only add: fields and files are never removed, renamed
# or retyped, so every reader of 1.x reads every 1.y run. A change that cannot be additive is a new major version.
FORMAT = dict(name="rigsim-stick-run", version="1.2")


def runtime_versions() -> dict:
    """What is actually running (a seed and a system image may come from different builds)."""
    r = dict(kernel=sh(["uname", "-r"])[1].strip())
    rc, drv = sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], timeout=60)
    if rc == 0 and drv.strip():
        r["nvidia_driver"] = drv.split("---")[0].strip().splitlines()[0]
    rc, dk = sh(["docker", "version", "--format", "{{.Server.Version}}"], timeout=60)
    if rc == 0:
        r["docker"] = dk.split("---")[0].strip()
    rc, vk = sh(["vulkaninfo", "--summary"], timeout=120)
    devs, cur = [], {}
    for line in vk.splitlines():
        m = re.match(r"\s*(deviceName|driverName|driverInfo|apiVersion|conformanceVersion)\s*=\s*(.*)", line)
        if m:
            if m.group(1) == "apiVersion" and cur:
                devs.append(cur); cur = {}
            cur[m.group(1)] = m.group(2).strip()
    if cur:
        devs.append(cur)
    r["vulkan_devices"] = [d for d in devs if d.get("deviceName")]
    return r


def load_manifest(data: Path, runtime: bool = True) -> dict:
    """The system image's and the seed's build manifests plus what runs now: written to each run as manifest.json."""
    def rd(p: Path) -> dict:
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return {}
    return dict(format=FORMAT, system=rd(OPT / "manifest.json"), seed=rd(data / "seed" / "MANIFEST.json"),
                runtime=runtime_versions() if runtime else {})


def _pkgs(man: dict, pattern: str) -> dict:
    return {k: v for k, v in man.get("system", {}).get("packages", {}).items() if re.match(pattern, k)}


def tools_for(step: str, man: dict, extra: dict | None = None) -> dict:
    """The exact versions behind one step's results: copied from the manifest into the step's entry in run.json."""
    sysm, seed, rt = man.get("system", {}), man.get("seed", {}), man.get("runtime", {})
    t: dict = dict(stick=dict((sysm.get("stick") or {}), format=FORMAT["version"]), kernel=rt.get("kernel", ""))
    images = seed.get("images", {})
    cuda = {k: v for k, v in images.items() if "cuda" in k}
    cpu = {k: v for k, v in images.items() if "cuda" not in k}
    if step.startswith("cpu-"):
        t["images"] = cpu
    elif step.startswith(("gpu-", "pcie-")):
        t.update(images=cuda, nvidia_driver=rt.get("nvidia_driver", ""))
    elif step == "opencl":
        t.update(packages=_pkgs(man, r"(clpeak|clinfo|ocl-icd|intel-opencl-icd|libnvidia-compute|mesa-opencl)"),
                 nvidia_driver=rt.get("nvidia_driver", ""))
    elif step == "vkprobe":
        t.update(vkprobe=sysm.get("files", {}).get("bin/vkprobe", {}),
                 packages=_pkgs(man, r"(libvulkan1|mesa-vulkan-drivers|libnvidia-gl-)"),
                 vulkan_devices=rt.get("vulkan_devices", []))
    elif step in ("rocm", "oneapi"):
        t["vendor"] = seed.get("vendor", {}).get(step, {})
    elif step == "llama-bench":
        t.update({"llama.cpp": seed.get("llama.cpp", {}), "models": seed.get("models", {}), "images": cuda,
                  "nvidia_driver": rt.get("nvidia_driver", ""), "vulkan_devices": rt.get("vulkan_devices", []),
                  "packages": _pkgs(man, r"(libvulkan1|mesa-vulkan-drivers|libnvidia-gl-)")})
    elif step == "mlc":
        t["mlc"] = seed.get("tools", {}).get("mlc", {})
    elif step == "blender":
        t["blender-benchmark"] = seed.get("tools", {}).get("blender-benchmark", {})
        t.update(nvidia_driver=rt.get("nvidia_driver", ""), packages=_pkgs(man, r"(libnvidia-gl-|libze|intel-level-zero-gpu)"))
        if seed.get("vendor", {}).get("rocm"):
            t["hip_runtime"] = seed["vendor"]["rocm"]               # Blender HIP renders inside the ROCm image
    t.update(extra or {})
    return t


def mlc_version(rundir: Path) -> str:
    f = rundir / "mlc" / "idle_latency.txt"
    m = re.search(r"Memory Latency Checker\s*-\s*(v[\w.]+)", f.read_text()) if f.exists() else None
    return m.group(1) if m else ""


def power_off() -> None:
    """systemd stops this service on the way down (SIGTERM); waiting for systemctl here would turn that into a
    'Stopped' traceback. Ignore the signal from now on and do not wait for the shutdown."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    subprocess.run(["systemctl", "--no-block", "poweroff"])


def main() -> int:
    def on_term(*_):
        raise Stopped()
    signal.signal(signal.SIGTERM, on_term)
    version = (OPT / "VERSION").read_text().strip() if (OPT / "VERSION").exists() else "unknown"
    cmdline = Path("/proc/cmdline").read_text() if Path("/proc/cmdline").exists() else ""
    data = mount_data()
    if data is None:
        say("no data partition labelled RIGSIM — nothing measured. Powering off in 5 minutes (Ctrl+Alt+F2: shell).")
        time.sleep(300)
        if "rigsim.nopoweroff" not in cmdline.split():
            power_off()
        return 1
    if not notice():
        return 0
    fp, stable = fingerprint(data)
    rundir = new_run_dir(data / "runs", fp)
    log = rundir / "stick.log"
    start = dt.datetime.now(dt.timezone.utc)
    man = load_manifest(data, runtime=False)          # runtime versions once drivers and Docker are up
    stick_info = dict(man["system"].get("stick") or dict(build=version))
    run = dict(format=FORMAT, run=rundir.name, stick=stick_info, seed=dict(build=man["seed"].get("build", "")),
               fingerprint=fp, fingerprint_stable=stable,
               start_utc=start.isoformat(timespec="seconds"), clock_note="machine RTC; may be wrong",
               questions={}, status="running")
    (rundir / "run.json").write_text(json.dumps(run, indent=1))
    say(f"run {rundir.name}", log)
    STATUS.start()
    STATUS.begin("inventory")
    try:
        run["hardware"] = collect_inventory(rundir / "inventory", log)
        say(f"cpu {run['hardware']['cpu']} | board {run['hardware']['board']} | {run['hardware']['ram_gib']} GiB | "
            f"gpus {run['hardware']['gpus'] or 'none'}", log)
        run["stick_check"] = stick_check()
        say(f"stick check: {run['stick_check'].get('result', run['stick_check'])}", log)
        run["stick_read"] = stick_read_speed(data)
        say(f"stick read: {run['stick_read']}", log)
        STATUS.begin("docker")
        run["docker"] = start_docker(data, log)
        man["runtime"] = runtime_versions()
        (rundir / "manifest.json").write_text(json.dumps(man, indent=1))
        RUN.mkdir(parents=True, exist_ok=True)
        tk = prepare_toolkits(data)
        cfg = RUN / "hwq.json"
        cfg.write_text(json.dumps(dict(toolkits=str(tk), image_mirror="", setup_proxy={}, gpu_names={},
                                       keep_running=[], testdir_parent=str(RUN / "testdir"),
                                       pause_file=str(RUN / "PAUSE"), min_free_gb=2)))
        nvidia, why = nvidia_state()
        run["displays"] = displays()
        say(f"screens: {[(x['card'], x['connector'], x['mode']) for x in run['displays']] or 'none connected'}", log)
        run["screens_blanked"] = blank_screens(True)
        say(f"screens off while measuring: {run['screens_blanked'] or 'not possible'}", log)
        cuda_image = re.search(r'(?m)^image\s*=\s*"([^"]+)"', (OPT / "questions" / "gpu-gemm.toml").read_text()).group(1)
        with DrmSampler(rundir / "samples-drm.jsonl"), NvidiaSampler(rundir / "samples-nvidia.csv", nvidia):
            qplan = list(plan(OPT / "questions", nvidia, why))
            STATUS.n = len(qplan) + 7
            for q, f, skip in qplan:
                STATUS.begin(q)
                if skip:
                    run["questions"][q] = dict(skipped=skip); say(f"{q}: skipped ({skip})", log); continue
                run["questions"][q] = run_question(q, f, rundir, cfg, log)
                run["questions"][q]["tools"] = tools_for(q, man)
                (rundir / "run.json").write_text(json.dumps(run, indent=1))
            for name, step in (("mlc", lambda: run_mlc(data, rundir, log)),
                               ("opencl", lambda: run_opencl(rundir, log)),
                               ("vkprobe", lambda: run_vulkan_probe(rundir, log)),
                               ("rocm", lambda: run_vendor("rocm", data, rundir, log)),
                               ("oneapi", lambda: run_vendor("oneapi", data, rundir, log)),
                               ("llama-bench", lambda: run_llama(data, rundir, cuda_image, nvidia, log)),
                               ("blender", lambda: run_blender(data, rundir, log))):
                STATUS.begin(name)
                run["questions"][name] = step()
                run["questions"][name]["tools"] = tools_for(name, man, dict(mlc_version=mlc_version(rundir)) if name == "mlc" else None)
                if run["questions"][name].get("skipped"):
                    say(f"{name}: skipped ({run['questions'][name]['skipped']})", log)
                (rundir / "run.json").write_text(json.dumps(run, indent=1))
        ok = all(v.get("skipped") or (v.get("rc") == 0) for v in run["questions"].values())
        run["status"] = "done" if ok else "done with failures"
    except Stopped:
        run["status"] = "stopped"
        say("stopped on request", log)
    except Exception as e:                              # noqa: BLE001 - always label what we have
        run["status"] = f"error: {type(e).__name__}: {e}"
        say(run["status"], log)
    blank_screens(False)
    run["end_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (rundir / "run.json").write_text(json.dumps(run, indent=1))
    hw = run.get("hardware", {})
    append_index(data, dict(run=rundir.name, start_utc=run["start_utc"], fingerprint=fp, cpu=hw.get("cpu", ""),
                            board=hw.get("board", ""), gpus="; ".join(hw.get("gpus", [])),
                            ram=f"{hw.get('ram_gib', '')} GiB", status=run["status"],
                            stick=f"{stick_info.get('version', '')} ({version})"))
    say(f"finished: {run['status']}", log)
    sh(["systemctl", "stop", "docker", "containerd"], timeout=300)
    sh(["umount", "/var/lib/containerd"], timeout=120)
    os.sync()
    STATUS.stop()
    off = should_power_off(data, cmdline) and run["status"] != "stopped"
    sh(["umount", str(MNT)], timeout=120)
    if off:
        say("powering off in 10 s")
        time.sleep(10)
        power_off()
    return 0


def preload(data: Path) -> int:
    """Run by flash.sh at flash time, inside the stick's own system image (chroot) with containerd and dockerd
    already started on the data partition: loads every seeded image — the vendor images too — so the stick's
    first boot on a machine finds them ready. Same binaries and paths as on the stick; the stick checks every
    image again before use (_loaded) and loads anything missing itself."""
    failed = 0
    for img in sorted((data / "seed" / "images").glob("*.tar"), key=lambda p: p.stat().st_size):
        r = load_image(img, data, None, every=15.0)
        if r.get("cached"):
            say(f"docker: {img.name} already on the stick")
        failed += r.get("rc", 0) != 0
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--preload":
        sys.exit(preload(Path(sys.argv[2])))
    sys.exit(main())
