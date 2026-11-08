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
PROFILE = ["cpu-mem-latency", "cpu-mem-ladder", "cpu-gemm", "cpu-gemm-shapes", "cpu-gemm-ladder", "cpu-power-time", "cpu-c2c",
           "pcie-copy", "pcie-xfer", "gpu-p2p", "gpu-p2p-xfer", "gpu-vram", "gpu-gemm", "gpu-gemm-shapes", "gpu-launch",
           "gpu-overlap"]
DRM = Path("/sys/class/drm")
FBDEV = Path("/sys/class/graphics")
SAMPLE_S = 0.2
MLC_RUNS = [("idle_latency", ["--idle_latency"]), ("max_bandwidth", ["--max_bandwidth"]),
            ("loaded_latency", ["--loaded_latency"]), ("latency_matrix", ["--latency_matrix"]),
            ("bandwidth_matrix", ["--bandwidth_matrix"])]
MLC_TIMEOUT_S = 15 * 60
# On small machines MLC gets a 200 MiB buffer (what it chose by itself on 16 GB Intel machines): on an 8 GB AMD laptop
# it chose 1800 MiB, asked for 8 GB of hugepages and stopped (run 05ea).
MLC_SMALL_GIB = 12
MLC_SMALL_BUFFER_KIB = 200 * 1024
# MLC wants 1000 hugepages (2 MiB) per NUMA node. On the Ryzen 5 2500U it stopped with rc 22 without them and ran with
# them (manual test 2026-11-04, idle latency 134.1 ns); on the Intel machines it ran either way. The stick reserves them
# for MLC (never more than leaves 2 GiB available) and gives them back afterwards.
MLC_HUGEPAGES_PER_NODE = 1000
HUGEPAGE = 2 << 20
NR_HUGEPAGES = Path("/proc/sys/vm/nr_hugepages")
NODES = Path("/sys/devices/system/node")


def reserve_hugepages(per_node: int = MLC_HUGEPAGES_PER_NODE, nr: Path = NR_HUGEPAGES,
                      nodes: Path = NODES) -> dict:
    """Ask the kernel for hugepages; returns what was there before, what was asked for and what it gave."""
    try:
        before = int(nr.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return dict(error="hugepages not available")
    n_nodes = max(1, len(list(nodes.glob("node[0-9]*"))))
    room = (meminfo().get("MemAvailable", 0) - 2 * GIB) // HUGEPAGE
    want = max(before, min(per_node * n_nodes, before + max(room, 0)))
    try:
        nr.write_text(str(want))
        got = int(nr.read_text().split()[0])
    except (OSError, ValueError, IndexError) as e:
        return dict(before=before, requested=want, error=str(e))
    return dict(before=before, requested=want, allocated=got)


def release_hugepages(h: dict, nr: Path = NR_HUGEPAGES) -> None:
    if "before" in h:
        try:
            nr.write_text(str(h["before"]))
        except OSError:
            pass
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
# the same tokens-in-flight sweep as gpu-gemm-shapes on CUDA (design/engine-primitives.md, item 2), plus the square peak
VENDOR_GEMM_SHAPES = [f"{m}x{n}x4096" for n in (4096, 14336) for m in (1, 4, 16, 64, 256, 1024, 4096, 8192)] + ["8192x8192x8192"]
PROBE_TIMEOUT_S = 10 * 60
AMD_DEVICES = ["--device", "/dev/kfd", "--device", "/dev/dri", "--security-opt", "seccomp=unconfined"]
INTEL_DEVICES = ["--device", "/dev/dri"]
# llama.cpp's SYCL backend hands batched GEMMs and flash attention to oneDNN. On an i7-13700K's UHD 770 (run d-9c93,
# 2026-10-31) oneDNN failed with "could not create a memory object" and took llama-bench, MUL_MAT, MUL_MAT_ID and
# FLASH_ATTN_EXT down; without oneDNN all four ran (manual test 2026-11-01). On exactly that error a oneAPI step runs
# once more without oneDNN: the first output is kept as <step>.onednn-failed.txt, the step records `retry`.
ONEDNN_ERROR = "could not create a memory object"
NO_ONEDNN = {"GGML_SYCL_ENABLE_DNN": "0", "GGML_SYCL_FA_ONEDNN": "0"}


def onednn_retry(rc: int, out: str, first: Path, again) -> tuple[int, str, dict | None]:
    """again(env) -> (rc, out). Retries once without oneDNN if the step failed on oneDNN's allocation error."""
    if rc == 0 or ONEDNN_ERROR not in out:
        return rc, out, None
    first.write_text(scrub(out))
    rc2, out2 = again(NO_ONEDNN)
    return rc2, out2, dict(reason=f"oneDNN: {ONEDNN_ERROR}", env=dict(NO_ONEDNN), first_rc=rc, first_output=first.name)


class OneDnnGuard:
    """One per oneAPI device. A llama.cpp step that fails on oneDNN runs again without it (onednn_retry); from then on
    every further llama.cpp step on that device runs without oneDNN right away (run c72d, 2026-11-01: FLASH_ATTN_EXT
    did not abort with oneDNN but hung for the full 30-min timeout, so the error text never came)."""

    def __init__(self):
        self.off_after = ""

    def run(self, step: str, first: Path, attempt) -> tuple[int, str, dict]:
        """attempt(env or None) -> (rc, out). Returns rc, out and what to add to the step's entry."""
        if self.off_after:
            rc, out = attempt(NO_ONEDNN)
            return rc, out, dict(onednn_off=dict(after=self.off_after, env=dict(NO_ONEDNN)))
        rc, out = attempt(None)
        rc, out, retry = onednn_retry(rc, out, first, attempt)
        if retry:
            self.off_after = step
        return rc, out, (dict(retry=retry) if retry else {})
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
        self.skippable = False
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

    def begin(self, step: str, skippable: bool = False) -> None:
        with self.lock:
            self.i += 1
            self.step, self.detail, self.t_step = step, "", time.time()
            self.skippable = skippable
            SKIP.clear()

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
                 self._t(now - self.t_step) if self.step else "", f"(run {self._t(now - self.t_run)})",
                 "· Enter twice: skip step" if self.skippable else ""]
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


# --- skipping a step from the console --------------------------------------------------------------------------------
# Pressing Enter twice (within 5 s) during a measurement step stops it and goes on with the next: the step's process
# group is killed, any container still running is stopped, and run.json records the step as skipped by the operator.
# Twice, because the first key press after the screens were switched off only wakes them.
SKIP = threading.Event()
SKIP_ARM_S = 5.0


class StepSkipped(Exception):
    pass


OPERATOR_SKIP = "skipped by the operator (Enter pressed twice during the step)"


def _kill_containers() -> None:
    try:
        ids = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True, timeout=30).stdout.split()
        if ids:
            subprocess.run(["docker", "kill", *ids], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pass


# --- memory per step (run 05ea, 2026-11-01: an 8 GB laptop without swap hung in Blender until it was shut down) ------
# Every measurement step runs under a memory limit: the available memory at its start minus a reserve, no swap.
# Processes on the host join a cgroup v2 with that limit; containers get --memory (the runner's session container via
# RIGSIM_MEMORY_BYTES). A step that needs more is ended by the kernel at once instead of thrashing the machine; the
# kernel's oom_kill counter tells the step it happened, and the run records it and goes on.
CGROUP = Path("/sys/fs/cgroup")
MEM_RESERVE = 768 << 20
MEM_MIN = 512 << 20
GIB = 1 << 30


def meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    """/proc/meminfo in bytes."""
    out = {}
    for line in (path.read_text().splitlines() if path.exists() else []):
        k, _, v = line.partition(":")
        f = v.split()
        if f and f[0].isdigit():
            out[k] = int(f[0]) * (1024 if f[1:] == ["kB"] else 1)
    return out


def oom_kills(path: Path = Path("/proc/vmstat")) -> int:
    """The kernel's count of OOM kills since boot (cgroup limits included)."""
    for line in (path.read_text().splitlines() if path.exists() else []):
        if line.startswith("oom_kill "):
            return int(line.split()[1])
    return 0


class StepMemory:
    """with StepMemory(name, log) as m: ... — the memory limit of one measurement step."""
    current: "StepMemory | None" = None

    def __init__(self, name: str, log: Path | None, root: Path = CGROUP):
        self.name, self.log, self.cg = name, log, root / "rigsim-step"
        self.cgroup = False
        self.kills = 0

    def __enter__(self) -> "StepMemory":
        self.available = meminfo().get("MemAvailable", 0)
        self.limit = max(self.available - MEM_RESERVE, MEM_MIN)
        try:
            self.cg.mkdir(exist_ok=True)
            (self.cg / "memory.max").write_text(str(self.limit))
            if (self.cg / "memory.swap.max").exists():
                (self.cg / "memory.swap.max").write_text("0")
            self.cgroup = True
        except OSError:
            pass
        self.kills0 = oom_kills()
        os.environ["RIGSIM_MEMORY_BYTES"] = str(self.limit)
        StepMemory.current = self
        say(f"{self.name}: {self.available / GIB:.1f} GiB available, step limit {self.limit / GIB:.1f} GiB", self.log)
        return self

    def join(self, pid: int) -> None:
        if self.cgroup:
            try:
                (self.cg / "cgroup.procs").write_text(str(pid))
            except OSError:
                pass

    def docker_args(self) -> list[str]:
        return ["--memory", str(self.limit), "--memory-swap", str(self.limit)]

    def __exit__(self, *exc) -> bool:
        StepMemory.current = None
        os.environ.pop("RIGSIM_MEMORY_BYTES", None)
        self.kills = oom_kills() - self.kills0
        try:
            self.cg.rmdir()
        except OSError:
            pass
        return False

    def record(self) -> dict:
        r = dict(available_gib=round(self.available / GIB, 2), limit_gib=round(self.limit / GIB, 2),
                 cgroup=self.cgroup)
        if self.kills:
            r["oom_kills"] = self.kills
        return r


def docker_mem() -> list[str]:
    """--memory for a container started inside a measurement step (none outside one)."""
    return StepMemory.current.docker_args() if StepMemory.current else []


def record_memory(entry: dict, mem: StepMemory, log: Path | None) -> None:
    """The step's memory limit into its entry; if the kernel ended processes at it, say so (format 1.6)."""
    entry["memory"] = mem.record()
    if mem.kills and not entry.get("skipped"):
        entry["out_of_memory"] = (f"the kernel ended {mem.kills} process(es) at the step's limit of "
                                  f"{mem.limit / GIB:.1f} GiB (the machine had {mem.available / GIB:.1f} GiB available)")
        say(f"{mem.name}: out of memory — {entry['out_of_memory']}", log)


def step_ok(v: dict) -> bool:
    """A step that counts as fine for the run's status: done, skipped with a reason, or ended by its memory limit (the
    machine cannot do it — recorded as out_of_memory, like a skip with a reason)."""
    return bool(v.get("skipped") or v.get("rc") == 0 or v.get("out_of_memory"))


def run_cmd(cmd: list[str], capture_output: bool = True, text: bool = True, timeout: float | None = None,
            env: dict | None = None, cwd: str | None = None) -> subprocess.CompletedProcess:
    """subprocess.run for measurement commands, but stoppable: raises StepSkipped when the operator skips the step
    (its whole process group is killed, running containers too)."""
    k0 = oom_kills()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE if capture_output else None,
                         stderr=subprocess.PIPE if capture_output else None, text=text, env=env, cwd=cwd,
                         start_new_session=True)
    if StepMemory.current:
        StepMemory.current.join(p.pid)
    t0 = time.time()
    while True:
        try:
            out, err = p.communicate(timeout=0.5)
            r = subprocess.CompletedProcess(cmd, p.returncode, out, err)
            r.oom = oom_kills() > k0                   # the kernel ended something at a memory limit meanwhile
            return r
        except subprocess.TimeoutExpired:
            if SKIP.is_set():
                _killpg(p); _kill_containers()
                raise StepSkipped()
            if timeout is not None and time.time() - t0 > timeout:
                out, err = _killpg(p)                  # what it printed until then is kept (run f78e: 30 min lost)
                raise subprocess.TimeoutExpired(cmd, timeout, output=out, stderr=err)


def _killpg(p: subprocess.Popen) -> tuple[str, str]:
    """Kill the process group; returns what it printed (empty where nothing was captured)."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        out, err = p.communicate(timeout=10)
        return out or "", err or ""
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return "", ""


def operator_keys() -> None:
    """Background reader of the console: Enter twice within SKIP_ARM_S skips the running measurement step."""
    armed = 0.0
    while True:
        try:
            line = sys.stdin.readline()
        except (OSError, ValueError):
            return
        if not line:
            return
        if not STATUS.skippable:
            continue
        now = time.time()
        if now - armed <= SKIP_ARM_S:
            SKIP.set()
            say(f"{STATUS.step}: skipping (Enter pressed twice)")
            armed = 0.0
        else:
            armed = now
            say(f"press Enter again within {SKIP_ARM_S:.0f} s to skip {STATUS.step}")


TIME_LIMIT_RC = 124                                    # like timeout(1): stopped at the time limit


def _timed_out(e: subprocess.TimeoutExpired) -> tuple[int, str]:
    out, err = e.output or "", e.stderr or ""
    out = out.decode(errors="replace") if isinstance(out, bytes) else out
    err = err.decode(errors="replace") if isinstance(err, bytes) else err
    return TIME_LIMIT_RC, out + (("\n--- stderr ---\n" + err) if err.strip() else "") + f"\n--- time limit: {e.timeout:.0f} s ---\n"


def sh(cmd: list[str], timeout: float = 120) -> tuple[int, str]:
    try:
        r = run_cmd(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.stderr.strip() else "")
    except subprocess.TimeoutExpired as e:
        return _timed_out(e)
    except OSError as e:
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


# Files whose columns look like IPv6 addresses but are not: lscpu -e's cache ids (L1d:L1i:L2:L3, e.g. 0:0:0:0) —
# masking them lost every machine's L3 topology. They carry no addresses.
NO_IPV6_SCRUB = {"lscpu-e.txt"}


def sanitize(text: str, ipv6: bool = True) -> str:
    """Drops lines naming serials, UUIDs or asset tags; masks MAC and IP addresses."""
    out = [l for l in text.splitlines() if not SENSITIVE.search(l)]
    t = "\n".join(out)
    t = MAC.sub("<mac>", t)
    if ipv6:
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


def has_nvidia_gpu(devices: Path = Path("/sys/bus/pci/devices")) -> bool:
    """An NVIDIA display device (vendor 0x10de, class 0x03xxxx) on the PCI bus."""
    for d in devices.glob("*"):
        try:
            if (d / "vendor").read_text().strip() == "0x10de" and (d / "class").read_text().startswith("0x03"):
                return True
        except OSError:
            continue
    return False


def collect_inventory(dst: Path, log: Path) -> dict:
    dst.mkdir(parents=True, exist_ok=True)
    nv = has_nvidia_gpu()
    # NVIDIA's modules and tools only where there is an NVIDIA card: elsewhere they just print "NVRM: No NVIDIA
    # GPU found" over the console
    for mod in ("i2c-i801", "i2c-piix4", "ee1004", "spd5118", "at24", "i2c-dev", "msr") + \
               (("nvidia_uvm", "nvidia_modeset", "nvidia_drm") if nv else ()):
        sh(["modprobe", mod], timeout=20)
    if nv and shutil.which("nvidia-modprobe"):           # device nodes the Vulkan driver opens (uvm, modeset)
        sh(["nvidia-modprobe", "-u", "-m", "-c", "0"], timeout=20)
    for k, (name, cmd) in enumerate(INVENTORY, 1):
        STATUS.set(f"{name} {k}/{len(INVENTORY)}")
        if cmd and cmd[0] == "nvidia-smi" and not nv:
            (dst / name).write_text("no NVIDIA GPU in this machine\n")
            continue
        rc, out = sh(cmd, timeout=180)
        (dst / name).write_text(sanitize(out, ipv6=name not in NO_IPV6_SCRUB) if not name.endswith(".json")
                                else MAC.sub("<mac>", out))
        if rc != 0:
            say(f"inventory: {name} rc={rc}", log)
    raw = spd_raw(dst)
    (dst / "pcie.json").write_text(json.dumps(pcie_topology(names=lspci_names()), indent=1))
    (dst / "powercap.json").write_text(json.dumps(powercap(), indent=1))
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
             dimms=dimms, cpufreq=cpufreq(), secure_boot=secure_boot(), kernel=sh(["uname", "-r"])[1].strip(),
             spd=spd_status(dst, raw), lockdown=lockdown())
    (dst / "summary.json").write_text(json.dumps(s, indent=1))
    return s


# --- PCIe topology ----------------------------------------------------------------------------------------------
# Files written by the inventory besides the INVENTORY commands (the validator allows exactly these).
INVENTORY_EXTRA = ["summary.json", "pcie.json", "powercap.json"]
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


def powercap(root: Path = Path("/sys/class/powercap")) -> dict:
    """The package power limits the machine runs under (Linux powercap / RAPL): per zone its name and each constraint's
    name ('long_term' = PL1, 'short_term' = PL2), limit in watts and time window in seconds. The board may set them
    away from the processor's TDP, and they influence what is measured: recorded as data. (rigsim models the package
    budget from the manufacturer's specification; these configured limits are context, not an input.)"""
    zones = {}
    for z in sorted(root.glob("*")):
        name = _read(z / "name")
        if not name or ":" not in z.name or z.name.count(":") > 1:       # package zones (intel-rapl:0), not sub-zones
            continue
        cons = []
        for c in sorted(z.glob("constraint_*_name")):
            i = c.name.split("_")[1]
            lim, win = _read(z / f"constraint_{i}_power_limit_uw"), _read(z / f"constraint_{i}_time_window_us")
            cons.append(dict(name=_read(c), limit_w=int(lim) / 1e6 if lim and lim.isdigit() else None,
                             window_s=int(win) / 1e6 if win and win.isdigit() else None))
        zones[z.name] = dict(name=name, enabled=_read(z / "enabled"), constraints=cons)
    return dict(zones=zones)


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


def spd_status(dst: Path, raw: list[str], i2c: Path = Path("/sys/bus/i2c/devices"),
               adapters: Path = Path("/sys/class/i2c-adapter")) -> dict:
    """Whether the modules' SPD could be read, and if not what was there to read it with (DDR item 2 of 2026-10-31:
    7 of 15 runs had "No EEPROM found", visible only in inventory/spd.txt)."""
    txt = (dst / "spd.txt").read_text() if (dst / "spd.txt").exists() else ""
    devs = sorted({(d / "driver").resolve().name for d in i2c.glob("*-00[5-9a-f]*") if (d / "driver").exists()}
                  & set(SPD_SERIAL))
    smbus = sorted({re.sub(r" at .*", "", _read(a / "name") or "") for a in adapters.glob("i2c-*")
                    if "smbus" in (_read(a / "name") or "").lower()})
    out = dict(spd_drivers=devs, smbus_adapters=smbus, raw_files=len(raw))
    if "Decoding EEPROM" in txt:
        return dict(out, status="decoded")
    if raw:
        return dict(out, status="raw", reason="decode-dimms decoded nothing; the raw SPD bytes are in inventory/spd-*.hex")
    # the first real line of decode-dimms' output, not the stick's "--- stderr ---" marker (run f78e, 2026-11-08)
    msg = next((l.strip() for l in txt.splitlines() if l.strip() and not l.startswith(("#", "---"))), "no output")
    why = [f"decode-dimms: {msg[:120]}", "SMBus adapters: " + (", ".join(smbus) or "none"),
           "SPD devices bound: " + (", ".join(devs) or "none")]
    return dict(out, status="unavailable", reason="; ".join(why))


def swap_off(swaps: Path = Path("/proc/swaps")) -> list[str]:
    """Switch off any swap that is on (the boot must not use the machine's disks; casper's 13swap is removed from the
    initramfs, this catches anything else). Returns what was on, by kind only (no device names)."""
    on = [l.split()[1] for l in (swaps.read_text().splitlines()[1:] if swaps.exists() else []) if l.strip()]
    if on:
        sh(["swapoff", "-a"], timeout=120)
    return on


def lockdown(path: Path = Path("/sys/kernel/security/lockdown")) -> str:
    """The kernel's lockdown mode ("none", "integrity", "confidentiality"), "" if the kernel has none: under lockdown
    /dev/mem, PCI config writes and debugfs are refused (what reading memory-controller registers would need)."""
    m = re.search(r"\[(\w+)\]", _read(path) or "")
    return m.group(1) if m else ""


def _dmi(k: str) -> str:
    try:
        return (DMI / k).read_text().strip()
    except OSError:
        return ""


# --- run plan --------------------------------------------------------------------------------------------------------
def question_gpus(qfile: Path) -> str:
    m = re.search(r'(?m)^gpus\s*=\s*"(\w+)"', qfile.read_text())
    return m.group(1) if m else "none"


def question_min_gpus(qfile: Path) -> int:
    """min_gpus = N in a question file: it needs at least N NVIDIA cards (gpu-p2p: 2)."""
    m = re.search(r'(?m)^min_gpus\s*=\s*(\d+)', qfile.read_text())
    return int(m.group(1)) if m else 1


def nvidia_count() -> int:
    rc, out = sh(["nvidia-smi", "-L"], timeout=60)
    return sum(1 for l in out.splitlines() if l.startswith("GPU ")) if rc == 0 else 0


def plan(qdir: Path, nvidia: bool, nvidia_reason: str, n_gpus: int | None = None) -> list[tuple[str, Path, str | None]]:
    """(question, file, skip reason or None), in profile order. n_gpus: NVIDIA cards the driver drives (None: unknown,
    no question skipped for it). A question this machine cannot answer is skipped with the reason, not run to fail
    (run d-9c93, 2026-10-31: gpu-p2p "failed" twice on a one-card machine)."""
    out = []
    for q in PROFILE:
        f = qdir / f"{q}.toml"
        if not f.exists():
            out.append((q, f, "question file missing"))
        elif question_gpus(f) != "none" and not nvidia:
            out.append((q, f, nvidia_reason))
        elif question_gpus(f) != "none" and n_gpus is not None and n_gpus < question_min_gpus(f):
            out.append((q, f, f"needs {question_min_gpus(f)} NVIDIA cards, this machine has {n_gpus}"))
        else:
            out.append((q, f, None))
    return out


def nvidia_state() -> tuple[bool, str]:
    rc, out = sh(["nvidia-smi", "-L"], timeout=60)
    if rc == 0 and "GPU" in out:
        return True, ""
    flavour = "legacy" if "legacy" in (OPT / "VERSION").read_text() else "open" if (OPT / "VERSION").exists() else "?"
    if flavour == "open" and _nvidia_pre_turing():
        if (SWAP_DIR / "FLAVOUR").exists():
            return False, "NVIDIA GPU older than Turing: measured after the switch to the proprietary modules (nvidia-swap)"
        return False, "NVIDIA GPU older than Turing: needs the legacy image (open modules support Turing and newer)"
    rc2, lspci = sh(["sh", "-c", "lspci -nn | grep -i nvidia"], timeout=30)
    if not lspci.strip():
        return False, "no NVIDIA GPU"
    return False, "NVIDIA GPU present but the driver did not load (card unsupported by the driver?)"


# --- NVIDIA module switch (old-GPU sprint, 2026-10-31) ----------------------------------------------------------------
# The open kernel modules drive Turing and newer, the proprietary ones Maxwell .. Ada. The stick boots one flavour and
# carries the other one's signed .ko files (chroot-setup.sh, same driver version, one user-space stack). After every
# step, cards the boot modules left undriven are measured in a second phase: the modules are swapped (the cards
# already measured are kept off the new driver with driver_override) and the CUDA questions and llama-bench run again
# for those cards only, into nvidia-swap/ (run format 1.7).
SWAP_DIR = Path("/usr/lib/rigsim/nvidia-swap")
SWAP_LOAD = ("nvidia", "nvidia-uvm", "nvidia-modeset")       # load order; nvidia-drm left out: nothing is displayed
SWAP_UNLOAD = ("nvidia_drm", "nvidia_peermem", "nvidia_modeset", "nvidia_uvm", "nvidia")


def norm_pci(bus: str) -> str:
    """'00000000:01:00.0' (nvidia-smi) and '0000:01:00.0' (sysfs) -> '0000:01:00.0'."""
    b = bus.strip().lower()
    m = re.match(r"^(?:([0-9a-f]+):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f])$", b)
    return f"{int(m.group(1) or '0', 16):04x}:{m.group(2)}:{m.group(3)}.{m.group(4)}" if m else b


def nvidia_pci_devices(devices: Path = Path("/sys/bus/pci/devices")) -> list[str]:
    """Every NVIDIA display device (vendor 0x10de, class 0x03xxxx), driven or not, by PCI address."""
    return sorted(d.name for d in devices.glob("*")
                  if _read(d / "vendor") == "0x10de" and (_read(d / "class") or "").startswith("0x03"))


def nvidia_driven() -> dict[str, str]:
    """PCI address -> nvidia-smi line ('name, memory, bus', as in hardware.gpus) for every card the loaded driver drives."""
    rc, out = sh(["nvidia-smi", "--query-gpu=name,memory.total,pci.bus_id", "--format=csv,noheader"], timeout=60)
    if rc != 0:
        return {}
    return {norm_pci(l.rsplit(",", 1)[-1]): l.strip() for l in out.splitlines() if l.count(",") >= 2}


def swap_modules(keep_off: list[str], log: Path | None, swap: Path = SWAP_DIR, sysfs: Path = Path("/sys"),
                 run=None) -> dict:
    """Replace the loaded NVIDIA modules with the other flavour. keep_off: cards measured already, kept off the new
    driver. Returns what happened; 'failed' says why it did not work (the cards are then recorded as not measured)."""
    run = run or (lambda cmd: sh(cmd, timeout=120))
    res = dict(flavour=_read(swap / "FLAVOUR") or "", version=_read(swap / "VERSION") or "", kept_off=keep_off)
    if not res["flavour"]:
        return dict(res, failed="the other NVIDIA module flavour is not on this stick")
    _kill_containers()
    run(["systemctl", "stop", "nvidia-persistenced"])
    for m in SWAP_UNLOAD:
        if not (sysfs / "module" / m).exists():
            continue
        rc, out = run(["rmmod", m])
        if rc != 0:                                    # a framebuffer console on nvidia-drm holds it: unbind, retry
            for v in sorted((sysfs / "class" / "vtconsole").glob("vtcon*")):
                if "frame buffer" in (_read(v / "name") or ""):
                    try:
                        (v / "bind").write_text("0")
                    except OSError:
                        pass
            rc, out = run(["rmmod", m])
        if rc != 0:
            return dict(res, failed=f"could not unload {m}: {out.strip()[:200]}")
    for pci in keep_off:
        try:
            (sysfs / "bus" / "pci" / "devices" / pci / "driver_override").write_text("none")
        except OSError as e:
            say(f"nvidia-swap: driver_override for {pci}: {e}", log)
    for m in SWAP_LOAD:
        f = next(iter(sorted(swap.glob(f"{m}.ko*"))), None)
        if f is None:
            return dict(res, failed=f"{m}.ko is not on this stick")
        _, deps = run(["modinfo", "-F", "depends", str(f)])
        for d in [x for x in deps.strip().split(",") if x]:
            run(["modprobe", d])
        rc, out = run(["insmod", str(f)])
        if rc != 0:
            return dict(res, failed=f"could not load {f.name}: {out.strip()[:200]}")
    run(["nvidia-modprobe", "-u", "-c=0"])
    return res


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
    r = run_cmd(["clinfo", "--list"], capture_output=True, text=True, env=env, timeout=120)
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
            r = run_cmd(["clpeak", "-p", str(plat), "-d", str(dev), *clpeak_tests()], capture_output=True,
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


# --- storage read speed (design/engine-primitives.md, item 4) ----------------------------------------------------------
# Read-only timings of the machine's internal drives with fio (--readonly refuses any write; O_DIRECT on the raw device).
# User decision 2026-11-04: read access yes, as long as no content and no identifying feature of a drive ends up in the
# report — so only timings and the drive's kind are kept (no serial, WWN/EUI, UUIDs, labels, partitions, filesystems,
# firmware; capacity rounded); nothing read is written anywhere. `rigsim.nodisks` on the kernel command line skips it.
SYS_BLOCK = Path("/sys/block")
STORAGE_JOBS = [  # name, fio rw, block size, queue depth
    ("seq-1m-qd1", "read", "1M", 1), ("seq-1m-qd8", "read", "1M", 8), ("seq-1m-qd32", "read", "1M", 32),
    ("rand-4k-qd1", "randread", "4k", 1), ("rand-4k-qd32", "randread", "4k", 32), ("rand-4k-qd128", "randread", "4k", 128),
    ("rand-128k-qd8", "randread", "128k", 8),
    ("rand-4m-qd1", "randread", "4M", 1), ("rand-4m-qd4", "randread", "4M", 4),
    ("rand-16m-qd1", "randread", "16M", 1), ("rand-16m-qd4", "randread", "16M", 4)]
STORAGE_JOB_S = 8
STORAGE_MAX_DRIVES = 4
CAPACITY_CLASSES_GB = (16, 32, 64, 128, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000)


def capacity_class(gb: float) -> int:
    """The nearest common drive size — a class, not the exact capacity (which could single out a drive)."""
    return min(CAPACITY_CLASSES_GB, key=lambda c: abs(c - gb))


def stick_disks(mounts: Path = Path("/proc/mounts")) -> set[str]:
    """The disks the stick itself is on (data partition, boot medium): never measured."""
    out = set()
    for line in (mounts.read_text().splitlines() if mounts.exists() else []):
        f = line.split()
        if len(f) > 1 and f[0].startswith("/dev/") and f[1].startswith(("/mnt/rigsim", "/cdrom", "/run/live", "/isodevice")):
            rc, parent = sh(["lsblk", "-no", "PKNAME", f[0]], timeout=10)
            out.add(parent.strip() or Path(f[0]).name)
    return out


def internal_drives(exclude: set[str], sysblock: Path = SYS_BLOCK) -> list[dict]:
    """Internal NVMe / SATA / SAS drives of at least 8 GB, without anything that identifies the drive."""
    out = []
    for d in sorted(sysblock.glob("*")):
        name = d.name
        if name.startswith(("loop", "ram", "zram", "sr", "dm-", "md", "fd", "nbd")) or name in exclude:
            continue
        dev = d / "device"
        path = str(dev.resolve()) if dev.exists() else ""
        if (_read(d / "removable") or "0") != "0" or "/usb" in path:
            continue
        sectors = int(_read(d / "size") or 0)
        gb = sectors * 512 / 1e9
        if gb < 8:
            continue
        kind = "nvme" if name.startswith("nvme") else ("sata" if "/ata" in path else "other")
        info = dict(drive=name, kind=kind, capacity_class_gb=capacity_class(gb),
                    rotational=(_read(d / "queue" / "rotational") or "0") == "1",
                    model=re.sub(r"[^A-Za-z0-9 ._-]", "", _read(dev / "model") or "").strip(),
                    logical_block=int(_read(d / "queue" / "logical_block_size") or 512))
        if kind == "nvme":                                # the controller's PCIe link
            pci = dev.resolve().parent if dev.exists() else None
            for p in (dev / "device", pci):
                if p and (p / "current_link_speed").exists():
                    info["link"] = dict(speed=_read(p / "current_link_speed"), width=_read(p / "current_link_width"),
                                        max_speed=_read(p / "max_link_speed"), max_width=_read(p / "max_link_width"))
                    break
        out.append(info)
    return out[:STORAGE_MAX_DRIVES]


def drive_temp_c(name: str) -> float | None:
    """The drive's temperature from hwmon (nvme / drivetemp), read-only; None where there is none."""
    ctrl = re.sub(r"n\d+$", "", name) if name.startswith("nvme") else name     # nvme0n1 -> nvme0
    for t in list(SYS_BLOCK.glob(f"{name}/device/hwmon*/temp1_input")) + \
             list(Path("/sys/class/nvme").glob(f"{ctrl}/hwmon*/temp1_input")):
        v = _read(t)
        if v and v.lstrip("-").isdigit():
            return int(v) / 1000
    return None


ZERO_SAMPLES = 256


def zero_share(dev: str, samples: int = ZERO_SAMPLES, block: int = 1 << 20, path: Path | None = None) -> float | None:
    """Share of blocks, sampled evenly over the device, that read as all zeros: never written or trimmed blocks, which
    a controller answers without reading flash (run f78e: two identical drives 0.5 vs 4.9 GB/s). Only the share is
    kept, nothing read leaves this function. None if the device cannot be read."""
    p = path or Path("/dev") / dev
    try:
        fd = os.open(p, os.O_RDONLY)
    except OSError:
        return None
    try:
        size = os.lseek(fd, 0, os.SEEK_END)
        n = max(1, min(samples, size // block))
        zero = bytes(block)
        hits = 0
        for i in range(n):
            off = (size // n * i) // block * block
            b = os.pread(fd, block, off)
            hits += b == zero[:len(b)]
        return round(hits / n, 3)
    except OSError:
        return None
    finally:
        os.close(fd)


def parse_fio(out: str) -> dict:
    """The numbers of one fio job (--output-format=json): read rate, IOPS, latency mean and percentiles."""
    try:
        j = json.loads(out[out.index("{"):])
        r = j["jobs"][0]["read"]
    except (ValueError, KeyError, IndexError):
        return {}
    clat = r.get("clat_ns", {}).get("percentile", {})
    pct = lambda p: clat.get(f"{p:.6f}")
    return dict(bytes_per_s=r.get("bw_bytes", 0), iops=r.get("iops", 0), lat_mean_us=r.get("lat_ns", {}).get("mean", 0) / 1e3,
                lat_p50_us=(pct(50) or 0) / 1e3, lat_p99_us=(pct(99) or 0) / 1e3, ios=r.get("total_ios", 0))


def fio_cmd(dev: str, rw: str, bs: str, qd: int, seconds: int, direct: bool = True) -> list[str]:
    return ["fio", "--readonly", "--name=rigsim", f"--filename=/dev/{dev}", f"--rw={rw}", f"--bs={bs}",
            f"--iodepth={qd}", "--ioengine=libaio", f"--direct={1 if direct else 0}", "--time_based",
            f"--runtime={seconds}", "--norandommap", "--randrepeat=0", "--group_reporting", "--output-format=json"] + \
           ([] if direct else ["--size=4G", "--invalidate=1"])


def run_storage(rundir: Path, log: Path, cmdline: str) -> dict:
    if "rigsim.nodisks" in cmdline.split():
        return dict(skipped="rigsim.nodisks on the kernel command line")
    if not shutil.which("fio"):
        return dict(skipped="fio not in the system image")
    drives = internal_drives(stick_disks())
    if not drives:
        return dict(skipped="no internal drive of 8 GB or more")
    d = rundir / "storage"
    d.mkdir(parents=True, exist_ok=True)
    runs, rc_all = {}, 0
    for info in drives:
        dev = info["drive"]
        info["temp_c_before"] = drive_temp_c(dev)
        jobs = [(n, rw, bs, qd, True) for n, rw, bs, qd in STORAGE_JOBS] + [("seq-1m-buffered", "read", "1M", 8, False)]
        for k, (n, rw, bs, qd, direct) in enumerate(jobs, 1):
            STATUS.set(f"storage {dev} {n} {k}/{len(jobs)}")
            if not direct:
                sh(["sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"], timeout=60)
            t0 = time.time()
            rc, out = sh(fio_cmd(dev, rw, bs, qd, STORAGE_JOB_S, direct), timeout=STORAGE_JOB_S * 4 + 60)
            res = parse_fio(out) if rc == 0 else {}
            runs[f"{dev}-{n}"] = dict(rc=rc, seconds=round(time.time() - t0, 1), **res,
                                      **({} if rc == 0 else dict(error=out.strip().splitlines()[-1][:200] if out.strip() else "")))
            rc_all |= rc != 0
        info["temp_c_after"] = drive_temp_c(dev)
        info["zero_share"] = zero_share(dev)
        say(f"storage {dev}: {info['kind']}, ~{info['capacity_class_gb']} GB, "
            f"seq 1M QD32 {runs.get(f'{dev}-seq-1m-qd32', {}).get('bytes_per_s', 0) / 1e9:.2f} GB/s", log)
    (d / "drives.json").write_text(json.dumps(drives, indent=1))
    (d / "results.json").write_text(json.dumps(runs, indent=1))
    return dict(rc=1 if rc_all else 0, drives=drives, runs=runs)


def run_mlc(data: Path, rundir: Path, log: Path) -> dict:
    mlc = data / "seed" / "tools" / "mlc" / "mlc"
    if not mlc.exists():
        return dict(skipped="Intel MLC not in the seed (build without MLC_URL or download failed)")
    d = rundir / "mlc"
    d.mkdir(parents=True, exist_ok=True)
    sh(["modprobe", "msr"], timeout=20)
    res = {}
    small = meminfo().get("MemTotal", 0) < MLC_SMALL_GIB * GIB
    buf = [f"-b{MLC_SMALL_BUFFER_KIB}"] if small else []
    huge = reserve_hugepages()
    say(f"mlc: hugepages {huge}", log)
    try:
        for k, (name, args) in enumerate(MLC_RUNS, 1):
            STATUS.set(f"{name} {k}/{len(MLC_RUNS)}")
            t0 = time.time()
            rc, out = sh([str(mlc), *args, *buf], timeout=MLC_TIMEOUT_S)
            (d / f"{name}.txt").write_text(scrub(out))
            res[name] = dict(rc=rc, seconds=round(time.time() - t0), values=parse_mlc(out))
            say(f"mlc {name}: rc={rc} ({time.time() - t0:.0f} s)", log)
    finally:
        release_hugepages(huge)
    prefetch = "unable" in " ".join((d / f).read_text().lower() for f in ("idle_latency.txt",) if (d / f).exists())
    return dict(rc=0 if all(v["rc"] == 0 for v in res.values()) else 1, runs={k: dict(rc=v["rc"], seconds=v["seconds"])
                for k, v in res.items()}, prefetchers_left_on=prefetch or bool(secure_boot()),
                hugepages=huge, **(dict(buffer_kib=MLC_SMALL_BUFFER_KIB) if small else {}))


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


def vulkan_integrated(summary: str) -> set[int]:
    """Indices of integrated GPUs in `vulkaninfo --summary` (they use the machine's RAM)."""
    out, idx = set(), None
    for line in summary.splitlines():
        m = re.match(r"\s*GPU(\d+):", line)
        if m:
            idx = int(m.group(1)); continue
        if idx is not None and re.match(r"\s*deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", line):
            out.add(idx)
    return out


GPU_LOST = ("ErrorDeviceLost", "device lost on")


def device_lost(text: str) -> str:
    """'' unless a GPU was lost (reset) during the command; then why, as the driver put it (run cfed, Vega 8)."""
    if not any(s in text for s in GPU_LOST):
        return ""
    cases = sum(1 for l in text.splitlines() if " runs - " in l)
    if "Not enough memory for command submission" in text:
        why = "out of GPU memory, the GPU was reset"
    elif "guilty of a hard recovery" in text:
        why = "the GPU was reset while running this (a hang or the driver's job timeout)"
    elif "innocent" in text:
        why = "the GPU context was lost after a reset caused elsewhere"
    else:
        why = "the GPU was lost"
    return why + (f"; {cases} case(s) measured before" if cases else "")


def item_ok(v: dict) -> bool:
    """A llama.cpp item that does not make its step fail: done, skipped with a reason, or ended by a limit of the
    machine (memory limit, GPU reset) — recorded with the reason."""
    measured_some = "; 0 case(s)" not in str(v.get("time_limit", "; 0 case(s)"))
    return bool(v.get("skipped") or v.get("rc") == 0 or v.get("out_of_memory") or v.get("device_lost")
                or (v.get("time_limit") and measured_some))


VK_MEMORY = re.compile(r"Device memory:\s*(\d+) MB \((\d+) MB free\)")


def vulkan_memory(tbo: Path, vi: int) -> int | None:
    """Free memory of Vulkan device vi in MiB as llama.cpp sees it (test-backend-ops with an op that matches nothing:
    it lists the device and runs no test). None if it cannot tell."""
    if not tbo.exists():
        return None
    try:
        r = run_cmd([str(tbo), "perf", "-o", "RIGSIM_NONE", "-b", "Vulkan0"], timeout=120,
                    env=dict(os.environ, GGML_VK_VISIBLE_DEVICES=str(vi)))
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = VK_MEMORY.search((r.stdout or "") + (r.stderr or ""))
    return int(m.group(2)) if m else None


def model_fits_gpu(m: Path, free_mib: int | None) -> str:
    """'' if the model fits the GPU's free memory (file size x 1.2 + 0.5 GiB for context and buffers), else why. On the
    Vega 8 (3996 MB) Llama 2 7B Q4_0 (3.56 GiB) ran out of GPU memory and reset the GPU (run cfed)."""
    if free_mib is None:
        return ""
    need = m.stat().st_size * 1.2 + 0.5 * GIB
    return "" if need <= free_mib * (1 << 20) else \
        f"needs about {need / GIB:.1f} GiB of GPU memory, the device has {free_mib / 1024:.1f} GiB free"


def model_fits(m: Path) -> str:
    """'' if the model fits in the RAM available now (file size x 1.3 + 1.5 GiB), else the reason it is skipped. An
    integrated GPU's buffers are not charged to the step's cgroup, so they need this check before (run 05ea: Llama 2
    7B on a Vega 8 with 7.3 GiB RAM ran out of memory and reset the GPU)."""
    need, avail = m.stat().st_size * 1.3 + 1.5 * GIB, meminfo().get("MemAvailable", 0)
    return "" if need <= avail else f"needs about {need / GIB:.1f} GiB of RAM, {avail / GIB:.1f} GiB available"


def llama_models(data: Path) -> list[Path]:
    return sorted((data / "seed" / "models").glob("*.gguf"), key=lambda p: p.stat().st_size)


def _run_env(cmd: list[str], timeout: float, env: dict) -> tuple[int, str]:
    try:
        r = run_cmd(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout + (("\n--- stderr ---\n" + r.stderr) if r.stderr.strip() else "")
    except subprocess.TimeoutExpired as e:
        return _timed_out(e)
    except OSError as e:
        return 127, f"{type(e).__name__}: {e}"


def backend_ops_plan(binary: str, backend: str) -> list[tuple[str, list[str]]]:
    """(key, command) per operation; perf mode with sql output — the format that carries the timings (csv has only
    the support columns); the caller retries without --output on builds that lack it."""
    out = []
    for op in BACKEND_OPS:
        flt = ["-p", BACKEND_OPS_FILTER[op]] if op in BACKEND_OPS_FILTER else []
        out.append((f"ops-{op.lower()}", [binary, "perf", "-o", op, *flt, "-b", backend, "--output", "sql"]))
    return out


def run_backend_ops(run, d: Path, prefix: str, binary: str, backend: str, log: Path,
                    onednn: "OneDnnGuard | None" = None) -> dict:
    """run(cmd, timeout[, env]) -> (rc, output). Older builds lack --output: then plain console output. A build that
    does not know the backend name: once more without -b (every backend it sees, CPU included). onednn (SYCL): the
    device's OneDnnGuard decides on oneDNN before anything else."""
    runs = {}
    ops = backend_ops_plan(binary, backend)
    for k, (key, cmd) in enumerate(ops, 1):
        STATUS.set(f"test-backend-ops {prefix} {key[4:]} {k}/{len(ops)}")
        t0 = time.time()
        extra = {}
        if onednn:
            rc, out, extra = onednn.run(f"{prefix}-{key}", d / f"{prefix}-{key}.onednn-failed.txt",
                                        lambda env, cmd=cmd: run(cmd, BACKEND_OPS_TIMEOUT_S, env))
        else:
            rc, out = run(cmd, BACKEND_OPS_TIMEOUT_S)
        if rc not in (0, 127, TIME_LIMIT_RC):                # not runnable or out of time: no point retrying
            cmd = cmd[:-2]; rc, out = run(cmd, BACKEND_OPS_TIMEOUT_S)
        if rc not in (0, 127, TIME_LIMIT_RC):
            i = cmd.index("-b"); cmd = cmd[:i] + cmd[i + 2:]; rc, out = run(cmd, BACKEND_OPS_TIMEOUT_S)
        (d / f"{prefix}-{key}.txt").write_text(scrub(" ".join(cmd) + "\n" + out))
        lost = device_lost(out) if rc != 0 else ""
        if lost:
            say(f"test-backend-ops {prefix} {key}: {lost}", log)
        if rc == TIME_LIMIT_RC:                          # what ran until the limit is measured (run f78e: FA on an iGPU)
            cases = sum(1 for l in out.splitlines() if " runs - " in l)
            extra = dict(extra, time_limit=f"stopped at the {BACKEND_OPS_TIMEOUT_S} s limit; {cases} case(s) measured before")
            say(f"test-backend-ops {prefix} {key}: {extra['time_limit']}", log)
        runs[f"{prefix}-{key}"] = dict(rc=rc, seconds=round(time.time() - t0, 1), **extra,
                                        **(dict(device_lost=lost) if lost else {}))
    say(f"test-backend-ops {prefix}: {sum(1 for v in runs.values() if v['rc'] == 0)}/{len(runs)} ops", log)
    return runs


def run_llama(data: Path, rundir: Path, cfg_image: str, nvidia: bool, log: Path, cuda_only: bool = False) -> dict:
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
            r = run_cmd(cmd, capture_output=True, text=True, timeout=LLAMA_TIMEOUT_S,
                               env=dict(os.environ, **(env or {})))
            rc, out, err = r.returncode, r.stdout, r.stderr
            oom = bool(getattr(r, "oom", False)) and rc != 0
        except subprocess.TimeoutExpired:
            rc, out, err, oom = -9, "", f"timeout after {LLAMA_TIMEOUT_S} s", False
        (d / f"{key}.jsonl").write_text(scrub(out))
        (d / f"{key}.log").write_text(scrub(err))
        lost = device_lost(out + err) if rc != 0 else ""
        runs[key] = dict(rc=rc, seconds=round(time.time() - t0), **(dict(out_of_memory=True) if oom else {}),
                         **(dict(device_lost=lost) if lost else {}))
        say(f"llama {key}: rc={rc} ({time.time() - t0:.0f} s)", log)

    if nvidia and (seed / "cuda" / "llama-bench").exists():
        rc, out = sh(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], timeout=60)
        for gi in [l.strip() for l in out.splitlines() if l.strip().isdigit()]:
            for m in models:
                one(f"cuda-gpu{gi}-{m.stem}", ["docker", "run", "--rm", *docker_mem(), "--gpus", "all", "--network", "none",
                    "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "-e", f"CUDA_VISIBLE_DEVICES={gi}",
                    "-v", f"{seed / 'cuda'}:/llama:ro", "-v", f"{m.parent}:/models:ro", cfg_image,
                    "/llama/llama-bench", "-m", f"/models/{m.name}", "-ngl", "99", *LLAMA_ARGS])
            if (seed / "cuda" / "test-backend-ops").exists():
                docker = ["docker", "run", "--rm", *docker_mem(), "--gpus", "all", "--network", "none", "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID",
                          "-e", f"CUDA_VISIBLE_DEVICES={gi}", "-v", f"{seed / 'cuda'}:/llama:ro", cfg_image]
                runs.update(run_backend_ops(lambda c, t: sh(docker + c, timeout=t), d, f"cuda-gpu{gi}",
                                            "/llama/test-backend-ops", "CUDA0", log))
    # forks tuned for Pascal (build.sh LLAMA_FORKS): on the cards of the architectures they were built for, as their own
    # backend "cuda+<fork>", same models and arguments as upstream
    forks = sorted(p for p in (seed / "forks").glob("*") if (p / "llama-bench").exists()) if nvidia else []
    if forks:
        _, out = sh(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"], timeout=60)
        caps = [(i.strip(), c.strip().replace(".", "")) for i, c in (l.split(",", 1) for l in out.splitlines() if "," in l)]
        for fk in forks:
            archs = (_read(fk / "ARCHS") or "60").split()
            for gi, cap in caps:
                if cap not in archs:
                    continue
                for m in models:
                    one(f"cuda+{fk.name}-gpu{gi}-{m.stem}", ["docker", "run", "--rm", *docker_mem(), "--gpus", "all", "--network", "none",
                        "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID", "-e", f"CUDA_VISIBLE_DEVICES={gi}",
                        "-v", f"{fk}:/llama:ro", "-v", f"{m.parent}:/models:ro", cfg_image,
                        "/llama/llama-bench", "-m", f"/models/{m.name}", "-ngl", "99", *LLAMA_ARGS])
    vk = seed / "vulkan" / "llama-bench"
    if vk.exists() and not cuda_only:
        _, summary = sh(["vulkaninfo", "--summary"], timeout=60)
        (d / "vulkaninfo-summary.txt").write_text(scrub(summary))
        tbo = seed / "vulkan" / "test-backend-ops"
        integrated = vulkan_integrated(summary)
        for vi, name in vulkan_devices(summary):
            free = vulkan_memory(tbo, vi)
            say(f"llama vulkan-dev{vi}: {free if free is not None else '?'} MiB GPU memory free", log)
            for m in models:
                why = (model_fits(m) if vi in integrated else "") or model_fits_gpu(m, free)
                if why:
                    runs[f"vulkan-dev{vi}-{m.stem}"] = dict(skipped=why)
                    say(f"llama vulkan-dev{vi}-{m.stem}: skipped ({why})", log)
                    continue
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
            why = model_fits(m)
            if why:
                runs[f"cpu-{m.stem}"] = dict(skipped=why)
                say(f"llama cpu-{m.stem}: skipped ({why})", log)
                continue
            one(f"cpu-{m.stem}", [str(vk), "-m", str(m), "-ngl", "0", *LLAMA_ARGS], env={"GGML_VK_VISIBLE_DEVICES": ""})
    if not runs:
        return dict(skipped="no usable llama.cpp backend")
    return dict(rc=0 if all(item_ok(v) for v in runs.values()) else 1, runs=runs)


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
    for d in ("h2d", "d2h") + (("bidir",) if kind == "rocm" else ()):    # the size sweep (engine primitives, item 3)
        for mem in ("pinned", "pageable"):
            out.append((f"xfer-{d}-{mem}", [probe, "xfer", "--dir", d, "--mem", mem, "--seconds", "0.2"]))
    if kind == "rocm":                                # the CUDA probe's HIP build has the overlap mode too
        out.append(("overlap", [probe, "overlap", "--mib", "256", "--seconds", "2"]))
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
    base = ["docker", "run", "--rm", *docker_mem(), "--network", "none", *devs, "-v", f"{data / 'seed' / 'models'}:/models:ro", tag]
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
    if not devices:                  # e.g. an iGPU the runtime does not support: not a failure of the stick
        return dict(res, skipped=f"the {'oneAPI' if kind == 'oneapi' else 'ROCm'} runtime sees no GPU on this machine (info.jsonl)")
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
        rc, _ = sh(["docker", "run", "-d", "--name", c, *docker_mem(), "--network", "none", *devs, *envargs,
                    "-v", f"{data / 'seed' / 'models'}:/models:ro", tag, "sleep", "infinity"], timeout=300)
        if rc != 0:
            runs[f"dev{i}"] = dict(rc=rc, error="container did not start"); continue
        plan = vendor_plan(kind) if have_probe else []
        plan += [(f"llama-{m.stem}", ["/rigsim/llama-bench", "-m", f"/models/{m.name}", "-ngl", "99", *LLAMA_ARGS]) for m in models]
        def ex(cmd, t, env=None, c=c):
            return sh(["docker", "exec", *[x for k, v in (env or {}).items() for x in ("-e", f"{k}={v}")], c,
                       "/rigsim/run.sh", *cmd], timeout=t)
        guard = OneDnnGuard() if kind == "oneapi" else None     # llama.cpp steps only; the probe has no oneDNN
        for k, (key, cmd) in enumerate(plan, 1):
            STATUS.set(f"dev{i} {key} {k}/{len(plan)}")
            t0 = time.time()
            t = LLAMA_TIMEOUT_S if key.startswith("llama") else PROBE_TIMEOUT_S
            extra = {}
            if guard and key.startswith("llama"):
                rc, out, extra = guard.run(f"dev{i}-{key}", d / f"dev{i}-{key}.onednn-failed.txt",
                                           lambda env, cmd=cmd, t=t: ex(cmd, t, env))
                if "retry" in extra:
                    say(f"oneapi dev{i} {key}: oneDNN failed, without oneDNN rc={rc}; oneDNN off for this device", log)
            else:
                rc, out = ex(cmd, t)
            (d / f"dev{i}-{key}.jsonl").write_text(scrub(out))
            runs[f"dev{i}-{key}"] = dict(rc=rc, seconds=round(time.time() - t0, 1), **extra)
        if "test-backend-ops" in meta.split():
            runs.update(run_backend_ops(ex, d, f"dev{i}", "/rigsim/test-backend-ops",
                                        "ROCm0" if kind == "rocm" else "SYCL0", log, onednn=guard))
        sh(["docker", "rm", "-f", c], timeout=120)
        say(f"{kind} device {i} ({dev.get('name')}): {sum(1 for k in runs if k.startswith(f'dev{i}-'))} runs"
            + (f", HSA_OVERRIDE_GFX_VERSION={ov}" if ov else "") + ("" if have_probe else ", probe not in the image"), log)
    return dict(res, rc=0 if runs and all(item_ok(v) for v in runs.values()) else 1, runs=runs)   # as run_llama


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
                    cmd = ["docker", "run", "--rm", *docker_mem(), "--network", "none", *AMD_DEVICES, "-v", f"{b}:{b}:ro",
                           "-w", str(script.parent), "-e", "HOME=/tmp", rocm_tag, "/rigsim/run.sh", *cmd]
                try:
                    rr = run_cmd(cmd, capture_output=True, text=True, cwd=str(script.parent), env=env,
                                        timeout=BLENDER_TIMEOUT_S)
                    rc, out, err = rr.returncode, rr.stdout, rr.stderr
                except subprocess.TimeoutExpired:
                    rc, out, err = -9, "", f"timeout after {BLENDER_TIMEOUT_S} s"
                (d / f"{key}.log").write_text(scrub(out + ("\n--- stderr ---\n" + err if err.strip() else "")))
                nodev = re.search(r"No matching device of type \w+ found", out + err)
                if rc == 0 and "Traceback (most recent call last)" in out + err:
                    rc = 1                                      # the script failed, Blender still exits 0
                say(f"blender {key}: rc={rc} ({time.time() - t0:.0f} s)", log)
                if t != "CPU" and (nodev or (rc != 0 and i == 0)):
                    # guessed type this GPU / build does not offer (run d-9c93: oneAPI on a UHD 770 — Blender's
                    # script found no such device and Blender exited 0 after a second)
                    unavailable.append(t)
                    say(f"blender {t}: not available here" + (f" ({nodev.group(0)})" if nodev else ""), log)
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
    r = run_cmd([str(cli), "devices", "--blender-version", ver], capture_output=True, text=True, env=env,
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
            rr = run_cmd([str(cli), "benchmark", "--blender-version", ver, "--device-type", t, "--json",
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
def stick_version() -> str:
    """The stick's release (build.sh writes stick/VERSION to /opt/rigsim/STICK_VERSION)."""
    return _read(OPT / "STICK_VERSION") or "unknown"


def notice(seconds: int = NOTICE_S) -> bool:
    """True = go on; False = a key was pressed (abort, nothing measured)."""
    print("\n" * 2 + "=" * 78)
    print(f" rigsim hardware stick {stick_version()}")
    print(" Measures this machine's CPU, memory and GPUs, then powers it off.")
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


DOCKER_STORAGE = {"features": {"containerd-snapshotter": False}, "storage-driver": "overlay2"}


def docker_config(d: dict, root: Path) -> dict:
    """dockerd's settings on the stick (and at flash time, flash.sh writes the same): data on the stick, and the
    classic overlay2 image store. The containerd image store keeps every image twice — the compressed layers (21 GiB)
    beside the unpacked ones (62 GiB): 83 GiB, more than a 128 GB stick holds next to the seed (2026-11-07)."""
    return dict(d, **DOCKER_STORAGE, **{"data-root": str(root)})


def start_docker(data: Path, log: Path) -> dict:
    """Docker and containerd keep everything on the data partition. The live system's root is an overlay; an
    overlay snapshot cannot sit on it ("not supported as upperdir"), so both stores move to the stick's ext4:
    Docker by data-root, containerd (its image store and snapshots, used by Docker's containerd image store) by a
    bind mount of /var/lib/containerd. A container is started once to prove it works."""
    sh(["systemctl", "stop", "docker", "docker.socket", "containerd"], timeout=120)
    store = mount_store(data, log)
    root = data / "docker"
    croot = data / "containerd"
    root.mkdir(exist_ok=True); croot.mkdir(exist_ok=True)
    cl = Path("/var/lib/containerd"); cl.mkdir(parents=True, exist_ok=True)
    if subprocess.run(["mountpoint", "-q", str(cl)]).returncode != 0:
        sh(["mount", "--bind", str(croot), str(cl)])
    cfg = Path("/etc/docker/daemon.json")
    d = docker_config(json.loads(cfg.read_text()) if cfg.exists() else {}, root)
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
    res = dict(ok=rc == 0, image=tags[0], storage=driver.strip()[:200], store=store)
    if rc != 0:
        res["error"] = out.strip()[-400:]
    say(f"docker: {'works' if rc == 0 else 'BROKEN — container questions will fail'} ({res['storage'][:60]})", log)
    return res


STORE_MNT = Path("/run/rigsim-store")


def mount_store(data: Path, log: Path | None) -> str:
    """The Docker / containerd store flash.sh built on the user's machine (store.img, ext4 in one file): mounted and
    bound to data/docker and data/containerd, so everything below works on it unchanged. Without it (or if it does
    not check clean) the store lives on the partition itself, as before."""
    img = data / "store.img"
    if not img.exists():
        return "partition"
    STORE_MNT.mkdir(parents=True, exist_ok=True)
    rc, out = sh(["e2fsck", "-p", str(img)], timeout=600)
    if rc not in (0, 1) or sh(["mount", "-o", "loop", str(img), str(STORE_MNT)], timeout=120)[0] != 0:
        say(f"docker: store.img not usable (e2fsck rc={rc}); using the partition", log)
        return "partition (store.img not usable)"
    for sub in ("docker", "containerd"):
        (STORE_MNT / sub).mkdir(exist_ok=True)
        (data / sub).mkdir(exist_ok=True)
        sh(["mount", "--bind", str(STORE_MNT / sub), str(data / sub)])
    say("docker: using the image store built at flash time (store.img)", log)
    return "store.img"


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
            if SKIP.is_set():
                p.kill(); p.wait(); raise StepSkipped()
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
                           "--config", str(cfg)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          start_new_session=True)
    if StepMemory.current:
        StepMemory.current.join(pr.pid)
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
            if SKIP.is_set():                                  # the operator skips: stop, then restore the machine
                _killpg(pr); _kill_containers()
                subprocess.run([sys.executable, str(OPT / "runner" / "hwq.py"), "restore", "--dir", str(d)],
                               capture_output=True, timeout=300)
                d.mkdir(parents=True, exist_ok=True)
                (d / "runner.log").write_text(scrub("".join(out_parts)) + "\nskipped by the operator\n")
                raise StepSkipped()
            STATUS.set(question_progress(d))
    reader.join(timeout=10)
    rc, out = (pr.returncode if pr.returncode is not None else -9), "".join(out_parts)
    d.mkdir(parents=True, exist_ok=True)
    (d / "runner.log").write_text(scrub(out))
    counts = json.loads(sh([sys.executable, str(OPT / "runner" / "hwq.py"), "status", "--dir", str(d)])[1] or "{}")
    say(f"{q}: rc={rc} {counts} ({time.time() - t0:.0f} s)", log)
    return dict(rc=rc, items=counts, seconds=round(time.time() - t0))


def nvidia_phase2(data: Path, rundir: Path, cfg: Path, cuda_image: str, man: dict, run: dict, log: Path) -> dict | None:
    """After every step: NVIDIA cards the boot modules did not drive are measured with the other flavour. None when
    every NVIDIA card was driven (nothing to do)."""
    driven = nvidia_driven()
    left = [p for p in nvidia_pci_devices() if p not in driven]
    if not left:
        return None
    say(f"nvidia-swap: cards the boot modules do not drive: {left}", log)
    STATUS.n += 1
    STATUS.begin("nvidia-swap")
    res = dict(swap_modules(sorted(driven), log), pci=left)
    if res.get("failed"):
        say(f"nvidia-swap: {res['failed']} — those cards are not measured", log)
        return res
    now = nvidia_driven()
    res["gpus"] = [line for pci, line in sorted(now.items())]
    missing = [p for p in left if p not in now]
    if missing:
        res["not_driven"] = missing
    if not now:
        res["failed"] = "the other modules loaded, but they drive none of the cards either"
        say(f"nvidia-swap: {res['failed']}", log)
        return res
    say(f"nvidia-swap: {res['flavour']} modules {res['version']} drive {res['gpus']}", log)
    # questions as for the first phase, for the cards the new driver drives (gpu-p2p needs two of them)
    gpu_q = [(q, f) for q, f, skip in plan(OPT / "questions", True, "", len(now)) if not skip and question_gpus(f) != "none"]
    STATUS.n += len(gpu_q) + 1
    sd = rundir / "nvidia-swap"
    sd.mkdir(parents=True, exist_ok=True)
    with NvidiaSampler(sd / "samples-nvidia.csv", True):
        steps = [(f"nvidia-swap/{q}", q, lambda q=q, f=f: run_question(q, f, sd, cfg, log)) for q, f in gpu_q]
        steps.append(("nvidia-swap/llama-bench", "llama-bench",
                      lambda: run_llama(data, sd, cuda_image, True, log, cuda_only=True)))
        for name, kind, step in steps:
            STATUS.begin(name, skippable=True)
            with StepMemory(name, log) as mem:                # the same memory limit as the first phase
                try:
                    run["questions"][name] = step()
                except StepSkipped:
                    run["questions"][name] = dict(skipped=OPERATOR_SKIP)
                    run.setdefault("operator_skips", []).append(name)
            record_memory(run["questions"][name], mem, log)
            run["questions"][name]["tools"] = tools_for(kind, man)
            (rundir / "run.json").write_text(json.dumps(run, indent=1))
    return res


# --- format and versions ---------------------------------------------------------------------------------------------
# The run folder's format (results/FORMAT.md). Minor versions only add: fields and files are never removed, renamed
# or retyped, so every reader of 1.x reads every 1.y run. A change that cannot be additive is a new major version.
FORMAT = dict(name="rigsim-stick-run", version="1.7")


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
        run["swap_was_on"] = swap_off()                # by type ("partition", "file"); nothing may use the disks
        if run["swap_was_on"]:
            say(f"swap was on ({run['swap_was_on']}), switched off", log)
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
            qplan = list(plan(OPT / "questions", nvidia, why, nvidia_count() if nvidia else None))
            STATUS.n = 2 + len(qplan) + 8        # inventory, docker, the questions, the eight steps below
            threading.Thread(target=operator_keys, daemon=True).start()
            for q, f, skip in qplan:
                STATUS.begin(q, skippable=not skip)
                if skip:
                    run["questions"][q] = dict(skipped=skip); say(f"{q}: skipped ({skip})", log); continue
                with StepMemory(q, log) as mem:
                    try:
                        run["questions"][q] = run_question(q, f, rundir, cfg, log)
                    except StepSkipped:
                        run["questions"][q] = dict(skipped=OPERATOR_SKIP)
                        run.setdefault("operator_skips", []).append(q)
                record_memory(run["questions"][q], mem, log)
                run["questions"][q]["tools"] = tools_for(q, man)
                (rundir / "run.json").write_text(json.dumps(run, indent=1))
            for name, step in (("mlc", lambda: run_mlc(data, rundir, log)),
                               ("storage", lambda: run_storage(rundir, log, cmdline)),
                               ("opencl", lambda: run_opencl(rundir, log)),
                               ("vkprobe", lambda: run_vulkan_probe(rundir, log)),
                               ("rocm", lambda: run_vendor("rocm", data, rundir, log)),
                               ("oneapi", lambda: run_vendor("oneapi", data, rundir, log)),
                               ("llama-bench", lambda: run_llama(data, rundir, cuda_image, nvidia, log)),
                               ("blender", lambda: run_blender(data, rundir, log))):
                STATUS.begin(name, skippable=True)
                with StepMemory(name, log) as mem:
                    try:
                        run["questions"][name] = step()
                    except StepSkipped:
                        run["questions"][name] = dict(skipped=OPERATOR_SKIP)
                        run.setdefault("operator_skips", []).append(name)
                record_memory(run["questions"][name], mem, log)
                run["questions"][name]["tools"] = tools_for(name, man, dict(mlc_version=mlc_version(rundir)) if name == "mlc" else None)
                if run["questions"][name].get("skipped"):
                    say(f"{name}: skipped ({run['questions'][name]['skipped']})", log)
                (rundir / "run.json").write_text(json.dumps(run, indent=1))
        swap = nvidia_phase2(data, rundir, cfg, cuda_image, man, run, log)
        if swap is not None:
            run["nvidia_swap"] = swap
            (rundir / "run.json").write_text(json.dumps(run, indent=1))
        ok = all(step_ok(v) for v in run["questions"].values()) and not (run.get("nvidia_swap") or {}).get("failed")
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
    for m in (data / "docker", data / "containerd", STORE_MNT):        # the flash-time store, if mounted
        sh(["umount", str(m)], timeout=120)
    os.sync()
    STATUS.stop()
    off = should_power_off(data, cmdline) and run["status"] != "stopped"
    sh(["umount", str(MNT)], timeout=120)
    if off:
        say("powering off in 10 s")
        time.sleep(10)
        power_off()
    return 0


def prune_images(data: Path, log: Path | None = None) -> list[str]:
    """Remove every image the seed no longer has (an older ROCm, an earlier build's vendor tag): the store is reused
    from flash to flash and only ever grew — 83 GiB on 2026-11-07, more than the stick could take."""
    want = {t for img in (data / "seed" / "images").glob("*.tar") for t in image_tags(img)}
    if not want:                                       # cannot tell what belongs: remove nothing
        return []
    rc, out = sh(["docker", "image", "ls", "--format", "{{.Repository}}:{{.Tag}}"], timeout=120)
    gone = [t for t in out.split() if rc == 0 and t not in want and not t.endswith(":<none>")]
    for t in gone:
        sh(["docker", "image", "rm", "-f", t], timeout=600)
    sh(["docker", "image", "prune", "-f"], timeout=600)      # the layers nothing refers to any more
    if gone:
        say(f"docker: removed images the seed no longer has: {gone}", log)
    return gone


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
    prune_images(data)
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--preload":
        sys.exit(preload(Path(sys.argv[2])))
    sys.exit(main())
