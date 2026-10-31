#!/usr/bin/env python3
"""rigsim hardware bench — the queue that runs on the box (stdlib only; started by ansible/hwbench.yml as the
transient systemd unit rigsim-hwbench, as root).

  hwq.py plan     QUESTION.toml [--smoke]                            expand and validate, change nothing
  hwq.py run      QUESTION.toml --dir RUNDIR --config run.json [--smoke] [--fresh]
  hwq.py restore  --dir RUNDIR                                       put back what baseline.json recorded
  hwq.py cleanup  --dir RUNDIR                                       remove what the run created (containers,
                                                                     images, volumes, test files); nothing else
  hwq.py status   --dir RUNDIR

Containers: no built images. Each run starts one session container from a common image (the question's `image`,
optionally through a mirror prefix), copies the question's toolkit in (probes/<toolkit>/: sources + setup.sh),
installs and compiles inside it (network on, optional proxy), disconnects it from the network, runs every item with
`docker exec`, and discards it at the end.

The run takes the box over: every running container is stopped (except keep_running), and no process may be left
on a GPU. The box is left as it was found: the state before the first item (GPU power limits, CPU frequency limits
and governors, Docker containers / images / volumes, running containers) is written once to baseline.json and
restored at the end, on SIGTERM and on any error; ansible/hwbench.yml restores again from the same file.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import itertools
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from pathlib import Path

GPU_FIELDS = "index,pci.bus_id,name,power.limit,power.min_limit,power.max_limit,power.default_limit"
SAMPLE_FIELDS = ("timestamp,index,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,utilization.memory,"
                 "temperature.gpu,pcie.link.gen.current,pcie.link.width.current")
LABEL = "rigsim.hwbench"
TOOLKIT_DIR = "/opt/rigsim-hwbench"     # where the toolkit lands inside the session container
SH = None                      # tests replace this with a fake


class Stop(Exception):
    """SIGTERM / SIGINT: stop after restoring."""


def sh(cmd: list[str], timeout: float | None = None, check: bool = False) -> subprocess.CompletedProcess:
    if SH is not None:
        return SH(cmd, timeout=timeout, check=check)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


def short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:n]


# --- plans -------------------------------------------------------------------------------------------------------
REQUIRED = ("question", "asks", "feeds", "image", "toolkit", "grid", "args")


def load_plan(path: str) -> dict:
    plan = tomllib.loads(Path(path).read_text())
    for k in REQUIRED:
        if k not in plan:
            raise ValueError(f"{path}: missing '{k}'")
    if "cmd" not in plan["args"]:
        raise ValueError(f"{path}: [args] needs 'cmd'")
    if plan.get("gpus", "none") not in ("item", "none"):
        raise ValueError(f"{path}: gpus must be 'item' (the grid's gpu key picks the card) or 'none'")
    if plan.get("gpus") == "item" and "gpu" not in plan["grid"]:
        raise ValueError(f"{path}: gpus = 'item' needs a 'gpu' grid key")
    return plan


def logical_cpus() -> int:
    """The logical CPUs this process may run on."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def cap_to_cpus(values: list, n: int) -> list:
    """A thread ladder on this machine: the values up to its logical CPU count, and the count itself (the all-CPU
    point), so no step asks for threads the machine does not have and the full machine is always measured."""
    return sorted({v for v in values if isinstance(v, int) and 0 < v <= n} | {n})


def expand(plan: dict, smoke: bool, gpus: list[dict] | None = None, cpus: int | None = None) -> list[dict]:
    """Grid -> items (cartesian product, in key order). In a gpu list, '@all' = one item per GPU found, '@together' =
    one item with every GPU visible at once. A grid key named in [limits] as "logical-cpus" is capped to the machine's
    logical CPU count (cap_to_cpus; `cpus` defaults to the real count). A smoke run uses the [smoke] grid and args
    overrides and one repeat (smokes stay short: one point per mechanism)."""
    grid, args = dict(plan["grid"]), dict(plan["args"])
    if smoke:
        grid.update((plan.get("smoke") or {}).get("grid", {}))
        args.update((plan.get("smoke") or {}).get("args", {}))
    for k, rule in (plan.get("limits") or {}).items():
        if rule != "logical-cpus":
            raise ValueError(f"{plan['question']}: unknown limit {rule!r} for {k}")
        if k in grid:
            grid[k] = cap_to_cpus(grid[k] if isinstance(grid[k], list) else [grid[k]], cpus or logical_cpus())
    if "gpu" in grid and gpus is not None:
        names: list[str] = []
        for g in grid["gpu"]:
            names += [x["name"] for x in gpus] if g == "@all" else [g]
        grid["gpu"] = names
    keys = list(grid)
    repeats = 1 if smoke else int(plan.get("repeats", 1))
    items = []
    for combo in itertools.product(*(grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys)):
        p = dict(zip(keys, combo))
        for r in range(repeats):
            iid = plan["question"] + "/" + "/".join(f"{k}={p[k]}" for k in keys) + f"/rep={r}"
            items.append(dict(id=iid, params=p, rep=r, args=args))
    return items


def command_for(item: dict) -> list[str]:
    fmt = {**item["args"], **item["params"]}
    return [str(x).format(**fmt) for x in item["args"]["cmd"]]


# --- host state ----------------------------------------------------------------------------------------------------
def gpus_found(names: dict[str, str]) -> list[dict]:
    """GPUs by nvidia-smi, named by the config's bus-id map (unmapped: gpu<index>)."""
    r = sh(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"], timeout=30)
    if r.returncode != 0:
        return []
    res = []
    for line in r.stdout.strip().splitlines():
        idx, bus, model, lim, lo, hi, dflt = [x.strip() for x in line.split(",")]
        key = bus.lower().replace("00000000:", "0000:")
        res.append(dict(index=int(idx), bus=key, model=model, name=names.get(key, f"gpu{idx}"),
                        power_limit_w=float(lim), min_w=float(lo), max_w=float(hi), default_w=float(dflt)))
    return res


CPUFREQ = "/sys/devices/system/cpu"


def cpufreq_state(root: str = CPUFREQ) -> dict:
    st = {}
    for d in sorted(glob.glob(f"{root}/cpu[0-9]*/cpufreq")):
        cpu = d.split("/")[-2]
        try:
            st[cpu] = {k: Path(d, f"scaling_{k}").read_text().strip() for k in ("governor", "min_freq", "max_freq")}
        except OSError:
            pass
    return st


def docker_lists() -> dict:
    def ids(cmd):
        r = sh(cmd, timeout=60)
        return sorted(set(r.stdout.split())) if r.returncode == 0 else []
    return dict(containers=ids(["docker", "ps", "-aq", "--no-trunc"]),
                running=ids(["docker", "ps", "-q", "--no-trunc"]),
                images=ids(["docker", "images", "-q", "--no-trunc"]),
                volumes=ids(["docker", "volume", "ls", "-q"]))


def capture_baseline(rundir: Path, names: dict, root: str = CPUFREQ) -> dict:
    """Written once per run directory: a resumed run keeps the state from before its first item."""
    f = rundir / "baseline.json"
    if f.exists():
        return json.loads(f.read_text())
    base = dict(t=time.time(), gpus=gpus_found(names), cpufreq=cpufreq_state(root), docker=docker_lists())
    f.write_text(json.dumps(base, indent=1))
    return base


def resolve_watts(value, gpu: dict) -> float:
    """A grid value in watts, or 'default' (the card's default limit) / 'max' (its highest allowed limit)."""
    if value == "default":
        return gpu["default_w"]
    if value == "max":
        return gpu["max_w"]
    return float(value)


def cpu_range_mhz(root: str = CPUFREQ) -> tuple[float, float]:
    try:
        d = Path(f"{root}/cpu0/cpufreq")
        return (int((d / "cpuinfo_min_freq").read_text()) / 1000, int((d / "cpuinfo_max_freq").read_text()) / 1000)
    except OSError:
        return (0.0, 0.0)


def set_power_limit(index: int, watts: float) -> None:
    r = sh(["nvidia-smi", "-i", str(index), "-pl", f"{watts:g}"], timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi -pl {watts:g} on {index}: {r.stderr.strip()}")


def set_cpu_mhz(mhz: float | None, base: dict, root: str = CPUFREQ) -> None:
    """Pins every CPU's scaling range to `mhz` (performance governor); None = back to the baseline. Writes are
    ordered so that min <= max holds at every step."""
    for cpu, st in base["cpufreq"].items():
        d = Path(f"{root}/{cpu}/cpufreq")
        if mhz is None:
            (d / "scaling_max_freq").write_text(st["max_freq"])
            (d / "scaling_min_freq").write_text(st["min_freq"])
            (d / "scaling_governor").write_text(st["governor"])
        else:
            khz = str(int(mhz * 1000))
            (d / "scaling_governor").write_text("performance")
            if int(khz) < int((d / "scaling_min_freq").read_text()):
                (d / "scaling_min_freq").write_text(khz); (d / "scaling_max_freq").write_text(khz)
            else:
                (d / "scaling_max_freq").write_text(khz); (d / "scaling_min_freq").write_text(khz)


def restore(rundir: Path, log=print, root: str = CPUFREQ) -> list[str]:
    """Back to baseline.json: power limits, CPU frequency limits and governors, containers stopped by the run.
    Idempotent; every step is attempted even if another fails. Returns the problems (empty = box as found)."""
    f = rundir / "baseline.json"
    if not f.exists():
        return ["no baseline.json: nothing to restore"]
    base = json.loads(f.read_text())
    problems: list[str] = []
    for g in base["gpus"]:
        try:
            set_power_limit(g["index"], g["power_limit_w"])
        except Exception as e:                                        # noqa: BLE001 - keep restoring the rest
            problems.append(f"power limit {g['name']}: {e}")
    if base.get("cpufreq"):
        try:
            set_cpu_mhz(None, base, root)
        except Exception as e:                                        # noqa: BLE001
            problems.append(f"cpufreq: {e}")
    stopped = json.loads((rundir / "stopped.json").read_text()) if (rundir / "stopped.json").exists() else []
    for c in stopped:
        r = sh(["docker", "start", c], timeout=300)
        if r.returncode != 0:
            problems.append(f"start {c[:12]}: {r.stderr.strip()}")
    running = set(docker_lists()["running"])
    missing = [c[:12] for c in base["docker"]["running"] if c not in running]
    if missing:
        problems.append(f"containers running before but not now: {missing}")
    current = {g["index"]: g for g in gpus_found({})}
    for g in base["gpus"]:
        cur = current.get(g["index"])
        if cur and abs(cur["power_limit_w"] - g["power_limit_w"]) > 0.5:
            problems.append(f"{g['name']}: power limit {cur['power_limit_w']:g} W, expected {g['power_limit_w']:g} W")
    if base.get("cpufreq") and cpufreq_state(root) != base["cpufreq"]:
        problems.append("CPU frequency limits or governors differ from the baseline")
    (rundir / "restore.json").write_text(json.dumps(dict(t=time.time(), problems=problems), indent=1))
    log("restore: box as found" if not problems else f"restore: {len(problems)} problem(s)")
    return problems


def cleanup(rundir: Path, log=print) -> int:
    """Removes what the run created and nothing uses: its labelled containers, images pulled during the run that no
    container uses, new dangling volumes, its test-file directory. Never anything present before the run."""
    base = json.loads((rundir / "baseline.json").read_text())
    n = 0
    for c in sh(["docker", "ps", "-aq", "--no-trunc", "--filter", f"label={LABEL}"], timeout=60).stdout.split():
        sh(["docker", "rm", "-f", c], timeout=120); log(f"cleanup: container {c[:12]}"); n += 1
    used = set(sh(["sh", "-c", "docker ps -aq | xargs -r docker inspect -f '{{.Image}}'"], timeout=60).stdout.split())
    for i in docker_lists()["images"]:
        if i not in base["docker"]["images"] and i not in used:
            if sh(["docker", "rmi", i], timeout=300).returncode == 0:
                log(f"cleanup: image {i[:19]}"); n += 1
    dangling = set(sh(["docker", "volume", "ls", "-q", "--filter", "dangling=true"], timeout=60).stdout.split())
    for v in sorted(dangling - set(base["docker"]["volumes"])):
        sh(["docker", "volume", "rm", v], timeout=60); log(f"cleanup: volume {v}"); n += 1
    for tf in json.loads((rundir / "testdirs.json").read_text()) if (rundir / "testdirs.json").exists() else []:
        if Path(tf).name.startswith("rigsim-hwbench-") and Path(tf).exists():
            shutil.rmtree(tf); log(f"cleanup: test files {tf}"); n += 1
    return n


# --- sampling ------------------------------------------------------------------------------------------------------
def start_sampler(path: Path, ms: int):
    if not shutil.which("nvidia-smi"):
        return None
    f = open(path, "w")
    return subprocess.Popen(["nvidia-smi", f"--query-gpu={SAMPLE_FIELDS}", "--format=csv,noheader,nounits",
                             "-lms", str(ms)], stdout=f, stderr=subprocess.DEVNULL)


def _ts(s: str) -> float | None:
    """nvidia-smi timestamps: '2026/10/27 12:00:01.123' (local time)."""
    try:
        main, _, frac = s.partition(".")
        return time.mktime(time.strptime(main, "%Y/%m/%d %H:%M:%S")) + (float("0." + frac) if frac else 0.0)
    except ValueError:
        return None


def summarise_samples(path: Path, t0: float, t1: float, gpus: list[dict]) -> dict:
    """Median / p10 / p90 per card of the samples inside [t0, t1]."""
    names = {g["index"]: g["name"] for g in gpus}
    cols = SAMPLE_FIELDS.split(",")
    per: dict[str, dict[str, list[float]]] = {}
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != len(cols):
            continue
        ts = _ts(parts[0])
        if ts is None or not (t0 <= ts <= t1):
            continue
        card = names.get(int(parts[1]), f"gpu{parts[1]}")
        for c, v in zip(cols[2:], parts[2:]):
            try:
                per.setdefault(card, {}).setdefault(c, []).append(float(v))
            except ValueError:
                pass
    out: dict[str, dict] = {}
    for card, series in per.items():
        out[card] = {}
        for c, vals in series.items():
            vals.sort()
            out[card][c] = dict(n=len(vals), median=statistics.median(vals), p10=vals[len(vals) // 10],
                                p90=vals[(9 * len(vals)) // 10])
    return out


# --- run -----------------------------------------------------------------------------------------------------------
def parse_probe_output(text: str) -> list[dict]:
    """Probes print one JSON object per result line; anything else stays in the raw log only."""
    rows = []
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def image_name(plan: dict, cfg: dict) -> str:
    """The question's common image, optionally through a mirror / pull-through proxy prefix (config image_mirror)."""
    prefix = (cfg.get("image_mirror") or "").rstrip("/")
    return f"{prefix}/{plan['image']}" if prefix else plan["image"]


def session_name(rundir: Path) -> str:
    return f"rigsim-hwbench-{rundir.name}"


def start_session(rundir: Path, plan: dict, cfg: dict, toolkit: Path, testdir: str | None, log) -> str:
    """One container per run: common image, kept alive, toolkit copied in, set up with the network on, then cut off
    from the network. Returns its name. A leftover session from an interrupted run is discarded first."""
    name = session_name(rundir)
    sh(["docker", "rm", "-f", name], timeout=120)
    image = image_name(plan, cfg)
    if sh(["docker", "image", "inspect", image], timeout=60).returncode != 0:
        r = sh(["docker", "pull", image], timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"cannot pull {image} (the box needs a registry mirror or egress for common images; "
                               f"config image_mirror): {r.stderr.strip()[-300:]}")
    cmd = ["docker", "run", "-d", "--name", name, "--label", f"{LABEL}={rundir.name}", "--ipc", "host",
           "--security-opt", "no-new-privileges", "--entrypoint", "sleep"]
    if plan.get("gpus", "none") != "none":
        cmd += ["--gpus", "all"]
    if testdir:
        cmd += ["-v", f"{testdir}:/data:rw"]
    r = sh(cmd + [image, "infinity"], timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"session container did not start: {r.stderr.strip()[-300:]}")
    digest = sh(["docker", "image", "inspect", "-f", "{{index .RepoDigests 0}}", image], timeout=60).stdout.strip()
    (rundir / "session.json").write_text(json.dumps(dict(name=name, image=image, digest=digest, t=time.time()), indent=1))
    if sh(["docker", "cp", f"{toolkit}/.", f"{name}:{TOOLKIT_DIR}"], timeout=300).returncode != 0:
        raise RuntimeError("could not copy the toolkit into the session container")
    env = []
    for k in ("http_proxy", "https_proxy", "no_proxy"):
        if cfg.get("setup_proxy", {}).get(k):
            env += ["-e", f"{k}={cfg['setup_proxy'][k]}", "-e", f"{k.upper()}={cfg['setup_proxy'][k]}"]
    log(f"session {name}: setting up from {image}")
    r = sh(["docker", "exec", *env, name, "sh", f"{TOOLKIT_DIR}/setup.sh"], timeout=float(plan.get("setup_timeout_s", 1800)))
    (rundir / "setup.log").write_text((r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""))
    if r.returncode != 0:
        raise RuntimeError(f"toolkit setup failed (setup.log): {(r.stderr or '').strip()[-300:]}")
    if not plan.get("network_during_items", False):
        for net in sh(["docker", "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}", name],
                      timeout=60).stdout.split():
            sh(["docker", "network", "disconnect", "-f", net, name], timeout=60)
    return name


def exec_cmd(session: str, item: dict, gpus: list[dict], plan: dict) -> list[str]:
    """One item inside the session. GPU questions pick the item's card by PCI order (= nvidia-smi index)."""
    cmd = ["docker", "exec", "-e", "CUDA_DEVICE_ORDER=PCI_BUS_ID"]
    if plan.get("gpus") == "item":
        if item["params"]["gpu"] == "@together":
            vis = ",".join(str(g["index"]) for g in sorted(gpus, key=lambda g: g["index"]))
        else:
            vis = str(next(g["index"] for g in gpus if g["name"] == item["params"]["gpu"]))
        cmd += ["-e", f"CUDA_VISIBLE_DEVICES={vis}"]
    return cmd + [session] + command_for(item)


def gpu_processes() -> list[str]:
    r = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], timeout=30)
    return [l.strip() for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []


def stop_containers(rundir: Path, base: dict, keep: list[str], log) -> None:
    f = rundir / "stopped.json"
    stopped = json.loads(f.read_text()) if f.exists() else []
    for c in base["docker"]["running"]:
        if c in stopped:
            continue
        name = sh(["docker", "inspect", "-f", "{{.Name}}", c], timeout=30).stdout.strip().lstrip("/")
        if any(fnmatch.fnmatch(name, k) for k in keep):
            continue
        if sh(["docker", "stop", "-t", "60", c], timeout=180).returncode == 0:
            stopped.append(c); log(f"stopped {name}")
            f.write_text(json.dumps(stopped))           # after each stop: a crash still restores what was stopped


def run(plan_path: str, rundir: Path, cfg: dict, smoke: bool, fresh: bool, log, root: str = CPUFREQ) -> int:
    plan = load_plan(plan_path)
    rundir.mkdir(parents=True, exist_ok=True)
    if fresh:
        for f in ("state.json", "results.jsonl"):
            (rundir / f).unlink(missing_ok=True)
    toolkit = Path(cfg.get("toolkits", str(Path(__file__).resolve().parent / "toolkits"))) / plan["toolkit"]
    if not (toolkit / "setup.sh").exists():
        raise ValueError(f"toolkit '{plan['toolkit']}' has no setup.sh under {toolkit.parent}")
    base = capture_baseline(rundir, cfg.get("gpu_names", {}), root)
    gpus = base["gpus"]
    items = expand(plan, smoke, gpus)
    sf = rundir / "state.json"
    state = json.loads(sf.read_text()) if sf.exists() else {}
    # every planned item is in the state from the start, so its size is the question's total (progress counters
    # showed '13/13', '14/14' while items were still being added)
    for it in items:
        state.setdefault(it["id"], {"status": "pending"})
    sf.write_text(json.dumps(state, indent=1))

    def on_signal(signum, frame):
        raise Stop()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    session = None
    try:
        if plan.get("exclusive", True):
            stop_containers(rundir, base, cfg.get("keep_running", []), log)
        if plan.get("gpus", "none") != "none" or plan.get("exclusive", True):
            procs = gpu_processes()
            if procs:
                raise RuntimeError(f"processes still on the GPUs after stopping the containers: {procs}")
        testdir = None
        if plan.get("testdir"):
            testdir = str(Path(cfg["testdir_parent"]) / f"rigsim-hwbench-{rundir.name}")
            Path(testdir).mkdir(parents=True, exist_ok=True)
            (rundir / "testdirs.json").write_text(json.dumps([testdir]))
        (rundir / "samples").mkdir(exist_ok=True)
        session = start_session(rundir, plan, cfg, toolkit, testdir, log)
        for rnd in range(int(plan.get("max_attempts", 2))):
            todo = [it for it in items if state.get(it["id"], {}).get("status") not in ("done", "skipped", "failed")]
            if not todo:
                break
            for item in todo:
                if Path(cfg.get("pause_file", "/nonexistent")).exists():
                    log("paused between items"); return 0
                run_item(item, plan, rundir, cfg, base, gpus, session, state, smoke, log, root)
                sf.write_text(json.dumps(state, indent=1))
        return 0
    except Stop:
        log("stopping on signal")
        return 130
    finally:
        sh(["docker", "rm", "-f", session_name(rundir)], timeout=120)      # the session is discarded every time
        for prob in restore(rundir, log, root):
            log(f"RESTORE PROBLEM: {prob}")


def run_item(item, plan, rundir, cfg, base, gpus, session, state, smoke, log, root=CPUFREQ):
    p = item["params"]
    st = state.get(item["id"], {})
    gpu = next((g for g in gpus if g["name"] == p.get("gpu")), None)
    watts = resolve_watts(p.get("power_w"), gpu) if gpu and "power_w" in p else None
    reason = None
    if gpu and "power_w" in p and not (gpu["min_w"] <= watts <= gpu["max_w"]):
        reason = f"{p['power_w']} W outside {gpu['min_w']:g}-{gpu['max_w']:g} W"
    if "cpu_mhz" in p:
        lo, hi = cpu_range_mhz(root)
        if lo and not (lo <= float(p["cpu_mhz"]) <= hi):
            reason = f"{p['cpu_mhz']} MHz outside {lo:g}-{hi:g} MHz"
    if reason:
        state[item["id"]] = dict(status="skipped", reason=reason)
        log(f"skip {item['id']}: {reason}")
        return
    if shutil.disk_usage(rundir).free < cfg.get("min_free_gb", 5) * 1e9:
        raise RuntimeError("free disk below min_free_gb")
    for g in gpus:                                  # this item's limit on its card, every other card at baseline
        set_power_limit(g["index"], watts if (g is gpu and watts is not None) else g["power_limit_w"])
    if "cpu_mhz" in p:
        set_cpu_mhz(float(p["cpu_mhz"]), base, root)
    sp = rundir / "samples" / f"{short(item['id'])}.csv"
    sampler = start_sampler(sp, int(plan.get("sample_ms", 100)))
    time.sleep(float(plan.get("settle_s", 1.0)))
    cmd = exec_cmd(session, item, gpus, plan)
    t0 = time.time()
    try:
        r = sh(cmd, timeout=float(plan.get("timeout_s", 600)))
        rc, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        rc, err = -9, "timeout"
        sh(["docker", "restart", "-t", "0", session], timeout=120)    # kills the hung probe; setup is kept
    t1 = time.time()
    if sampler:
        time.sleep(0.3); sampler.terminate(); sampler.wait(timeout=10)
    if "cpu_mhz" in p:
        set_cpu_mhz(None, base, root)
    samples = summarise_samples(sp, t0, t1, gpus)
    limit = float(plan.get("max_gpu_temp_c", 87))
    hot = sorted(c for c, s in samples.items() if s.get("temperature.gpu", {}).get("p90", 0) > limit)
    attempts = st.get("attempts", 0) + 1
    ok = rc == 0 and not hot
    row = dict(id=item["id"], question=plan["question"], params=p, rep=item["rep"], attempt=attempts, rc=rc,
               t_start=t0, t_end=t1, results=parse_probe_output(out), stderr_tail=(err or "")[-2000:],
               samples=samples, sample_file=sp.name, hot=hot, smoke=smoke, command=cmd[cmd.index(session) + 1:])
    with open(rundir / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    final = attempts >= int(plan.get("max_attempts", 2))
    state[item["id"]] = dict(status="done" if ok else ("failed" if final else "retry"), attempts=attempts)
    log(f"{'ok  ' if ok else 'FAIL'} {item['id']} rc={rc}" + (f" hot {hot}" if hot else ""))
    if hot:
        time.sleep(float(plan.get("cooldown_s", 120)))


def status(rundir: Path) -> dict:
    sf = rundir / "state.json"
    counts: dict[str, int] = {}
    for v in (json.loads(sf.read_text()) if sf.exists() else {}).values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "restore", "cleanup", "status"])
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--dir")
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text()) if a.config else {}
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    if a.cmd == "plan":
        plan = load_plan(a.plan)
        items = expand(plan, a.smoke, None)
        print(json.dumps(dict(question=plan["question"], items=len(items),
                              first=items[0]["id"] if items else None,
                              command=command_for(items[0]) if items else []), indent=1))
        return 0
    d = Path(a.dir)
    if a.cmd == "run":
        return run(a.plan, d, cfg, a.smoke, a.fresh, log)
    if a.cmd == "restore":
        return 1 if restore(d, log) else 0
    if a.cmd == "cleanup":
        log(f"cleanup: {cleanup(d, log)} removed"); return 0
    print(json.dumps(status(d))); return 0


if __name__ == "__main__":
    sys.exit(main())
