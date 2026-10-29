#!/usr/bin/env python3
"""Named, comparable values from a run folder — what the data page shows per run and what rigsim is asked to
predict (predictions/README.md). Reads every run format 1.x; a value a run does not have is simply absent.

  python3 tools/measurements.py results/runs/<run>        prints the run's values as JSON

Every GPU value carries its card's PCI address (conditions.pci) where the run recorded it, so two cards of the same
model stay apart; the device part of a key is the index within that backend (gpuN, devN), not a model name.

Keys are stable (they are how a prediction finds its measurement) and versioned with VERSION; a new kind of value
gets a new key, an existing key never changes meaning:
  llama-bench/<backend>/<device>/<model>/<test>     tokens/s        backend: cuda, vulkan, hip, sycl, cpu
  gpu-vram/<gpu>/<op>                               GB/s            op: read, write, copy (median over repeats)
  cpu-mem-ladder/<kernel>/<threads>                 GB/s            likwid kernel (load_*, store_avx, store_mem_avx)
  mlc/idle_latency                                  ns
  mlc/max_bandwidth/<traffic>                       GB/s            traffic: all_reads, 3to1_rw, 2to1_rw, ..."""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

VERSION = "2"                  # 2: cpu-mem-ladder values added. Any change to the set of values bumps it.
LLAMA_FILE = re.compile(r"^(cuda-gpu\d+|vulkan-dev\d+|cpu)-(.+)\.jsonl$")
VENDOR_FILE = re.compile(r"^dev(\d+)-llama-(.+)\.jsonl$")


def _json_lines(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def _norm_pci(bus: str) -> str:
    b = bus.strip().lower()
    m = re.match(r"^(?:([0-9a-f]+):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f])$", b)
    return f"{int(m.group(1) or '0', 16):04x}:{m.group(2)}:{m.group(3)}.{m.group(4)}" if m else b


def device_pci(run: Path) -> dict[tuple[str, str], str]:
    """(backend, device) -> PCI address, from what the run recorded: NVIDIA by nvidia-smi order (CUDA runs in PCI
    bus order), Vulkan from vkprobe's list, ROCm / oneAPI from their probes' device lists."""
    out: dict[tuple[str, str], str] = {}
    try:
        gpus = json.loads((run / "run.json").read_text()).get("hardware", {}).get("gpus", [])
    except (OSError, ValueError):
        gpus = []
    nv = [g for g in gpus if g.count(",") >= 2 and re.search(r"[0-9a-fA-F]+:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", g)]
    for i, g in enumerate(nv):
        out[("cuda", f"gpu{i}")] = _norm_pci(g.rsplit(",", 1)[-1])
    for f, backend, prefix in ((run / "vkprobe" / "list.jsonl", "vulkan", "dev"), (run / "rocm" / "info.jsonl", "hip", "dev"),
                               (run / "oneapi" / "info.jsonl", "sycl", "dev")):
        if f.exists():
            for r in _json_lines(f):
                if r.get("pci") and "index" in r:
                    out[(backend, f"{prefix}{r['index']}")] = _norm_pci(r["pci"])
    return out


def _llama_rows(rows: list[dict], backend: str, device: str, model: str, source: str) -> list[dict]:
    out = []
    for r in rows:
        if "avg_ts" not in r:
            continue
        n_p, n_g = int(r.get("n_prompt") or 0), int(r.get("n_gen") or 0)
        test = f"pp{n_p}" if n_p and not n_g else f"tg{n_g}" if n_g and not n_p else f"pp{n_p}+tg{n_g}"
        out.append(dict(key=f"llama-bench/{backend}/{device}/{model}/{test}", value=float(r["avg_ts"]), unit="tok/s",
                        spread=float(r.get("stddev_ts") or 0), source=source,
                        conditions=dict(backend=backend, device=device, gpu=r.get("gpu_info", ""), model=model,
                                        model_file=r.get("model_filename", ""), n_prompt=n_p, n_gen=n_g,
                                        n_gpu_layers=r.get("n_gpu_layers"), n_threads=r.get("n_threads"),
                                        n_batch=r.get("n_batch"), n_ubatch=r.get("n_ubatch"),
                                        flash_attn=r.get("flash_attn"), build_commit=r.get("build_commit", ""))))
    return out


def llama(run: Path) -> list[dict]:
    out = []
    pci = device_pci(run)
    for f in sorted((run / "llama").glob("*.jsonl")):
        m = LLAMA_FILE.match(f.name)
        if not m:
            continue
        backend, _, dev = m.group(1).partition("-")
        out += _llama_rows(_json_lines(f), backend, dev or "cpu", m.group(2), f"llama/{f.name}")
        for r in out:
            r["conditions"].setdefault("pci", pci.get((r["conditions"]["backend"], r["conditions"]["device"]), ""))
    for kind, backend in (("rocm", "hip"), ("oneapi", "sycl")):
        for f in sorted((run / kind).glob("dev*-llama-*.jsonl")):
            m = VENDOR_FILE.match(f.name)
            if m:
                out += _llama_rows(_json_lines(f), backend, f"dev{m.group(1)}", m.group(2), f"{kind}/{f.name}")
    for r in out:
        r["conditions"].setdefault("pci", pci.get((r["conditions"]["backend"], r["conditions"]["device"]), ""))
    return out


def gpu_vram(run: Path) -> list[dict]:
    f = run / "q" / "gpu-vram" / "results.jsonl"
    if not f.exists():
        return []
    by: dict[tuple[str, str], list[float]] = {}
    bus = {}
    for b in sorted((run / "q" / "gpu-vram").glob("baseline.json")):
        try:
            bus = {g["name"]: _norm_pci(g.get("bus", "")) for g in json.loads(b.read_text()).get("gpus", [])}
        except (ValueError, KeyError):
            pass
    for row in _json_lines(f):
        if row.get("rc") != 0 or row.get("smoke"):
            continue
        for r in row.get("results") or []:
            if r.get("mode") == "stream" and r.get("bytes_per_s"):
                by.setdefault((str(row.get("params", {}).get("gpu", "")), r.get("op", "")), []).append(r["bytes_per_s"])
    return [dict(key=f"gpu-vram/{g}/{op}", value=statistics.median(v) / 1e9, unit="GB/s", spread=0.0,
                 source="q/gpu-vram/results.jsonl", conditions=dict(gpu=g, op=op, repeats=len(v), pci=bus.get(g, "")))
            for (g, op), v in sorted(by.items())]


def ladder(run: Path) -> list[dict]:
    """The bench runner's RAM ladder (likwid-bench): bandwidth per kernel and thread count, median over repeats.
    likwid counts the bytes the program moves (a store kernel: the bytes written)."""
    f = run / "q" / "cpu-mem-ladder" / "results.jsonl"
    if not f.exists():
        return []
    by: dict[tuple[str, int], list[float]] = {}
    for row in _json_lines(f):
        if row.get("rc") != 0 or row.get("smoke"):
            continue
        for r in row.get("results") or []:
            if r.get("mode") == "ladder" and r.get("bytes_per_s"):
                by.setdefault((r["kernel"], int(r["threads"])), []).append(r["bytes_per_s"])
    return [dict(key=f"cpu-mem-ladder/{k}/{t}", value=statistics.median(v) / 1e9, unit="GB/s", spread=0.0,
                 source="q/cpu-mem-ladder/results.jsonl", conditions=dict(kernel=k, threads=t, repeats=len(v)))
            for (k, t), v in sorted(by.items())]


def mlc(run: Path) -> list[dict]:
    out = []
    f = run / "mlc" / "idle_latency.txt"
    if f.exists():
        m = re.search(r"Each iteration took .*?\(\s*([\d.]+)\s*ns\)", f.read_text())
        if m:
            out.append(dict(key="mlc/idle_latency", value=float(m.group(1)), unit="ns", spread=0.0,
                            source="mlc/idle_latency.txt", conditions={}))
    f = run / "mlc" / "max_bandwidth.txt"
    if f.exists():
        for name, val in re.findall(r"^\s*(ALL Reads|3:1 Reads-Writes|2:1 Reads-Writes|1:1 Reads-Writes|"
                                    r"Stream-triad like)\s*:\s*([\d.]+)", f.read_text(), re.M):
            key = {"ALL Reads": "all_reads", "Stream-triad like": "stream_triad"}.get(
                name, name.split()[0].replace(":", "to") + "_rw")
            out.append(dict(key=f"mlc/max_bandwidth/{key}", value=float(val) / 1000, unit="GB/s", spread=0.0,
                            source="mlc/max_bandwidth.txt", conditions={}))
    return out


def extract(run: Path) -> dict:
    return dict(version=VERSION, run=run.name, measurements=llama(run) + gpu_vram(run) + ladder(run) + mlc(run))


if __name__ == "__main__":
    print(json.dumps(extract(Path(sys.argv[1])), indent=1))
