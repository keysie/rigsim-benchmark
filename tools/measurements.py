#!/usr/bin/env python3
"""Named, comparable values from a run folder — what the data page shows per run and what rigsim is asked to
predict (predictions/README.md). Reads every run format 1.x; a value a run does not have is simply absent.

  python3 tools/measurements.py results/runs/<run>        prints the run's values as JSON

Every GPU value carries its card's PCI address (conditions.pci) where the run recorded it, so two cards of the same
model stay apart; the device part of a key is the index within that backend (gpuN, devN), not a model name.

Keys are stable (they are how a prediction finds its measurement) and versioned with VERSION; a new kind of value
gets a new key, an existing key never changes meaning:
  gpu-gemm/<gpu>/<prec>/<M>x<N>x<K>                 TFLOP/s (int8: TOP/s)   cuBLAS, with SM clock and power
  cpu-gemm/<prec>/<M>x<N>x<K>/t<threads>            GFLOP/s (int8: GOP/s)   OpenBLAS / oneDNN, with ISA and kernel
  xfer/<gpu>/<dir>/<mem>/<size>/latency|rate        µs | GB/s       pcie-xfer: one copy + wait, queued copies
  xfer/<gpuA>-<gpuB>/d2d/peer|host/<size>/…         µs | GB/s       gpu-p2p-xfer: with / without peer access
  gpu-overlap/<gpu>/kernel_share|copy_share         ratio      each with the other running / alone
  storage/<drive>/<job>/rate|iops|lat_p50|lat_p99   GB/s | IO/s | µs   read-only fio jobs (seq / rand, block, QD)
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

VERSION = "6"                  # 2: ladder; 3: sycl_onednn; 4: gpu-gemm, cpu-gemm; 5: xfer, gpu-overlap; 6: storage.
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
    try:
        vruns = json.loads((run / "run.json").read_text()).get("questions", {}).get("oneapi", {}).get("runs") or {}
    except (OSError, ValueError):
        vruns = {}
    for kind, backend in (("rocm", "hip"), ("oneapi", "sycl")):
        for f in sorted((run / kind).glob("dev*-llama-*.jsonl")):
            m = VENDOR_FILE.match(f.name)
            if m:
                rows = _llama_rows(_json_lines(f), backend, f"dev{m.group(1)}", m.group(2), f"{kind}/{f.name}")
                e = vruns.get(f.name[:-len(".jsonl")]) or {}
                if kind == "oneapi" and (e.get("retry") or e.get("onednn_off")):
                    for r in rows:                   # measured without oneDNN after it failed (format 1.5)
                        r["conditions"]["sycl_onednn"] = False
                out += rows
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


def _gemm_rows(run: Path, questions: tuple[str, ...]) -> dict[tuple, dict]:
    """Rows of the GEMM questions, grouped by (device, prec, M, N, K, threads)."""
    by: dict[tuple, dict] = {}
    for q in questions:
        f = run / "q" / q / "results.jsonl"
        for row in (_json_lines(f) if f.exists() else []):
            if row.get("rc") != 0 or row.get("smoke"):
                continue
            p = row.get("params", {})
            for r in row.get("results") or []:
                if r.get("unsupported") or not r.get("ops_per_s"):
                    continue
                m, n, k = r.get("m", r.get("n")), r.get("nn", r.get("n")), r.get("k", r.get("n"))
                key = (str(p.get("gpu", "")), r.get("prec", p.get("prec", "")), m, n, k, r.get("threads", 0))
                e = by.setdefault(key, dict(v=[], src=q, impl=r.get("impl", ""), isa=r.get("isa"), lib=r.get("lib", ""),
                                            clock=[], power=[]))
                e["v"].append(r["ops_per_s"])
                s = (row.get("samples") or {}).get(str(p.get("gpu", "")), {})
                if s.get("clocks.sm", {}).get("median") is not None:
                    e["clock"].append(s["clocks.sm"]["median"])
                if s.get("power.draw", {}).get("median") is not None:
                    e["power"].append(s["power.draw"]["median"])
    return by


def gpu_gemm(run: Path) -> list[dict]:
    """cuBLAS rates per card, format and shape (gpu-gemm, gpu-gemm-shapes): TFLOP/s (int8: TOP/s), with the card's SM
    clock and power during the point (format 1.x samples)."""
    bus = {}
    for b in sorted((run / "q").glob("gpu-gemm*/baseline.json")):
        try:
            bus.update({g["name"]: _norm_pci(g.get("bus", "")) for g in json.loads(b.read_text()).get("gpus", [])})
        except (ValueError, KeyError):
            pass
    out = []
    for (g, prec, m, n, k, _), e in sorted(_gemm_rows(run, ("gpu-gemm", "gpu-gemm-shapes")).items(), key=str):
        c = dict(gpu=g, prec=prec, m=m, n=n, k=k, repeats=len(e["v"]), pci=bus.get(g, ""))
        if e["clock"]:
            c["sm_clock_mhz"] = statistics.median(e["clock"])
        if e["power"]:
            c["power_w"] = statistics.median(e["power"])
        out.append(dict(key=f"gpu-gemm/{g}/{prec}/{m}x{n}x{k}", value=statistics.median(e["v"]) / 1e12,
                        unit="TOP/s" if prec == "int8" else "TFLOP/s", spread=0.0, source=f"q/{e['src']}/results.jsonl",
                        conditions=c))
    return out


def cpu_gemm(run: Path) -> list[dict]:
    """CPU rates per format, shape and thread count (cpu-gemm, cpu-gemm-shapes): GFLOP/s (int8: GOP/s), with the
    library, oneDNN's ISA code and kernel."""
    out = []
    for (_, prec, m, n, k, t), e in sorted(_gemm_rows(run, ("cpu-gemm", "cpu-gemm-shapes")).items(), key=str):
        out.append(dict(key=f"cpu-gemm/{prec}/{m}x{n}x{k}/t{t}", value=statistics.median(e["v"]) / 1e9,
                        unit="GOP/s" if prec == "int8" else "GFLOP/s", spread=0.0, source=f"q/{e['src']}/results.jsonl",
                        conditions=dict(prec=prec, m=m, n=n, k=k, threads=t, repeats=len(e["v"]), lib=e["lib"],
                                        isa=e["isa"], impl=e["impl"])))
    return out


def _size(b: int) -> str:
    return f"{b >> 20}MiB" if b >= 1 << 20 else f"{b >> 10}KiB"


def xfer(run: Path) -> list[dict]:
    """Transfer sweeps (pcie-xfer, gpu-p2p-xfer): per size the latency of one copy + wait and the rate of queued ones."""
    out = []
    for q in ("pcie-xfer", "gpu-p2p-xfer"):
        f = run / "q" / q / "results.jsonl"
        for row in (_json_lines(f) if f.exists() else []):
            if row.get("rc") != 0 or row.get("smoke"):
                continue
            g = str(row.get("params", {}).get("gpu", ""))
            for r in row.get("results") or []:
                if r.get("mode") != "xfer" or not r.get("bytes"):
                    continue
                if r.get("dir") == "d2d":
                    dev, path = f"gpu{r.get('src')}-gpu{r.get('dst')}", "peer" if r.get("peer_access") else "host"
                    base = f"xfer/{dev}/d2d/{path}/{_size(r['bytes'])}"
                else:
                    dev, path = g, r.get("mem", "")
                    base = f"xfer/{dev}/{r.get('dir')}/{path}/{_size(r['bytes'])}"
                c = dict(device=dev, dir=r.get("dir"), path=path, bytes=r["bytes"], samples=r.get("latency_samples"))
                if r.get("latency_s", -1) > 0:
                    out.append(dict(key=f"{base}/latency", value=r["latency_s"] * 1e6, unit="µs", spread=0.0,
                                    source=f"q/{q}/results.jsonl",
                                    conditions=dict(c, p10_us=r.get("latency_p10_s", 0) * 1e6,
                                                    p90_us=r.get("latency_p90_s", 0) * 1e6)))
                if r.get("rate_bytes_per_s", 0) > 0:
                    out.append(dict(key=f"{base}/rate", value=r["rate_bytes_per_s"] / 1e9, unit="GB/s", spread=0.0,
                                    source=f"q/{q}/results.jsonl", conditions=dict(c, copies=r.get("copies"))))
    return out


def gpu_overlap(run: Path) -> list[dict]:
    """gpu-overlap: the share of its own rate a GEMM keeps with a copy beside it, and the copy with the GEMM."""
    out, f = [], run / "q" / "gpu-overlap" / "results.jsonl"
    for row in (_json_lines(f) if f.exists() else []):
        if row.get("rc") != 0 or row.get("smoke"):
            continue
        g = str(row.get("params", {}).get("gpu", ""))
        for r in row.get("results") or []:
            if r.get("mode") != "overlap" or not r.get("kernel_alone_ops_per_s") or not r.get("copy_alone_bytes_per_s"):
                continue
            c = dict(gpu=g, copy_engines=r.get("copy_engines"), kernel_alone_tflops=r["kernel_alone_ops_per_s"] / 1e12,
                     copy_alone_gbs=r["copy_alone_bytes_per_s"] / 1e9)
            for name, a, b in (("kernel_share", "kernel_with_copy_ops_per_s", "kernel_alone_ops_per_s"),
                               ("copy_share", "copy_with_kernel_bytes_per_s", "copy_alone_bytes_per_s")):
                out.append(dict(key=f"gpu-overlap/{g}/{name}", value=r.get(a, 0) / r[b], unit="ratio", spread=0.0,
                                source="q/gpu-overlap/results.jsonl", conditions=c))
    return out


def storage(run: Path) -> list[dict]:
    """Read-only drive timings (format 1.7): per drive and fio job its rate, IOPS and latency."""
    try:
        drives = {d["drive"]: d for d in json.loads((run / "storage" / "drives.json").read_text())}
        res = json.loads((run / "storage" / "results.json").read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return []
    out = []
    for key, r in sorted(res.items()):
        drive, _, job = key.partition("-")
        if r.get("rc") != 0 or drive not in drives:
            continue
        d = drives[drive]
        c = dict(drive=drive, job=job, kind=d.get("kind"), capacity_class_gb=d.get("capacity_class_gb"),
                 link=d.get("link"), ios=r.get("ios"))
        for name, v, unit in (("rate", r.get("bytes_per_s", 0) / 1e9, "GB/s"), ("iops", r.get("iops", 0), "IO/s"),
                              ("lat_p50", r.get("lat_p50_us", 0), "µs"), ("lat_p99", r.get("lat_p99_us", 0), "µs")):
            if v:
                out.append(dict(key=f"storage/{drive}/{job}/{name}", value=v, unit=unit, spread=0.0,
                                source="storage/results.json", conditions=c))
    return out


def extract(run: Path) -> dict:
    return dict(version=VERSION, run=run.name,
                measurements=llama(run) + gpu_vram(run) + ladder(run) + mlc(run) + gpu_gemm(run) + cpu_gemm(run)
                + xfer(run) + gpu_overlap(run) + storage(run))


if __name__ == "__main__":
    print(json.dumps(extract(Path(sys.argv[1])), indent=1))
