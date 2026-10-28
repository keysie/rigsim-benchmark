#!/usr/bin/env python3
"""Checks runs before they enter results/runs/: the exact layout the stick writes (no other file), completeness,
nothing identifying, and — for a pull request — that it only adds new run folders.

  python3 tools/validate.py results/runs/<run> [...]     check run folders
  python3 tools/validate.py --pr origin/main             check a branch against main (the CI does this)

Exit code 0 = all good; every problem is printed with the file it concerns."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "stick" / "payload"))
from stick import INVENTORY, INVENTORY_EXTRA, MAC, PRIVATE_IPV4, SENSITIVE  # noqa: E402  (the stick's own definitions: one source of truth)

FORMAT_NAME = "rigsim-stick-run"
KNOWN_VERSIONS = {"1.0", "1.1", "1.2"}                           # results/FORMAT.md: minor versions only add
RUN_ID = re.compile(r"^(\d{8}T\d{6}Z)-([0-9a-f]{12})-([0-9a-f]{4})$")
COMPLETE = {"done", "done with failures"}          # finished; "stopped" and "error: ..." are not complete
MAX_FILE = 25 * 2**20
MAX_RUN = 300 * 2**20

N = r"[A-Za-z0-9][A-Za-z0-9._+-]*"                  # a name the stick builds from questions, devices, models, ops
INVENTORY_FILES = sorted({name for name, _ in INVENTORY} | set(INVENTORY_EXTRA))
RUNNER_FILES = "baseline.json|restore.json|results.jsonl|session.json|setup.log|state.json|stopped.json|testdirs.json|runner.log"
LAYOUT = [re.compile(p) for p in (
    r"run\.json", r"manifest\.json", r"stick\.log", r"samples-drm\.jsonl", r"samples-nvidia\.csv",
    r"inventory/spd-\d+-[0-9a-f]{4}\.hex",
    r"inventory/(" + "|".join(map(re.escape, INVENTORY_FILES)) + r")",
    rf"q/{N}/({RUNNER_FILES})", rf"q/{N}/samples/[0-9a-f]{{10}}\.csv",
    r"mlc/(idle_latency|max_bandwidth|loaded_latency|latency_matrix|bandwidth_matrix)\.txt",
    rf"opencl/clinfo-list\.txt", rf"opencl/clpeak-{N}\.(txt|json)",
    rf"llama/{N}\.(jsonl|log|txt)",
    rf"vkprobe/(list|dev\d+-{N})\.jsonl",
    rf"(rocm|oneapi)/(info\.jsonl|image\.txt|sycl-ls\.txt|dev\d+-{N}\.(jsonl|txt))",
    rf"blender/(devices\.txt|{N}\.(json|log))",
)]
STRICT_JSON = re.compile(r"(run|manifest|inventory/summary|inventory/pcie|inventory/lshw|q/[^/]+/(baseline|restore|session|state|stopped|testdirs)"
                         r"|opencl/clpeak-[^/]+)\.json$")


def check_run(d: Path) -> list[str]:
    errs: list[str] = []
    e = lambda p, msg: errs.append(f"{d.name}/{p}: {msg}" if p else f"{d.name}: {msg}")
    m = RUN_ID.match(d.name)
    if not m:
        return [f"{d.name}: not a run folder name (<UTC time>-<12 hex>-<4 hex>)"]
    if not d.is_dir():
        return [f"{d.name}: not a folder"]
    total = 0
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d).as_posix()
        if p.is_symlink():
            e(rel, "symbolic link"); continue
        if p.is_dir():
            continue
        if not any(r.fullmatch(rel) for r in LAYOUT):
            e(rel, "not a file the stick writes"); continue
        size = p.stat().st_size
        total += size
        if size > MAX_FILE:
            e(rel, f"{size / 2**20:.0f} MiB, more than {MAX_FILE // 2**20} MiB"); continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            e(rel, "not UTF-8 text"); continue
        if STRICT_JSON.search(rel):
            try:
                json.loads(text)
            except ValueError as x:
                e(rel, f"not valid JSON ({x})")
        bad = [l for l in text.splitlines() if MAC.search(l)]
        if bad:
            e(rel, f"contains a MAC address: {bad[0].strip()[:80]}")
        bad = [l for l in text.splitlines() if PRIVATE_IPV4.search(l)]
        if bad:
            e(rel, f"contains a local IP address: {bad[0].strip()[:80]}")
        if rel.startswith("inventory/") and rel.endswith(".txt"):
            bad = [l for l in text.splitlines() if SENSITIVE.search(l)]
            if bad:
                e(rel, f"contains an identifier line: {bad[0].strip()[:80]}")
    if total > MAX_RUN:
        e("", f"{total / 2**20:.0f} MiB in total, more than {MAX_RUN // 2**20} MiB")
    # completeness
    rj = d / "run.json"
    if not rj.is_file():
        e("run.json", "missing"); return errs
    try:
        r = json.loads(rj.read_text())
    except ValueError:
        return errs                                                     # reported above
    fmt = r.get("format") or {}
    if fmt.get("name") != FORMAT_NAME:
        e("run.json", "no run format (stick builds before 0.9.0 are not accepted)")
        return errs
    if fmt.get("version") not in KNOWN_VERSIONS:
        major = str(fmt.get("version", "")).split(".")[0]
        e("run.json", f"format {fmt.get('version')!r} is " + ("newer than this repository knows: update it"
                                                              if major == "1" else "not a known version"))
        return errs
    st = r.get("stick") or {}
    if not (isinstance(st, dict) and re.fullmatch(r"\d+\.\d+\.\d+", str(st.get("version", "")))
            and re.fullmatch(r"[0-9a-f]{40}", str(st.get("commit", ""))) and st.get("scripts_sha256")):
        e("run.json", "stick without version, commit and scripts_sha256")
    if r.get("run") != d.name:
        e("run.json", f"names run {r.get('run')!r}, folder is {d.name!r}")
    if r.get("fingerprint") != m.group(2):
        e("run.json", "fingerprint does not match the folder name")
    if r.get("status") not in COMPLETE:
        e("run.json", f"run not complete: status {r.get('status')!r}")
    for k in ("start_utc", "end_utc", "stick", "hardware", "questions"):
        if not r.get(k):
            e("run.json", f"missing {k!r}")
    hw = r.get("hardware") or {}
    for k in ("cpu", "board", "ram_gib", "gpus"):
        if k not in hw:
            e("run.json", f"hardware without {k!r}")
    for q, v in (r.get("questions") or {}).items():
        if not isinstance(v, dict) or not ("rc" in v or "skipped" in v):
            e("run.json", f"step {q!r} has no result")
        elif not isinstance(v.get("tools"), dict) or "stick" not in v["tools"]:
            e("run.json", f"step {q!r} has no tool versions")
    try:
        man = json.loads((d / "manifest.json").read_text())
        for k in ("system", "seed", "runtime"):
            if not isinstance(man.get(k), dict):
                e("manifest.json", f"no {k!r} part")
    except (OSError, ValueError):
        pass                                                            # missing / invalid: reported below / above
    for f in ("manifest.json", "stick.log", "inventory/summary.json", "inventory/cpuinfo.txt", "inventory/dmidecode.txt"):
        if not (d / f).is_file():
            e(f, "missing")
    return errs


def check_pr(base: str, root: Path = ROOT) -> list[str]:
    """Only additions, only inside new run folders under results/runs/."""
    out = subprocess.run(["git", "diff", "--name-status", "--no-renames", f"{base}...HEAD"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    errs, runs = [], set()
    existing = set(subprocess.run(["git", "ls-tree", "--name-only", f"{base}:results/runs"], cwd=root,
                                  capture_output=True, text=True).stdout.split())
    for line in out.splitlines():
        status, path = line.split("\t", 1)
        parts = path.split("/")
        if status != "A":
            errs.append(f"{path}: {'changed' if status == 'M' else 'deleted' if status == 'D' else status} — a contribution only adds new run folders")
            continue
        if len(parts) < 4 or parts[:2] != ["results", "runs"]:
            errs.append(f"{path}: outside results/runs/<run>/ — a contribution only adds run folders")
            continue
        if parts[2] in existing:
            errs.append(f"{path}: run {parts[2]} is already in the repository")
            continue
        runs.add(parts[2])
    if not runs and not errs:
        errs.append("no run folders added")
    for run in sorted(runs):
        errs += check_run(root / "results" / "runs" / run)
    return errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--pr", metavar="BASE", help="check the commits since BASE (e.g. origin/main)")
    a = ap.parse_args()
    errs = check_pr(a.pr) if a.pr else [x for d in a.runs for x in check_run(d)]
    if not a.pr and not a.runs:
        ap.error("give run folders or --pr BASE")
    for x in errs:
        print(f"::error::{x}" if a.pr else f"ERROR {x}")
    print("all runs valid" if not errs else f"{len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
