#!/usr/bin/env python3
"""Builds docs/data.json (the table on the GitHub Pages site) from every run in results/runs/*/run.json.
Reads every format version 1.x (fields a run lacks are left empty).

  python3 tools/index.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "docs" / "data.json"


def flat(v, prefix: str = "") -> list[list[str]]:
    """Nested tool versions as [key, value] rows; hashes shortened to 12 characters."""
    rows: list[list[str]] = []
    if isinstance(v, dict):
        for k, x in v.items():
            rows += flat(x, f"{prefix} / {k}" if prefix else str(k))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            label = x.get("deviceName") if isinstance(x, dict) and x.get("deviceName") else str(i)
            if isinstance(x, dict) and x.get("deviceName"):
                x = {k: y for k, y in x.items() if k != "deviceName"}
            rows += flat(x, f"{prefix} / {label}" if prefix else label)
    elif v not in (None, "", {}):
        s = str(v)
        if len(s) in (40, 64) and all(c in "0123456789abcdef" for c in s):
            s = s[:12]
        rows.append([prefix, s])
    return rows


def summary(steps: dict, man: dict) -> dict:
    """The few versions worth a column: NVIDIA driver, kernel, llama.cpp commit, CUDA, Mesa."""
    rt = man.get("runtime", {})
    pk = man.get("system", {}).get("packages", {})
    llama = man.get("seed", {}).get("llama.cpp", {}) or (steps.get("llama-bench", {}).get("tools", {}).get("llama.cpp", {}))
    commits = sorted({b.get("tag") or b.get("commit", "")[:9] for b in llama.values()
                      if isinstance(b, dict) and (b.get("tag") or b.get("commit"))})
    cuda = ""
    for img in man.get("seed", {}).get("images", {}).values():
        for k, v in (img.get("packages") or {}).items():
            if k.startswith("cuda-nvcc"):
                cuda = v.split("-")[0]
    return {k: v for k, v in dict(nvidia_driver=rt.get("nvidia_driver", ""), kernel=rt.get("kernel", ""),
                                  llama_cpp=", ".join(commits), cuda=cuda,
                                  mesa=pk.get("mesa-vulkan-drivers", "")).items() if v}


def row(run_json: Path) -> dict:
    r = json.loads(run_json.read_text())
    man_f = run_json.parent / "manifest.json"
    man = json.loads(man_f.read_text()) if man_f.exists() else {}
    hw = r.get("hardware", {})
    qs = r.get("questions", {})
    st = r.get("stick")
    stick = st if isinstance(st, dict) else dict(build=st or "")
    steps = []
    for name, v in qs.items():
        if not isinstance(v, dict):
            continue
        result = "skipped" if v.get("skipped") else "ok" if v.get("rc") == 0 else "failed"
        steps.append(dict(name=name, result=result, note=v.get("skipped") or v.get("error") or "",
                          tools=[x for x in flat(v.get("tools", {})) if not x[0].startswith("stick")]))
    return dict(run=r.get("run", run_json.parent.name), date=(r.get("start_utc") or "")[:10], machine=r.get("fingerprint", ""),
                cpu=hw.get("cpu", ""), board=hw.get("board", ""), ram_gib=hw.get("ram_gib"),
                gpus=[g.split(",")[0].strip() for g in hw.get("gpus", [])],
                stick=stick.get("version", ""), stick_build=stick.get("build", ""), stick_commit=stick.get("commit", ""),
                stick_dirty=bool(stick.get("dirty")), format=(r.get("format") or {}).get("version", ""),
                versions=summary(qs, man), status=r.get("status", ""),
                steps_ok=sum(1 for s in steps if s["result"] == "ok"),
                steps_failed=sum(1 for s in steps if s["result"] == "failed"),
                steps_skipped=sum(1 for s in steps if s["result"] == "skipped"), steps=steps)


def build() -> list[dict]:
    rows = [row(p) for p in sorted(RUNS.glob("*/run.json"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
    return rows


if __name__ == "__main__":
    print(f"{len(build())} runs -> {OUT.relative_to(ROOT)}")
