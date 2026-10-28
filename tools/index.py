#!/usr/bin/env python3
"""Builds docs/data.json (the table on the GitHub Pages site) from every run in results/runs/*/run.json.

  python3 tools/index.py"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "docs" / "data.json"


def row(run_json: Path) -> dict:
    r = json.loads(run_json.read_text())
    hw = r.get("hardware", {})
    qs = r.get("questions", {})
    ok = sum(1 for v in qs.values() if isinstance(v, dict) and not v.get("skipped") and v.get("rc") == 0)
    failed = sum(1 for v in qs.values() if isinstance(v, dict) and not v.get("skipped") and v.get("rc") not in (0, None))
    skipped = sum(1 for v in qs.values() if isinstance(v, dict) and v.get("skipped"))
    return dict(run=r.get("run", run_json.parent.name), date=(r.get("start_utc") or "")[:10], machine=r.get("fingerprint", ""),
                cpu=hw.get("cpu", ""), board=hw.get("board", ""), ram_gib=hw.get("ram_gib"),
                gpus=[g.split(",")[0].strip() for g in hw.get("gpus", [])], stick=r.get("stick", ""),
                status=r.get("status", ""), steps_ok=ok, steps_failed=failed, steps_skipped=skipped)


def build() -> list[dict]:
    rows = [row(p) for p in sorted(RUNS.glob("*/run.json"))]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(runs=rows), indent=1) + "\n")
    return rows


if __name__ == "__main__":
    print(f"{len(build())} runs -> {OUT.relative_to(ROOT)}")
