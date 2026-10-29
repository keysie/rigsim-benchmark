#!/usr/bin/env python3
"""rigsim's predictions for the measured runs (predictions/README.md has the contract).

  python3 tools/predict.py [--rigsim <checkout>] [--tag vX.Y.Z]

Finds rigsim's newest engine release (tag vX.Y.Z), and for every run in results/runs/ that has no prediction file
for that tag yet, asks rigsim to predict the run's measurements (tools/measurements.py). Writes
predictions/<tag>/<run>.json; files of older tags are never touched. Without any engine tag it does nothing."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import measurements  # noqa: E402

RIGSIM = "https://github.com/keysie/rigsim"
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "predictions"
TIMEOUT_S = 600


def vkey(tag: str) -> tuple[int, int, int]:
    m = TAG.match(tag)
    return tuple(int(x) for x in m.groups()) if m else (-1, -1, -1)


def newest_tag(url: str = RIGSIM) -> str | None:
    out = subprocess.run(["git", "ls-remote", "--tags", "--refs", url], capture_output=True, text=True).stdout
    tags = [l.split("refs/tags/")[-1] for l in out.splitlines() if "refs/tags/" in l]
    tags = [t for t in tags if TAG.match(t)]
    return max(tags, key=vkey) if tags else None


def checkout(tag: str, where: Path, url: str = RIGSIM) -> str:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", tag, url, str(where)], check=True)
    return subprocess.run(["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def check_output(out: dict, meas: dict) -> str | None:
    """Predictions only for keys that were measured, in the measured unit, as finite numbers."""
    known = {m["key"]: m["unit"] for m in meas["measurements"]}
    for p in out.get("predictions", []):
        if p.get("key") not in known:
            return f"prediction for {p.get('key')!r}, which was not measured"
        if p.get("unit") != known[p["key"]]:
            return f"{p['key']}: unit {p.get('unit')!r}, measured in {known[p['key']]!r}"
        if not isinstance(p.get("value"), (int, float)) or not math.isfinite(p["value"]):
            return f"{p['key']}: value {p.get('value')!r} is not a finite number"
    return None


def predict_run(run: Path, rigsim: Path, tag: str, commit: str) -> dict:
    meas = measurements.extract(run)
    rec = dict(rigsim=dict(tag=tag, commit=commit), run=run.name, measurements_version=meas["version"],
               computed_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    if not meas["measurements"]:
        return dict(rec, predictions=[], unsupported=[], note="the run has no comparable values")
    with tempfile.TemporaryDirectory() as t:
        mf, of = Path(t) / "measurements.json", Path(t) / "out.json"
        mf.write_text(json.dumps(meas))
        try:
            r = subprocess.run([sys.executable, "-m", "app.predict", "--run", str(run), "--measurements", str(mf),
                                "--out", str(of)], cwd=rigsim / "backend", capture_output=True, text=True,
                               timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return dict(rec, error=f"rigsim did not finish within {TIMEOUT_S} s")
        if r.returncode != 0 or not of.exists():
            return dict(rec, error=f"rigsim failed (rc={r.returncode}): {(r.stderr or r.stdout).strip()[-400:]}")
        try:
            out = json.loads(of.read_text())
        except ValueError as e:
            return dict(rec, error=f"rigsim wrote no valid JSON ({e})")
    bad = check_output(out, meas)
    if bad:
        return dict(rec, error=f"rigsim's output rejected: {bad}")
    return dict(rec, predictions=out.get("predictions", []), unsupported=out.get("unsupported", []))


def needs_prediction(f: Path) -> bool:
    """No file yet for this tag, or one made from an older set of measured values (measurements.VERSION): the
    newest release then predicts the run again, so values added to the measurements get predicted too. Files of
    older releases are never touched."""
    if not f.exists():
        return True
    try:
        return json.loads(f.read_text()).get("measurements_version") != measurements.VERSION
    except ValueError:
        return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rigsim", type=Path, help="an existing rigsim checkout (at --tag) instead of cloning")
    ap.add_argument("--tag", help="rigsim tag to use (default: the newest vX.Y.Z)")
    a = ap.parse_args()
    tag = a.tag or newest_tag()
    if not tag:
        print("no rigsim engine release (vX.Y.Z) yet: nothing to predict")
        return 0
    todo = [r for r in sorted(RUNS.glob("*/run.json")) if needs_prediction(OUT / tag / f"{r.parent.name}.json")]
    if not todo:
        print(f"rigsim {tag}: every run already predicted")
        return 0
    with tempfile.TemporaryDirectory() as t:
        rigsim = a.rigsim or Path(t) / "rigsim"
        commit = checkout(tag, rigsim) if not a.rigsim else subprocess.run(
            ["git", "-C", str(rigsim), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        req = rigsim / "backend" / "requirements.txt"
        if req.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)], check=False)
        if not (rigsim / "backend" / "app" / "predict.py").exists():
            print(f"rigsim {tag} has no prediction entry point (backend/app/predict.py): nothing written")
            return 0
        (OUT / tag).mkdir(parents=True, exist_ok=True)
        for rj in todo:
            rec = predict_run(rj.parent, rigsim, tag, commit)
            (OUT / tag / f"{rj.parent.name}.json").write_text(json.dumps(rec, indent=1) + "\n")
            print(f"{tag} {rj.parent.name}: " + (rec.get("error") or f"{len(rec.get('predictions', []))} predictions"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
