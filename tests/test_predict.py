"""tools/predict.py and the prediction columns of the table."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
load = lambda name, path: (lambda s: (s.loader.exec_module(m := importlib.util.module_from_spec(s)), m)[1])(
    importlib.util.spec_from_file_location(name, path))
P = load("predict", ROOT / "tools" / "predict.py")
IX = load("ix2", ROOT / "tools" / "index.py")
TV = load("tv2", ROOT / "tests" / "test_validate.py")

FAKE_RIGSIM = '''
import argparse, json
a = argparse.ArgumentParser(); [a.add_argument(x) for x in ("--run", "--measurements", "--out")]; a = a.parse_args()
m = json.load(open(a.measurements))["measurements"]
preds = [dict(key=x["key"], value=x["value"] * 0.9, unit=x["unit"], calibration=x["key"].startswith("mlc"), basis="fake")
         for x in m if not x["key"].endswith("pp512")]
json.dump(dict(predictions=preds, unsupported=[dict(key=x["key"], reason="not modelled") for x in m if x["key"].endswith("pp512")]),
          open(a.out, "w"))
'''


def run_with_llama(tmp_path):
    d = TV.make_run(tmp_path / "runs")
    (d / "llama" / "cuda-gpu0-llama-2-7b-q4_0.jsonl").write_text(
        '{"n_prompt": 512, "n_gen": 0, "avg_ts": 5000.0, "gpu_info": "NVIDIA GeForce RTX 3090"}\n'
        '{"n_prompt": 0, "n_gen": 128, "avg_ts": 160.0, "gpu_info": "NVIDIA GeForce RTX 3090"}\n')
    (d / "mlc").mkdir()
    (d / "mlc" / "idle_latency.txt").write_text("Each iteration took 218.9 base frequency clocks (\t64.1\tns)\n")
    return d


def fake_rigsim(tmp_path, code=FAKE_RIGSIM):
    r = tmp_path / "rigsim"; (r / "backend" / "app").mkdir(parents=True)
    (r / "backend" / "app" / "__init__.py").write_text("")
    (r / "backend" / "app" / "predict.py").write_text(code)
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    return r


def test_tags_and_measurements(tmp_path):
    assert max(["v0.1.0", "v0.10.0", "v0.9.3", "engine-x"], key=P.vkey) == "v0.10.0"
    keys = {m["key"]: m for m in P.measurements.extract(run_with_llama(tmp_path))["measurements"]}
    assert keys["llama-bench/cuda/gpu0/llama-2-7b-q4_0/tg128"]["value"] == 160.0 and keys["mlc/idle_latency"]["value"] == 64.1


def test_a_run_predicted_and_shown(tmp_path, monkeypatch):
    run, rig = run_with_llama(tmp_path), fake_rigsim(tmp_path)
    rec = P.predict_run(run, rig, "v0.1.0", "c" * 40)
    assert "error" not in rec and len(rec["predictions"]) == 2 and rec["unsupported"][0]["reason"] == "not modelled"
    pred = tmp_path / "predictions"
    (pred / "v0.1.0").mkdir(parents=True); (pred / "v0.1.0" / f"{run.name}.json").write_text(json.dumps(rec))
    older = dict(rec, rigsim=dict(tag="v0.0.9"), predictions=[dict(p, value=p["value"] * 2) for p in rec["predictions"]])
    (pred / "v0.0.9").mkdir(); (pred / "v0.0.9" / f"{run.name}.json").write_text(json.dumps(older))
    rows, summary = IX.results(run, pred)
    tg = next(r for r in rows if r["key"].endswith("tg128"))
    assert [p["tag"] for p in tg["predictions"]] == ["v0.1.0", "v0.0.9"]                 # newest first, older kept
    assert abs(tg["predictions"][0]["error"] + 0.1) < 1e-9 and abs(tg["predictions"][1]["error"] - 0.8) < 1e-9
    pp = next(r for r in rows if r["key"].endswith("pp512"))
    assert pp["predictions"][0]["unsupported"] == "not modelled"
    assert summary == dict(tag="v0.1.0", n=1, median_abs_error=summary["median_abs_error"])  # mlc = calibration: not counted
    assert abs(summary["median_abs_error"] - 0.1) < 1e-9


def test_bad_rigsim_output_is_an_error_not_data(tmp_path):
    run = run_with_llama(tmp_path)
    bad = FAKE_RIGSIM.replace('unit=x["unit"]', 'unit="GB/s"')
    assert "unit" in P.predict_run(run, fake_rigsim(tmp_path, bad), "v0.1.0", "c")["error"]
    crash = "raise SystemExit(3)\n"
    (tmp_path / "rigsim").rename(tmp_path / "old")
    assert "rigsim failed (rc=3)" in P.predict_run(run, fake_rigsim(tmp_path, crash), "v0.1.0", "c")["error"]


def test_nothing_without_an_engine_release(monkeypatch, capsys):
    monkeypatch.setattr(P, "newest_tag", lambda url=None: None)
    monkeypatch.setattr(sys, "argv", ["predict.py"])
    assert P.main() == 0 and "no rigsim engine release" in capsys.readouterr().out
    wf = (ROOT / ".github" / "workflows" / "predictions.yml").read_text()
    assert "cron:" in wf and "tools/predict.py" in wf and "tools/index.py" in wf


def test_newest_release_repredicts_when_the_measured_values_change(tmp_path):
    f = tmp_path / "x.json"
    assert P.needs_prediction(f)
    f.write_text(json.dumps(dict(measurements_version=P.measurements.VERSION)))
    assert not P.needs_prediction(f)
    f.write_text(json.dumps(dict(measurements_version="1")))
    assert P.needs_prediction(f)
