"""tools/index.py builds the table the GitHub Pages site shows."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("index", ROOT / "tools" / "index.py")
index = importlib.util.module_from_spec(spec); spec.loader.exec_module(index)


def test_index_rows_from_run_json(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "20261027T234840Z-2ad20e343d87-7b6a"; run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(dict(
        run=run.name, fingerprint="2ad20e343d87", start_utc="2026-10-27T23:48:40+00:00", stick="abc123", status="done with failures",
        hardware=dict(cpu="13th Gen Intel(R) Core(TM) i7-13700K", board="Z790 AORUS ELITE AX", ram_gib=31.1,
                      gpus=["NVIDIA GeForce RTX 3080, 10240 MiB, 00000000:01:00.0"]),
        questions={"cpu-gemm": dict(rc=0), "gpu-gemm": dict(rc=1), "rocm": dict(skipped="no AMD GPU")})))
    monkeypatch.setattr(index, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(index, "OUT", tmp_path / "data.json")
    rows = index.build()
    assert rows == json.loads((tmp_path / "data.json").read_text())["runs"]
    r = rows[0]
    assert r["date"] == "2026-10-27" and r["gpus"] == ["NVIDIA GeForce RTX 3080"] and r["machine"] == "2ad20e343d87"
    assert (r["steps_ok"], r["steps_failed"], r["steps_skipped"]) == (1, 1, 1)


def test_site_reads_the_generated_table():
    html = (ROOT / "docs" / "index.html").read_text()
    assert 'fetch("data.json")' in html and "results/runs/" in html
    assert json.loads((ROOT / "docs" / "data.json").read_text())["runs"] == []    # until runs are imported
