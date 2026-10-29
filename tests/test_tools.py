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


def test_site_reads_the_generated_table(tmp_path, monkeypatch):
    html = (ROOT / "docs" / "index.html").read_text()
    assert 'fetch("data.json", { cache: "no-store" })' in html and "results/runs/" in html
    folders = {p.name for p in (ROOT / "results" / "runs").iterdir() if p.is_dir()}
    # the committed table may lag behind: a submission's pull request adds run folders, and the table is rebuilt
    # only after the merge (workflow runs.yml) — but it never lists a run that is not there
    committed = {r["run"] for r in json.loads((ROOT / "docs" / "data.json").read_text())["runs"]}
    assert committed <= folders
    # and a rebuild covers every run folder
    monkeypatch.setattr(index, "OUT", tmp_path / "data.json")
    assert {r["run"] for r in index.build()} == folders


def test_measurements_carry_the_cards_pci_address(tmp_path):
    import importlib.util, json as js
    spec = importlib.util.spec_from_file_location("m", ROOT / "tools" / "measurements.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    run = tmp_path / "r"; (run / "llama").mkdir(parents=True); (run / "vkprobe").mkdir()
    (run / "run.json").write_text(js.dumps(dict(hardware=dict(gpus=[
        "NVIDIA GeForce RTX 3090, 24576 MiB, 00000000:17:00.0", "NVIDIA GeForce RTX 3090, 24576 MiB, 00000000:65:00.0",
        "Intel Corporation Raptor Lake-S GT1 [UHD Graphics 770] [8086:a780]"]))))
    (run / "vkprobe" / "list.jsonl").write_text('{"mode":"list","index":0,"pci":"0000:65:00.0"}\n{"mode":"list","index":1,"pci":"0000:17:00.0"}\n')
    line = '{"n_prompt":0,"n_gen":128,"avg_ts":%s,"gpu_info":"NVIDIA GeForce RTX 3090"}\n'
    for name, v in (("cuda-gpu0", 150), ("cuda-gpu1", 140), ("vulkan-dev0", 120), ("vulkan-dev1", 130)):
        (run / "llama" / f"{name}-m.jsonl").write_text(line % v)
    got = {m["key"]: m["conditions"]["pci"] for m in M.extract(run)["measurements"]}
    assert got == {"llama-bench/cuda/gpu0/m/tg128": "0000:17:00.0", "llama-bench/cuda/gpu1/m/tg128": "0000:65:00.0",
                   "llama-bench/vulkan/dev0/m/tg128": "0000:65:00.0", "llama-bench/vulkan/dev1/m/tg128": "0000:17:00.0"}


def test_page_shows_the_credit(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ix3", ROOT / "tools" / "index.py")
    IX = importlib.util.module_from_spec(spec); spec.loader.exec_module(IX)
    (tmp_path / "a.json").write_text('{"github": "keysie"}'); (tmp_path / "b.json").write_text('{"withheld": true}')
    assert IX.contributor("a", tmp_path) == {"github": "keysie"} and IX.contributor("b", tmp_path) == {"withheld": True}
    assert IX.contributor("c", tmp_path) == {}
    html = (ROOT / "docs" / "index.html").read_text()
    assert 'href="https://github.com/${encodeURIComponent(c.github)}"' in html and "withheld at their request" in html
