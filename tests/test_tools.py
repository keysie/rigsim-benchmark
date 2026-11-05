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
    (tmp_path / "a.json").write_text('{"github": "keysie"}')
    assert IX.contributor("a", tmp_path) == {"github": "keysie"} and IX.contributor("c", tmp_path) == {}
    html = (ROOT / "docs" / "index.html").read_text()
    assert 'href="https://github.com/${encodeURIComponent(c.github)}"' in html and "Submitted by someone" in html


def test_conventional_commit_checker():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cc", ROOT / ".github" / "scripts" / "check-commits.py")
    C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
    assert C.problem("feat(stick): skip a step\n\nbody") is None and C.problem("chore(runs)!: x") is None
    for bad in ("stick: no type", "Feat: caps", "feat(stick):x", "feat: x\nbody without blank line"):
        assert C.problem(bad), bad
    for wf, msg in (("predictions.yml", 'git commit -q -m "chore(predictions): '), ("runs.yml", '"chore(table): '),
                    ("submission.yml", 'git commit -q -m "chore(runs): ')):
        assert msg in (ROOT / ".github" / "workflows" / wf).read_text()


def test_bots_push_with_their_commit_message_check():
    """The ruleset on main requires 'Check Conventional Commits' on every commit; the bots check their own commit,
    report the check, then push."""
    helper = (ROOT / ".github" / "scripts" / "push-checked.sh").read_text()
    assert 'check-commits.py "$sha^!"' in helper and "check-runs" in helper and 'name="Check Conventional Commits"' in helper
    assert helper.index("check-commits.py") < helper.index("check-runs") < helper.index('git push -q origin "HEAD:$target"')
    for wf in ("runs.yml", "predictions.yml"):
        s = (ROOT / ".github" / "workflows" / wf).read_text()
        assert ".github/scripts/push-checked.sh main" in s and "checks: write" in s and "\n          git push\n" not in s


def test_sycl_values_measured_without_onednn_are_flagged(tmp_path):
    """Format 1.5: a oneAPI llama-bench that ran again without oneDNN carries conditions.sycl_onednn = false."""
    import importlib.util, json as js
    spec = importlib.util.spec_from_file_location("m", ROOT / "tools" / "measurements.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    run = tmp_path / "r"; (run / "oneapi").mkdir(parents=True)
    line = '{"n_prompt":0,"n_gen":128,"avg_ts":%s}\n'
    (run / "oneapi" / "dev0-llama-tiny.jsonl").write_text(line % 38.2)
    (run / "oneapi" / "dev1-llama-tiny.jsonl").write_text(line % 50)
    (run / "run.json").write_text(js.dumps(dict(questions=dict(oneapi=dict(runs={
        "dev0-llama-tiny": dict(rc=0, retry=dict(env={"GGML_SYCL_ENABLE_DNN": "0"})), "dev1-llama-tiny": dict(rc=0),
        "dev0-llama-big": dict(rc=0, onednn_off=dict(after="dev0-llama-tiny"))})))))
    (run / "oneapi" / "dev0-llama-big.jsonl").write_text(line % 9)
    got = {m["key"]: m["conditions"] for m in M.extract(run)["measurements"]}
    assert got["llama-bench/sycl/dev0/tiny/tg128"]["sycl_onednn"] is False
    assert "sycl_onednn" not in got["llama-bench/sycl/dev1/tiny/tg128"]
    assert got["llama-bench/sycl/dev0/big/tg128"]["sycl_onednn"] is False


def test_gemm_values_with_clock_isa_and_unsupported_left_out(tmp_path):
    import importlib.util, json as js
    spec = importlib.util.spec_from_file_location("m", ROOT / "tools" / "measurements.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    run = tmp_path / "r"
    g = run / "q" / "gpu-gemm-shapes"; g.mkdir(parents=True)
    (g / "baseline.json").write_text(js.dumps(dict(gpus=[dict(name="gpu0", bus="00000000:01:00.0")])))
    rows = [dict(rc=0, params=dict(gpu="gpu0", prec="bf16", shape="1x4096x4096"),
                 results=[dict(mode="gemm", prec="bf16", m=1, n=4096, k=4096, ops_per_s=2e12)],
                 samples=dict(gpu0={"clocks.sm": dict(median=1710.0), "power.draw": dict(median=180.0)})),
            dict(rc=0, params=dict(gpu="gpu0", prec="fp8", shape="1x4096x4096"),
                 results=[dict(mode="gemm", prec="fp8", m=1, n=4096, k=4096, unsupported=True)])]
    (g / "results.jsonl").write_text("".join(js.dumps(r) + "\n" for r in rows))
    c = run / "q" / "cpu-gemm-shapes"; c.mkdir(parents=True)
    (c / "results.jsonl").write_text(js.dumps(dict(rc=0, params=dict(prec="bf16", shape="512x4096x4096", threads=0),
        results=[dict(mode="cgemm", prec="bf16", n=4096, m=512, nn=4096, k=4096, threads=8, ops_per_s=6.35e11,
                      isa=4079, impl="brg:avx512_core_amx", lib="onednn")])) + "\n")
    old = run / "q" / "cpu-gemm"; old.mkdir(parents=True)                    # before shapes: square --n only
    (old / "results.jsonl").write_text(js.dumps(dict(rc=0, params=dict(prec="fp32", n=512, threads=1),
        results=[dict(mode="cgemm", prec="fp32", n=512, threads=1, ops_per_s=5.27e10, isa=15, lib="openblas")])) + "\n")
    got = {m["key"]: m for m in M.extract(run)["measurements"]}
    b = got["gpu-gemm/gpu0/bf16/1x4096x4096"]
    assert b["value"] == 2.0 and b["unit"] == "TFLOP/s" and b["conditions"]["sm_clock_mhz"] == 1710.0
    assert b["conditions"]["pci"] == "0000:01:00.0" and not any("fp8" in k for k in got)
    a = got["cpu-gemm/bf16/512x4096x4096/t8"]
    assert abs(a["value"] - 635) < 1 and a["conditions"]["impl"] == "brg:avx512_core_amx"
    assert "cpu-gemm/fp32/512x512x512/t1" in got and M.VERSION == "5"


def test_transfer_sweep_and_overlap_values(tmp_path):
    import importlib.util, json as js
    spec = importlib.util.spec_from_file_location("m", ROOT / "tools" / "measurements.py")
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    run = tmp_path / "r"
    for q, rows in (("pcie-xfer", [dict(rc=0, params=dict(gpu="gpu0", dir="h2d", mem="pinned"), results=[
                        dict(mode="xfer", dir="h2d", mem="pinned", src=-1, dst=0, peer_access=False, bytes=4096,
                             latency_s=9e-6, latency_p10_s=8e-6, latency_p90_s=1.2e-5, latency_samples=900, copies=4000,
                             rate_bytes_per_s=1.1e9),
                        dict(mode="xfer", dir="h2d", mem="pinned", bytes=268435456, latency_s=0.0215,
                             rate_bytes_per_s=2.5e10)])]),
                    ("gpu-p2p-xfer", [dict(rc=0, params=dict(gpu="@together", peer="off"), results=[
                        dict(mode="xfer", dir="d2d", mem="device", src=0, dst=1, peer_access=False, bytes=1 << 20,
                             latency_s=6e-5, rate_bytes_per_s=1.2e10)])]),
                    ("gpu-overlap", [dict(rc=0, params=dict(gpu="gpu0"), results=[
                        dict(mode="overlap", kernel_alone_ops_per_s=1e14, copy_alone_bytes_per_s=2.5e10,
                             kernel_with_copy_ops_per_s=9.5e13, copy_with_kernel_bytes_per_s=2.4e10, copy_engines=2)])])):
        d = run / "q" / q; d.mkdir(parents=True)
        (d / "results.jsonl").write_text("".join(js.dumps(r) + "\n" for r in rows))
    got = {m["key"]: m for m in M.extract(run)["measurements"]}
    assert got["xfer/gpu0/h2d/pinned/4KiB/latency"]["value"] == 9.0 and got["xfer/gpu0/h2d/pinned/4KiB/rate"]["unit"] == "GB/s"
    assert got["xfer/gpu0/h2d/pinned/256MiB/rate"]["value"] == 25.0
    assert "xfer/gpu0-gpu1/d2d/host/1MiB/latency" in got
    assert got["gpu-overlap/gpu0/kernel_share"]["value"] == 0.95 and got["gpu-overlap/gpu0/copy_share"]["value"] == 0.96
