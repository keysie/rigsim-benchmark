"""tools/validate.py: the checks every contributed run passes."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("validate", ROOT / "tools" / "validate.py")
V = importlib.util.module_from_spec(spec); spec.loader.exec_module(V)
RUN = "20261027T234840Z-2ad20e343d87-7b6a"


def make_run(base: Path, name: str = RUN, **over) -> Path:
    d = base / name
    (d / "inventory").mkdir(parents=True)
    (d / "q" / "cpu-gemm" / "samples").mkdir(parents=True)
    (d / "vkprobe").mkdir(); (d / "llama").mkdir()
    run = dict(run=name, fingerprint=name.split("-")[1], start_utc="2026-10-27T23:48:40+00:00",
               end_utc="2026-10-28T01:30:00+00:00", stick="abc123", status="done with failures",
               hardware=dict(cpu="i7-13700K", board="Z790", ram_gib=31.1, gpus=["NVIDIA GeForce RTX 3080"]),
               questions={"cpu-gemm": dict(rc=0), "rocm": dict(skipped="no AMD GPU"), "gpu-gemm": dict(rc=1)})
    run.update(over)
    (d / "run.json").write_text(json.dumps(run))
    (d / "stick.log").write_text("23:48:40 run\n")
    (d / "samples-drm.jsonl").write_text('{"t": 1}\n')
    for f in ("cpuinfo.txt", "dmidecode.txt", "lspci.txt"):
        (d / "inventory" / f).write_text("Base Board Information\n\tManufacturer: Gigabyte\n")
    (d / "inventory" / "summary.json").write_text("{}")
    (d / "inventory" / "lshw.json").write_text('{"id": "x", "serial": "<mac>"}')
    (d / "q" / "cpu-gemm" / "results.jsonl").write_text("{}\n")
    (d / "q" / "cpu-gemm" / "samples" / "0123456789.csv").write_text("t,v\n")
    (d / "vkprobe" / "dev0-gemm-f16-4096x4096x4096.jsonl").write_text("{}\n--- stderr ---\nnote\n")
    (d / "llama" / "cuda-gpu0-ops-mul_mat.txt").write_text("MUL_MAT ...\n")
    return d


def test_a_stick_run_passes(tmp_path):
    assert V.check_run(make_run(tmp_path)) == []


@pytest.mark.parametrize("mutate, expect", [
    (lambda d: (d / "notes.txt").write_text("hi"), "not a file the stick writes"),
    (lambda d: (d / "inventory" / "extra.txt").write_text("x"), "not a file the stick writes"),
    (lambda d: (d / "q" / "cpu-gemm" / "samples" / "x.csv").write_text("x"), "not a file the stick writes"),
    (lambda d: (d / "run.json").write_text(json.dumps(dict(json.loads((d / "run.json").read_text()), status="stopped"))), "not complete"),
    (lambda d: (d / "run.json").write_text(json.dumps(dict(json.loads((d / "run.json").read_text()), status="error: X"))), "not complete"),
    (lambda d: (d / "run.json").write_text(json.dumps(dict(json.loads((d / "run.json").read_text()), end_utc=None))), "missing 'end_utc'"),
    (lambda d: (d / "inventory" / "summary.json").unlink(), "inventory/summary.json: missing"),
    (lambda d: (d / "inventory" / "dmidecode.txt").write_text("\tSerial Number: ABC123\n"), "identifier line"),
    (lambda d: (d / "stick.log").write_text("eth0 aa:bb:cc:dd:ee:ff\n"), "MAC address"),
    (lambda d: (d / "inventory" / "summary.json").write_text("{broken"), "not valid JSON"),
    (lambda d: (d / "mlc").mkdir() or (d / "mlc" / "idle_latency.txt").write_bytes(b"\xff\xfe\x00"), "not UTF-8"),
    (lambda d: (d / "llama" / "x.log").symlink_to("/etc/passwd"), "symbolic link"),
])
def test_anything_else_is_rejected(tmp_path, mutate, expect):
    d = make_run(tmp_path)
    mutate(d)
    errs = V.check_run(d)
    assert errs and any(expect in e for e in errs), errs


def test_folder_name_and_run_json_must_agree(tmp_path):
    assert "not a run folder name" in V.check_run(make_run(tmp_path, "my-run"))[0]
    d = make_run(tmp_path, RUN, run="20261027T000000Z-2ad20e343d87-0000")
    assert any("names run" in e for e in V.check_run(d))


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / "results" / "runs").mkdir(parents=True); (r / "README.md").write_text("x")
    (r / "results" / "runs" / ".gitkeep").write_text("")
    g = lambda *a: subprocess.run(["git", *a], cwd=r, check=True, capture_output=True)
    g("init", "-q", "-b", "main"); g("config", "user.email", "t@t"); g("config", "user.name", "t")
    g("add", "-A"); g("commit", "-qm", "base"); g("switch", "-qc", "pr")
    return r, g


def test_pull_request_may_only_add_new_run_folders(tmp_path):
    r, g = _repo(tmp_path)
    make_run(r / "results" / "runs"); g("add", "-A"); g("commit", "-qm", "run")
    assert V.check_pr("main", r) == []
    (r / "README.md").write_text("changed"); (r / "docs").mkdir(); (r / "docs" / "data.json").write_text("{}")
    g("add", "-A"); g("commit", "-qm", "more")
    errs = V.check_pr("main", r)
    assert any("README.md: changed" in e for e in errs) and any("docs/data.json: outside results/runs" in e for e in errs)


def test_pull_request_cannot_touch_existing_runs(tmp_path):
    r, g = _repo(tmp_path)
    g("switch", "-q", "main"); make_run(r / "results" / "runs"); g("add", "-A"); g("commit", "-qm", "merged run")
    g("switch", "-qC", "pr")
    (r / "results" / "runs" / RUN / "stick.log").write_text("edited\n"); g("add", "-A"); g("commit", "-qm", "edit")
    assert any("changed" in e for e in V.check_pr("main", r))


def test_contribution_tools_and_licence_statement():
    readme = (ROOT / "README.md").read_text()
    assert "## Contributing runs" in readme and "stick/README.md" in readme.split("## Contributing runs")[1]
    assert "By sharing runs you agree that they are published under [CC BY 4.0]" in readme
    assert "CC BY 4.0" in (ROOT / ".github" / "pull_request_template.md").read_text()
    wf = (ROOT / ".github" / "workflows" / "runs.yml").read_text()
    assert 'validate.py --pr "origin/${{ github.base_ref }}"' in wf and "tools/index.py" in wf
    assert shutil.which("bash") and subprocess.run(["bash", "-n", str(ROOT / "tools" / "contribute.sh")]).returncode == 0
