"""tools/submission.py: a zip dropped into a 'Submit runs' issue."""
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("submission", ROOT / "tools" / "submission.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
spec = importlib.util.spec_from_file_location("tv", ROOT / "tests" / "test_validate.py")
TV = importlib.util.module_from_spec(spec); spec.loader.exec_module(TV)

BODY = """### Archive with the runs

[runs.zip](https://github.com/user-attachments/files/123456/runs.zip)

### Confirmations

- [X] I agree that the runs are published under CC BY 4.0
- [X] The owner of every measured machine agreed to publication
- [X] The run folders are unchanged from the stick
"""


def zip_of(tree: Path, extra: dict | None = None, prefix: str = "runs/") -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for f in sorted(tree.rglob("*")):
            if f.is_file():
                z.write(f, prefix + f.relative_to(tree).as_posix())
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return b.getvalue()


@pytest.fixture
def stick(tmp_path):
    src = tmp_path / "stick"
    TV.make_run(src)
    TV.make_run(src, "20261028T010000Z-2ad20e343d87-aaaa")
    return src


def run(monkeypatch, tmp_path, data, body=BODY):
    monkeypatch.setattr(S, "fetch", lambda url, limit=0: data)
    dest = tmp_path / "repo-runs"; dest.mkdir(exist_ok=True)
    return S.process(body, dest), dest


def test_two_runs_with_os_clutter_become_one_submission(monkeypatch, tmp_path, stick):
    data = zip_of(stick, {"__MACOSX/runs/._x": "junk", "runs/.DS_Store": "junk"})
    (report, added), dest = run(monkeypatch, tmp_path, data)
    assert sorted(added) == sorted(p.name for p in stick.iterdir()) and all((dest / a / "run.json").exists() for a in added)
    assert "2 run(s) accepted" in report and "ignored" in report


def test_a_run_already_in_the_repository_is_skipped(monkeypatch, tmp_path, stick):
    (tmp_path / "repo-runs" / TV.RUN).mkdir(parents=True)
    (report, added), _ = run(monkeypatch, tmp_path, zip_of(stick))
    assert TV.RUN not in added and len(added) == 1 and "already in the repository" in report


@pytest.mark.parametrize("extra, expect", [
    ({"../evil.txt": "x"}, "path outside the archive"),
    ({"/etc/x": "x"}, "path outside the archive"),
    ({"notes.txt": "hello"}, "files outside a run folder"),
])
def test_unsafe_or_foreign_content_refused(monkeypatch, tmp_path, stick, extra, expect):
    (report, added), _ = run(monkeypatch, tmp_path, zip_of(stick, extra))
    assert not added and expect in report


def test_link_in_zip_refused(monkeypatch, tmp_path, stick):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        i = zipfile.ZipInfo(f"runs/{TV.RUN}/stick.log"); i.external_attr = (0o120777 << 16)
        z.writestr(i, "/etc/passwd")
    (report, added), _ = run(monkeypatch, tmp_path, b.getvalue())
    assert not added and "a link" in report


def test_invalid_run_reported_valid_one_kept(monkeypatch, tmp_path, stick):
    (stick / TV.RUN / "notes.txt").write_text("extra")
    (report, added), _ = run(monkeypatch, tmp_path, zip_of(stick))
    assert added == ["20261028T010000Z-2ad20e343d87-aaaa"] and "notes.txt: not a file the stick writes" in report


def test_confirmations_and_zip_link_required(monkeypatch, tmp_path, stick):
    (report, added), _ = run(monkeypatch, tmp_path, zip_of(stick), BODY.replace("- [X] I agree", "- [ ] I agree"))
    assert not added and "tick all confirmations" in report and "CC BY 4.0" in report
    (report, added), _ = run(monkeypatch, tmp_path, zip_of(stick), BODY.replace("https://github.com/user-attachments/files/123456/runs.zip", "https://evil.example/runs.zip"))
    assert not added and "No archive found" in report


def test_size_limits(monkeypatch, tmp_path, stick):
    monkeypatch.setattr(S, "MAX_TOTAL", 1000)
    (report, added), _ = run(monkeypatch, tmp_path, zip_of(stick))
    assert not added and "unpacked" in report


def test_workflow_and_form():
    wf = (ROOT / ".github" / "workflows" / "submission.yml").read_text()
    assert "BODY: ${{ github.event.issue.body }}" in wf and '"$BODY"' in wf            # body via env, never inlined
    assert "tools/submission.py" in wf and "gh pr create" in wf and "Closes #%s" in wf and "index.py" not in wf
    form = (ROOT / ".github" / "ISSUE_TEMPLATE" / "submit-runs.yml").read_text()
    for c in S.CONFIRMATIONS:
        assert c in form
    assert "label: Archive with the runs" in form and "### Archive with the runs" in wf



def tgz_of(tree, extra=None, link=None):
    import tarfile
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for f in sorted(tree.rglob("*")):
            t.add(f, "runs/" + f.relative_to(tree).as_posix(), recursive=False)
        for name, data in (extra or {}).items():
            i = tarfile.TarInfo(name); i.size = len(data); t.addfile(i, io.BytesIO(data.encode()))
        if link:
            i = tarfile.TarInfo(link); i.type = tarfile.SYMTYPE; i.linkname = "/etc/passwd"; t.addfile(i)
    return b.getvalue()


def test_tar_gz_accepted_by_content(monkeypatch, tmp_path, stick):
    (report, added), _ = run(monkeypatch, tmp_path, tgz_of(stick, {"runs/.DS_Store": "x"}),
                             BODY.replace("runs.zip", "runs.tar.gz"))
    assert len(added) == 2 and "2 run(s) accepted" in report


@pytest.mark.parametrize("kind, expect", [
    ("link", "a link"), ("dotdot", "path outside the archive"), ("plain-gz", "does not hold a tar"), ("junk", "neither a zip"),
])
def test_tar_gz_unsafe_refused(monkeypatch, tmp_path, stick, kind, expect):
    import gzip
    data = {"link": lambda: tgz_of(stick, link=f"runs/{TV.RUN}/x.log"),
            "dotdot": lambda: tgz_of(stick, {"../evil": "x"}),
            "plain-gz": lambda: gzip.compress(b"just text"),
            "junk": lambda: b"7z\xbc\xaf'\x1c"}[kind]()
    (report, added), _ = run(monkeypatch, tmp_path, data, BODY.replace("runs.zip", "runs.tgz"))
    assert not added and expect in report


def test_gzip_bomb_bounded(monkeypatch, tmp_path):
    import gzip
    monkeypatch.setattr(S, "MAX_TOTAL", 1 << 20)
    (report, added), _ = run(monkeypatch, tmp_path, gzip.compress(b"\0" * (4 << 20)), BODY.replace("runs.zip", "x.gz"))
    assert not added and "unpacked" in report


def test_only_github_archive_types_are_fetched():
    assert S.zip_urls("[a](https://github.com/user-attachments/files/1/a.tar.gz) [b](https://github.com/user-attachments/files/2/b.tgz) "
                      "[c](https://github.com/user-attachments/files/3/c.zip) [d](https://github.com/user-attachments/files/4/d.tar.xz)") == \
        ["https://github.com/user-attachments/files/1/a.tar.gz", "https://github.com/user-attachments/files/2/b.tgz",
         "https://github.com/user-attachments/files/3/c.zip"]


def test_credit_by_default_withheld_on_request(monkeypatch, tmp_path, stick):
    monkeypatch.setattr(S, "fetch", lambda url, limit=0: zip_of(stick))
    dest, credits = tmp_path / "runs", tmp_path / "attribution"; dest.mkdir()
    report, added = S.process(BODY, dest, author="keysie", credits=credits)
    assert len(added) == 2 and all(json.loads((credits / f"{a}.json").read_text()) == {"github": "keysie"} for a in added)
    assert "Credited on the results page to @keysie" in report
    body = BODY + "\n### Your name on the results\n\n- [X] Keep my username out of the stored result data and do not display it on the results page\n"
    dest2, credits2 = tmp_path / "runs2", tmp_path / "attribution2"; dest2.mkdir()
    report, added = S.process(body, dest2, author="keysie", credits=credits2)
    assert not credits2.exists() and "kept out" in report                     # withheld: no file at all
    assert S.attribution("bad name!", BODY) == {}                              # not a GitHub username: no credit


def test_form_offers_the_choice_unticked():
    form = (ROOT / ".github" / "ISSUE_TEMPLATE" / "submit-runs.yml").read_text()
    block = form.split("id: attribution", 1)[1].split("- type:", 1)[0]
    assert S.WITHHOLD in block and "required: true" not in block
