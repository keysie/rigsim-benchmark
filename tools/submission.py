#!/usr/bin/env python3
"""Runs submitted through the GitHub UI: an issue from the "Submit runs" form with one or more archives attached —
the types GitHub accepts as issue attachments: .zip, .tar.gz / .tgz (GitHub refuses .tar.xz, .7z, .rar).
The submission workflow (.github/workflows/submission.yml) calls this; it never trusts the zip:

  python3 tools/submission.py <issue body file> <report.md> <runs.txt>

- the three confirmations in the form must be ticked (licence, owner's consent, unchanged from the stick);
- only archives uploaded to GitHub itself (github.com/user-attachments/...) are fetched, size-limited; the format
  is taken from the content (zip, or gzip around a tar), not from the name;
- extraction refuses absolute paths, "..", links and anything beyond the size and file-count limits;
- every folder named like a run is taken; operating-system clutter outside the run folders (__MACOSX, .DS_Store,
  Thumbs.db, desktop.ini) is ignored, any other file outside a run folder is refused;
- each run is checked by tools/validate.py exactly like a pull request; runs already in the repository are skipped.

Valid new runs are copied to results/runs/ (their names go to runs.txt); report.md says what happened to each."""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

ZIP_URL = re.compile(r"https://github\.com/user-attachments/files/\d+/[A-Za-z0-9._%+-]+\.(?:zip|tgz|gz)")
CONFIRMATIONS = ("CC BY 4.0", "agreed to publication", "unchanged")
WITHHOLD = "Keep my username out of the stored result data"
LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")


def withheld(body: str) -> bool:
    """The submitter ticked 'Keep my username out of the stored result data ...'."""
    line = next((l for l in body.splitlines() if WITHHOLD in l and l.strip().startswith("- [")), "")
    return bool(re.match(r"\s*- \[[xX]\]", line))


def attribution(author: str | None, body: str) -> dict:
    """What results/attribution/<run>.json says: the submitter's GitHub username. Withheld: no file at all (no
    attribution found counts as withheld everywhere)."""
    if withheld(body):
        return {}
    return {"github": author} if author and LOGIN.match(author) else {}
MAX_ZIP = 100 * 2**20            # per zip, as downloaded
MAX_TOTAL = 400 * 2**20          # all zips of one submission, unpacked
MAX_FILES = 20000
CLUTTER = re.compile(r"(^|/)(__MACOSX(/|$)|\.DS_Store$|Thumbs\.db$|desktop\.ini$|\._)")


class Refused(Exception):
    pass


def confirmations(body: str) -> list[str]:
    """The confirmations in the form that are not ticked."""
    missing = []
    for c in CONFIRMATIONS:
        line = next((l for l in body.splitlines() if c in l and l.strip().startswith("- [")), "")
        if not re.match(r"\s*- \[[xX]\]", line):
            missing.append(c)
    return missing


def zip_urls(body: str) -> list[str]:
    return list(dict.fromkeys(ZIP_URL.findall(body)))


def fetch(url: str, limit: int = MAX_ZIP) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "rigsim-benchmark-submission"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise Refused(f"{url.rsplit('/', 1)[-1]}: larger than {limit // 2**20} MiB")
    return data


def _check_name(name: str) -> PurePosixPath:
    p = PurePosixPath(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", name):
        raise Refused(f"{name}: path outside the archive")
    return p


def safe_extract(data: bytes, dst: Path, budget: list[int]) -> list[str]:
    """Extracts a zip or a gzipped tar below dst; returns the names it skipped as clutter. Raises Refused on anything
    unsafe."""
    if data[:2] == b"\x1f\x8b":
        return _extract_tar_gz(data, dst, budget)
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise Refused("neither a zip nor a .tar.gz file")
    skipped = []
    infos = z.infolist()
    if len(infos) > MAX_FILES:
        raise Refused(f"more than {MAX_FILES} files")
    for i in infos:
        name = i.filename.replace("\\", "/")
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", name):
            raise Refused(f"{name}: path outside the archive")
        if (i.external_attr >> 16) & 0o170000 == 0o120000:
            raise Refused(f"{name}: a link")
        if CLUTTER.search(name):
            skipped.append(name)
            continue
        budget[0] -= i.file_size
        if budget[0] < 0:
            raise Refused(f"more than {MAX_TOTAL // 2**20} MiB unpacked")
        target = dst / p
        if i.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with z.open(i) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
    return skipped


def _extract_tar_gz(data: bytes, dst: Path, budget: list[int]) -> list[str]:
    try:
        raw = gzip.GzipFile(fileobj=io.BytesIO(data)).read(MAX_TOTAL + 1)   # bounded: no gzip bomb
    except (OSError, EOFError):
        raise Refused("damaged .gz file")
    if len(raw) > MAX_TOTAL:
        raise Refused(f"more than {MAX_TOTAL // 2**20} MiB unpacked")
    try:
        t = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
        members = t.getmembers()
    except tarfile.TarError:
        raise Refused("a .gz file that does not hold a tar archive (pack the run folders with tar -czf)")
    if len(members) > MAX_FILES:
        raise Refused(f"more than {MAX_FILES} files")
    skipped = []
    for m in members:
        p = _check_name(m.name)
        if m.issym() or m.islnk():
            raise Refused(f"{m.name}: a link")
        if not (m.isfile() or m.isdir()):
            raise Refused(f"{m.name}: not a regular file")
        if CLUTTER.search(m.name):
            skipped.append(m.name)
            continue
        budget[0] -= m.size
        if budget[0] < 0:
            raise Refused(f"more than {MAX_TOTAL // 2**20} MiB unpacked")
        target = dst / p
        if m.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with t.extractfile(m) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
    return skipped


def find_runs(tree: Path) -> tuple[list[Path], list[str]]:
    """Run folders anywhere in the unpacked tree (e.g. runs/<run>/ or <run>/), and files outside any run folder."""
    runs = sorted({p for p in tree.rglob("*") if p.is_dir() and validate.RUN_ID.match(p.name)
                   and not any(validate.RUN_ID.match(a.name) for a in p.parents)})
    stray = [str(f.relative_to(tree)) for f in tree.rglob("*")
             if f.is_file() and not any(r in f.parents for r in runs)]
    return runs, stray


def process(body: str, dest: Path = ROOT / "results" / "runs", author: str | None = None,
            credits: Path = ROOT / "results" / "attribution") -> tuple[str, list[str]]:
    lines = []
    missing = confirmations(body)
    if missing:
        return ("Not accepted: please tick all confirmations in the form (" + ", ".join(missing) + ")."), []
    urls = zip_urls(body)
    if not urls:
        return ("No archive found. Drag a .zip or .tar.gz with your run folders into the form's upload field "
                "(it then shows a github.com/user-attachments/... link). GitHub does not accept .tar.xz, .7z or .rar."), []
    added, budget = [], [MAX_TOTAL]
    with tempfile.TemporaryDirectory() as t:
        for n, url in enumerate(urls):
            name = url.rsplit("/", 1)[-1]
            part = Path(t) / str(n)
            try:
                skipped = safe_extract(fetch(url), part, budget)
            except Refused as e:
                lines.append(f"- ❌ `{name}`: {e}")
                continue
            except OSError as e:
                lines.append(f"- ❌ `{name}`: could not be downloaded ({e})")
                continue
            runs, stray = find_runs(part)
            if stray:
                lines.append(f"- ❌ `{name}`: files outside a run folder — " + ", ".join(f"`{s}`" for s in stray[:5]))
                continue
            if not runs:
                lines.append(f"- ❌ `{name}`: no run folder in it (folders are named like `20261027T234840Z-…-…`)")
                continue
            if skipped:
                lines.append(f"- `{name}`: ignored {len(skipped)} system file(s) (e.g. `{skipped[0]}`)")
            for r in runs:
                if (dest / r.name).exists() or r.name in added:
                    lines.append(f"- ↩️ `{r.name}`: already in the repository")
                    continue
                pruned = validate.prune_interrupted(r)       # a stopped run: keep the steps that finished
                if pruned:
                    steps = sorted({validate.step_of(x) for x in pruned})
                    lines.append(f"- ✂️ `{r.name}` was stopped: dropped the unfinished step(s) {', '.join(steps)} "
                                 f"({len(pruned)} file(s)); the finished steps are kept")
                errs = validate.check_run(r)
                if errs:
                    short = [e[len(r.name) + 1:] if e.startswith(r.name + "/") else e.removeprefix(r.name + ": ")
                             for e in errs]
                    lines.append(f"- ❌ `{r.name}`:\n" + "\n".join(f"  - {e}" for e in short[:10]))
                    continue
                shutil.copytree(r, dest / r.name)
                added.append(r.name)
                credit = attribution(author, body)
                if credit:
                    credits.mkdir(parents=True, exist_ok=True)
                    (credits / f"{r.name}.json").write_text(json.dumps(credit) + "\n")
                hw = json.loads((r / "run.json").read_text()).get("hardware", {})
                lines.append(f"- ✅ `{r.name}`: {hw.get('cpu', '?')}, " + (", ".join(hw.get("gpus", [])) or "no GPU"))
    credit = attribution(author, body)
    who = ("Credited on the results page to @" + credit["github"] if credit.get("github")
           else "Your username is kept out of the result data and the results page" if withheld(body) else "")
    head = (f"**{len(added)} run(s) accepted**" if added else "**No run accepted**") + \
        " — checked like every contribution (exact stick format, complete, nothing identifying).\n\n"
    return head + "\n".join(lines) + (f"\n\n{who}." if added and who else ""), added


def main() -> int:
    body = Path(sys.argv[1]).read_text()
    report, added = process(body, author=os.environ.get("AUTHOR"))
    Path(sys.argv[2]).write_text(report + "\n")
    Path(sys.argv[3]).write_text("".join(f"{a}\n" for a in added))
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
