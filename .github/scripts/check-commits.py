#!/usr/bin/env python3
"""Conventional Commits (https://www.conventionalcommits.org/en/v1.0.0/) for every commit message.

  check-commits.py --file <message file>      the commit-msg hook (.githooks/commit-msg)
  check-commits.py <rev range>                CI: every commit in the range, e.g. origin/main..HEAD

First line: <type>[(<scope>)][!]: <description>, type one of TYPES, lower-case. Merge commits (more than one parent)
are not checked."""
from __future__ import annotations

import re
import subprocess
import sys

TYPES = ("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert")
HEADER = re.compile(rf"^(?:{'|'.join(TYPES)})(?:\([a-z0-9][a-z0-9._/-]*\))?!?: \S.*$")


def problem(message: str) -> str | None:
    lines = [l for l in message.splitlines() if not l.startswith("#")]
    head = lines[0] if lines else ""
    if not HEADER.match(head):
        return (f"not a conventional commit: {head!r}\n  expected <type>[(<scope>)][!]: <description>, "
                f"type one of {', '.join(TYPES)}")
    if len(lines) > 1 and lines[1].strip():
        return "the line after the header must be blank"
    return None


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--file":
        p = problem(open(argv[1], encoding="utf-8").read())
        if p:
            print(p, file=sys.stderr)
        return 1 if p else 0
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    revs = subprocess.run(["git", "rev-list", "--no-merges", argv[0]], capture_output=True, text=True, check=True)
    bad = 0
    for sha in revs.stdout.split():
        msg = subprocess.run(["git", "log", "-1", "--format=%B", sha], capture_output=True, text=True, check=True).stdout
        p = problem(msg)
        if p:
            bad += 1
            print(f"{sha[:10]}: {p}")
    print(f"{bad} commit(s) not conventional" if bad else "all commits conventional")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
