#!/usr/bin/env python3
"""Reads back what flash.sh wrote and compares it with the source, byte for byte (SHA-256 per item), with one
progress bar over everything. The page cache is dropped first, so the stick itself is read, not what Linux still
holds from the write.

  verify.py --iso <iso> <device>            the system image at the start of the device (the MBR partition table,
                                            which flash.sh changes after writing, is left out)
            --tree <src dir> <dst dir>      every file below src, at the same place below dst
            --file <src> <dst>              one file (e.g. the image store)

Exit code 0: everything identical. 1: differences (listed). 2: usage."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

CHUNK = 8 << 20
MBR_TABLE = (446, 512)                    # partition entries + signature: rewritten by flash.sh after the image


class Bar:
    def __init__(self, total: int):
        self.total, self.done, self.t0, self.label, self.last = max(total, 1), 0, time.time(), "", 0.0
        self.tty = sys.stdout.isatty()

    def add(self, n: int, label: str = "") -> None:
        self.done += n
        if label:
            self.label = label
        now = time.time()
        if now - self.last < 0.2 and self.done < self.total:
            return
        self.last = now
        frac = min(self.done / self.total, 1.0)
        rate = self.done / max(now - self.t0, 1e-6)
        eta = (self.total - self.done) / rate if rate else 0
        width = max(10, min(40, shutil.get_terminal_size((100, 20)).columns - 60))
        bar = "#" * int(frac * width) + "-" * (width - int(frac * width))
        line = (f"verify [{bar}] {frac:5.1%}  {self.done / 2**30:6.1f}/{self.total / 2**30:.1f} GiB  "
                f"{rate / 2**20:5.0f} MB/s  ETA {int(eta) // 60}:{int(eta) % 60:02d}  {self.label[-30:]}")
        if self.tty:
            sys.stdout.write("\r\033[K" + line); sys.stdout.flush()
        elif self.done >= self.total or int(frac * 10) != int((self.done - n) / self.total * 10):
            print(line, flush=True)

    def end(self) -> None:
        if self.tty:
            sys.stdout.write("\n"); sys.stdout.flush()


def digest(path: str, length: int, bar: Bar, label: str, skip: tuple[int, int] | None = None) -> str:
    """SHA-256 of the first `length` bytes of path; the bytes in `skip` count as zero."""
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        pos = 0
        while pos < length:
            b = os.read(fd, min(CHUNK, length - pos))
            if not b:
                break
            if skip and pos < skip[1] and pos + len(b) > skip[0]:
                ba = bytearray(b)
                for i in range(max(skip[0], pos), min(skip[1], pos + len(b))):
                    ba[i - pos] = 0
                b = bytes(ba)
            h.update(b)
            pos += len(b)
            bar.add(len(b), label)
    finally:
        os.close(fd)
    return h.hexdigest() if pos == length else f"short read ({pos} of {length} bytes)"


def main(argv: list[str]) -> int:
    items: list[tuple[str, str, int, tuple | None, str]] = []     # src, dst, length, skip, label
    i = 0
    try:
        while i < len(argv):
            kind, src, dst = argv[i], argv[i + 1], argv[i + 2]
            i += 3
            if kind == "--iso":
                items.append((src, dst, os.path.getsize(src), MBR_TABLE, "system image"))
            elif kind == "--file":
                items.append((src, dst, os.path.getsize(src), None, Path(src).name))
            elif kind == "--tree":
                for p in sorted(Path(src).rglob("*")):
                    if p.is_file() and not p.is_symlink():
                        rel = p.relative_to(src)
                        items.append((str(p), str(Path(dst) / rel), p.stat().st_size, None, str(rel)))
            else:
                raise IndexError
    except (IndexError, OSError) as e:
        print(f"usage: verify.py --iso ISO DEV | --tree SRC DST | --file SRC DST ... ({e})", file=sys.stderr)
        return 2
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n")       # read the stick, not what is still cached
    except OSError:
        pass
    bar = Bar(2 * sum(n for _, _, n, _, _ in items))
    bad = []
    for src, dst, n, skip, label in items:
        if not os.path.exists(dst):
            bad.append(f"{label}: missing on the stick"); bar.add(2 * n); continue
        if skip is None and os.path.isfile(dst) and os.path.getsize(dst) != n:    # the image: a prefix of the device
            bad.append(f"{label}: size {os.path.getsize(dst)} on the stick, {n} here"); bar.add(2 * n); continue
        if digest(src, n, bar, label, skip) != digest(dst, n, bar, label, skip):
            bad.append(f"{label}: content differs")
    bar.end()
    if bad:
        print(f"verification FAILED — {len(bad)} item(s) differ from what was written:")
        for b in bad[:20]:
            print("  " + b)
        return 1
    print(f"verification passed: {len(items)} item(s), {bar.total / 2 / 2**30:.1f} GiB read back and identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
