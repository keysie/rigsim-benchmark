#!/usr/bin/env python3
"""Reads back what flash.sh wrote and compares it with the source, byte for byte (SHA-256 per item), with one
progress bar over everything. The page cache is dropped first, so the stick itself is read, not what Linux still
holds from the write.

  verify.py --iso <iso> <device>            the system image at the start of the device (the partition table, which
                                            flash.sh changes after writing, is left out: the MBR table, and for a GPT
                                            image also the GPT header and entries, sectors 1-33; flash.sh checks the
                                            table itself with sfdisk --verify)
            --tree <src dir> <dst dir>      every file below src, at the same place below dst
            --file <src> <dst>              one file (e.g. the image store)

The progress counts what is read from the stick. Holes of sparse files (the image store: 96 GiB apparent, about half
of it used) are hashed as zeros without being read, and not counted.

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
GPT_TABLE = (446, 34 * 512)               # + GPT header (sector 1) and entry array (2-33), since stick 0.13.5
ZERO = bytes(CHUNK)


def table_skip(iso: str) -> tuple[int, int]:
    """The bytes flash.sh rewrites when it adds the data partition behind this image."""
    with open(iso, "rb") as f:
        f.seek(512)
        return GPT_TABLE if f.read(8) == b"EFI PART" else MBR_TABLE


def data_ranges(fd: int, length: int) -> list[tuple[int, int]]:
    """(start, end) of the data in a file; holes left out. Whole range where holes are not reported."""
    out, pos = [], 0
    try:
        while pos < length:
            start = os.lseek(fd, pos, os.SEEK_DATA)
            if start >= length:
                break
            end = min(os.lseek(fd, start, os.SEEK_HOLE), length)
            out.append((start, end))
            pos = end
    except OSError as e:
        if e.errno == 6:                               # ENXIO: no data after pos
            return out
        return [(0, length)]
    return out


def allocated(path: str, length: int) -> int:
    """Bytes that will actually be read from path (a device: all of length)."""
    if not os.path.isfile(path):
        return length
    fd = os.open(path, os.O_RDONLY)
    try:
        return sum(e - s for s, e in data_ranges(fd, length))
    finally:
        os.close(fd)


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


def digest(path: str, length: int, bar: Bar | None, label: str, skip: tuple[int, int] | None = None) -> str:
    """SHA-256 of the first `length` bytes of path; the bytes in `skip` count as zero, holes are hashed as zeros
    without reading them. Progress (bar) counts the bytes actually read; bar None: not counted (the source side)."""
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        ranges = data_ranges(fd, length) if os.path.isfile(path) else [(0, length)]
        pos = 0
        for start, end in ranges + [(length, length)]:
            while pos < start:                                # a hole: zeros, nothing read
                n = min(CHUNK, start - pos)
                h.update(ZERO[:n]); pos += n
            os.lseek(fd, pos, os.SEEK_SET)
            while pos < end:
                b = os.read(fd, min(CHUNK, end - pos))
                if not b:
                    return f"short read ({pos} of {length} bytes)"
                if skip and pos < skip[1] and pos + len(b) > skip[0]:
                    lo, hi = max(skip[0], pos) - pos, min(skip[1], pos + len(b)) - pos
                    b = b[:lo] + bytes(hi - lo) + b[hi:]
                h.update(b)
                pos += len(b)
                if bar:
                    bar.add(len(b), label)
    finally:
        os.close(fd)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    items: list[tuple[str, str, int, tuple | None, str]] = []     # src, dst, length, skip, label
    i = 0
    try:
        while i < len(argv):
            kind, src, dst = argv[i], argv[i + 1], argv[i + 2]
            i += 3
            if kind == "--iso":
                items.append((src, dst, os.path.getsize(src), table_skip(src), "system image"))
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
    sizes = [allocated(dst, n) if os.path.exists(dst) else 0 for _, dst, n, _, _ in items]
    bar = Bar(sum(sizes))                                     # what is read from the stick
    bad = []
    for (src, dst, n, skip, label), size in zip(items, sizes):
        if not os.path.exists(dst):
            bad.append(f"{label}: missing on the stick"); continue
        if skip is None and os.path.isfile(dst) and os.path.getsize(dst) != n:    # the image: a prefix of the device
            bad.append(f"{label}: size {os.path.getsize(dst)} on the stick, {n} here"); bar.add(size); continue
        if digest(src, n, None, label, skip) != digest(dst, n, bar, label, skip):
            bad.append(f"{label}: content differs")
    bar.end()
    if bad:
        print(f"verification FAILED — {len(bad)} item(s) differ from what was written:")
        for b in bad[:20]:
            print("  " + b)
        return 1
    print(f"verification passed: {len(items)} item(s), {bar.total / 2**30:.1f} GiB read back from the stick and identical")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
