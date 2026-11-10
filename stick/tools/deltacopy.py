#!/usr/bin/env python3
"""Bring a copy of a sparse file up to date by writing only the chunks that differ (flash.sh: the image store).

  deltacopy.py SRC DST [--chunk-mib 4]

The image store is one ext4 file of ~44 GiB. Between two flashes most of it stays the same (images that did not
change), and the stick itself writes into its copy while it runs (container layers). Copying it whole cost 44 GiB of
writes every time; this compares chunk by chunk and writes only what differs:

- a chunk the same on both sides: nothing written (only read);
- a chunk with data in SRC that differs: written;
- a chunk that is a hole or all zeros in SRC but holds data in DST: punched out of DST (stays sparse, no stale data —
  rsync --inplace --sparse leaves such data in place, tested 2026-11-10).

DST is created if missing and cut to SRC's size. Prints one progress line every few seconds and a summary.
Exit 0: DST equals SRC. 2: usage.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time

FALLOC_FL_KEEP_SIZE, FALLOC_FL_PUNCH_HOLE = 0x01, 0x02
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]


def punch(fd: int, off: int, length: int) -> None:
    """Make [off, off+length) a hole; where the file system cannot, write zeros."""
    if _libc.fallocate(fd, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, off, length) != 0:
        os.pwrite(fd, bytes(length), off)


def data_ranges(fd: int, size: int) -> list[tuple[int, int]]:
    """(start, end) of the data in a sparse file; the whole file where holes are not reported."""
    out, pos = [], 0
    try:
        while pos < size:
            s = os.lseek(fd, pos, os.SEEK_DATA)
            e = min(os.lseek(fd, s, os.SEEK_HOLE), size)
            out.append((s, e))
            pos = e
    except OSError as e:
        if e.errno != 6:                                   # ENXIO: no more data
            return [(0, size)]
    return out


def has_data(ranges: list[tuple[int, int]], a: int, b: int) -> bool:
    return any(s < b and e > a for s, e in ranges)


def copy(src: str, dst: str, chunk: int = 4 << 20, out=sys.stdout) -> dict:
    fs = os.open(src, os.O_RDONLY)
    fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fs).st_size
        os.ftruncate(fd, size)
        src_map, dst_map = data_ranges(fs, size), data_ranges(fd, size)
        zero = bytes(chunk)
        st = dict(size=size, read=0, written=0, punched=0, same=0)
        t0 = last = time.time()
        for off in range(0, size, chunk):
            n = min(chunk, size - off)
            s_has, d_has = has_data(src_map, off, off + n), has_data(dst_map, off, off + n)
            if not s_has and not d_has:                     # a hole on both sides
                continue
            a = os.pread(fs, n, off) if s_has else zero[:n]
            b = os.pread(fd, n, off) if d_has else zero[:n]
            st["read"] += n * (s_has + d_has)
            if a == b:
                st["same"] += n
            elif a == zero[:n]:
                punch(fd, off, n); st["punched"] += n
            else:
                os.pwrite(fd, a, off); st["written"] += n
            if time.time() - last > 5:
                last = time.time()
                print(f"  {off / size:6.1%}  written {st['written'] >> 20} MiB, unchanged {st['same'] >> 20} MiB",
                      file=out, flush=True)
        os.fsync(fd)
        st["seconds"] = round(time.time() - t0, 1)
        return st
    finally:
        os.close(fs)
        os.close(fd)


def main(argv: list[str]) -> int:
    import argparse
    a = argparse.ArgumentParser(description="write only the chunks of DST that differ from SRC")
    a.add_argument("src"); a.add_argument("dst"); a.add_argument("--chunk-mib", type=int, default=4)
    o = a.parse_args(argv)
    st = copy(o.src, o.dst, o.chunk_mib << 20)
    print(f"store: {st['written'] >> 20} MiB written, {st['punched'] >> 20} MiB cleared, "
          f"{st['same'] >> 20} MiB unchanged ({st['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
