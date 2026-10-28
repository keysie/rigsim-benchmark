#!/usr/bin/env python3
"""Mask a fetched run and pack it (controller side). The raw fetch stays private where it is; the masked copy
<dir>-public and <dir>-public.tar.gz are what may be shared or committed.

  archive.py DIR --gpu-names '{"0000:65:00.0": "3090-gigabyte"}' --hosts '["ai-server", "10.0.0.5", ...]'

Masked: IPv4 addresses, the given host names and addresses, home paths, GPU UUIDs, MAC addresses, PCI bus ids
(mapped to card names where known). Then checked clean: the archive is not written if anything is left.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tarfile
from pathlib import Path

IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID = re.compile(r"\bGPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
MAC = re.compile(r"\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")
PCI = re.compile(r"\b(?:[0-9a-fA-F]{4,8}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b")
HOME = re.compile(r"/home/[^/\s\"']+|/root(?=/)")


def mask_text(text: str, gpu_names: dict[str, str], hosts: list[str]) -> str:
    def pci(m):
        key = m.group(0).lower()
        if key.count(":") == 1:
            key = "0000:" + key
        key = re.sub(r"^0{4}(?=0000:)", "", key)
        return gpu_names.get(key) or gpu_names.get(key.replace("0000:", "00000000:")) or "<pci>"
    text = UUID.sub("GPU-<uuid>", text)
    text = PCI.sub(pci, text)
    for h in sorted((h for h in hosts if h), key=len, reverse=True):
        text = text.replace(h, "<host>")
    text = IPV4.sub("x.x.x.x", text)
    text = MAC.sub("<mac>", text)
    return HOME.sub("~", text)


def leftovers(text: str, hosts: list[str]) -> list[str]:
    found = [m for m in IPV4.findall(text) if m != "x.x.x.x"] + UUID.findall(text) + MAC.findall(text) + PCI.findall(text)
    found += [h for h in hosts if h and h in text]
    found += HOME.findall(text)
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--gpu-names", default="{}")
    ap.add_argument("--hosts", default="[]")
    a = ap.parse_args(argv)
    src = Path(a.dir).expanduser()
    names = {k.lower(): v for k, v in json.loads(a.gpu_names).items()}
    hosts = [h for h in json.loads(a.hosts) if isinstance(h, str) and len(h) >= 3]
    out = src.with_name(src.name + "-public")
    if out.exists():
        shutil.rmtree(out)
    problems = []
    for f in sorted(p for p in src.rglob("*") if p.is_file()):
        rel = f.relative_to(src)
        dst = out / Path(mask_text(str(rel), names, hosts))
        dst.parent.mkdir(parents=True, exist_ok=True)
        masked = mask_text(f.read_text(errors="replace"), names, hosts)
        left = leftovers(masked, hosts)
        if left:
            problems.append(f"{rel}: {sorted(set(left))[:5]}")
        dst.write_text(masked)
    if problems:
        print("NOT CLEAN — no archive written:"); [print("  " + p) for p in problems]
        return 1
    tgz = out.with_suffix(".tar.gz")
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname=out.name)
    print(f"masked copy: {out}")
    print(f"archive:     {tgz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
