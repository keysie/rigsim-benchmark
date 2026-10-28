#!/usr/bin/env python3
"""Exact versions of everything the stick runs, recorded when the stick is built.

  manifest.py rootfs <rootfs> <payload dir> <out.json>    the system image: packages, the stick's own files
  manifest.py seed <seed dir> <out.json>                  the seed: llama.cpp builds, images, models, tools

The stick merges both (and what it detects at run time) into each run's manifest.json and copies the entries a
step used into that step's "tools" in run.json. Build settings come from the environment (SUITE, NVIDIA_*, ...)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# rootfs packages whose versions matter for a measurement (drivers, runtimes, the tools the stick calls)
ROOTFS_PACKAGES = re.compile(
    r"^(linux-image-\S+|linux-modules-nvidia-\S+|nvidia-\S+|libnvidia-\S+|mesa-\S+|libvulkan1|vulkan-tools|"
    r"libgl1-mesa-dri|intel-opencl-icd|intel-level-zero-gpu|libze\S*|libigdgmm\S*|clinfo|clpeak|ocl-icd-\S+|"
    r"docker\.io|containerd|runc|i2c-tools|dmidecode|lshw|hwloc|numactl|pciutils|lm-sensors|python3|casper|"
    r"shim-signed|grub-efi-amd64-signed|linux-firmware)$")
CUDA_PACKAGES = re.compile(r"^(cuda-nvcc\S*|cuda-cudart-\S+|libcublas-\S+|cuda-toolkit-\S+)$")
SETTINGS = ("SUITE", "KERNEL", "NVIDIA_BRANCH", "NVIDIA_FLAVOUR", "MIRROR", "LLAMA_REF", "AMDGPU_TARGETS",
            "ROCM_IMAGE", "ONEAPI_IMAGE", "BLENDER_VERSION", "BLENDER_SCRIPT_URL")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    """One hash over every file (path + content) under root: the stick's scripts as one fingerprint."""
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + bytes.fromhex(sha256(p)))
    return h.hexdigest()


def sh(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def dpkg_list(text: str, keep: re.Pattern) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] and keep.match(parts[0]):
            out[parts[0]] = parts[1]
    return dict(sorted(out.items()))


def rootfs(root: Path, payload: Path, out: Path) -> dict:
    pk = sh(["chroot", str(root), "dpkg-query", "-W", "-f", "${Package}\t${Version}\n"])
    m = dict(
        part="rootfs",
        stick=dict(version=(payload / "STICK_VERSION").read_text().strip(), build=(payload / "VERSION").read_text().strip(),
                   commit=os.environ.get("STICK_COMMIT", ""), dirty=os.environ.get("STICK_DIRTY", "") == "1",
                   scripts_sha256=tree_sha256(payload)),
        settings={k: os.environ[k] for k in SETTINGS if os.environ.get(k)},
        packages=dpkg_list(pk, ROOTFS_PACKAGES),
        files={f: dict(sha256=sha256(payload / f)) for f in ("stick.py", "runner/hwq.py", "bin/vkprobe") if (payload / f).exists()},
        kernel=sorted(p.name for p in (root / "lib" / "modules").iterdir()) if (root / "lib" / "modules").exists() else [],
    )
    out.write_text(json.dumps(m, indent=1) + "\n")
    return m


def image_info(ref: str) -> dict:
    fmt = "{{.Id}}\t{{index .RepoDigests 0}}"
    i = sh(["docker", "image", "inspect", "-f", fmt, ref]).split("\t")
    return dict(ref=ref, id=i[0] if i and i[0] else "", digest=i[1] if len(i) > 1 else "")


def seed(seed_dir: Path, out: Path) -> dict:
    m: dict = dict(part="seed", build=(seed_dir / "VERSION").read_text().strip() if (seed_dir / "VERSION").exists() else "",
                   settings={k: os.environ[k] for k in SETTINGS if os.environ.get(k)})
    # container images of the bench questions, with the package versions inside that the probes are built from
    imgs = {}
    for ref in os.environ.get("IMAGES", "").split():           # the seeded question images, by reference
        info = image_info(ref)
        if "cuda" in ref:
            info["packages"] = dpkg_list(sh(["docker", "run", "--rm", "--network", "none", ref, "dpkg-query", "-W", "-f",
                                             "${Package}\t${Version}\n"]), CUDA_PACKAGES)
        imgs[ref] = info
    debs = {}
    for d in sorted((seed_dir / "debs").glob("*.deb")):
        name, ver = (d.name.split("_") + ["", ""])[:2]
        debs[name] = ver.replace("%3a", ":")
    if debs and imgs:
        for ref, info in imgs.items():
            if "cuda" not in ref:
                info["offline_packages"] = debs
    m["images"] = imgs
    # llama.cpp per backend
    llama = {}
    for b in sorted(p for p in (seed_dir / "llama").glob("*") if p.is_dir()):
        llama[b.name] = dict(commit=(b / "COMMIT").read_text().strip() if (b / "COMMIT").exists() else "",
                             files={f.name: sha256(f) for f in sorted(b.iterdir()) if f.is_file() and f.name != "COMMIT"})
    m["llama.cpp"] = llama
    # vendor images (ROCm, oneAPI): versions inside, as built
    vendor = {}
    vt = seed_dir / "images" / "VENDOR.tsv"
    for line in (vt.read_text().splitlines() if vt.exists() else []):
        name, tag, base = (line.split("\t") + ["", "", ""])[:3]
        if not (seed_dir / "images" / f"vendor-{name}.tar").exists():
            continue
        inside = sh(["docker", "run", "--rm", "--network", "none", tag, "sh", "-c",
                     "for f in ROCM_VERSION ONEAPI_VERSION LLAMA_COMMIT TARGETS; do [ -f /rigsim/$f ] && "
                     "printf '%s\\t%s\\n' $f \"$(head -1 /rigsim/$f)\"; done; cd /rigsim && sha256sum probe probe_sycl "
                     "llama-bench test-backend-ops 2>/dev/null | sed 's/  /\\tsha256:/'"])
        kv = dict(l.split("\t", 1) for l in inside.splitlines() if "\t" in l)
        vendor[name] = dict(image=tag, base=image_info(base), **{k.lower(): v for k, v in kv.items()})
    m["vendor"] = vendor
    sums = seed_dir / "models" / "SHA256SUMS"
    m["models"] = {l.split()[1]: dict(sha256=l.split()[0]) for l in (sums.read_text().splitlines() if sums.exists() else [])
                   if len(l.split()) == 2}
    tools = {}
    mlc = seed_dir / "tools" / "mlc" / "mlc"
    if mlc.exists():
        tools["mlc"] = dict(sha256=sha256(mlc))
    bl = seed_dir / "blender"
    if (bl / "benchmark-launcher-cli").exists():
        bb = dict(launcher_sha256=sha256(bl / "benchmark-launcher-cli"),
                  blender=(bl / "BLENDER_VERSION").read_text().strip() if (bl / "BLENDER_VERSION").exists() else "")
        if (bl / "benchmark-script.tar.gz").exists():
            bb["script"] = dict(url=os.environ.get("BLENDER_SCRIPT_URL", ""), sha256=sha256(bl / "benchmark-script.tar.gz"))
        bb["scenes"] = {p.parent.name: sha256(p) for p in sorted((bl / "home").rglob("*.blend"))}
        tools["blender-benchmark"] = bb
    m["tools"] = tools
    out.write_text(json.dumps(m, indent=1) + "\n")
    return m


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "rootfs":
        rootfs(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    elif len(sys.argv) == 4 and sys.argv[1] == "seed":
        seed(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)
