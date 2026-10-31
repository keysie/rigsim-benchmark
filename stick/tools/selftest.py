#!/usr/bin/env python3
"""Build-time self-test of the stick: is every part the build was asked to ship in the seed, and does every native
binary find its libraries where it will run? A missing part stops the build (exit 1), so it becomes an issue to fix
instead of runs that lack the expected data (user decision 2026-10-31). ALLOW_INCOMPLETE=1 builds anyway, for testing.

Parts left out on purpose by a build option (MLC_URL or BLENDER_LAUNCHER_URL empty, SKIP_MODELS, SKIP_ROCM,
SKIP_ONEAPI) are reported as "left out", not as failures.

Where things run on the stick (stick.py): Blender, MLC, vkprobe and llama.cpp Vulkan in the system image; llama.cpp
CUDA inside the CUDA image. `ldd` is therefore run in the system image (chroot, binary bind-mounted) or in that image.

    sudo -E python3 stick/tools/selftest.py --seed OUT/seed --rootfs ROOTFS --payload ROOTFS/opt/rigsim \\
        --cuda-image IMAGE --image IMAGE ... --out OUT/seed/SELFTEST.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
LLAMA_BINS = ("llama-bench", "test-backend-ops")
# The NVIDIA driver's user-space libraries are never in a CUDA image: the container toolkit hands them in from the
# system image at `docker run --gpus` (first real build, 2026-10-31: libcuda.so.1 "missing" in the image). Missing in
# the image they are fine, provided the system image has them.
DRIVER_LIB_PREFIXES = ("libcuda.so", "libnvidia-")
MOUNT = "mnt/rigsim-selftest"


def load_stick():
    """The stick's own code, so the test looks for each part exactly where a run will."""
    spec = importlib.util.spec_from_file_location("stick", HERE.parent / "payload" / "stick.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def missing_libs(ldd_output: str) -> list[str]:
    """Library names `ldd` could not resolve."""
    return sorted({ln.split()[0] for ln in ldd_output.splitlines() if "not found" in ln and ln.split()})


def ldd_in_rootfs(rootfs: Path, binary: Path) -> list[str]:
    """`ldd` of a binary as the system image sees it: its folder bind-mounted into the rootfs, ldd in a chroot."""
    m = rootfs / MOUNT
    m.mkdir(parents=True, exist_ok=True)
    subprocess.run(["mount", "--bind", str(binary.parent), str(m)], check=True)
    try:
        r = subprocess.run(["chroot", str(rootfs), "ldd", f"/{MOUNT}/{binary.name}"], capture_output=True, text=True)
        return missing_libs(r.stdout + r.stderr)
    finally:
        subprocess.run(["umount", str(m)], check=False)
        try:
            m.rmdir()
        except OSError:
            pass


def ldd_in_image(image: str, binary: Path) -> list[str]:
    """`ldd` of a binary inside a container image (llama.cpp CUDA runs in the CUDA image)."""
    r = subprocess.run(["docker", "run", "--rm", "--network", "none", "-v", f"{binary.parent}:/x:ro", image,
                        "ldd", f"/x/{binary.name}"], capture_output=True, text=True)
    return missing_libs(r.stdout + r.stderr)


def rootfs_has_lib(rootfs: Path, lib: str) -> bool:
    """Whether the system image's dynamic linker knows a library (ldconfig cache in the chroot)."""
    r = subprocess.run(["chroot", str(rootfs), "ldconfig", "-p"], capture_output=True, text=True)
    return any(ln.strip().startswith(lib + " ") for ln in r.stdout.splitlines())


def is_driver_lib(lib: str) -> bool:
    return lib.startswith(DRIVER_LIB_PREFIXES)


def nonempty(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0


def image_tar(seed: Path, image: str) -> Path:
    return seed / "images" / (image.replace("/", "_").replace(":", "_") + ".tar")


class SelfTest:
    def __init__(self, seed: Path, payload: Path, env: dict, cuda_image: str, images: list[str],
                 ldd_root: Callable[[Path], list[str]], ldd_image: Callable[[str, Path], list[str]], stick=None,
                 root_has: Callable[[str], bool] = lambda lib: True):
        self.seed, self.payload, self.env = seed, payload, env
        self.cuda_image, self.images = cuda_image, images
        self.ldd_root, self.ldd_image, self.root_has = ldd_root, ldd_image, root_has
        self.stick = stick or load_stick()
        self.results: dict[str, dict] = {}

    def opt_off(self, name: str) -> bool:
        return self.env.get(name, "0") == "1"

    def set(self, part: str, problems: list[str], left_out: str = "") -> None:
        if left_out:
            self.results[part] = dict(status="left out", why=left_out)
        else:
            self.results[part] = dict(status="missing" if problems else "ok", problems=problems)

    def binaries(self, part: str, paths: list[Path], ldd: Callable[[Path], list[str]]) -> list[str]:
        problems = []
        for p in paths:
            if not (p.is_file() and os.access(p, os.X_OK)):
                problems.append(f"{p.name}: not in the seed or not executable")
                continue
            libs = ldd(p)
            if libs:
                problems.append(f"{p.name}: libraries missing: {' '.join(libs)}")
        return problems

    def ldd_in_cuda_image(self, p: Path) -> list[str]:
        """Missing in the CUDA image, minus the driver libraries the container toolkit brings in from the system image
        — those are checked there instead."""
        out = []
        for lib in self.ldd_image(self.cuda_image, p):
            if not is_driver_lib(lib):
                out.append(lib)
            elif not self.root_has(lib):
                out.append(f"{lib} (driver library, not in the system image either)")
        return out

    def run(self) -> dict[str, dict]:
        s, e = self.seed, self.env
        # container images the questions run in, and the CPU toolkit's offline packages
        self.set("images", [f"{img}: image archive missing" for img in self.images if not nonempty(image_tar(s, img))])
        self.set("debs", [] if list((s / "debs").glob("*.deb")) else ["no offline packages for the CPU toolkit"])
        # the stick's own probe, in the system image
        self.set("vkprobe", self.binaries("vkprobe", [self.payload / "bin" / "vkprobe"], self.ldd_root))
        # llama.cpp: Vulkan in the system image, CUDA in the CUDA image
        self.set("llama.cpp vulkan", self.binaries("llama", [s / "llama" / "vulkan" / b for b in LLAMA_BINS], self.ldd_root))
        self.set("llama.cpp cuda", self.binaries("llama", [s / "llama" / "cuda" / b for b in LLAMA_BINS],
                                                 self.ldd_in_cuda_image))
        # models
        if self.opt_off("SKIP_MODELS"):
            self.set("models", [], "SKIP_MODELS=1")
        else:
            got = [p for p in (s / "models").glob("*.gguf") if nonempty(p)]
            self.set("models", [] if len(got) >= 2 else [f"{len(got)} of 2 models in the seed"])
        # Blender: the offline route's pieces, found the way the stick finds them, and its libraries
        if not e.get("BLENDER_LAUNCHER_URL"):
            self.set("blender", [], "BLENDER_LAUNCHER_URL is empty")
        else:
            b = s / "blender"
            ver = (b / "BLENDER_VERSION").read_text().strip() if (b / "BLENDER_VERSION").is_file() else ""
            exe, scenes = self.stick.blender_offline(b, ver) if ver else (None, {})
            problems = [] if ver else ["BLENDER_VERSION not in the seed"]
            if exe is None:
                problems.append("Blender executable not in the seed")
            else:
                problems += self.binaries("blender", [exe], self.ldd_root)
            problems += [f"scene {n} not in the seed" for n in self.stick.BLENDER_SCENES if n not in scenes]
            if not self.stick.benchmark_script(b).is_file():
                problems.append("benchmark script not in the seed")
            self.set("blender", problems)
        # Intel MLC
        if not e.get("MLC_URL"):
            self.set("mlc", [], "MLC_URL is empty")
        else:
            self.set("mlc", self.binaries("mlc", [s / "tools" / "mlc" / "mlc"], self.ldd_root))
        # vendor images (AMD ROCm, Intel oneAPI)
        for v in ("rocm", "oneapi"):
            opt = f"SKIP_{v.upper()}"
            if self.opt_off(opt):
                self.set(v, [], f"{opt}=1")
            else:
                self.set(v, [] if nonempty(s / "images" / f"vendor-{v}.tar") else [f"vendor-{v}.tar not in the seed"])
        return self.results


def report(results: dict[str, dict]) -> str:
    lines = []
    for part, r in results.items():
        detail = r.get("why") or "; ".join(r.get("problems", []))
        lines.append(f"  {r['status']:<9} {part}{'  — ' + detail if detail else ''}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--seed", type=Path, required=True)
    a.add_argument("--rootfs", type=Path, required=True)
    a.add_argument("--payload", type=Path, required=True)
    a.add_argument("--cuda-image", required=True)
    a.add_argument("--image", action="append", default=[])
    a.add_argument("--out", type=Path, required=True)
    o = a.parse_args(argv)
    t = SelfTest(o.seed, o.payload, dict(os.environ), o.cuda_image, o.image,
                 lambda p: ldd_in_rootfs(o.rootfs, p), ldd_in_image, root_has=lambda lib: rootfs_has_lib(o.rootfs, lib))
    results = t.run()
    bad = [p for p, r in results.items() if r["status"] == "missing"]
    o.out.write_text(json.dumps(dict(complete=not bad, parts=results), indent=1) + "\n")
    print("seed self-test:\n" + report(results))
    if bad:
        if os.environ.get("ALLOW_INCOMPLETE", "0") == "1":
            print(f"WARN: incomplete stick ({', '.join(bad)}); building anyway because ALLOW_INCOMPLETE=1", file=sys.stderr)
            return 0
        print(f"the stick would not collect the expected data: {', '.join(bad)} — fix the build "
              "(or ALLOW_INCOMPLETE=1 for a test stick)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
