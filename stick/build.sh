#!/usr/bin/env bash
# Build the rigsim hardware stick on Ubuntu: stick/out/rigsim-stick[-legacy].iso and stick/out/seed/.
# Ubuntu, because its prebuilt NVIDIA modules are signed by Canonical and load under Secure Boot through the
# Microsoft-signed shim with nothing enrolled on the machine. The boot chain (shim, GRUB, kernel) is Ubuntu's own
# signed binaries; only grub.cfg and the squashfs are ours.
#
#   stick/build.sh                    NVIDIA open modules: Turing (RTX 20) .. Blackwell (RTX 50)
#   NVIDIA_FLAVOUR=legacy stick/build.sh   proprietary modules: Maxwell (GTX 900) .. Ada, not Blackwell
#   stick/build.sh -v                 every step's output live (as before the ticker), logs kept as well
#   REBUILD_ROOTFS=1 stick/build.sh   a fresh rootfs even if the cached one is younger than ROOTFS_MAX_AGE_DAYS (7)
#   SUITE=resolute stick/build.sh     another Ubuntu release (default: noble, 24.04 LTS)
#   KERNEL=ga                         Ubuntu's release kernel instead of the newer HWE kernel (default: hwe — newer
#                                     AMD and Intel GPUs need it; NVIDIA's signed modules exist for both)
# Seed extras (each optional; the stick records "not in the seed" when one is missing):
#   LLAMA_REF=release|latest-release|<tag|branch|commit>   llama.cpp to build; default "release": the newest patch
#                            of the previous release line (see below); tag and commit are recorded
#   SKIP_MODELS=1            no GGUF models (TinyLlama 1.1B Q4_0, Llama 2 7B Q4_0 — the scoreboards' model)
#   BLENDER_VERSION=4.2.0  BLENDER_LAUNCHER_URL=...   Blender Benchmark launcher CLI, Blender and scenes
#   BLENDER_SCRIPT_URL=...   the benchmark script for the offline route (default: script 3.1.1 from opendata)
#   MLC_URL=...              Intel Memory Latency Checker archive from Intel's download page (license: Intel's,
#                            downloaded by you at build time, not redistributed by this repository)
#   ROCM_IMAGE=...  SKIP_ROCM=1        AMD: ROCm image the HIP probe (hipBLAS) and llama.cpp HIP are built in and run
#                                      from; AMDGPU_TARGETS (space-separated gfx targets) picks what is compiled
#   ONEAPI_IMAGE=...  SKIP_ONEAPI=1    Intel: oneAPI image for the SYCL probe (oneMKL) and llama.cpp SYCL
# Needs: Docker and internet (everything else comes in the build container; BUILD_IN_CONTAINER=0: mmdebstrap,
# squashfs-tools, xorriso, mtools, dosfstools on a Debian / Ubuntu host, with sudo).
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$S/.." && pwd)"
# Tools the build downloads itself, straight from their makers (nothing of them is in this repository). Update the
# links here when newer versions come out; set a variable to "" to leave that tool off the stick.
#   Intel MLC is Intel's software under Intel's licence (shown on Intel's MLC download page): building the stick
#   with it means accepting that licence. The Blender Benchmark launcher is open source.
MLC_URL="${MLC_URL-https://downloadmirror.intel.com/926327/mlc_v3.13.tgz}"
BLENDER_LAUNCHER_URL="${BLENDER_LAUNCHER_URL-https://download.blender.org/release/BlenderBenchmark2.0/launcher/benchmark-launcher-cli-3.3.0-linux.tar.gz}"
[[ "${1:-}" == -v || "${1:-}" == --verbose ]] && { export VERBOSE=1; shift; }   # every step's output live
# By default the whole build runs inside an Ubuntu build container (builder/Dockerfile): the host needs only Docker,
# whatever its distribution or SELinux. The repository is mounted at the same path, so paths handed to the host's
# Docker daemon (for the inner build containers) mean the same thing inside and out. BUILD_IN_CONTAINER=0 builds on
# the host directly (Debian / Ubuntu hosts; needs the tools listed below).
if [[ "${RIGSIM_IN_CONTAINER:-0}" != 1 && "${BUILD_IN_CONTAINER:-1}" == 1 ]]; then
  command -v docker >/dev/null || { echo "Docker is missing: install Docker, then run again" >&2; exit 2; }
  SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"
  [[ -S "$SOCK" ]] || { echo "no Docker socket at $SOCK (DOCKER_SOCK=... to point elsewhere)" >&2; exit 2; }
  BUILDER="rigsim/stick-builder:$(sha256sum "$S/builder/Dockerfile" | cut -c1-12)"
  docker image inspect "$BUILDER" >/dev/null 2>&1 || { echo "building the build container $BUILDER"; docker build -q -t "$BUILDER" "$S/builder"; }
  envs=()
  for v in VERBOSE REBUILD_ROOTFS ROOTFS_MAX_AGE_DAYS SUITE NVIDIA_BRANCH NVIDIA_FLAVOUR MIRROR KERNEL ROCM_IMAGE ONEAPI_IMAGE AMDGPU_TARGETS LLAMA_REF \
           SKIP_MODELS SKIP_ROCM SKIP_ONEAPI BLENDER_LAUNCHER_URL BLENDER_VERSION MLC_URL; do
    [[ -n "${!v+x}" ]] && envs+=(-e "$v=${!v}")          # set, even if empty ("" leaves a tool out)
  done
  tty=(); [[ -t 0 && -t 1 ]] && tty=(-it)
  rc=0
  docker run --rm "${tty[@]}" --privileged --security-opt label=disable \
    -v "$SOCK:/var/run/docker.sock" -v "$REPO:$REPO" -w "$REPO" -e RIGSIM_IN_CONTAINER=1 "${envs[@]}" \
    "$BUILDER" bash "$S/build.sh" "$@" || rc=$?
  # the build ran as root in the container: hand the results back to the user who started it
  [[ -d "$S/out" ]] && docker run --rm --security-opt label=disable -v "$S/out:$S/out" "$BUILDER" \
    chown -R "${SUDO_UID:-$(id -u)}:${SUDO_GID:-$(id -g)}" "$S/out"
  exit $rc
fi
source "$S/tools/deps.sh"
need mmdebstrap:mmdebstrap mksquashfs:squashfs-tools xorriso:xorriso mcopy:mtools mmd:mtools mkfs.vfat:dosfstools \
     curl:curl sha256sum:coreutils tar:tar python3:python3-minimal
command -v docker >/dev/null || { echo "Docker is missing: install Docker (docker.io or Docker's own docker-ce), then run again" >&2; exit 2; }
LOGS="$S/out/logs/$(date +%Y%m%d-%H%M%S)"; source "$S/tools/ticker.sh"
sudo -v                                                  # ask for the password once, now; keep it fresh until we exit
( while kill -0 $$ 2>/dev/null; do sudo -n true 2>/dev/null; sleep 50; done ) &
echo "logs: $LOGS"
SUITE="${SUITE:-noble}"; NVIDIA_BRANCH="${NVIDIA_BRANCH:-580}"; NVIDIA_FLAVOUR="${NVIDIA_FLAVOUR:-open}"
MIRROR="${MIRROR:-http://archive.ubuntu.com/ubuntu}"; KERNEL="${KERNEL:-hwe}"
ROCM_IMAGE="${ROCM_IMAGE:-rocm/dev-ubuntu-24.04:6.4.1-complete}"
ONEAPI_IMAGE="${ONEAPI_IMAGE:-intel/oneapi-basekit:2025.2.0-0-devel-ubuntu24.04}"
AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx900 gfx906 gfx908 gfx90a gfx942 gfx1010 gfx1012 gfx1030 gfx1031 gfx1032 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201}"
case "$NVIDIA_FLAVOUR" in open) SUFFIX=""; MODSUFFIX="-open";; legacy) SUFFIX="-legacy"; MODSUFFIX="";;
  *) echo "NVIDIA_FLAVOUR is open or legacy" >&2; exit 2;; esac
OUT="$S/out"; W="$OUT/work$SUFFIX"; ROOT="$W/rootfs"; ISO="$W/iso"
CUDA_IMAGE="$(sed -n 's/^image = "\(nvidia[^"]*\)"/\1/p' "$S"/questions/*.toml | head -1)"
CPU_IMAGE="$(sed -n 's/^image = "\(debian[^"]*\)"/\1/p' "$S"/questions/*.toml | head -1)"
VERSION="$(git -C "$REPO" rev-parse --short HEAD)$(git -C "$REPO" diff --quiet -- stick bench/hw || echo -dirty)$SUFFIX"
STICK_VERSION="$(cat "$S/VERSION")"                   # the stick's own release number (semver, see VERSION)
export STICK_COMMIT="$(git -C "$REPO" rev-parse HEAD)"
export STICK_DIRTY="$(git -C "$REPO" diff --quiet -- stick bench/hw && echo 0 || echo 1)"
export SUITE KERNEL NVIDIA_BRANCH NVIDIA_FLAVOUR MIRROR
sudo rm -rf "$W" && mkdir -p "$W" "$ISO"/{casper,boot/grub,EFI/boot,.disk} "$OUT/seed/images" "$OUT/seed/debs"

# --- caches: a part is rebuilt only when what it is built from changes (its key); out/cache holds downloads ------
C="$OUT/cache"; mkdir -p "$C"
key() { printf '%s\n' "$@" | sha256sum | cut -c1-16; }
srckey() { find "$@" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16; }   # content of files
cache_ok() { [[ -f "$1" && "$(cat "$1")" == "$2" ]]; }                                         # marker holds the key
cached_step() {   # marker key label command...: skip when the marker holds the key, else run and record it
  local marker="$1" k="$2" label="$3"; shift 3
  if cache_ok "$marker" "$k"; then printf ' ✓  %s  (cached)\n' "$label"; return 0; fi
  step "$label" "$@"
  echo "$k" > "$marker"
}
APT_CACHE="$C/apt-$SUITE"; mkdir -p "$APT_CACHE"

# The finished rootfs (base + packages, before the stick's own files) is reused for ROOTFS_MAX_AGE_DAYS (default 7)
# while its inputs are unchanged (release, kernel, NVIDIA branch and flavour, mirror, chroot-setup.sh, the overlay);
# older, it is built again so driver and security updates come in. REBUILD_ROOTFS=1 forces a fresh one. The build
# date is recorded in the manifest (ROOTFS_BUILT), the packages as always.
RKEY="$(key "$SUITE" "$KERNEL" "$NVIDIA_BRANCH" "$NVIDIA_FLAVOUR" "$MODSUFFIX" "$MIRROR" "$(srckey "$S/ubuntu/chroot-setup.sh" "$S/ubuntu/rootfs")")"
RCACHE="$C/rootfs/$RKEY"; MAX_DAYS="${ROOTFS_MAX_AGE_DAYS:-7}"
rootfs_mounts() {
  sudo mkdir -p "$ROOT/dev" "$ROOT/proc" "$ROOT/sys"
  sudo mount --bind /dev "$ROOT/dev"; sudo mount -t proc proc "$ROOT/proc"; sudo mount -t sysfs sys "$ROOT/sys"
  trap 'sudo umount -l "$ROOT/var/cache/apt/archives" "$ROOT/dev" "$ROOT/proc" "$ROOT/sys" 2>/dev/null || true' EXIT
}
built=0; [[ -f "$RCACHE.built" ]] && built="$(cat "$RCACHE.built")"
age_days=$(( ($(date +%s) - built) / 86400 ))
if [[ "${REBUILD_ROOTFS:-0}" != 1 && -d "$RCACHE" && $built -gt 0 && $age_days -lt $MAX_DAYS ]]; then
  sudo mkdir -p "$ROOT"
  step "rootfs (cached, built $age_days day(s) ago; REBUILD_ROOTFS=1 for a fresh one)" sudo cp -a "$RCACHE/." "$ROOT/"
  rootfs_mounts
else
step "rootfs base ($SUITE)" sudo mmdebstrap --variant=minbase --components="main restricted universe multiverse" \
  --include=apt,ca-certificates,curl,gnupg --skip=essential/unlink \
  --setup-hook='mkdir -p "$1"/var/cache/apt/archives' --setup-hook="sync-in $APT_CACHE /var/cache/apt/archives" \
  --customize-hook="sync-out /var/cache/apt/archives $APT_CACHE" "$SUITE" "$ROOT" "$MIRROR"
sudo chroot "$ROOT" test -x /usr/bin/apt-get || { echo "the rootfs has no apt-get: mmdebstrap did not produce a usable base" \
  "(see the rootfs-base log); its top level: $(ls "$ROOT" 2>/dev/null | tr '\n' ' ')" >&2; exit 1; }
sudo mkdir -p "$ROOT/opt/rigsim" "$ROOT/tmp/rigsim-build"
sudo cp "$S/ubuntu/chroot-setup.sh" "$ROOT/tmp/rigsim-build/"
sudo cp -r "$S/ubuntu/rootfs/." "$ROOT/"
rootfs_mounts
sudo mkdir -p "$ROOT/var/cache/apt/archives"; sudo mount --bind "$APT_CACHE" "$ROOT/var/cache/apt/archives"
step "rootfs packages (kernel, NVIDIA $NVIDIA_BRANCH $NVIDIA_FLAVOUR, tools)" \
  sudo chroot "$ROOT" env SUITE="$SUITE" NVIDIA_BRANCH="$NVIDIA_BRANCH" MODSUFFIX="$MODSUFFIX" KERNEL="$KERNEL" MIRROR="$MIRROR" \
  KEEP_APT_CACHE=1 sh /tmp/rigsim-build/chroot-setup.sh
sudo umount "$ROOT/var/cache/apt/archives"
echo "apt cache: $(ls "$APT_CACHE"/*.deb 2>/dev/null | wc -l) packages, $(du -sh "$APT_CACHE" | cut -f1)"
# keep it: one rootfs per key, older ones (other settings, outdated) removed; -x leaves /dev /proc /sys out
save_rootfs() {
  sudo rm -rf "$C/rootfs" && mkdir -p "$C/rootfs"
  sudo cp -ax "$ROOT/." "$RCACHE.tmp/" && sudo mv "$RCACHE.tmp" "$RCACHE" && date +%s > "$RCACHE.built"
}
step "rootfs cache (reused for $MAX_DAYS days)" save_rootfs
built="$(cat "$RCACHE.built")"
fi
export ROOTFS_BUILT="$(date -u -d "@$built" +%Y-%m-%dT%H:%MZ)"
sudo mkdir -p "$ROOT/opt/rigsim" "$ROOT/tmp/rigsim-build"

P="$ROOT/opt/rigsim"
sudo cp -r "$REPO/bench/hw/runner" "$REPO/bench/hw/toolkits" "$P/"
sudo cp -r "$S/overlay/toolkits/." "$P/toolkits/"
sudo cp -r "$S/questions" "$P/questions"
sudo install -m 0755 "$S/payload/stick.py" "$P/stick.py"
echo "$VERSION" | sudo tee "$P/VERSION" >/dev/null
echo "$STICK_VERSION" | sudo tee "$P/STICK_VERSION" >/dev/null
sudo mkdir -p "$P/bin"
VKK="$(key "$SUITE" "$(srckey "$S/src/vkprobe")")"; mkdir -p "$C/vkprobe"
cached_step "$C/vkprobe/.key" "$VKK" "vkprobe (Vulkan probe)" \
  docker run --rm --security-opt label=disable -v "$S/src/vkprobe:/src:ro" -v "$C/vkprobe:/out" "ubuntu:$SUITE" sh -c '
  set -e; export DEBIAN_FRONTEND=noninteractive; apt-get update -qq
  apt-get install -y -qq gcc libc6-dev libvulkan-dev glslc >/dev/null
  cp -r /src /tmp/vk && sh /tmp/vk/build.sh /out/vkprobe'
sudo install -m 0755 "$C/vkprobe/vkprobe" "$P/bin/vkprobe"
# exact versions of the system image: packages, the stick's own files (merged with the seed's on every run)
step "manifest (system image)" sudo -E python3 "$S/tools/manifest.py" rootfs "$ROOT" "$P" "$W/manifest-rootfs.json"
sudo install -m 0644 "$W/manifest-rootfs.json" "$P/manifest.json"

# A file of the rootfs, links resolved INSIDE the rootfs. Many of Ubuntu's boot files are links with absolute
# targets (alternatives, /etc/alternatives/..., /boot/vmlinuz-...): read from outside, they would resolve against
# the build host's own filesystem — missing in the build container, the wrong file on a Debian/Ubuntu host.
rootfile() {
  local p; p="$(sudo chroot "$ROOT" readlink -e "$1" 2>/dev/null)" || { echo "not in the rootfs: $1" >&2; return 1; }
  echo "$ROOT$p"
}
boot_files() {
# the newest kernel and its initrd inside the rootfs, by file name (never through a link: an absolute link would
# resolve on the build host)
KV="$(ls "$ROOT"/boot/vmlinuz-* | sort -V | tail -1)"; KV="${KV##*/vmlinuz-}"
echo "kernel $KV"
sudo cp "$(rootfile "/boot/vmlinuz-$KV")" "$ISO/casper/vmlinuz"
sudo cp "$(rootfile "/boot/initrd.img-$KV")" "$ISO/casper/initrd"
SHIM="$(rootfile /usr/lib/shim/shimx64.efi.signed.latest 2>/dev/null || rootfile /usr/lib/shim/shimx64.efi.signed)"
echo "shim $SHIM"
sudo cp "$SHIM" "$ISO/EFI/boot/bootx64.efi"
sudo cp "$(rootfile /usr/lib/shim/mmx64.efi)" "$ISO/EFI/boot/mmx64.efi"
sudo cp "$(rootfile /usr/lib/grub/x86_64-efi-signed/gcdx64.efi.signed)" "$ISO/EFI/boot/grubx64.efi"
printf 'rigsim hardware stick %s (Ubuntu %s)\n' "$VERSION" "$SUITE" > "$ISO/.disk/info"
cp "$S/ubuntu/grub.cfg" "$ISO/boot/grub/grub.cfg"
sudo chroot "$ROOT" grub-mkstandalone --format=i386-pc --output=/tmp/core.img \
  --install-modules="linux normal iso9660 biosdisk search configfile" --modules="linux normal iso9660 biosdisk search" \
  --locales="" --fonts="" "boot/grub/grub.cfg=/tmp/rigsim-build/bios.cfg" 2>/dev/null || \
  { printf 'search --set=root --file /.disk/info\nset prefix=($root)/boot/grub\nconfigfile /boot/grub/grub.cfg\n' | sudo tee "$ROOT/tmp/rigsim-build/bios.cfg" >/dev/null;
    sudo chroot "$ROOT" grub-mkstandalone --format=i386-pc --output=/tmp/core.img \
      --install-modules="linux normal iso9660 biosdisk search configfile" --modules="linux normal iso9660 biosdisk search" \
      --locales="" --fonts="" "boot/grub/grub.cfg=/tmp/rigsim-build/bios.cfg"; }
cat "$(rootfile /usr/lib/grub/i386-pc/cdboot.img)" "$(rootfile /tmp/core.img)" > "$ISO/boot/grub/bios.img"
cp "$(rootfile /usr/lib/grub/i386-pc/boot_hybrid.img)" "$W/boot_hybrid.img"
dd if=/dev/zero of="$ISO/boot/grub/efiboot.img" bs=1M count=10 status=none
mkfs.vfat "$ISO/boot/grub/efiboot.img" >/dev/null
mmd -i "$ISO/boot/grub/efiboot.img" ::/EFI ::/EFI/boot
mcopy -i "$ISO/boot/grub/efiboot.img" "$ISO"/EFI/boot/*.efi ::/EFI/boot/
}
step "boot files (signed shim, GRUB, kernel)" boot_files
sudo rm -rf "$ROOT/tmp/rigsim-build" "$ROOT/tmp/core.img" "$ROOT"/var/lib/apt/lists/* "$ROOT"/var/cache/apt/archives/*.deb
sudo umount -l "$ROOT/dev" "$ROOT/proc" "$ROOT/sys"; trap - EXIT
step "squashfs" sudo mksquashfs "$ROOT" "$ISO/casper/filesystem.squashfs" -comp zstd -noappend -e boot
sudo chown -R "$(id -u):$(id -g)" "$ISO"

# checksums of every file on the ISO: casper-md5check verifies the stick on each boot (a corrupted stick shows in
# the run's record instead of as odd measurements)
# the BIOS boot image is left out: xorriso writes the boot-info table into it (-boot-info-table, --grub2-boot-info),
# so its bytes on the ISO differ from the file here (first stick check after the fix: "fail")
(cd "$ISO" && find . -type f ! -name md5sum.txt ! -path ./boot/grub/bios.img ! -path ./boot/grub/boot.cat -print0 \
  | sort -z | xargs -0 md5sum > md5sum.txt)
step "ISO (BIOS + UEFI)" xorriso -as mkisofs -iso-level 3 -full-iso9660-filenames -volid "RIGSIM-STICK" \
  -eltorito-boot boot/grub/bios.img -no-emul-boot -boot-load-size 4 -boot-info-table --eltorito-catalog boot/grub/boot.cat \
  --grub2-boot-info --grub2-mbr "$W/boot_hybrid.img" \
  -eltorito-alt-boot -e boot/grub/efiboot.img -no-emul-boot -append_partition 2 0xef "$ISO/boot/grub/efiboot.img" \
  -output "$OUT/rigsim-stick$SUFFIX.iso" -graft-points "$ISO" /boot/grub/bios.img="$ISO/boot/grub/bios.img" \
  /boot/grub/efiboot.img="$ISO/boot/grub/efiboot.img"
# the same check the stick runs at boot, here: every listed file read back out of the finished ISO
verify_iso() {
  local x="$W/iso-check"; rm -rf "$x"; mkdir -p "$x"
  xorriso -osirrox on -indev "$OUT/rigsim-stick$SUFFIX.iso" -extract / "$x" >/dev/null 2>&1
  (cd "$x" && md5sum --quiet -c md5sum.txt); local rc=$?
  chmod -R u+w "$x"; rm -rf "$x"; return $rc
}
step "ISO checksums (as casper-md5check sees them)" verify_iso

: > "$OUT/seed/images/DIGESTS"
image_id() { docker image inspect -f '{{.Id}}' "$1"; }
for img in "$CUDA_IMAGE" "$CPU_IMAGE"; do
  step "pull $img" docker pull "$img"
  docker image inspect -f '{{index .RepoDigests 0}}' "$img" >> "$OUT/seed/images/DIGESTS"
  tar="$OUT/seed/images/$(echo "$img" | tr '/:' '__').tar"
  [[ -f "$tar" ]] || rm -f "$tar.key"
  cached_step "$tar.key" "$(image_id "$img")" "save $img" docker save "$img" -o "$tar"
done
DEBS="likwid fio python3-minimal gcc libc6-dev libopenblas-dev libdnnl-dev"
debs() { rm -f "$OUT"/seed/debs/*.deb; docker run --rm --security-opt label=disable -v "$OUT/seed/debs:/out" "$CPU_IMAGE" sh -c \
  "apt-get update -qq && apt-get install -y -qq --download-only --no-install-recommends $DEBS >/dev/null && cp /var/cache/apt/archives/*.deb /out/"; }
cached_step "$OUT/seed/debs/.key" "$(key "$(image_id "$CPU_IMAGE")" "$DEBS")" "offline debs (cpu toolkit)" debs

# llama.cpp: the ref is resolved to a commit first, so every backend (and the vendor images) builds the same one, and
# a build is reused while commit, image and build script stay the same
# Releases only (vX.Y.Z, every one to two weeks, with release notes; the b<N> tags are pre-releases, many a day),
# and defensively: "release" (default) takes the previous release line — when v0.8.0 is out, the newest v0.7.x —
# so the line has had time for its hotfixes (v0.4.1 followed v0.4.0); "latest-release" takes the newest release.
# Any other LLAMA_REF (tag, branch, commit) is built as given. The rule and the result are recorded.
LLAMA_URL=https://github.com/ggml-org/llama.cpp
LLAMA_REF="${LLAMA_REF:-release}"; L="$OUT/seed/llama"; mkdir -p "$L/cuda" "$L/vulkan"
export LLAMA_SELECT="$LLAMA_REF"
pick_release() {   # release | latest-release
  local tags lines line
  tags="$(git ls-remote --tags --refs "$LLAMA_URL" 'refs/tags/v*' | awk '{print $2}' | sed 's#^refs/tags/##' \
          | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | sort -V || true)"
  [[ -n "$tags" ]] || return 1
  lines="$(sed -E 's/^(v[0-9]+\.[0-9]+)\..*/\1/' <<<"$tags" | uniq)"
  if [[ "$1" == release && "$(wc -l <<<"$lines")" -ge 2 ]]; then line="$(tail -2 <<<"$lines" | head -1)"
  else line="$(tail -1 <<<"$lines")"; fi
  awk -v l="$line" 'index($0, l ".") == 1' <<<"$tags" | tail -1
}
if [[ "$LLAMA_REF" == release || "$LLAMA_REF" == latest-release ]]; then
  LLAMA_REF="$(pick_release "$LLAMA_REF")"
  [[ -n "$LLAMA_REF" ]] || { echo "no llama.cpp release tag found" >&2; exit 1; }
fi
LLAMA_TAG=""
if [[ "$LLAMA_REF" =~ ^[0-9a-f]{7,40}$ ]]; then LLAMA_COMMIT="$LLAMA_REF"
else
  refs="$(git ls-remote "$LLAMA_URL" "refs/tags/$LLAMA_REF^{}" "refs/tags/$LLAMA_REF" "refs/heads/$LLAMA_REF")"
  # an annotated tag: the commit it points to (^{}), not the tag object
  LLAMA_COMMIT="$( (grep -F "refs/tags/$LLAMA_REF^{}" <<<"$refs" || grep -F "refs/tags/$LLAMA_REF" <<<"$refs" \
                    || grep -F "refs/heads/$LLAMA_REF" <<<"$refs" || true) | head -1 | cut -f1)"
  grep -qF "refs/tags/$LLAMA_REF" <<<"$refs" && LLAMA_TAG="$LLAMA_REF"
fi
[[ -n "$LLAMA_COMMIT" ]] || { echo "llama.cpp ref $LLAMA_REF not found" >&2; exit 1; }
echo "llama.cpp ${LLAMA_TAG:-$LLAMA_REF} = $LLAMA_COMMIT"
LLAMA_CUDA='
  set -e; apt-get update -qq && apt-get install -y -qq git cmake >/dev/null
  git clone -q https://github.com/ggml-org/llama.cpp /src && cd /src && git checkout -q "$REF"
  cmake -B b -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="52;61;70;75;80;86;89;90;120" -DCMAKE_CUDA_FLAGS=-Wno-deprecated-gpu-targets -DBUILD_SHARED_LIBS=OFF         -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF >/dev/null
  cmake --build b --target llama-bench test-backend-ops -j"$(nproc)"
  cp b/bin/llama-bench b/bin/test-backend-ops /out/ && git rev-parse HEAD > /out/COMMIT'
LLAMA_VULKAN='
  set -e; export DEBIAN_FRONTEND=noninteractive; apt-get update -qq
  apt-get install -y -qq git cmake build-essential libvulkan-dev glslc spirv-headers >/dev/null
  git clone -q https://github.com/ggml-org/llama.cpp /src && cd /src && git checkout -q "$REF"
  cmake -B b -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DGGML_NATIVE=OFF >/dev/null
  cmake --build b --target llama-bench test-backend-ops -j"$(nproc)"
  cp b/bin/llama-bench b/bin/test-backend-ops /out/ && git rev-parse HEAD > /out/COMMIT'
llama_build() {   # backend image script
  rm -f "$L/$1"/*; mkdir -p "$L/$1"
  docker run --rm --security-opt label=disable -v "$L/$1:/out" -e REF="$LLAMA_COMMIT" "$2" sh -c "$3"
  echo "$LLAMA_TAG" > "$L/$1/TAG"
}
docker pull -q "ubuntu:$SUITE" >/dev/null
cached_step "$C/llama-cuda.key" "$(key "$LLAMA_COMMIT" "$(image_id "$CUDA_IMAGE")" "$LLAMA_CUDA")" \
  "llama.cpp CUDA (9 architectures)" llama_build cuda "$CUDA_IMAGE" "$LLAMA_CUDA"
cached_step "$C/llama-vulkan.key" "$(key "$LLAMA_COMMIT" "$(image_id "ubuntu:$SUITE")" "$LLAMA_VULKAN")" \
  "llama.cpp Vulkan" llama_build vulkan "ubuntu:$SUITE" "$LLAMA_VULKAN"
: > "$OUT/seed/images/VENDOR.tsv"
vendor_build() {  # name base src tag
  local c="rigsim-build-$1"
  docker rm -f "$c" >/dev/null 2>&1 || true
  if docker run --security-opt label=disable --name "$c" -v "$S/src/$1/build-in-image.sh:/build.sh:ro" -v "$3:/src:ro" \
       -e LLAMA_REF="$LLAMA_COMMIT" -e LLAMA_TAG="$LLAMA_TAG" -e AMDGPU_TARGETS="$AMDGPU_TARGETS" "$2" bash /build.sh; then
    docker commit "$c" "$4" >/dev/null && docker rm "$c" >/dev/null
  else
    docker rm -f "$c" >/dev/null 2>&1 || true; return 1
  fi
}
vendor() {   # name base-image label source-dir. Image reused while base, llama.cpp commit, targets and sources stay.
  local name="$1" base="$2" label="$3" src="$4" tar="$OUT/seed/images/vendor-$1.tar"
  step "$label: pull $base" docker pull "$base" || return 1
  local digest; digest="$(docker image inspect -f '{{index .RepoDigests 0}}' "$base")"; echo "$digest" >> "$OUT/seed/images/DIGESTS"
  local k; k="$(key "$digest" "$LLAMA_COMMIT" "$AMDGPU_TARGETS" "$(srckey "$S/src/$name" "$src")")"
  local tag="rigsim/$name:$k"
  if docker image inspect "$tag" >/dev/null 2>&1; then printf ' ✓  %s: build  (cached)\n' "$label"
  else step "$label: build" vendor_build "$name" "$base" "$src" "$tag" || return 1; fi
  if [[ -f "$tar" ]] && cache_ok "$tar.key" "$k"; then printf ' ✓  %s: save  (cached)\n' "$label"
  else rm -f "$tar.key"; step "$label: save" docker save "$tag" -o "$tar" || return 1; echo "$k" > "$tar.key"; fi
  printf '%s\t%s\t%s\n' "$name" "$tag" "$base" >> "$OUT/seed/images/VENDOR.tsv"
}
for v in rocm oneapi; do                                  # skipped this time: not on this stick (local images stay)
  u="SKIP_${v^^}"; [[ "${!u:-0}" == 1 ]] && rm -f "$OUT/seed/images/vendor-$v.tar" "$OUT/seed/images/vendor-$v.tar.key"
done
[[ "${SKIP_ROCM:-0}" == 1 ]] || vendor rocm "$ROCM_IMAGE" "AMD ROCm image" "$REPO/bench/hw/toolkits/cuda/src" \
  || echo "    WARN: no ROCm image; the stick records it as not in the seed"
[[ "${SKIP_ONEAPI:-0}" == 1 ]] || vendor oneapi "$ONEAPI_IMAGE" "Intel oneAPI image" "$S/src/syclprobe" \
  || echo "    WARN: no oneAPI image; the stick records it as not in the seed"
if [[ "${SKIP_MODELS:-0}" != 1 ]]; then
  M="$OUT/seed/models"; mkdir -p "$M"
  fetch() { [[ -s "$M/$2" ]] || curl -fL --retry 3 -# -o "$M/$2" "$1"; }
  step "model TinyLlama 1.1B" fetch https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf tinyllama-1.1b-q4_0.gguf
  step "model Llama 2 7B" fetch https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf llama-2-7b-q4_0.gguf
  (cd "$M" && sha256sum *.gguf > SHA256SUMS)
fi
if [[ -n "${BLENDER_LAUNCHER_URL:-}" ]]; then
  blender_seed() {
  B="$OUT/seed/blender"; mkdir -p "$B/home"
  curl -fL "$BLENDER_LAUNCHER_URL" | tar -xz -C "$B" --strip-components=0
  cli="$(find "$B" -maxdepth 2 -name benchmark-launcher-cli -type f | head -1)"; mv -f "$cli" "$B/benchmark-launcher-cli" 2>/dev/null || true
  echo "$BLENDER_VERSION" > "$B/BLENDER_VERSION"
  HOME="$B/home" XDG_CACHE_HOME="$B/home/.cache" "$B/benchmark-launcher-cli" blender download "$BLENDER_VERSION"
  HOME="$B/home" XDG_CACHE_HOME="$B/home/.cache" "$B/benchmark-launcher-cli" scenes download     --blender-version "$BLENDER_VERSION" monster junkshop classroom
  # the official offline route (opendata.blender.org/about, "Run Offline"): Blender + this script + a scene, no
  # launcher, no network; the launcher needs the server even to list devices
  rm -rf "$B/script" && mkdir -p "$B/script"
  curl -fL "$BLENDER_SCRIPT_URL" -o "$B/benchmark-script.tar.gz"
  tar -xzf "$B/benchmark-script.tar.gz" -C "$B/script"
  m="$(find "$B/script" -name main.py | head -1)"; [[ -n "$m" ]] || { echo "no main.py in the benchmark script" >&2; return 1; }
  [[ "$(dirname "$m")" == "$B/script" ]] || { mv "$(dirname "$m")"/* "$B/script/"; }
  # the launcher may keep Blender and the scenes as archives in its cache: unpack them once for the offline route
  rm -rf "$B/unpacked"; mkdir -p "$B/unpacked/blender" "$B/unpacked/scenes"
  find "$B/home" -type f \( -name 'blender-*linux*.tar.xz' -o -name 'blender-*linux*.tar.gz' \) -print0 \
    | xargs -0 -r -I{} tar -xf {} -C "$B/unpacked/blender" 2>/dev/null || true
  find "$B/home" -type f \( -name '*.tar.bz2' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.zip' \) ! -name 'blender-*' \
    | while read -r a; do
        n="$(basename "$a")"; n="${n%%.*}"; mkdir -p "$B/unpacked/scenes/$n"
        case "$a" in *.zip) python3 -c "import sys,zipfile;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$a" "$B/unpacked/scenes/$n" ;;
                     *) tar -xf "$a" -C "$B/unpacked/scenes/$n" ;; esac
      done
  echo "blender offline route: $(find "$B" -type f -name blender -perm -u+x | head -1); scenes: $(find "$B" -name '*.blend' | wc -l)"
  }
  : "${BLENDER_VERSION:=4.2.0}"; mkdir -p "$OUT/seed/blender"
  : "${BLENDER_SCRIPT_URL:=https://opendata.blender.org/cdn/BlenderBenchmark2.0/script/blender-benchmark-script-3.1.1.tar.gz}"
  cached_step "$OUT/seed/blender/.key" "$(key "$BLENDER_LAUNCHER_URL" "$BLENDER_VERSION" "$BLENDER_SCRIPT_URL" unpack-1)" "Blender Benchmark $BLENDER_VERSION" blender_seed
  # Does Blender find every library in the stick's system image? (ldd inside the rootfs; a run is too late to learn)
  blender_libs() {
    local exe missing
    exe="$(find "$OUT/seed/blender" -type f -name blender -perm -u+x | head -1)"
    [[ -n "$exe" ]] || { echo "blender: no executable in the seed"; return 0; }
    sudo mkdir -p "$ROOT/mnt/blender-check"
    sudo mount --bind "$(dirname "$exe")" "$ROOT/mnt/blender-check"
    missing="$(sudo chroot "$ROOT" ldd /mnt/blender-check/blender 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
    sudo umount "$ROOT/mnt/blender-check"; sudo rmdir "$ROOT/mnt/blender-check"
    if [[ -n "$missing" ]]; then
      echo "WARN: Blender will not start on the stick — libraries missing in the system image: $missing" >&2
    else
      echo "blender: every library it links is in the system image"
    fi
  }
  step "Blender libraries" blender_libs
else
  echo "== seed: Blender Benchmark left out (BLENDER_LAUNCHER_URL is empty)"
fi
if [[ -n "${MLC_URL:-}" ]]; then
  mlc_seed() {
  T="$OUT/seed/tools/mlc"; rm -rf "$T" && mkdir -p "$T"
  curl -fL "$MLC_URL" | tar -xz -C "$T"
  m="$(find "$T" -path '*Linux*' -name mlc -type f | head -1)"; cp "$m" "$T/mlc" && chmod +x "$T/mlc"
  }
  mkdir -p "$OUT/seed/tools"
  echo "    Intel MLC is downloaded from Intel under Intel's licence; building with it means accepting that licence."
  cached_step "$OUT/seed/tools/mlc.key" "$(key "$MLC_URL")" "Intel MLC" mlc_seed
else
  echo "== seed: Intel MLC left out (MLC_URL is empty)"
fi
echo "$VERSION" > "$OUT/seed/VERSION"
export LLAMA_REF LLAMA_SELECT AMDGPU_TARGETS ROCM_IMAGE ONEAPI_IMAGE BLENDER_VERSION BLENDER_SCRIPT_URL
step "manifest (seed)" env IMAGES="$CUDA_IMAGE $CPU_IMAGE" python3 "$S/tools/manifest.py" seed "$OUT/seed" "$OUT/seed/MANIFEST.json"
echo "built $OUT/rigsim-stick$SUFFIX.iso ($VERSION) and $OUT/seed"
