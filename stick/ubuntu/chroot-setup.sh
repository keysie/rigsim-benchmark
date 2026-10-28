#!/bin/sh
# Runs inside the rootfs chroot (build.sh). Ubuntu's signed generic kernel, casper (live boot), the precompiled
# Canonical-signed NVIDIA modules for that kernel (no DKMS: DKMS modules would be unsigned), Docker + NVIDIA
# container toolkit, the vendor-neutral OpenCL stack (NVIDIA, Intel compute-runtime, Mesa rusticl for AMD), the
# inventory tools.
set -eu
export DEBIAN_FRONTEND=noninteractive
cat > /etc/apt/sources.list <<EOF
deb ${MIRROR:-http://archive.ubuntu.com/ubuntu} $SUITE main restricted universe multiverse
deb ${MIRROR:-http://archive.ubuntu.com/ubuntu} $SUITE-updates main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu $SUITE-security main restricted universe multiverse
EOF
apt-get update
# kernel: the HWE series (newer AMD and Intel GPUs need it) unless KERNEL=ga; NVIDIA's signed modules follow it
if [ "${KERNEL:-hwe}" = hwe ]; then KFL="generic-hwe-$(. /etc/os-release; echo "$VERSION_ID")"; else KFL=generic; fi
apt-cache show "linux-image-$KFL" >/dev/null 2>&1 || { echo "no linux-image-$KFL on $SUITE: release kernel"; KFL=generic; }
apt-get install -y --no-install-recommends \
  "linux-image-$KFL" linux-firmware casper initramfs-tools \
  shim-signed grub-efi-amd64-signed grub-pc-bin grub-common \
  systemd-sysv sudo less nano rsync python3 python3-yaml kbd console-setup \
  dmidecode lshw pciutils usbutils i2c-tools lm-sensors nvme-cli smartmontools util-linux hwloc numactl cpuid msr-tools \
  docker.io containerd \
  ocl-icd-libopencl1 clinfo clpeak mesa-opencl-icd intel-opencl-icd
# the extra modules (many GPU, USB and network drivers) for exactly the kernel the meta package pulled in; there is
# no meta package for them in every series (none for the 24.04 HWE kernel), so by version
KVER="$(ls /lib/modules | sort -V | tail -1)"
if apt-cache show "linux-modules-extra-$KVER" >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends "linux-modules-extra-$KVER"
else
  echo "no linux-modules-extra-$KVER: the image package carries all modules for this kernel"
fi
# NVIDIA: precompiled, signed modules matching the installed kernel + userspace without DKMS (meta package for the
# series if there is one, else the one for this exact kernel)
NVMOD="linux-modules-nvidia-${NVIDIA_BRANCH}${MODSUFFIX}-$KFL"
apt-cache show "$NVMOD" >/dev/null 2>&1 || NVMOD="linux-modules-nvidia-${NVIDIA_BRANCH}${MODSUFFIX}-$KVER"
apt-get install -y --no-install-recommends \
  "$NVMOD" \
  "nvidia-headless-no-dkms-${NVIDIA_BRANCH}${MODSUFFIX}" "nvidia-utils-${NVIDIA_BRANCH}" "libnvidia-compute-${NVIDIA_BRANCH}"
# Vulkan (llama-bench on every vendor; NVIDIA's ICD is in libnvidia-gl), OpenMP for llama.cpp's CPU path,
# Level Zero for Blender's oneAPI device — optional: a missing package must not stop the build
apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers vulkan-tools libgomp1
apt-get install -y --no-install-recommends "libnvidia-gl-${NVIDIA_BRANCH}" || echo "WARN: no NVIDIA Vulkan ICD"
# nvidia-modprobe creates the device nodes the Vulkan driver opens (nvidia-uvm, nvidia-modeset) without X;
# first trial run: the NVIDIA ICD was installed but the loader could not create an instance through it
apt-get install -y --no-install-recommends nvidia-modprobe || echo "WARN: no nvidia-modprobe"
# NVIDIA's Vulkan ICD points at libGLX_nvidia, which cannot create an instance without an X server (second trial
# run: modules, device nodes and libraries all there, still "Could not get vkCreateInstance"). The EGL library
# carries the same Vulkan driver and works headless (NVIDIA container toolkit issue 191).
ICD=/usr/share/vulkan/icd.d/nvidia_icd.json
if [ -f "$ICD" ] && [ -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]; then
  sed -i 's/libGLX_nvidia\.so\.0/libEGL_nvidia.so.0/' "$ICD"; grep library_path "$ICD"
fi
GLX=/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0
if [ -e "$GLX" ] && ldd "$GLX" | grep -q 'not found'; then
  echo "the NVIDIA Vulkan driver misses libraries:" >&2; ldd "$GLX" | grep 'not found' >&2; exit 1
fi
apt-get install -y --no-install-recommends libze1 intel-level-zero-gpu || echo "WARN: no Level Zero (Blender oneAPI)"
apt-get install -y --no-install-recommends libxrender1 libxi6 libxkbcommon0 libsm6 libgl1 || echo "WARN: Blender libs"
if dpkg -l | grep -q '^ii  nvidia-dkms'; then echo "DKMS slipped in: its modules would not load under Secure Boot" >&2; exit 1; fi
# NVIDIA container toolkit (docker --gpus)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
# Docker starts once the orchestrator has put its data root on the stick's data partition
systemctl disable docker.service docker.socket containerd.service || true
systemctl enable rigsim-stick.service
systemctl mask getty@tty1.service
echo rigsim-stick > /etc/hostname
update-initramfs -u -k all
# relative links: seen from outside the chroot, an absolute /boot/... would point at the build host's own /boot
ln -sf "$(basename "$(ls /boot/vmlinuz-* | sort -V | tail -1)")" /boot/vmlinuz
ln -sf "$(basename "$(ls /boot/initrd.img-* | sort -V | tail -1)")" /boot/initrd.img
# the package cache is build.sh's (bind-mounted, kept between builds); it empties the mount point itself
[ "${KEEP_APT_CACHE:-0}" = 1 ] || apt-get clean
