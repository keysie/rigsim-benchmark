#!/usr/bin/env bash
# Boot test of the stick's disk layout (stick/tools/layout.sh) in QEMU, without building the whole stick: a small
# image with the same boot chain (Ubuntu's signed shim and GRUB from the host's packages, GRUB's BIOS core), the same
# xorriso layout and the data partition flash.sh adds. It boots it as a USB stick under
#   - UEFI (OVMF), Secure Boot off
#   - UEFI (OVMF with Microsoft's keys), Secure Boot on
#   - BIOS (SeaBIOS)
# and passes when GRUB reaches the image's grub.cfg each time. Catches layout regressions; picky real firmwares (the
# reason for the GPT layout, 2026-10-31) still need a real machine.
#
# Needs root and: xorriso mtools dosfstools fdisk e2fsprogs qemu-system-x86 ovmf seabios grub-pc-bin
#                 grub-efi-amd64-signed shim-signed  (Debian / Ubuntu package names)
set -euo pipefail
S="$(cd "$(dirname "$0")/../.." && pwd)"
source "$S/tools/layout.sh"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
ISO="$W/iso"; mkdir -p "$ISO"/{EFI/boot,boot/grub,.disk}
MARK=RIGSIM-BOOT-OK

# the boot chain build.sh copies from the system image, here from the host's packages
SHIM=/usr/lib/shim/shimx64.efi.signed.latest; [[ -f $SHIM ]] || SHIM=/usr/lib/shim/shimx64.efi.signed
cp "$SHIM" "$ISO/EFI/boot/bootx64.efi"
cp /usr/lib/shim/mmx64.efi "$ISO/EFI/boot/mmx64.efi"
cp /usr/lib/grub/x86_64-efi-signed/gcdx64.efi.signed "$ISO/EFI/boot/grubx64.efi"
echo "rigsim boot test" > "$ISO/.disk/info"
cat > "$ISO/boot/grub/grub.cfg" <<CFG
if [ "\$grub_platform" = pc ]; then serial --unit=0 --speed=115200; terminal_output --append serial; fi
echo "$MARK \$grub_platform"
sleep 600
CFG
# BIOS core as in build.sh, plus serial so the test can read it
printf 'search --set=root --file /.disk/info\nset prefix=($root)/boot/grub\nconfigfile /boot/grub/grub.cfg\n' > "$W/bios.cfg"
grub-mkstandalone --format=i386-pc --output="$W/core.img" \
  --install-modules="linux normal iso9660 biosdisk search configfile serial terminal echo sleep test" \
  --modules="linux normal iso9660 biosdisk search configfile serial terminal echo sleep test" --locales="" --fonts="" "boot/grub/grub.cfg=$W/bios.cfg"
cat /usr/lib/grub/i386-pc/cdboot.img "$W/core.img" > "$ISO/boot/grub/bios.img"

make_efiboot "$ISO"
make_iso "$ISO" /usr/lib/grub/i386-pc/boot_hybrid.img "$W/test.iso" RIGSIM-TEST >/dev/null 2>&1

# the stick as flash.sh writes it: image, then the data partition (ext4, label RIGSIM)
STICK="$W/stick.img"; truncate -s 2G "$STICK"
dd if="$W/test.iso" of="$STICK" conv=notrunc status=none
add_data_part "$STICK" 1024MiB >/dev/null
sfdisk -d "$STICK" > "$W/table"
grep -q '^label: gpt' "$W/table" || { echo "FAIL: not a GPT"; cat "$W/table"; exit 1; }
grep -qi 'type=C12A7328-F81F-11D2-BA4B-00A0C93EC93B' "$W/table" || { echo "FAIL: no EFI system partition"; cat "$W/table"; exit 1; }
grep -qi 'type=0FC63DAF-8483-4772-8E79-3D69D8477DE4' "$W/table" || { echo "FAIL: no Linux data partition"; cat "$W/table"; exit 1; }
sfdisk --verify "$STICK" >/dev/null || { echo "FAIL: sfdisk --verify"; sfdisk --verify "$STICK"; exit 1; }
start=$(awk -F'[=,]' '/type=0FC63DAF/ {gsub(/ /,"",$2); print $2}' "$W/table" | tail -1)
mkfs.ext4 -q -F -L RIGSIM -E offset=$((start * 512)) "$STICK" 512M
echo "ok: layout (GPT, ESP, data partition at sector $start, backup header at the end)"
# flash.sh's read-back: the image compared with the stick, its partition table left out (flash.sh rewrote it)
python3 "$S/tools/verify.py" --iso "$W/test.iso" "$STICK" >/dev/null || { echo "FAIL: verify.py rejects a correctly flashed stick"; exit 1; }
echo "ok: read-back verification passes after the data partition was added"
# flash.sh [u]pdate: the image written again in front, the data partition restored at the same sectors, untouched
size=$(awk -F'[=,]' '/type=0FC63DAF/ {gsub(/ /,"",$4); print $4}' "$W/table" | tail -1)
dd if="$W/test.iso" of="$STICK" conv=notrunc status=none
add_data_part "$STICK" "$start" "$size" >/dev/null 2>&1
[[ "$(blkid -p -O $((start * 512)) -o value -s LABEL "$STICK" 2>/dev/null)" == RIGSIM ]] && sfdisk --verify "$STICK" >/dev/null \
  || { echo "FAIL: data partition not intact after an update"; sfdisk -d "$STICK"; exit 1; }
echo "ok: update in place keeps the data partition"

boot() {   # name, firmware args...
  local name="$1"; shift
  local log="$W/$name.log"
  timeout 120 qemu-system-x86_64 -machine q35 -m 512 -nographic -no-reboot "$@" \
    -device qemu-xhci -drive if=none,id=stick,format=raw,file="$STICK",snapshot=on \
    -device usb-storage,drive=stick,removable=on,bootindex=0 > "$log" 2>&1 &
  local pid=$! t=0
  while kill -0 $pid 2>/dev/null && (( t < 110 )); do
    grep -aq "$MARK" "$log" && break; sleep 1; t=$((t + 1))
  done
  kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
  if grep -aq "$MARK" "$log"; then echo "ok: boots under $name"
  else echo "FAIL: no boot under $name; last output:"; tr -d '\r' < "$log" | tail -15; return 1; fi
}
OV=/usr/share/OVMF
cp "$OV/OVMF_VARS_4M.fd" "$W/vars.fd"; cp "$OV/OVMF_VARS_4M.ms.fd" "$W/vars-ms.fd"
rc=0
boot "UEFI" -drive if=pflash,format=raw,readonly=on,file="$OV/OVMF_CODE_4M.fd" \
  -drive if=pflash,format=raw,file="$W/vars.fd" || rc=1
boot "UEFI with Secure Boot" -global driver=cfi.pflash01,property=secure,value=on \
  -machine q35,smm=on -drive if=pflash,format=raw,readonly=on,file="$OV/OVMF_CODE_4M.secboot.fd" \
  -drive if=pflash,format=raw,file="$W/vars-ms.fd" || rc=1
boot "BIOS" || rc=1
exit $rc
