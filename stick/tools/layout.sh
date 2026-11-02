# The stick's disk layout, shared by build.sh (the ISO), flash.sh (the data partition) and tests/boot/boot-test.sh.
#
# Since stick 0.13.5 the image uses the layout of current Ubuntu ISOs: a GPT with the ISO 9660 filesystem as partition
# 1 (starting after 16 sectors, so no partition contains the table itself), the EFI system partition as a real GPT
# ESP (partition 2), a protective MBR with the boot flag set, and GRUB's hybrid MBR code for BIOS boot. Before, the
# image was an MBR-only hybrid (partition 1 from sector 0, ESP as MBR type 0xEF, no boot flag): some UEFI firmwares
# saw the stick but offered no boot option for it without CSM (user report 2026-10-31).

# UEFI system partition image (FAT16: some firmwares handle FAT12 ESPs badly) with every .efi file of $1/EFI/boot.
# make_efiboot <iso dir>   -> <iso dir>/boot/grub/efiboot.img
make_efiboot() {
  local img="$1/boot/grub/efiboot.img"
  rm -f "$img"
  dd if=/dev/zero of="$img" bs=1M count=16 status=none
  mkfs.vfat -F 16 -n RIGSIM-EFI "$img" >/dev/null
  mmd -i "$img" ::/EFI ::/EFI/boot
  mcopy -i "$img" "$1"/EFI/boot/*.efi ::/EFI/boot/
}

# The hybrid ISO. Needs <iso dir>/boot/grub/bios.img (El Torito BIOS image) and efiboot.img (make_efiboot).
# make_iso <iso dir> <boot_hybrid.img> <output> <volume id>
make_iso() {
  xorriso -as mkisofs -iso-level 3 -full-iso9660-filenames -volid "$4" \
    --grub2-mbr "$2" -partition_offset 16 --mbr-force-bootable \
    -append_partition 2 28732ac11ff8d211ba4b00a0c93ec93b "$1/boot/grub/efiboot.img" -appended_part_as_gpt \
    -iso_mbr_part_type a2a0d0ebe5b9334487c068b6b72699c7 \
    -eltorito-boot boot/grub/bios.img -no-emul-boot -boot-load-size 4 -boot-info-table --grub2-boot-info \
    --eltorito-catalog boot/grub/boot.cat \
    -eltorito-alt-boot -e boot/grub/efiboot.img -no-emul-boot \
    -output "$3" -graft-points "$1" /boot/grub/bios.img="$1/boot/grub/bios.img" \
    /boot/grub/efiboot.img="$1/boot/grub/efiboot.img"
}

# The data partition behind the image just written to <dev> (a disk or an image file). A GPT image ends with its
# backup header at the end of the ISO, so it is moved to the end of the stick first; then the partition is appended
# (type "L" = Linux filesystem, in a GPT and in an MBR). An MBR image (sticks written before 0.13.5) works the same.
# add_data_part <dev> <sfdisk start> [<sfdisk size>]
add_data_part() {
  if sfdisk -d "$1" 2>/dev/null | grep -q '^label: gpt'; then
    sfdisk --relocate gpt-bak-std "$1"
  fi
  echo "start=$2${3:+, size=$3}, type=L" | sfdisk --append --no-reread "$1"
}
