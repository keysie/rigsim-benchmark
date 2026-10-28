#!/usr/bin/env bash
# Write or update a rigsim stick.
#   sudo stick/flash.sh /dev/sdX                       new stick, or asks what to do if it already is one
#   sudo ISO=stick/out/rigsim-stick-legacy.iso stick/flash.sh /dev/sdX
#   sudo stick/flash.sh --reseed /dev/sdX              only refresh the seed (no questions)
#
# On an existing stick (a partition labelled RIGSIM) it shows the stick's version and number of runs and offers:
#   [u]pdate  new system image + new seed, results kept: the data partition is left where it is, only the image in
#             front of it is rewritten and the partition entry restored at the same sectors (no mkfs)
#   [s]eed    only the seed (models, images, tools); system image unchanged
#   [e]rase   start from scratch
#   [a]bort
# Before update or erase the runs are always copied to $BACKUP (default ~/rigsim-stick-backup-<time>).
# New sticks keep room for larger images: the data partition starts at RESERVE_MIB (default 8192) or 2 GiB after
# the image, whichever is later.
# Needs: util-linux (sfdisk, partx, lsblk), e2fsprogs, rsync.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"; OUT="$S/out"; ISO="${ISO:-$OUT/rigsim-stick.iso}"
source "$S/tools/deps.sh"
need sfdisk:fdisk partx:util-linux blockdev:util-linux lsblk:util-linux e2fsck:e2fsprogs \
     mkfs.ext4:e2fsprogs rsync:rsync
RESERVE_MIB="${RESERVE_MIB:-8192}"
reseed=0; [[ "${1:-}" == --reseed ]] && { reseed=1; shift; }
DEV="${1:?device, e.g. /dev/sdX}"
[[ -b "$DEV" ]] || { echo "$DEV is not a block device" >&2; exit 2; }
[[ "$(lsblk -dno RM "$DEV")" == 1 || "${FORCE:-0}" == 1 ]] || { echo "$DEV is not removable (FORCE=1 to override)" >&2; exit 2; }
[[ $reseed == 1 || -s "$ISO" ]] || { echo "no image at $ISO (run stick/build.sh first)" >&2; exit 2; }
for p in $(lsblk -lnpo NAME "$DEV" | tail -n +2); do
  mountpoint -q "$(lsblk -no MOUNTPOINT "$p" | head -1)" 2>/dev/null && umount "$p" || true
done

MNT=""; cleanup() { [[ -n "$MNT" ]] && { umount "$MNT" 2>/dev/null || true; rmdir "$MNT" 2>/dev/null || true; }; }
trap cleanup EXIT
# The kernel's view of the new partition table. Not partprobe: libparted rejects a hybrid ISO's table, whose first
# partition starts at sector 0 and so contains the table itself ("recursive partition"); the kernel accepts it.
reread() {
  blockdev --rereadpt "$DEV" 2>/dev/null || { partx -u "$DEV" 2>/dev/null; partx -a "$DEV" 2>/dev/null; } || true
  if command -v udevadm >/dev/null; then udevadm settle; else sleep 2; fi
}
data_part() { lsblk -lnpo NAME,LABEL "$DEV" | awk '$2 == "RIGSIM" {print $1; exit}'; }
mount_data() { MNT="$(mktemp -d)"; mount "$1" "$MNT"; }

seed_stick() {                                       # $1 = data partition
  mount_data "$1"
  mkdir -p "$MNT/runs" "$MNT/seed"
  rsync -a --delete --info=progress2 "$OUT/seed/" "$MNT/seed/"
  [[ -e "$MNT/index.tsv" ]] || printf 'run\tstart_utc\tfingerprint\tcpu\tboard\tgpus\tram\tstatus\tstick\n' > "$MNT/index.tsv"
  sync; umount "$MNT"; rmdir "$MNT"; MNT=""
}

backup_runs() {                                      # $1 = data partition; copies runs + index, read-only mount
  local dst="${BACKUP:-$HOME/rigsim-stick-backup-$(date -u +%Y%m%dT%H%M%SZ)}"
  MNT="$(mktemp -d)"; mount -o ro "$1" "$MNT"
  if [[ -d "$MNT/runs" && -n "$(ls -A "$MNT/runs" 2>/dev/null)" ]]; then
    mkdir -p "$dst"; rsync -a "$MNT/runs" "$MNT/index.tsv" "$MNT/.salt" "$dst/" 2>/dev/null || rsync -a "$MNT/runs" "$dst/"
    echo "results backed up to $dst ($(ls "$dst/runs" | wc -l) runs)"
  fi
  umount "$MNT"; rmdir "$MNT"; MNT=""
}

fresh() {
  lsblk "$DEV"; read -rp "ERASE $DEV completely? type yes: " a; [[ "$a" == yes ]] || exit 1
  dd if="$ISO" of="$DEV" bs=4M conv=fsync status=progress
  local iso_mib=$(( $(stat -c %s "$ISO") / 1048576 + 1 ))
  local start=$(( iso_mib + 2048 > RESERVE_MIB ? iso_mib + 2048 : RESERVE_MIB ))
  echo "start=${start}MiB, type=83" | sfdisk --append --no-reread "$DEV"
  reread
  local part; part="$(lsblk -lnpo NAME,TYPE "$DEV" | awk '$2 == "part" {p = $1} END {print p}')"
  [[ "$(lsblk -lnpo TYPE "$DEV" | grep -c part)" -ge 3 && -b "$part" ]] || {
    echo "the kernel does not show the new data partition on $DEV (unplug and replug the stick, then run again)" >&2; exit 4; }
  mkfs.ext4 -F -L RIGSIM "$part"
  seed_stick "$part"
  echo "stick ready: $DEV (data partition $part from ${start} MiB)"
}

update() {                                           # $1 = data partition
  local part="$1" nr start sectors ss
  nr="$(cat "/sys/class/block/$(basename "$part")/partition")"
  read -r start sectors < <(partx -g -o START,SECTORS --nr "$nr" "$DEV")
  ss="$(blockdev --getss "$DEV")"
  local iso_bytes; iso_bytes="$(stat -c %s "$ISO")"
  if (( iso_bytes + 1048576 > start * ss )); then
    echo "the new image ($(( iso_bytes / 1048576 )) MiB) does not fit in front of the data partition" \
         "($(( start * ss / 1048576 )) MiB): choose [s]eed only, or [e]rase (results are backed up first)." >&2
    return 3
  fi
  backup_runs "$part"
  echo "writing the new image in front of the data partition (sectors from $start stay untouched)"
  dd if="$ISO" of="$DEV" bs=4M conv=fsync status=progress
  echo "start=${start}, size=${sectors}, type=83" | sfdisk --append --no-reread "$DEV"
  reread
  part="$(lsblk -lnpo NAME,LABEL "$DEV" | awk '$2 == "RIGSIM" {print $1; exit}')"
  if [[ -z "$part" ]] || ! e2fsck -n "$part" >/dev/null 2>&1; then
    echo "the data partition did not come back cleanly — the backup above has every run; run with [e]rase." >&2
    return 4
  fi
  seed_stick "$part"
  echo "stick updated: $DEV (results kept, $part)"
}

part="$(data_part || true)"
if [[ $reseed == 1 ]]; then
  [[ -n "$part" ]] || { echo "$DEV is not a rigsim stick (no RIGSIM partition)" >&2; exit 2; }
  seed_stick "$part"; echo "seed refreshed on $DEV"; exit 0
fi
if [[ -z "$part" ]]; then fresh; exit 0; fi

MNT="$(mktemp -d)"; mount -o ro "$part" "$MNT"
old="$(cat "$MNT/seed/VERSION" 2>/dev/null || echo unknown)"
runs="$(ls "$MNT/runs" 2>/dev/null | wc -l)"
machines="$(ls "$MNT/runs" 2>/dev/null | awk -F- '{print $2}' | sort -u | wc -l)"
umount "$MNT"; rmdir "$MNT"; MNT=""
new="$(cat "$OUT/seed/VERSION" 2>/dev/null || echo unknown)"
echo "$DEV is already a rigsim stick: version $old, $runs runs from $machines machines. New build: $new."
while true; do
  read -rp "[u]pdate (keep results), [s]eed only, [e]rase everything, [a]bort? " c
  case "$c" in
    u) update "$part" && exit 0 || true ;;
    s) seed_stick "$part"; echo "seed refreshed on $DEV"; exit 0 ;;
    e) backup_runs "$part"; fresh; exit 0 ;;
    a) exit 1 ;;
  esac
done
