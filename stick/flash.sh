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
# After seeding, the container images are loaded into the stick's Docker store here (NO_PRELOAD=1: on first boot).
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

# Container images are loaded here, on this machine, into the stick's Docker store kept as ONE file: STORE
# (default stick/out/store.img, sparse, ext4 inside). Unpacking an image means tens of thousands of small files —
# hours on a USB stick, minutes on this machine's disk — and the finished file then goes to the stick as one
# sequential copy. The stick mounts it at boot (loop device) as its Docker / containerd store. It is the stick's
# own containerd and dockerd that load the images (from the ISO's system image, in a chroot, at the paths the stick
# uses), so the store is exactly what the stick would have written; the stick still checks every image and loads
# whatever is missing. The file is reused on later flashes (only new or changed images are loaded).
# NO_PRELOAD=1 skips all this (the stick loads the images on its first boot).
STORE="${STORE:-$OUT/store.img}"; STORE_GB="${STORE_GB:-96}"
PRELOAD_IN_CHROOT='
containerd >/tmp/containerd.log 2>&1 & cpid=$!
for i in $(seq 60); do [ -S /run/containerd/containerd.sock ] && break; sleep 1; done
dockerd --data-root /mnt/rigsim/docker --containerd=/run/containerd/containerd.sock --iptables=false --bridge=none \
  >/tmp/dockerd.log 2>&1 & dpid=$!
for i in $(seq 90); do docker info >/dev/null 2>&1 && break; sleep 1; done
if ! docker info >/dev/null 2>&1; then
  echo "dockerd from the stick did not start here:"; tail -15 /tmp/dockerd.log; kill $dpid $cpid 2>/dev/null; exit 3
fi
python3 /opt/rigsim/stick.py --preload /mnt/rigsim; rc=$?
kill $dpid; wait $dpid 2>/dev/null; kill $cpid; wait $cpid 2>/dev/null
exit $rc'
# Every mount for the chroot happens inside a private mount namespace (unshare): nothing can propagate to this
# machine's own mounts, and the kernel drops them all when the namespace ends. (Before, the recursive binds of /dev
# and /sys were made in the host's namespace; on hosts with shared mount propagation — Fedora / Nobara, systemd —
# unmounting them took the host's own /dev/pts and cgroup mounts with them: terminals and shutdown stopped working.)
STORE_IN_NAMESPACE='
set -e
mount --make-rprivate /
mkdir -p "$t/iso" "$t/sq" "$t/up" "$t/wk" "$t/store" "$r"
mount -o loop,ro "$ISO" "$t/iso"
mount -o loop,ro "$t/iso/casper/filesystem.squashfs" "$t/sq"
mount -t overlay overlay -o "lowerdir=$t/sq,upperdir=$t/up,workdir=$t/wk" "$r"
mount -o loop "$STORE" "$t/store"
mkdir -p "$t/store/docker" "$t/store/containerd"
mount --rbind /dev "$r/dev"; mount -t proc proc "$r/proc"; mount --rbind /sys "$r/sys"
mount -t tmpfs tmpfs "$r/run"; mount -t tmpfs tmpfs "$r/tmp"
mount -t tmpfs tmpfs "$r/mnt"; mkdir -p "$r/mnt/rigsim/seed" "$r/mnt/rigsim/docker" "$r/var/lib/containerd"
mount --bind -o ro "$OUT/seed" "$r/mnt/rigsim/seed"
mount --bind "$t/store/docker" "$r/mnt/rigsim/docker"
mount --bind "$t/store/containerd" "$r/var/lib/containerd"
set +e
chroot "$r" /bin/bash -c "$PRELOAD_IN_CHROOT"; rc=$?
sync
umount "$t/store" 2>/dev/null || umount -l "$t/store"
exit $rc'
build_store() {                                      # loads the seed's images into $STORE (on this machine)
  local t rc=0
  modprobe overlay 2>/dev/null || true; modprobe squashfs 2>/dev/null || true; modprobe loop 2>/dev/null || true
  if [[ ! -f "$STORE" ]]; then
    truncate -s "${STORE_GB}G" "$STORE" && mkfs.ext4 -q -F -L RIGSIM-STORE -m 0 "$STORE" || return 1
  fi
  e2fsck -fy "$STORE" >/dev/null 2>&1 || { echo "images: $STORE is damaged; delete it to start a fresh one" >&2; return 1; }
  t="$(mktemp -d /var/tmp/rigsim-preload.XXXXXX)"
  t="$t" r="$t/root" ISO="$ISO" STORE="$STORE" OUT="$OUT" PRELOAD_IN_CHROOT="$PRELOAD_IN_CHROOT" \
    unshare --mount --propagation private bash -c "$STORE_IN_NAMESPACE" || rc=$?
  sync
  # the namespace is gone, so are its mounts; the work folder is only removed if nothing is mounted below it
  if findmnt -rn -o TARGET | grep -q "^$t"; then echo "WARN: $t still has mounts; left in place" >&2; else rm -rf "$t"; fi
  return $rc
}
preload_images() {                                   # $1 = data partition (not mounted)
  if [[ "${NO_PRELOAD:-0}" == 1 ]]; then echo "images: loaded on the stick's first boot (NO_PRELOAD=1)"; return 0; fi
  if [[ ! -s "$ISO" ]]; then echo "images: no ISO at $ISO here — they load on the stick's first boot"; return 0; fi
  echo "images: loading them into $STORE on this machine (progress every 15 s)"
  if ! build_store; then
    echo "images: preloading failed — the stick loads what is missing on its first boot"; return 0
  fi
  local used; used="$(du -B1 "$STORE" | cut -f1)"
  mount_data "$1"
  local old free; old="$(du -B1 "$MNT/store.img" 2>/dev/null | cut -f1)"
  free="$(( $(df -B1 --output=avail "$MNT" | tail -1) + ${old:-0} ))"
  if (( used + 1073741824 > free )); then
    echo "images: the store needs $((used >> 30)) GiB, the stick has $((free >> 30)) GiB free — not copied" >&2
    umount "$MNT"; rmdir "$MNT"; MNT=""; return 0
  fi
  echo "images: copying the store to the stick ($((used >> 30)) GiB, one sequential write)"
  rm -f "$MNT/store.img"
  # images an older stick version unpacked onto the partition itself are replaced by the store: freed
  rm -rf "$MNT/docker" "$MNT/containerd"
  dd if="$STORE" of="$MNT/store.img" bs=16M conv=sparse status=progress && sync
  umount "$MNT"; rmdir "$MNT"; MNT=""
  return 0
}

seed_stick() {                                       # $1 = data partition
  mount_data "$1"
  mkdir -p "$MNT/runs" "$MNT/seed"
  rsync -a --delete --info=progress2 "$OUT/seed/" "$MNT/seed/"
  [[ -e "$MNT/index.tsv" ]] || printf 'run\tstart_utc\tfingerprint\tcpu\tboard\tgpus\tram\tstatus\tstick\n' > "$MNT/index.tsv"
  sync; umount "$MNT"; rmdir "$MNT"; MNT=""
  preload_images "$1"
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
