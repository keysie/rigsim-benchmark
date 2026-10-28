#!/usr/bin/env bash
# Copy runs off a stick to a results folder. Never deletes anything on the stick; existing runs are not re-copied.
#   stick/tools/collect.sh /media/RIGSIM ~/rigsim-stick-results
set -euo pipefail
SRC="${1:?mounted data partition}"; DST="${2:?results folder}"
mkdir -p "$DST/runs"
rsync -a --ignore-existing "$SRC/runs/" "$DST/runs/"
cp "$SRC/index.tsv" "$DST/index-$(date -u +%Y%m%dT%H%M%SZ).tsv"
echo "runs on the stick: $(ls "$SRC/runs" | wc -l), in $DST: $(ls "$DST/runs" | wc -l)"
