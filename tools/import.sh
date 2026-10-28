#!/usr/bin/env bash
# Takes runs from a stick into results/runs/ and rebuilds the table. Never deletes anything on the stick; runs
# already here are not copied again. Review the new runs before committing (see results/README.md).
#   tools/import.sh /run/media/$USER/RIGSIM
set -euo pipefail
SRC="${1:?mounted RIGSIM data partition}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
before="$(ls "$ROOT/results/runs" | grep -v '^\.' | sort)"
"$ROOT/stick/tools/collect.sh" "$SRC" "$ROOT/results"
rm -f "$ROOT"/results/index-*.tsv                       # the stick's own index; data.json replaces it here
after="$(ls "$ROOT/results/runs" | grep -v '^\.' | sort)"
new="$(comm -13 <(echo "$before") <(echo "$after") | sed '/^$/d')"
python3 "$ROOT/tools/index.py"
if [[ -n "$new" ]]; then
  echo "new runs — review them before committing (owner's consent, nothing identifying):"
  echo "$new" | sed 's/^/  results\/runs\//'
else
  echo "no new runs"
fi
