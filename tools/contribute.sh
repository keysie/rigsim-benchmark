#!/usr/bin/env bash
# Offers the runs on a stick to this repository: copies runs that are not here yet, keeps those that pass the
# checks (tools/validate.py), commits them on a new branch and opens a pull request (with the GitHub CLI `gh`;
# without it, prints the two commands to do it yourself). Nothing on the stick is changed.
#   tools/contribute.sh /run/media/$USER/RIGSIM
set -euo pipefail
SRC="${1:?mounted RIGSIM data partition of the stick}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[[ -d "$SRC/runs" ]] || { echo "$SRC has no runs/ folder: is it the stick's RIGSIM partition?" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "this checkout has uncommitted changes: commit or stash them first" >&2; exit 2; }
git fetch -q origin main 2>/dev/null || true
BASE="$(git rev-parse -q --verify origin/main || git rev-parse HEAD)"
BRANCH="runs-$(date -u +%Y%m%d-%H%M%S)"
git switch -q -c "$BRANCH" "$BASE"

added=(); rejected=()
for d in "$SRC"/runs/*/; do
  id="$(basename "$d")"
  [[ -e "results/runs/$id" ]] && continue
  cp -r "$d" "results/runs/$id"
  if out="$(python3 tools/validate.py "results/runs/$id" 2>&1)"; then
    added+=("$id")
  else
    rejected+=("$id"); rm -rf "results/runs/$id"
    echo "not included: $id"; { echo "$out" | grep '^ERROR' || true; } | sed 's/^ERROR /    /' | head -8
  fi
done

if [[ ${#added[@]} -eq 0 ]]; then
  git switch -q - && git branch -q -D "$BRANCH"
  echo "no new complete runs to contribute (${#rejected[@]} not included)"; exit 0
fi
git add results/runs
git commit -q -m "runs: ${#added[@]} new run(s)" -m "$(printf '%s\n' "${added[@]}")"
echo "${#added[@]} run(s) committed on branch $BRANCH (${#rejected[@]} not included)"

BODY="$(cat .github/pull_request_template.md 2>/dev/null)"
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  gh pr create --base main --title "runs: ${#added[@]} new run(s)" --body "$BODY"   # offers to fork if needed
else
  cat <<EOT
To open the pull request: fork https://github.com/keysie/rigsim-benchmark on GitHub, then
  git remote add fork https://github.com/<your-account>/rigsim-benchmark.git
  git push fork $BRANCH
and open the pull request GitHub shows you. (With the GitHub CLI logged in, this script does it for you.)
EOT
fi
