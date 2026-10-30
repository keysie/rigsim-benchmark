#!/usr/bin/env bash
# Push the workflow's own commit (HEAD) to a protected branch: the ruleset requires the check "Check Conventional
# Commits" on every commit that lands there, and a commit pushed straight from a workflow has none. So the commit is
#   1. checked with the same checker the CI runs (.github/scripts/check-commits.py) — a failure stops here,
#   2. pushed to a short-lived branch, so GitHub knows it,
#   3. given the check's result through the Checks API (checks: write),
#   4. pushed to the target branch, then the short-lived branch is deleted.
# Needs GH_TOKEN (the workflow token) and GITHUB_REPOSITORY. Usage: push-checked.sh [branch, default main]
set -euo pipefail
target="${1:-main}"
sha="$(git rev-parse HEAD)"
python3 .github/scripts/check-commits.py "$sha^!"
tmp="bot/checked-${sha:0:12}"
git push -q origin "HEAD:refs/heads/$tmp"
trap 'git push -q origin --delete "$tmp" 2>/dev/null || true' EXIT
gh api -X POST "repos/$GITHUB_REPOSITORY/check-runs" \
  -f name="Check Conventional Commits" -f head_sha="$sha" -f status=completed -f conclusion=success \
  -f "output[title]=Conventional commit" \
  -f "output[summary]=Checked by .github/scripts/check-commits.py in the workflow that made this commit." >/dev/null
git push -q origin "HEAD:$target"
echo "pushed $sha to $target with its commit-message check"
