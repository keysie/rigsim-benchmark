# Sourced by build.sh. step "label" command...  runs the command with its output in $LOGS/NN-label.log and shows
# one live line: spinner, step, percentage (the last "NN%" the tool printed: make, mksquashfs, curl), elapsed time.
# On failure the log's last lines and its path are shown. Without a terminal: plain "== label" lines.
# VERBOSE=1 (build.sh -v): every step's output live on the terminal as well as in its log, no ticker.
LOGS="${LOGS:-$PWD/logs}"; mkdir -p "$LOGS"; _STEP=0
step() {
  local label="$1"; shift
  _STEP=$((_STEP + 1))
  local log; log="$LOGS/$(printf '%02d' "$_STEP")-$(printf '%s' "$label" | tr -cs 'A-Za-z0-9' '-' | sed 's/-$//').log"
  local t0=$SECONDS rc=0
  # always in the background: `set -e` then still holds inside shell functions run as a step
  if [[ "${VERBOSE:-0}" == 1 ]]; then
    echo "== $label"
    "$@" > >(tee "$log") 2>&1 & local pid=$!
  else
    "$@" >"$log" 2>&1 & local pid=$!
  fi
  local i=0 spin='|/-\' pct
  if [[ ! -t 1 || "${VERBOSE:-0}" == 1 ]]; then
    [[ "${VERBOSE:-0}" == 1 ]] || echo "== $label"
  else
    while kill -0 "$pid" 2>/dev/null; do
      pct="$(tail -c 4096 "$log" 2>/dev/null | tr '\r' '\n' | grep -aoE '[0-9]{1,3}(\.[0-9]+)?%' | tail -1 | cut -d. -f1 | tr -d '%' || true)"   # no "NN%" yet is normal
      [[ -n "$pct" && "$pct" -le 100 ]] || pct=""
      printf '\r %s  %s%s  %d:%02d\033[K' "${spin:i++%4:1}" "$label" "${pct:+  $pct%}" $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60))
      sleep 0.25
    done
  fi
  wait "$pid" || rc=$?
  local took; took="$(printf '%d:%02d' $(((SECONDS - t0) / 60)) $(((SECONDS - t0) % 60)))"
  if [[ $rc -eq 0 ]]; then
    if [[ -t 1 ]]; then printf '\r ✓  %s  %s\033[K\n' "$label" "$took"; fi
  else
    if [[ -t 1 ]]; then printf '\r ✗  %s  %s\033[K\n' "$label" "$took"; fi
    tail -n 40 "$log" | sed 's/^/    /'; echo "    full log: $log"
  fi
  return $rc
}
