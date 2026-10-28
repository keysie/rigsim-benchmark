# Sourced by build.sh and flash.sh: install the host packages a script needs, asking first.
# need <command>:<package> ...   — checks each command; missing ones are installed with apt-get (no -y: apt shows
# what it will install and asks). Without apt-get the missing packages are listed and the script stops.
need() {
  local missing=() pair cmd pkg
  for pair in "$@"; do
    cmd="${pair%%:*}"; pkg="${pair#*:}"
    command -v "$cmd" >/dev/null 2>&1 || [[ " ${missing[*]} " == *" $pkg "* ]] || missing+=("$pkg")
  done
  [[ ${#missing[@]} -eq 0 ]] && return 0
  echo "missing packages: ${missing[*]}"
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "no apt-get here: install them with your package manager, then run again" >&2; exit 2
  fi
  local sudo=""; [[ $EUID -ne 0 ]] && sudo=sudo
  $sudo apt-get update -qq
  $sudo apt-get install --no-install-recommends "${missing[@]}"
  for pair in "$@"; do
    command -v "${pair%%:*}" >/dev/null 2>&1 || { echo "still missing: ${pair%%:*} (${pair#*:})" >&2; exit 2; }
  done
}
