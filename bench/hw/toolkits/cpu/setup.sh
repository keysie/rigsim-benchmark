#!/bin/sh
# Runs inside the session container (common image debian:trixie): likwid, fio and a compiler from the Debian
# archive (network or the configured proxy needed during setup only), then the pointer chase compiled.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends likwid fio python3-minimal gcc libc6-dev >/dev/null
cd /opt/rigsim-hwbench
gcc -O2 -o /usr/local/bin/chase src/chase.c
install -m 0755 src/cpuprobe /usr/local/bin/cpuprobe
likwid-bench -a | head -3 || true
echo "cpu toolkit ready"
