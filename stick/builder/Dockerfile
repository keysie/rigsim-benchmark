# The stick's build environment: everything build.sh needs, so the host needs only Docker (any x86_64 Linux).
# build.sh builds this image itself (tag = hash of this file) and re-runs itself inside it.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      mmdebstrap ubuntu-keyring squashfs-tools xorriso mtools dosfstools zstd \
      curl ca-certificates gnupg git python3-minimal coreutils tar sudo docker.io \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'
