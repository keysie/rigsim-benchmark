#!/bin/sh
# Builds vkprobe: shaders to SPIR-V (embedded as C arrays), then the binary. Needs glslc, libvulkan-dev, gcc.
# A cooperative-matrix shader the compiler cannot build is left out (the probe then reports it as unavailable).
set -eu
cd "$(dirname "$0")"; OUT="${1:-./vkprobe}"; G="$(mktemp -d)"
glslc --target-env=vulkan1.2 -O -mfmt=num -o "$G/stream.inc" shaders/stream.comp
glslc --target-env=vulkan1.2 -O -mfmt=num -o "$G/empty.inc" shaders/empty.comp
glslc --target-env=vulkan1.2 -O -mfmt=num -o "$G/fill.inc" shaders/fill.comp
glslc --target-env=vulkan1.2 -O -mfmt=num -o "$G/fma32.inc" shaders/fma.comp
glslc --target-env=vulkan1.2 -O -mfmt=num -DF16 -o "$G/fma16.inc" shaders/fma.comp
DEFS=""
cm() { if glslc --target-env=vulkan1.3 -O -mfmt=num -DAT="$2" -DCT="$3" -o "$G/cm_$1.inc" shaders/coopmat.comp 2>"$G/cm_$1.err"; then
         DEFS="$DEFS -DHAVE_CM_$1"; else echo "coopmat $1 not built: $(head -1 "$G/cm_$1.err")"; : > "$G/cm_$1.inc"; fi; }
cm F16 float16_t float16_t; cm F16F32 float16_t float; cm I8 int8_t int32_t
gcc -O3 -std=c11 -Wall -I"$G" $DEFS -o "$OUT" vkprobe.c -lvulkan -lpthread -lm
rm -rf "$G"; echo "vkprobe built: $OUT"
