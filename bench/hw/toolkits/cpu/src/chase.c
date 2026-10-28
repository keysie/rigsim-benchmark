/* Pointer chase over a random cyclic permutation of 64-byte lines: ns per dependent load (one thread).
   chase MIB PASSES -> one JSON line. Transparent huge pages requested to keep TLB misses out of the number. */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ULL;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmp(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

int main(int argc, char** argv) {
    size_t mib = argc > 1 ? strtoull(argv[1], 0, 10) : 1024;
    int passes = argc > 2 ? atoi(argv[2]) : 5;
    size_t bytes = mib << 20, lines = bytes / 64;
    char* buf = aligned_alloc(1 << 21, bytes);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    madvise(buf, bytes, MADV_HUGEPAGE);
    uint64_t* perm = malloc(lines * sizeof(uint64_t));
    for (size_t i = 0; i < lines; ++i) perm[i] = i;
    for (size_t i = lines - 1; i > 0; --i) { size_t j = xr() % i; uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; } /* Sattolo: one cycle */
    for (size_t i = 0; i < lines; ++i) *(uint64_t*)(buf + perm[i] * 64) = perm[(i + 1) % lines] * 64;
    free(perm);
    size_t steps = lines < 20000000 ? 20000000 : lines;
    double* ns = malloc(passes * sizeof(double));
    volatile uint64_t sink = 0;
    for (int p = 0; p < passes; ++p) {
        uint64_t off = 0;
        double t0 = now();
        for (size_t s = 0; s < steps; ++s) off = *(uint64_t*)(buf + off);
        double t1 = now();
        sink += off;
        ns[p] = (t1 - t0) / steps * 1e9;
    }
    qsort(ns, passes, sizeof(double), cmp);
    printf("{\"mode\":\"latency\",\"mib\":%zu,\"passes\":%d,\"steps\":%zu,\"ns_median\":%.3f,\"ns_min\":%.3f,\"ns_max\":%.3f}\n",
           mib, passes, steps, ns[passes / 2], ns[0], ns[passes - 1]);
    return (int)(sink & 0);
}
