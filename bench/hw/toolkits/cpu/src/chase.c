/* Pointer chase over a random cyclic permutation of 64-byte lines: ns per dependent load (one thread).
   chase MIB PASSES [CPU] -> one JSON line per core it ran on. Transparent huge pages requested to keep TLB misses out
   of the number. The thread is pinned (DDR item 3, 2026-10-31: unpinned, a hybrid CPU's scheduler picked P- or E-cores
   at random): to CPU if given, else to the core with the highest maximum clock; on hybrid Intel parts once more on an
   E-core. Each line says which core, its type (P/E/-) and its mean clock during the passes (scaling_cur_freq). */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ULL;
static uint64_t xr(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmp(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return (x > y) - (x < y); }

static long read_long(int cpu, const char* leaf) {
    char p[160]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/%s", cpu, leaf);
    FILE* f = fopen(p, "r"); long v = -1;
    if (f) { if (fscanf(f, "%ld", &v) != 1) v = -1; fclose(f); }
    return v;
}
static int in_list(const char* path, int cpu) {
    FILE* f = fopen(path, "r"); if (!f) return 0;
    char b[4096]; int hit = 0;
    if (fgets(b, sizeof b, f))
        for (char* t = strtok(b, ",\n"); t; t = strtok(NULL, ",\n")) {
            int a, z; if (sscanf(t, "%d-%d", &a, &z) == 2) { if (cpu >= a && cpu <= z) hit = 1; }
            else if (sscanf(t, "%d", &a) == 1 && a == cpu) hit = 1;
        }
    fclose(f); return hit;
}
static char core_type(int cpu) {
    if (in_list("/sys/devices/cpu_core/cpus", cpu)) return 'P';
    if (in_list("/sys/devices/cpu_atom/cpus", cpu)) return 'E';
    return '-';
}
static void run(char* buf, size_t lines, size_t mib, int passes, int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); sched_setaffinity(0, sizeof s, &s);
    size_t steps = lines < 20000000 ? 20000000 : lines;
    double* ns = malloc(passes * sizeof(double)); double khz = 0; int nk = 0;
    volatile uint64_t sink = 0;
    for (int p = 0; p < passes; ++p) {
        uint64_t off = 0;
        long f0 = read_long(cpu, "cpufreq/scaling_cur_freq");
        double t0 = now();
        for (size_t st = 0; st < steps; ++st) off = *(uint64_t*)(buf + off);
        double t1 = now();
        long f1 = read_long(cpu, "cpufreq/scaling_cur_freq");
        if (f0 > 0 && f1 > 0) { khz += (f0 + f1) / 2.0; ++nk; }
        sink += off;
        ns[p] = (t1 - t0) / steps * 1e9;
    }
    qsort(ns, passes, sizeof(double), cmp);
    long mx = read_long(cpu, "cpufreq/cpuinfo_max_freq");
    printf("{\"mode\":\"latency\",\"mib\":%zu,\"passes\":%d,\"steps\":%zu,\"ns_median\":%.3f,\"ns_min\":%.3f,\"ns_max\":%.3f,"
           "\"cpu\":%d,\"core_type\":\"%c\",\"mhz_mean\":%.0f,\"mhz_max\":%ld}\n",
           mib, passes, steps, ns[passes / 2], ns[0], ns[passes - 1], cpu, core_type(cpu), nk ? khz / nk / 1e3 : -1.0,
           mx > 0 ? mx / 1000 : -1L);
    free(ns); (void)sink;
}

int main(int argc, char** argv) {
    size_t mib = argc > 1 ? strtoull(argv[1], 0, 10) : 1024;
    int passes = argc > 2 ? atoi(argv[2]) : 5;
    int want = argc > 3 ? atoi(argv[3]) : -1;
    size_t bytes = mib << 20, lines = bytes / 64;
    char* buf = aligned_alloc(1 << 21, bytes);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 2; }
    madvise(buf, bytes, MADV_HUGEPAGE);
    uint64_t* perm = malloc(lines * sizeof(uint64_t));
    for (size_t i = 0; i < lines; ++i) perm[i] = i;
    for (size_t i = lines - 1; i > 0; --i) { size_t j = xr() % i; uint64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; } /* Sattolo: one cycle */
    for (size_t i = 0; i < lines; ++i) *(uint64_t*)(buf + perm[i] * 64) = perm[(i + 1) % lines] * 64;
    free(perm);
    cpu_set_t all; CPU_ZERO(&all); sched_getaffinity(0, sizeof all, &all);
    int fast = -1, ecore = -1; long best = -2;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &all)) continue;
        long mx = read_long(c, "cpufreq/cpuinfo_max_freq");
        if (mx > best) { best = mx; fast = c; }
        if (ecore < 0 && core_type(c) == 'E') ecore = c;
    }
    if (want >= 0) { run(buf, lines, mib, passes, want); return 0; }
    run(buf, lines, mib, passes, fast < 0 ? 0 : fast);
    if (ecore >= 0 && ecore != fast) run(buf, lines, mib, passes, ecore);
    return 0;
}
