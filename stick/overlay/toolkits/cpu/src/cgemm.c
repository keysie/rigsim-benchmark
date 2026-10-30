// rigsim CPU GEMM probe (stick), one JSON line per result.
//   cgemm --prec fp32|fp64|int8 --n 4096 --threads 0 (0 = all logical CPUs) [--reps 5]
//   cgemm --sustain 120 --idle 30 --n 2048 --threads 0 [--interval 0.5]
//         after `idle` seconds without load, sgemm back to back for `sustain` seconds; one line per interval with
//         its throughput, the mean core clock (cpufreq) and, where Linux exposes it, the package energy counter
//         (powercap / RAPL energy_uj — the processor's own estimate, context only)
// fp32 / fp64: OpenBLAS sgemm / dgemm. int8: oneDNN u8 x s8 -> s32 (the VNNI / AMX path; llama.cpp's quantised dot
// products on the CPU are u8 x s8 as well). Which instruction set oneDNN picked is in "isa".
#include <cblas.h>
#include <oneapi/dnnl/dnnl.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
extern void openblas_set_num_threads(int);
static const char* arg(int c, char** v, const char* k, const char* d) {
    for (int i = 1; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1];
    return d;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static double mean_clock_mhz(int ncpu) {           // mean of the cores' current clocks (cpufreq, kHz)
    double s = 0; int k = 0;
    for (int i = 0; i < ncpu; ++i) {
        char p[96]; snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", i);
        FILE* f = fopen(p, "r"); long khz;
        if (f) { if (fscanf(f, "%ld", &khz) == 1) { s += khz / 1000.0; ++k; } fclose(f); }
    }
    return k ? s / k : -1;
}
static long long energy_uj(void) {                   // package 0's energy counter, or -1
    FILE* f = fopen("/sys/class/powercap/intel-rapl:0/energy_uj", "r"); long long e = -1;
    if (f) { if (fscanf(f, "%lld", &e) != 1) e = -1; fclose(f); }
    return e;
}
static int sustain(long n, int threads, double secs, double idle, double interval) {
    size_t sz = (size_t)n * n * 4;
    float *a = aligned_alloc(64, sz), *b = aligned_alloc(64, sz), *cc = aligned_alloc(64, sz);
    if (!a || !b || !cc) { printf("{\"mode\":\"sustain\",\"error\":\"alloc\"}\n"); return 1; }
    for (size_t i = 0; i < (size_t)n * n; ++i) { a[i] = 1e-3f * (i % 7); b[i] = 1e-3f * (i % 5); cc[i] = 0; }
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    struct timespec ts = { (time_t)idle, (long)((idle - (time_t)idle) * 1e9) };
    nanosleep(&ts, NULL);                              // a period of low load first: the power budget recovers
    double ops = 2.0 * n * n * n, t0 = now(), last = t0, done = 0;
    long long e_last = energy_uj();
    while (now() - t0 < secs) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n, b, n, 0.0f, cc, n);
        done += ops;
        double t = now();
        if (t - last >= interval) {
            long long e = energy_uj();
            double w = (e >= 0 && e_last >= 0 && e >= e_last) ? (e - e_last) * 1e-6 / (t - last) : -1;
            printf("{\"mode\":\"sustain\",\"t\":%.3f,\"ops_per_s\":%.6g,\"clock_mhz\":%.0f,\"package_w\":%.2f,"
                   "\"n\":%ld,\"threads\":%d,\"idle_s\":%.1f}\n", t - t0, done / (t - last), mean_clock_mhz(ncpu), w,
                   n, threads, idle);
            fflush(stdout);
            done = 0; last = t; e_last = e;
        }
    }
    return 0;
}
static int cmp(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
int main(int c, char** v) {
    const char* prec = arg(c, v, "--prec", "fp32");
    long n = atol(arg(c, v, "--n", "4096"));
    int threads = atoi(arg(c, v, "--threads", "0")), reps = atoi(arg(c, v, "--reps", "5"));
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (threads > ncpu) {                             // a ladder step beyond this machine: nothing to measure
        printf("{\"mode\":\"cgemm\",\"threads\":%d,\"error\":\"more threads than logical CPUs (%d)\"}\n", threads, ncpu);
        return 1;
    }
    if (threads <= 0) threads = ncpu;
    openblas_set_num_threads(threads);
    omp_set_num_threads(threads);
    double sus = atof(arg(c, v, "--sustain", "0"));
    if (sus > 0)
        return sustain(n, threads, sus, atof(arg(c, v, "--idle", "30")), atof(arg(c, v, "--interval", "0.5")));
    int dbl = !strcmp(prec, "fp64"), i8 = !strcmp(prec, "int8");
    size_t es = dbl ? 8 : i8 ? 1 : 4, sz = (size_t)n * n * (i8 ? 4 : es);
    void *a = aligned_alloc(64, sz), *b = aligned_alloc(64, sz), *cc = aligned_alloc(64, sz);
    if (!a || !b || !cc) { printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"n\":%ld,\"error\":\"alloc\"}\n", prec, n); return 1; }
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        if (i8) { ((uint8_t*)a)[i] = (uint8_t)(i * 2654435761u >> 24); ((int8_t*)b)[i] = (int8_t)(i * 40503u >> 8); }
        else if (dbl) { ((double*)a)[i] = 1e-3 * (i % 7); ((double*)b)[i] = 1e-3 * (i % 5); ((double*)cc)[i] = 0; }
        else { ((float*)a)[i] = 1e-3f * (i % 7); ((float*)b)[i] = 1e-3f * (i % 5); ((float*)cc)[i] = 0; }
    }
    double* t = malloc(sizeof(double) * (reps + 1));
    for (int r = -1; r < reps; ++r) {                 // r = -1: warm-up
        double t0 = now();
        if (i8) { int32_t co = 0;
            if (dnnl_gemm_u8s8s32('N', 'N', 'F', n, n, n, 1.f, a, n, 0, b, n, 0, 0.f, cc, n, &co) != dnnl_success) {
                printf("{\"mode\":\"cgemm\",\"prec\":\"int8\",\"n\":%ld,\"error\":\"dnnl\"}\n", n); return 1; } }
        else if (dbl) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, a, n, b, n, 0.0, cc, n);
        else cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, a, n, b, n, 0.0f, cc, n);
        if (r >= 0) t[r] = now() - t0;
    }
    qsort(t, reps, sizeof(double), cmp);
    double ops = 2.0 * n * n * n;
    printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"n\":%ld,\"threads\":%d,\"reps\":%d,\"ops\":%.0f,"
           "\"t_median_s\":%.9g,\"t_min_s\":%.9g,\"ops_per_s\":%.6g,\"isa\":%d,\"lib\":\"%s\"}\n",
           prec, n, threads, reps, ops, t[reps / 2], t[0], ops / t[reps / 2], (int)dnnl_get_effective_cpu_isa(),
           i8 ? "onednn" : "openblas");
    return 0;
}
