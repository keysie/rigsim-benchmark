// rigsim CPU GEMM probe (stick), one JSON line per result.
//   cgemm --prec fp32|fp64|int8 --n 4096 --threads 0 (0 = all logical CPUs) [--reps 5]
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
static int cmp(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
int main(int c, char** v) {
    const char* prec = arg(c, v, "--prec", "fp32");
    long n = atol(arg(c, v, "--n", "4096"));
    int threads = atoi(arg(c, v, "--threads", "0")), reps = atoi(arg(c, v, "--reps", "5"));
    if (threads <= 0) threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    openblas_set_num_threads(threads);
    omp_set_num_threads(threads);
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
