// rigsim CPU GEMM probe (stick), one JSON line per result.
//   cgemm --prec fp32|fp64|int8|bf16|fp16 --n 4096 --threads 0 (0 = all logical CPUs) [--reps 5]
//   cgemm --prec ... --shape MxNxK ...    an M x K by K x N product (M = tokens in flight, N x K = a layer's weights)
//   cgemm --sustain 120 --idle 30 --n 2048 --threads 0 [--interval 0.5]
//         after `idle` seconds without load, sgemm back to back for `sustain` seconds; one line per interval with
//         its throughput, the mean core clock (cpufreq) and, where Linux exposes it, the package energy counter
//         (powercap / RAPL energy_uj — the processor's own estimate, context only) and the package temperature
//         (hwmon: Intel coretemp "Package id 0", AMD k10temp "Tctl")
// fp32 / fp64: OpenBLAS sgemm / dgemm. int8: oneDNN u8 x s8 -> s32 (the VNNI / AMX path; llama.cpp's quantised dot
// products on the CPU are u8 x s8 as well). bf16 / fp16: oneDNN matmul, f32 out, the weights reordered once into the
// layout oneDNN prefers (as engines prepack them) — AVX512_BF16 / AMX-BF16 / AVX512-FP16 / AMX-FP16 where the CPU has
// them; where oneDNN would fall back to its reference code ("ref:"), the result is "unsupported" (it would take
// minutes and say nothing about the hardware). Which instruction set oneDNN picked is in "isa", the kernel in "impl".
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
// The package temperature sensor: hwmon coretemp's "Package id 0" (Intel) or k10temp's "Tctl" (AMD), else the first
// temperature of either driver. Found once; "" if there is none.
static char temp_path[256];
static void find_temp(void) {
    temp_path[0] = 0;
    for (int h = 0; h < 32 && !temp_path[0]; ++h) {
        char p[128], name[64] = "";
        snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/name", h);
        FILE* f = fopen(p, "r");
        if (!f) continue;
        if (fscanf(f, "%63s", name) != 1) name[0] = 0;
        fclose(f);
        if (strcmp(name, "coretemp") && strcmp(name, "k10temp")) continue;
        for (int i = 1; i < 64; ++i) {
            char lab[64] = "";
            snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/temp%d_label", h, i);
            f = fopen(p, "r");
            if (f) { if (!fgets(lab, sizeof lab, f)) lab[0] = 0; fclose(f); }
            snprintf(p, sizeof p, "/sys/class/hwmon/hwmon%d/temp%d_input", h, i);
            if (access(p, R_OK) != 0) continue;
            if (!strncmp(lab, "Package id 0", 12) || !strncmp(lab, "Tctl", 4) || i == 1) {
                snprintf(temp_path, sizeof temp_path, "%s", p);
                if (lab[0]) break;
            }
        }
    }
}
static double temp_c(void) {
    if (!temp_path[0]) return -1000;
    FILE* f = fopen(temp_path, "r"); long mc = 0;
    if (!f) return -1000;
    if (fscanf(f, "%ld", &mc) != 1) mc = -1000000;
    fclose(f);
    return mc / 1000.0;
}
static const char* temp_json(char* buf, size_t n) {    // the temperature, or null where there is no sensor
    double t = temp_c();
    if (t < -500) snprintf(buf, n, "null"); else snprintf(buf, n, "%.1f", t);
    return buf;
}
static int sustain(long n, int threads, double secs, double idle, double interval) {
    char tb[32];
    size_t sz = (size_t)n * n * 4;
    float *a = aligned_alloc(64, sz), *b = aligned_alloc(64, sz), *cc = aligned_alloc(64, sz);
    if (!a || !b || !cc) { printf("{\"mode\":\"sustain\",\"error\":\"alloc\"}\n"); return 1; }
    for (size_t i = 0; i < (size_t)n * n; ++i) { a[i] = 1e-3f * (i % 7); b[i] = 1e-3f * (i % 5); cc[i] = 0; }
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    struct timespec ts = { (time_t)idle, (long)((idle - (time_t)idle) * 1e9) };
    find_temp();
    printf("{\"mode\":\"sustain\",\"t\":%.3f,\"phase\":\"idle start\",\"temp_c\":%s,\"clock_mhz\":%.0f}\n",
           -idle, temp_json(tb, sizeof tb), mean_clock_mhz(ncpu));
    fflush(stdout);
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
                   "\"temp_c\":%s,\"n\":%ld,\"threads\":%d,\"idle_s\":%.1f}\n", t - t0, done / (t - last),
                   mean_clock_mhz(ncpu), w, temp_json(tb, sizeof tb), n, threads, idle);
            fflush(stdout);
            done = 0; last = t; e_last = e;
        }
    }
    return 0;
}
// bf16 / fp16 through oneDNN's matmul primitive. Returns 0 and the times, 2 = unsupported (reference code only).
static int matmul(int f16, long M, long N, long K, int reps, double* t, char* impl, size_t implsz) {
    dnnl_engine_t eng; dnnl_stream_t st;
    if (dnnl_engine_create(&eng, dnnl_cpu, 0) != dnnl_success) return 1;
    if (dnnl_stream_create(&st, eng, dnnl_stream_default_flags) != dnnl_success) return 1;
    dnnl_data_type_t dt = f16 ? dnnl_f16 : dnnl_bf16;
    dnnl_dims_t ad = {M, K}, bd = {K, N}, cd = {M, N};
    dnnl_memory_desc_t a_md, b_plain, b_any, c_md;
    dnnl_memory_desc_create_with_tag(&a_md, 2, ad, dt, dnnl_ab);
    dnnl_memory_desc_create_with_tag(&b_plain, 2, bd, dt, dnnl_ab);
    dnnl_memory_desc_create_with_tag(&b_any, 2, bd, dt, dnnl_format_tag_any);
    dnnl_memory_desc_create_with_tag(&c_md, 2, cd, dnnl_f32, dnnl_ab);
    dnnl_primitive_desc_t pd;
    if (dnnl_matmul_primitive_desc_create(&pd, eng, a_md, b_any, NULL, c_md, NULL) != dnnl_success) return 2;
    const char* info = "";
    dnnl_primitive_desc_query(pd, dnnl_query_impl_info_str, 0, &info);
    snprintf(impl, implsz, "%s", info);
    if (!strncmp(info, "ref", 3)) return 2;
    const_dnnl_memory_desc_t b_md = dnnl_primitive_desc_query_md(pd, dnnl_query_weights_md, 0);
    dnnl_memory_t am, bp, bm, cm;
    dnnl_memory_create(&am, a_md, eng, DNNL_MEMORY_ALLOCATE);
    dnnl_memory_create(&bp, b_plain, eng, DNNL_MEMORY_ALLOCATE);
    dnnl_memory_create(&bm, b_md, eng, DNNL_MEMORY_ALLOCATE);
    dnnl_memory_create(&cm, c_md, eng, DNNL_MEMORY_ALLOCATE);
    uint16_t one = f16 ? 0x3c00 : 0x3f80;                // 1.0 in fp16 / bf16; sign alternates, no denormals
    void* h;
    dnnl_memory_get_data_handle(am, &h);
    for (long i = 0; i < M * K; ++i) ((uint16_t*)h)[i] = one | (uint16_t)((i & 1) << 15);
    dnnl_memory_get_data_handle(bp, &h);
    for (long i = 0; i < K * N; ++i) ((uint16_t*)h)[i] = one | (uint16_t)(((i >> 1) & 1) << 15);
    dnnl_primitive_desc_t rpd; dnnl_primitive_t rp;             // the weights into oneDNN's layout, once
    if (dnnl_reorder_primitive_desc_create(&rpd, b_plain, eng, b_md, eng, NULL) != dnnl_success) return 1;
    dnnl_primitive_create(&rp, rpd);
    dnnl_exec_arg_t ra[2] = {{DNNL_ARG_FROM, bp}, {DNNL_ARG_TO, bm}};
    dnnl_primitive_execute(rp, st, 2, ra); dnnl_stream_wait(st);
    dnnl_primitive_t p;
    if (dnnl_primitive_create(&p, pd) != dnnl_success) return 1;
    dnnl_exec_arg_t args[3] = {{DNNL_ARG_SRC, am}, {DNNL_ARG_WEIGHTS, bm}, {DNNL_ARG_DST, cm}};
    for (int r = -1; r < reps; ++r) {
        double t0 = now();
        dnnl_primitive_execute(p, st, 3, args); dnnl_stream_wait(st);
        if (r >= 0) t[r] = now() - t0;
    }
    return 0;
}
static int cmp(const void* a, const void* b) { double x = *(double*)a, y = *(double*)b; return (x > y) - (x < y); }
int main(int c, char** v) {
    const char* prec = arg(c, v, "--prec", "fp32");
    long n = atol(arg(c, v, "--n", "4096"));
    long M = n, N = n, K = n;                            // --shape MxNxK; --n n: square
    const char* shape = arg(c, v, "--shape", "");
    if (*shape && sscanf(shape, "%ldx%ldx%ld", &M, &N, &K) != 3) {
        printf("{\"mode\":\"cgemm\",\"error\":\"--shape wants MxNxK\"}\n"); return 1; }
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
    int bf = !strcmp(prec, "bf16"), hf = !strcmp(prec, "fp16");
    if (!dbl && !i8 && !bf && !hf && strcmp(prec, "fp32")) {
        printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"error\":\"unknown prec\"}\n", prec); return 1; }
    double* t = malloc(sizeof(double) * (reps + 1));
    char impl[256] = "";
    const char* lib = i8 || bf || hf ? "onednn" : "openblas";
    if (bf || hf) {
        int rc = matmul(hf, M, N, K, reps, t, impl, sizeof impl);
        if (rc == 2) {                                  // no such hardware path: say so, do not time the reference
            printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"m\":%ld,\"n\":%ld,\"k\":%ld,\"threads\":%d,"
                   "\"unsupported\":true,\"impl\":\"%s\",\"isa\":%d,\"lib\":\"onednn\"}\n",
                   prec, M, N, K, threads, impl, (int)dnnl_get_effective_cpu_isa());
            return 0; }
        if (rc) { printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"error\":\"dnnl\"}\n", prec); return 1; }
    } else {
        size_t es = dbl ? 8 : i8 ? 1 : 4;
        void *a = aligned_alloc(64, ((size_t)M * K * es + 63) / 64 * 64), *b = aligned_alloc(64, ((size_t)K * N * es + 63) / 64 * 64);
        void *cc = aligned_alloc(64, ((size_t)M * N * (i8 ? 4 : es) + 63) / 64 * 64);
        if (!a || !b || !cc) { printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"error\":\"alloc\"}\n", prec); return 1; }
        for (size_t i = 0; i < (size_t)M * K; ++i) {
            if (i8) ((uint8_t*)a)[i] = (uint8_t)(i * 2654435761u >> 24);
            else if (dbl) ((double*)a)[i] = 1e-3 * (i % 7); else ((float*)a)[i] = 1e-3f * (i % 7); }
        for (size_t i = 0; i < (size_t)K * N; ++i) {
            if (i8) ((int8_t*)b)[i] = (int8_t)(i * 40503u >> 8);
            else if (dbl) ((double*)b)[i] = 1e-3 * (i % 5); else ((float*)b)[i] = 1e-3f * (i % 5); }
        for (int r = -1; r < reps; ++r) {             // r = -1: warm-up
            double t0 = now();
            if (i8) { int32_t co = 0;
                if (dnnl_gemm_u8s8s32('N', 'N', 'F', M, N, K, 1.f, a, K, 0, b, N, 0, 0.f, cc, N, &co) != dnnl_success) {
                    printf("{\"mode\":\"cgemm\",\"prec\":\"int8\",\"error\":\"dnnl\"}\n"); return 1; } }
            else if (dbl) cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, a, K, b, N, 0.0, cc, N);
            else cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a, K, b, N, 0.0f, cc, N);
            if (r >= 0) t[r] = now() - t0;
        }
    }
    qsort(t, reps, sizeof(double), cmp);
    double ops = 2.0 * M * N * K;
    printf("{\"mode\":\"cgemm\",\"prec\":\"%s\",\"n\":%ld,\"m\":%ld,\"nn\":%ld,\"k\":%ld,\"threads\":%d,\"reps\":%d,"
           "\"ops\":%.0f,\"t_median_s\":%.9g,\"t_min_s\":%.9g,\"ops_per_s\":%.6g,\"isa\":%d,\"impl\":\"%s\",\"lib\":\"%s\"}\n",
           prec, n, M, N, K, threads, reps, ops, t[reps / 2], t[0], ops / t[reps / 2], (int)dnnl_get_effective_cpu_isa(),
           impl, lib);
    return 0;
}
