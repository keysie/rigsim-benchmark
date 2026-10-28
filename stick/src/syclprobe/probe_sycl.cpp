// rigsim SYCL probe (stick, Intel GPUs): oneMKL GEMM as the vendor library reference, plus bandwidth, launch cost
// and host copies. Same JSON lines as the CUDA probe. The device is picked by ONEAPI_DEVICE_SELECTOR (the stick sets
// level_zero:<index>); `info` lists the GPUs the runtime sees.
//   probe_sycl info
//   probe_sycl gemm   --prec fp32|tf32|fp16|fp16_fp32acc|bf16|int8 --shape MxNxK [--reps 20]
//   probe_sycl stream --op read|write|copy [--mib 1024] [--reps 20]
//   probe_sycl launch [--count 2000]
//   probe_sycl copy   --dir h2d|d2h [--mib 512] [--reps 10]
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static std::string arg(int argc, char** argv, const char* key, const char* dflt) {
    for (int i = 2; i + 1 < argc; ++i) if (!strcmp(argv[i], key)) return argv[i + 1];
    return dflt;
}
static double med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

static sycl::queue make_queue() {
    return sycl::queue(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(),
                                                                  sycl::property::queue::enable_profiling()});
}
static double event_s(sycl::event& e) {
    auto a = e.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto b = e.get_profiling_info<sycl::info::event_profiling::command_end>();
    return (b - a) * 1e-9;
}
static void head(sycl::queue& q, const char* mode) {
    auto d = q.get_device();
    printf("{\"mode\":\"%s\",\"device\":\"%s\",\"driver\":\"%s\",\"units\":%u,\"clock_mhz\":%u", mode,
           d.get_info<sycl::info::device::name>().c_str(), d.get_info<sycl::info::device::driver_version>().c_str(),
           d.get_info<sycl::info::device::max_compute_units>(), d.get_info<sycl::info::device::max_clock_frequency>());
}

template <typename T> static void fill_rand(sycl::queue& q, T* p, size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        uint64_t x = i[0] * 0x9e3779b97f4a7c15ULL; x ^= x >> 31; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 29;
        if constexpr (std::is_same_v<T, int8_t>) p[i] = (int8_t)((x % 255) - 127);
        else p[i] = T((float)(x & 0xffff) / 65536.0f - 0.5f);
    }).wait();
}

// ---- GEMM (oneMKL, row-major C(MxN) = A(MxK) B(KxN)) -------------------------------------------------------------
template <typename TA, typename TC, typename TS>
static int gemm_t(sycl::queue& q, const char* name, int64_t m, int64_t n, int64_t k, int reps,
                  oneapi::mkl::blas::compute_mode mode = oneapi::mkl::blas::compute_mode::standard) {
    TA* a = sycl::malloc_device<TA>(m * k, q); TA* b = sycl::malloc_device<TA>(k * n, q); TC* c = sycl::malloc_device<TC>(m * n, q);
    if (!a || !b || !c) { head(q, "gemm"); printf(",\"prec\":\"%s\",\"error\":\"alloc\"}\n", name); return 0; }
    fill_rand(q, a, m * k); fill_rand(q, b, k * n);
    TS alpha = TS(1), beta = TS(0);
    auto call = [&] {
        return oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
                                                  m, n, k, alpha, a, k, b, n, beta, c, n, mode);
    };
    head(q, "gemm");
    try { call().wait(); }
    catch (const std::exception& e) {
        printf(",\"prec\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,\"unsupported\":true,\"why\":\"%.80s\"}\n", name,
               (long long)m, (long long)n, (long long)k, e.what());
        sycl::free(a, q); sycl::free(b, q); sycl::free(c, q); return 0;
    }
    for (int i = 0; i < 5; ++i) call();
    q.wait();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { auto e = call(); e.wait(); t.push_back(event_s(e)); }
    double ops = 2.0 * m * (double)n * k, md = med(t);
    printf(",\"prec\":\"%s\",\"m\":%lld,\"n\":%lld,\"k\":%lld,\"reps\":%d,\"ops\":%.0f,\"t_median_s\":%.9g,"
           "\"t_min_s\":%.9g,\"ops_per_s\":%.6g,\"lib\":\"onemkl\"}\n", name, (long long)m, (long long)n, (long long)k,
           reps, ops, md, *std::min_element(t.begin(), t.end()), ops / md);
    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    return 0;
}

static int gemm_mode(sycl::queue& q, int argc, char** argv) {
    std::string p = arg(argc, argv, "--prec", "fp16");
    long long m, n, k;
    if (sscanf(arg(argc, argv, "--shape", "4096x4096x4096").c_str(), "%lldx%lldx%lld", &m, &n, &k) != 3) return 1;
    int reps = atoi(arg(argc, argv, "--reps", "20").c_str());
    using bf = oneapi::mkl::bfloat16;
    if (p == "fp32") return gemm_t<float, float, float>(q, "fp32", m, n, k, reps);
    if (p == "tf32") return gemm_t<float, float, float>(q, "tf32", m, n, k, reps, oneapi::mkl::blas::compute_mode::float_to_tf32);
    if (p == "fp16") return gemm_t<sycl::half, sycl::half, sycl::half>(q, "fp16", m, n, k, reps);
    if (p == "fp16_fp32acc") return gemm_t<sycl::half, float, float>(q, "fp16_fp32acc", m, n, k, reps);
    if (p == "bf16") return gemm_t<bf, float, float>(q, "bf16", m, n, k, reps);
#ifdef RIGSIM_MKL_INT8
    if (p == "int8") return gemm_t<int8_t, int32_t, float>(q, "int8", m, n, k, reps);
#else
    if (p == "int8") { head(q, "gemm"); printf(",\"prec\":\"int8\",\"unsupported\":true,\"why\":\"oneMKL build without int8 gemm\"}\n"); return 0; }
#endif
    fprintf(stderr, "unknown precision %s\n", p.c_str());
    return 1;
}

// ---- streaming ------------------------------------------------------------------------------------------------------
static int stream_mode(sycl::queue& q, int argc, char** argv) {
    std::string op = arg(argc, argv, "--op", "read");
    size_t bytes = strtoull(arg(argc, argv, "--mib", "1024").c_str(), nullptr, 10) << 20;
    size_t maxa = q.get_device().get_info<sycl::info::device::max_mem_alloc_size>();
    if (bytes > maxa) bytes = maxa & ~(size_t)0xfffff;
    int reps = atoi(arg(argc, argv, "--reps", "20").c_str());
    size_t n = bytes / sizeof(sycl::float4);
    auto* x = sycl::malloc_device<sycl::float4>(n, q); auto* y = sycl::malloc_device<sycl::float4>(n, q);
    float* out = sycl::malloc_device<float>(1, q);
    fill_rand(q, (float*)x, n * 4);
    size_t threads = std::min<size_t>(n, (size_t)q.get_device().get_info<sycl::info::device::max_compute_units>() * 2048);
    threads = (threads + 255) / 256 * 256;
    const char opc = op[0];                     // a plain char: std::string cannot go into a kernel
    auto run = [&]() -> sycl::event {
        return q.parallel_for(sycl::nd_range<1>(threads, 256), [=](sycl::nd_item<1> it) {
            size_t i0 = it.get_global_id(0), st = it.get_global_range(0);
            if (opc == 'r') {
                float acc = 0.f;
                for (size_t i = i0; i < n; i += st) { auto v = x[i]; acc += v.x() + v.y() + v.z() + v.w(); }
                if (acc == 12345.678f) out[0] = acc;
            } else if (opc == 'w') {
                for (size_t i = i0; i < n; i += st) y[i] = sycl::float4(1.f, 2.f, 3.f, (float)i);
            } else {
                for (size_t i = i0; i < n; i += st) y[i] = x[i];
            }
        });
    };
    for (int i = 0; i < 3; ++i) run().wait();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) { auto e = run(); e.wait(); t.push_back(event_s(e)); }
    double moved = op == "copy" ? 2.0 * bytes : (double)bytes, md = med(t);
    head(q, "stream");
    printf(",\"op\":\"%s\",\"bytes\":%.0f,\"reps\":%d,\"t_median_s\":%.9g,\"bytes_per_s\":%.6g}\n", op.c_str(), moved, reps, md, moved / md);
    sycl::free(x, q); sycl::free(y, q); sycl::free(out, q);
    return 0;
}

// ---- launch cost ----------------------------------------------------------------------------------------------------
static int launch_mode(sycl::queue& q, int argc, char** argv) {
    int count = atoi(arg(argc, argv, "--count", "2000").c_str());
    int* d = sycl::malloc_device<int>(64, q);
    auto k = [&](int i) { return q.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) { if (i == -1) d[it.get_local_id(0)] = 1; }); };
    for (int i = 0; i < count; ++i) k(i);
    q.wait();
    std::vector<double> bb, host, sync;
    for (int r = 0; r < 7; ++r) {
        auto h0 = clk::now();
        auto first = k(0);
        sycl::event last;
        for (int i = 1; i < count; ++i) last = k(i);
        double hs = secs(h0);
        last.wait();
        auto a = first.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto b = last.get_profiling_info<sycl::info::event_profiling::command_end>();
        bb.push_back((b - a) * 1e-9 / count); host.push_back(hs / count);
    }
    for (int r = 0; r < 201; ++r) { auto h0 = clk::now(); k(r).wait(); sync.push_back(secs(h0)); }
    head(q, "launch");
    printf(",\"count\":%d,\"per_kernel_s\":%.6g,\"host_per_launch_s\":%.6g,\"launch_sync_s\":%.6g}\n",
           count, med(bb), med(host), med(sync));
    sycl::free(d, q);
    return 0;
}

// ---- host copies ----------------------------------------------------------------------------------------------------
static int copy_mode(sycl::queue& q, int argc, char** argv) {
    std::string dir = arg(argc, argv, "--dir", "h2d");
    size_t bytes = strtoull(arg(argc, argv, "--mib", "512").c_str(), nullptr, 10) << 20;
    int reps = atoi(arg(argc, argv, "--reps", "10").c_str());
    char* h = sycl::malloc_host<char>(bytes, q); char* d = sycl::malloc_device<char>(bytes, q);
    memset(h, 1, bytes);
    auto once = [&] { return dir == "h2d" ? q.memcpy(d, h, bytes) : q.memcpy(h, d, bytes); };
    once().wait();
    auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) once();
    q.wait();
    double s = secs(t0);
    head(q, "copy");
    printf(",\"dir\":\"%s\",\"bytes\":%.0f,\"seconds\":%.6g,\"bytes_per_s\":%.6g}\n", dir.c_str(), (double)bytes * reps, s, bytes * (double)reps / s);
    sycl::free(h, q); sycl::free(d, q);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: probe_sycl info|gemm|stream|launch|copy ...\n"); return 1; }
    std::string mode = argv[1];
    try {
        if (mode == "info") {
            int i = 0;
            for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu)) {
                // which physical card: identical models stay apart (Intel's PCI-address extension, where offered)
                std::string pci;
                if (d.has(sycl::aspect::ext_intel_pci_address)) pci = d.get_info<sycl::ext::intel::info::device::pci_address>();
                printf("{\"mode\":\"info\",\"index\":%d,\"pci\":\"%s\",\"name\":\"%s\",\"backend\":\"%s\",\"units\":%u,\"clock_mhz\":%u,"
                       "\"mem_bytes\":%llu}\n", i++, pci.c_str(), d.get_info<sycl::info::device::name>().c_str(),
                       d.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "other",
                       d.get_info<sycl::info::device::max_compute_units>(), d.get_info<sycl::info::device::max_clock_frequency>(),
                       (unsigned long long)d.get_info<sycl::info::device::global_mem_size>());
            }
            return 0;
        }
        sycl::queue q = make_queue();
        if (mode == "gemm") return gemm_mode(q, argc, argv);
        if (mode == "stream") return stream_mode(q, argc, argv);
        if (mode == "launch") return launch_mode(q, argc, argv);
        if (mode == "copy") return copy_mode(q, argc, argv);
    } catch (const std::exception& e) {
        printf("{\"mode\":\"%s\",\"error\":\"%.200s\"}\n", mode.c_str(), e.what());
        return 2;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
