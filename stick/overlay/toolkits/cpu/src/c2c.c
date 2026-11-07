// rigsim CPU core-to-core probe (stick, design/engine-primitives.md item 3), one JSON line per result.
//   c2c pairs   [--max-cores 32] [--ms 4]
//       one-way latency of a cache line handed between two cores (ping-pong on one atomic), for pairs of physical
//       cores (one logical CPU each; more than max-cores: an even sample), plus one SMT sibling pair. Each pair says
//       how the two cores relate: smt, same L3, other L3 in the package, other package; on hybrid Intel parts also the
//       core types (P/E, from /sys/devices/cpu_core and cpu_atom).
//   c2c barrier [--iters 20000]
//       the cost of one barrier over T threads (T = 2, 4, 8, ... all physical cores, pinned): a spin barrier (what
//       llama.cpp's CPU backend does after every operation) and a futex barrier (pthread_barrier), placed on the
//       cores in order, on one L3 only and spread over the L3s where the machine has several, on P-cores only on
//       hybrid parts.
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define MAXCPU 1024
static const char* arg(int c, char** v, const char* k, const char* d) {
    for (int i = 2; i + 1 < c; ++i) if (!strcmp(v[i], k)) return v[i + 1];
    return d;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int read_int(const char* path, int dflt) {
    FILE* f = fopen(path, "r"); int v = dflt;
    if (f) { if (fscanf(f, "%d", &v) != 1) v = dflt; fclose(f); }
    return v;
}
static int in_list(const char* path, int cpu) {           // cpu in a list like "0-7,16-23"; -1: no such file
    FILE* f = fopen(path, "r"); if (!f) return -1;
    char buf[4096]; int found = 0;
    if (fgets(buf, sizeof buf, f)) {
        for (char* t = strtok(buf, ",\n"); t; t = strtok(NULL, ",\n")) {
            int a, b; if (sscanf(t, "%d-%d", &a, &b) == 2) { if (cpu >= a && cpu <= b) found = 1; }
            else if (sscanf(t, "%d", &a) == 1 && a == cpu) found = 1;
        }
    }
    fclose(f); return found;
}

struct cpu { int id, pkg, core, l3, smt_first; char type; };
static struct cpu cpus[MAXCPU];
static int ncpu;

static void topology(void) {                               // the CPUs this process may run on, with their place
    cpu_set_t set; CPU_ZERO(&set); sched_getaffinity(0, sizeof set, &set);
    ncpu = 0;
    for (int c = 0; c < MAXCPU && ncpu < MAXCPU; ++c) {
        if (!CPU_ISSET(c, &set)) continue;
        char p[256]; struct cpu x = {c, 0, c, 0, 1, '?'};
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c); x.pkg = read_int(p, 0);
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/topology/core_id", c); x.core = read_int(p, c);
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d/cache/index3/id", c); x.l3 = read_int(p, x.pkg);
        int pc = in_list("/sys/devices/cpu_core/cpus", c), ec = in_list("/sys/devices/cpu_atom/cpus", c);
        x.type = pc == 1 ? 'P' : ec == 1 ? 'E' : '-';
        for (int i = 0; i < ncpu; ++i)                      // a later sibling of a core already listed
            if (cpus[i].pkg == x.pkg && cpus[i].core == x.core && cpus[i].type == x.type) { x.smt_first = 0; break; }
        cpus[ncpu++] = x;
    }
}
static void pin(int cpu) { cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); pthread_setaffinity_np(pthread_self(), sizeof s, &s); }

// ---- ping-pong -----------------------------------------------------------------------------------------------------
static _Alignas(64) _Atomic long flag;
struct pp { int cpu; long rounds; };
static void* pong(void* a) {
    struct pp* p = a; pin(p->cpu);
    for (long r = 0; r < p->rounds; ++r) {
        while (atomic_load_explicit(&flag, memory_order_acquire) != 2 * r + 1) ;
        atomic_store_explicit(&flag, 2 * r + 2, memory_order_release);
    }
    return NULL;
}
static double pingpong_ns(int a, int b, double ms) {      // one-way: half a round trip
    long rounds = 2000;
    for (int pass = 0; pass < 2; ++pass) {                  // pass 0 sizes the run to about `ms`
        atomic_store(&flag, 0);
        struct pp p = {b, rounds}; pthread_t t; pthread_create(&t, NULL, pong, &p);
        pin(a);
        double t0 = now();
        for (long r = 0; r < rounds; ++r) {
            atomic_store_explicit(&flag, 2 * r + 1, memory_order_release);
            while (atomic_load_explicit(&flag, memory_order_acquire) != 2 * r + 2) ;
        }
        double el = now() - t0;
        pthread_join(t, NULL);
        if (pass == 1) return el / rounds / 2 * 1e9;
        long want = (long)(rounds * (ms / 1e3) / (el > 1e-6 ? el : 1e-6));
        rounds = want < 200 ? 200 : want > 5000000 ? 5000000 : want;
    }
    return -1;
}
static const char* relation(struct cpu* x, struct cpu* y) {
    if (x->pkg != y->pkg) return "other-package";
    if (x->core == y->core && x->type == y->type) return "smt";
    return x->l3 == y->l3 ? "same-l3" : "other-l3";
}
static int pairs_mode(int c, char** v) {
    int maxc = atoi(arg(c, v, "--max-cores", "32")); double ms = atof(arg(c, v, "--ms", "4"));
    int phys[MAXCPU], np = 0;
    for (int i = 0; i < ncpu; ++i) if (cpus[i].smt_first) phys[np++] = i;
    int pick[MAXCPU], k = 0;
    if (np <= maxc) { for (int i = 0; i < np; ++i) pick[k++] = phys[i]; }
    else for (int i = 0; i < maxc; ++i) pick[k++] = phys[(long)i * np / maxc];   // an even sample over the cores
    if (k < 2) { printf("{\"mode\":\"c2c\",\"error\":\"fewer than two physical cores (%d)\"}\n", k); return 0; }
    for (int i = 0; i < k; ++i)
        for (int j = i + 1; j < k; ++j) {
            struct cpu *x = &cpus[pick[i]], *y = &cpus[pick[j]];
            printf("{\"mode\":\"c2c\",\"a\":%d,\"b\":%d,\"relation\":\"%s\",\"types\":\"%c%c\",\"ns\":%.2f}\n",
                   x->id, y->id, relation(x, y), x->type, y->type, pingpong_ns(x->id, y->id, ms));
            fflush(stdout);
        }
    for (int i = 0; i < ncpu; ++i) if (!cpus[i].smt_first)  // one SMT sibling pair, if the machine has SMT
        for (int j = 0; j < ncpu; ++j)
            if (cpus[j].smt_first && cpus[j].pkg == cpus[i].pkg && cpus[j].core == cpus[i].core) {
                printf("{\"mode\":\"c2c\",\"a\":%d,\"b\":%d,\"relation\":\"smt\",\"types\":\"%c%c\",\"ns\":%.2f}\n",
                       cpus[j].id, cpus[i].id, cpus[j].type, cpus[i].type, pingpong_ns(cpus[j].id, cpus[i].id, ms));
                return 0;
            }
    return 0;
}

// ---- barriers ------------------------------------------------------------------------------------------------------
struct bar { int n; _Alignas(64) _Atomic int count; _Alignas(64) _Atomic int sense; };
static void spin_wait(struct bar* b, int* local) {         // sense-reversing spin barrier
    *local = !*local;
    if (atomic_fetch_add_explicit(&b->count, 1, memory_order_acq_rel) == b->n - 1) {
        atomic_store_explicit(&b->count, 0, memory_order_relaxed);
        atomic_store_explicit(&b->sense, *local, memory_order_release);
    } else while (atomic_load_explicit(&b->sense, memory_order_acquire) != *local) ;
}
struct job { int cpu, iters, futex; struct bar* b; pthread_barrier_t* pb; pthread_barrier_t* start; double secs; };
static void* worker(void* a) {
    struct job* j = a; pin(j->cpu); int local = 0;
    pthread_barrier_wait(j->start);
    double t0 = now();
    for (int i = 0; i < j->iters; ++i) { if (j->futex) pthread_barrier_wait(j->pb); else spin_wait(j->b, &local); }
    j->secs = now() - t0;
    return NULL;
}
static double barrier_ns(const int* cpu_list, int n, int iters, int futex) {
    struct bar b = {n, 0, 0}; pthread_barrier_t pb, start;
    pthread_barrier_init(&pb, NULL, n); pthread_barrier_init(&start, NULL, n);
    pthread_t t[MAXCPU]; struct job j[MAXCPU];
    for (int i = 0; i < n; ++i) { j[i] = (struct job){cpu_list[i], iters, futex, &b, &pb, &start, 0}; pthread_create(&t[i], NULL, worker, &j[i]); }
    double worst = 0;
    for (int i = 0; i < n; ++i) { pthread_join(t[i], NULL); if (j[i].secs > worst) worst = j[i].secs; }
    pthread_barrier_destroy(&pb); pthread_barrier_destroy(&start);
    return worst / iters * 1e9;
}
static int barrier_mode(int c, char** v) {
    int iters = atoi(arg(c, v, "--iters", "20000"));
    int phys[MAXCPU], np = 0, pc[MAXCPU], npc = 0;
    for (int i = 0; i < ncpu; ++i) if (cpus[i].smt_first) { phys[np++] = cpus[i].id; if (cpus[i].type == 'P') pc[npc++] = cpus[i].id; }
    int l3first = cpus[0].l3, one[MAXCPU], none = 0, nl3 = 0, l3s[64];
    for (int i = 0; i < ncpu; ++i) if (cpus[i].smt_first) {
        if (cpus[i].l3 == l3first) one[none++] = cpus[i].id;
        int seen = 0; for (int k = 0; k < nl3; ++k) if (l3s[k] == cpus[i].l3) seen = 1;
        if (!seen && nl3 < 64) l3s[nl3++] = cpus[i].l3;
    }
    int spread[MAXCPU], ns = 0;                             // round robin over the L3s
    for (int round = 0; ns < np && round < MAXCPU; ++round)
        for (int k = 0; k < nl3; ++k) {
            int seen = 0;
            for (int i = 0; i < ncpu; ++i) if (cpus[i].smt_first && cpus[i].l3 == l3s[k] && seen++ == round) spread[ns++] = cpus[i].id;
        }
    struct { const char* name; int* list; int n; } pl[] = {{"in-order", phys, np}, {"one-l3", one, nl3 > 1 ? none : 0},
                                                            {"spread-l3", spread, nl3 > 1 ? ns : 0}, {"p-cores", pc, npc && npc < np ? npc : 0}};
    if (np < 2) { printf("{\"mode\":\"barrier\",\"error\":\"fewer than two physical cores (%d)\"}\n", np); return 0; }
    for (int p = 0; p < 4; ++p)
        for (int n = 2; n <= pl[p].n; n = n * 2 > pl[p].n && n < pl[p].n ? pl[p].n : n * 2) {
            for (int fx = 0; fx < 2; ++fx)
                printf("{\"mode\":\"barrier\",\"kind\":\"%s\",\"placement\":\"%s\",\"threads\":%d,\"iters\":%d,\"ns\":%.1f}\n",
                       fx ? "futex" : "spin", pl[p].name, n, fx ? iters / 10 : iters,
                       barrier_ns(pl[p].list, n, fx ? iters / 10 : iters, fx));
            fflush(stdout);
            if (n == pl[p].n) break;
        }
    return 0;
}

int main(int c, char** v) {
    if (c < 2) { fprintf(stderr, "usage: c2c pairs|barrier ...\n"); return 2; }
    topology();
    if (!strcmp(v[1], "pairs")) return pairs_mode(c, v);
    if (!strcmp(v[1], "barrier")) return barrier_mode(c, v);
    fprintf(stderr, "unknown mode %s\n", v[1]); return 2;
}
