/* Bounded, opt-in decode trace. GPU events bracket stream intervals, not
 * individual kernel execution: an interval can include submission/scheduling
 * gaps. CPU and GPU tracks have separate clock domains, approximately aligned
 * at origin enqueue. Do not infer exact cross-domain overlap or device idle.
 *
 * Events are preallocated before the warmup window. Finish runs after the
 * runtime's existing drain, uses queries only, and reports every lost record.
 * Keep this C-compatible so a mock GPU can test the actual implementation. */
#include "ds4_rocm_trace.h"
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef DS4_ROCM_TRACE_CAP
#define DS4_ROCM_TRACE_CAP 32768u
#endif
#define DS4_ROCM_TRACE_EVENTS (2u * DS4_ROCM_TRACE_CAP + 1u)

typedef struct {
    const char *name;
    uint64_t lane, bytes;
    uint32_t layer, position, start_event, end_event;
    double start_us, end_us;
    int gpu, complete, ok;
} ds4_rocm_trace_record;

static struct {
    int initialized, enabled, active;
    FILE *file;
    ds4_rocm_trace_record *records;
    cudaEvent_t *events;
    uint32_t count, created, used, previous;
    uint32_t skip, tokens, step, last_position, layer, position;
    uint32_t captured_tokens, dropped, incomplete, errors;
    double cpu_origin_us, origin_enqueue_us, setup_us;
} g_rocm_trace;
static pthread_mutex_t g_rocm_trace_mutex = PTHREAD_MUTEX_INITIALIZER;

static double rocm_trace_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1000.0;
}

static int rocm_trace_option(const char *key, uint32_t fallback,
                              uint32_t maximum, uint32_t *out) {
    const char *s = getenv(key);
    if (!s) { *out = fallback; return 1; }
    if (!*s || *s == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(s, &end, 10);
    if (errno || end == s || *end || value > maximum) return 0;
    *out = (uint32_t)value;
    return 1;
}

static void rocm_trace_init(void) {
    g_rocm_trace.initialized = 1;
    const char *path = getenv("DS4_ROCM_DECODE_TRACE");
    if (!path || !*path) return;
    if (!rocm_trace_option("DS4_ROCM_DECODE_TRACE_SKIP", 64, UINT32_MAX,
                            &g_rocm_trace.skip) ||
        !rocm_trace_option("DS4_ROCM_DECODE_TRACE_TOKENS", 8, 16,
                            &g_rocm_trace.tokens) || !g_rocm_trace.tokens) {
        fprintf(stderr, "ds4: invalid ROCm decode trace window\n");
        return;
    }
    /* Never overwrite an earlier trace or user file. */
    g_rocm_trace.file = fopen(path, "wx");
    if (!g_rocm_trace.file) {
        fprintf(stderr, "ds4: cannot create ROCm trace %s: %s\n", path, strerror(errno));
        return;
    }
    const double start = rocm_trace_now_us();
    g_rocm_trace.records = (ds4_rocm_trace_record *)calloc(DS4_ROCM_TRACE_CAP, sizeof(*g_rocm_trace.records));
    g_rocm_trace.events = (cudaEvent_t *)calloc(DS4_ROCM_TRACE_EVENTS, sizeof(*g_rocm_trace.events));
    if (!g_rocm_trace.records || !g_rocm_trace.events) { g_rocm_trace.errors++; return; }
    for (; g_rocm_trace.created < DS4_ROCM_TRACE_EVENTS; g_rocm_trace.created++) {
        if (cudaEventCreate(&g_rocm_trace.events[g_rocm_trace.created]) != cudaSuccess) {
            g_rocm_trace.errors++;
            return;
        }
    }
    g_rocm_trace.cpu_origin_us = rocm_trace_now_us();
    if (cudaEventRecord(g_rocm_trace.events[0], 0) != cudaSuccess) {
        g_rocm_trace.errors++;
        return;
    }
    g_rocm_trace.origin_enqueue_us = rocm_trace_now_us() - g_rocm_trace.cpu_origin_us;
    g_rocm_trace.setup_us = rocm_trace_now_us() - start;
    g_rocm_trace.used = 1;
    g_rocm_trace.last_position = UINT32_MAX;
    g_rocm_trace.enabled = 1;
}

/* All pool access is protected by the trace mutex. Active is also atomic for
 * the disabled fast path in read workers; they never initialize the tracer. */
static uint32_t rocm_trace_marker(cudaStream_t stream) {
    if (g_rocm_trace.used == g_rocm_trace.created) {
        g_rocm_trace.dropped++;
        return UINT32_MAX;
    }
    uint32_t index = g_rocm_trace.used++;
    if (cudaEventRecord(g_rocm_trace.events[index], stream) != cudaSuccess) {
        g_rocm_trace.errors++;
        return UINT32_MAX;
    }
    return index;
}

static uint32_t rocm_trace_record(const char *name, uint64_t lane, uint64_t bytes, int gpu) {
    if (g_rocm_trace.count == DS4_ROCM_TRACE_CAP) {
        g_rocm_trace.dropped++;
        return 0;
    }
    ds4_rocm_trace_record *r = &g_rocm_trace.records[g_rocm_trace.count++];
    r->name = name; r->lane = lane; r->bytes = bytes; r->gpu = gpu;
    r->layer = g_rocm_trace.layer; r->position = g_rocm_trace.position;
    return g_rocm_trace.count;
}

#ifdef __cplusplus
extern "C"
#endif
void ds4_rocm_trace_layer(uint32_t layer, uint32_t position) {
    if (!g_rocm_trace.initialized) rocm_trace_init();
    if (!g_rocm_trace.enabled) return;
    pthread_mutex_lock(&g_rocm_trace_mutex);
    if (position != g_rocm_trace.last_position) {
        g_rocm_trace.last_position = position;
        g_rocm_trace.step++;
        if (g_rocm_trace.step > g_rocm_trace.skip &&
            g_rocm_trace.step - g_rocm_trace.skip <= g_rocm_trace.tokens)
            g_rocm_trace.captured_tokens++;
    }
    const int active = g_rocm_trace.step > g_rocm_trace.skip &&
        g_rocm_trace.step - g_rocm_trace.skip <= g_rocm_trace.tokens;
    g_rocm_trace.layer = layer;
    g_rocm_trace.position = position;
    if (active) g_rocm_trace.previous = rocm_trace_marker(0);
    __atomic_store_n(&g_rocm_trace.active, active, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_rocm_trace_mutex);
}

#ifdef __cplusplus
extern "C"
#endif
void ds4_rocm_trace_stage(const char *name) {
    if (!__atomic_load_n(&g_rocm_trace.active, __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&g_rocm_trace_mutex);
    const uint32_t end = rocm_trace_marker(0);
    if (g_rocm_trace.previous != UINT32_MAX && end != UINT32_MAX) {
        uint32_t id = rocm_trace_record(name, 0, 0, 1);
        if (id) {
            ds4_rocm_trace_record *r = &g_rocm_trace.records[id - 1];
            r->start_event = g_rocm_trace.previous; r->end_event = end;
            r->complete = r->ok = 1;
        }
    }
    g_rocm_trace.previous = end;
    /* Selected uploads have drained before this layer's final HC stage. */
    if (strcmp(name, "ffn_hc_post") == 0)
        __atomic_store_n(&g_rocm_trace.active, 0, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_rocm_trace_mutex);
}

static uint32_t rocm_trace_cpu_begin(const char *name, uint64_t lane, uint64_t bytes) {
    if (!__atomic_load_n(&g_rocm_trace.active, __ATOMIC_ACQUIRE)) return 0;
    pthread_mutex_lock(&g_rocm_trace_mutex);
    uint32_t id = __atomic_load_n(&g_rocm_trace.active, __ATOMIC_RELAXED) ?
        rocm_trace_record(name, lane, bytes, 0) : 0;
    if (id) g_rocm_trace.records[id - 1].start_us = rocm_trace_now_us() - g_rocm_trace.cpu_origin_us;
    pthread_mutex_unlock(&g_rocm_trace_mutex);
    return id;
}

static void rocm_trace_cpu_end(uint32_t id, int ok) {
    if (!id) return;
    pthread_mutex_lock(&g_rocm_trace_mutex);
    ds4_rocm_trace_record *r = &g_rocm_trace.records[id - 1];
    r->end_us = rocm_trace_now_us() - g_rocm_trace.cpu_origin_us;
    r->complete = 1; r->ok = ok;
    pthread_mutex_unlock(&g_rocm_trace_mutex);
}

static uint32_t rocm_trace_upload_begin(cudaStream_t stream, uint64_t bytes) {
    if (!__atomic_load_n(&g_rocm_trace.active, __ATOMIC_ACQUIRE)) return 0;
    pthread_mutex_lock(&g_rocm_trace_mutex);
    uint32_t id = __atomic_load_n(&g_rocm_trace.active, __ATOMIC_RELAXED) ?
        rocm_trace_record("upload", (uint64_t)(uintptr_t)stream, bytes, 1) : 0;
    if (id) g_rocm_trace.records[id - 1].start_event = rocm_trace_marker(stream);
    pthread_mutex_unlock(&g_rocm_trace_mutex);
    return id;
}

static void rocm_trace_upload_end(uint32_t id, cudaStream_t stream, int ok) {
    if (!id) return;
    pthread_mutex_lock(&g_rocm_trace_mutex);
    ds4_rocm_trace_record *r = &g_rocm_trace.records[id - 1];
    r->end_event = rocm_trace_marker(stream);
    r->complete = 1; r->ok = ok;
    pthread_mutex_unlock(&g_rocm_trace_mutex);
}

static void rocm_trace_finish(void) {
    /* Called after existing shutdown synchronization and read-worker join. */
    __atomic_store_n(&g_rocm_trace.active, 0, __ATOMIC_RELEASE);
    if (!g_rocm_trace.initialized) return;
    FILE *fp = g_rocm_trace.file;
    if (fp) {
        fprintf(fp, "{\"traceEvents\":[\n");
        uint32_t written = 0;
        for (uint32_t i = 0; i < g_rocm_trace.count; i++) {
            ds4_rocm_trace_record *r = &g_rocm_trace.records[i];
            if (!r->complete) { g_rocm_trace.incomplete++; continue; }
            if (r->gpu) {
                if (r->start_event == UINT32_MAX || r->end_event == UINT32_MAX) continue;
                cudaError_t origin = cudaEventQuery(g_rocm_trace.events[0]);
                cudaError_t begin = cudaEventQuery(g_rocm_trace.events[r->start_event]);
                cudaError_t end = cudaEventQuery(g_rocm_trace.events[r->end_event]);
                if (origin == cudaErrorNotReady || begin == cudaErrorNotReady || end == cudaErrorNotReady) {
                    g_rocm_trace.incomplete++; continue;
                }
                float start_ms = 0, duration_ms = 0;
                if (origin != cudaSuccess || begin != cudaSuccess || end != cudaSuccess ||
                    cudaEventElapsedTime(&start_ms, g_rocm_trace.events[0], g_rocm_trace.events[r->start_event]) != cudaSuccess ||
                    cudaEventElapsedTime(&duration_ms, g_rocm_trace.events[r->start_event], g_rocm_trace.events[r->end_event]) != cudaSuccess ||
                    !isfinite(start_ms) || !isfinite(duration_ms) || duration_ms < 0) {
                    g_rocm_trace.errors++; continue;
                }
                r->start_us = (double)start_ms * 1000.0;
                r->end_us = r->start_us + (double)duration_ms * 1000.0;
            }
            /* Names are internal static identifiers, never request text. */
            fprintf(fp, "%s{\"name\":\"%s\",\"cat\":\"%s\",\"ph\":\"X\",\"pid\":%d,\"tid\":%llu,\"ts\":%.3f,\"dur\":%.3f,\"args\":{\"layer\":%u,\"position\":%u,\"bytes\":%llu,\"ok\":%s}}\n",
                    written++ ? "," : "", r->name, r->gpu ? "gpu_stream" : "cpu",
                    r->gpu ? 1 : 2, (unsigned long long)r->lane,
                    r->start_us, r->end_us - r->start_us, r->layer, r->position,
                    (unsigned long long)r->bytes, r->ok ? "true" : "false");
        }
        fprintf(fp, "],\"metadata\":{\"records\":%u,\"written\":%u,\"dropped\":%u,\"incomplete\":%u,\"errors\":%u,\"captured_tokens\":%u,\"skip\":%u,\"requested_tokens\":%u,\"setup_us\":%.3f,\"origin_enqueue_us\":%.3f,\"clock_alignment\":\"CPU and GPU have separate origins, approximately aligned at event enqueue; GPU intervals include submission gaps, not pure kernel busy time\"}}\n",
                g_rocm_trace.count, written, g_rocm_trace.dropped, g_rocm_trace.incomplete,
                g_rocm_trace.errors, g_rocm_trace.captured_tokens, g_rocm_trace.skip,
                g_rocm_trace.tokens, g_rocm_trace.setup_us, g_rocm_trace.origin_enqueue_us);
        if (ferror(fp)) fprintf(stderr, "ds4: ROCm trace write failed\n");
        if (fclose(fp)) fprintf(stderr, "ds4: ROCm trace close failed\n");
        fprintf(stderr, "ds4: ROCm trace records=%u written=%u dropped=%u incomplete=%u errors=%u\n",
                g_rocm_trace.count, written, g_rocm_trace.dropped, g_rocm_trace.incomplete, g_rocm_trace.errors);
    }
    for (uint32_t i = 0; i < g_rocm_trace.created; i++) (void)cudaEventDestroy(g_rocm_trace.events[i]);
    free(g_rocm_trace.events);
    free(g_rocm_trace.records);
    memset(&g_rocm_trace, 0, sizeof(g_rocm_trace));
}
