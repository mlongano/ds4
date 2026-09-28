/* Compile the actual trace implementation against non-synchronizing GPU mocks. */
#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DS4_ROCM_TRACE_CAP 64u
#define cudaSuccess 0
#define cudaErrorNotReady 1
typedef int cudaError_t;
typedef void *cudaStream_t;
typedef struct fake_event { float milliseconds; } *cudaEvent_t;
static unsigned creates, live_events, records;
static int create_failure, record_failure, query_status, elapsed_failure;
static float gpu_clock;

static cudaError_t cudaEventCreate(cudaEvent_t *event) {
    creates++;
    if (create_failure && creates == 4) return 2;
    *event = calloc(1, sizeof(**event));
    assert(*event);
    live_events++;
    return 0;
}
static cudaError_t cudaEventDestroy(cudaEvent_t event) {
    assert(event && live_events);
    live_events--;
    free(event);
    return 0;
}
static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    (void)stream;
    records++;
    if (record_failure && records == 3) return 2;
    gpu_clock += 0.1f;
    event->milliseconds = gpu_clock;
    return 0;
}
static cudaError_t cudaEventQuery(cudaEvent_t event) {
    assert(event);
    return query_status;
}
static cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b) {
    assert(a && b);
    if (elapsed_failure == 1) return 2;
    *ms = elapsed_failure == 2 ? NAN : b->milliseconds - a->milliseconds;
    return 0;
}
#include "../rocm/ds4_rocm_trace.cuh"

static void *cpu_worker(void *arg) {
    uint64_t lane = (uintptr_t)arg;
    for (unsigned i = 0; i < 8; i++) {
        uint32_t id = rocm_trace_cpu_begin("read", lane, 4096);
        assert(id);
        rocm_trace_cpu_end(id, 1);
    }
    return NULL;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    const char *mode = argv[1], *path = argv[2];
    unsetenv("DS4_ROCM_DECODE_TRACE");
    unsetenv("DS4_ROCM_DECODE_TRACE_SKIP");
    unsetenv("DS4_ROCM_DECODE_TRACE_TOKENS");
    if (!strcmp(mode, "disabled")) {
        ds4_rocm_trace_layer(0, 100);
        ds4_rocm_trace_stage("q_path");
        assert(!rocm_trace_cpu_begin("read", 1, 32));
        assert(!rocm_trace_upload_begin((void *)1, 32));
        rocm_trace_cpu_end(0, 1);
        rocm_trace_upload_end(0, (void *)1, 1);
        rocm_trace_finish();
        assert(!creates && !records);
        return 0;
    }
    setenv("DS4_ROCM_DECODE_TRACE", path, 1);
    setenv("DS4_ROCM_DECODE_TRACE_SKIP", "1", 1);
    setenv("DS4_ROCM_DECODE_TRACE_TOKENS", "2", 1);
    if (!strcmp(mode, "invalid-window")) {
        setenv("DS4_ROCM_DECODE_TRACE_TOKENS", "17", 1);
        ds4_rocm_trace_layer(0, 100);
        rocm_trace_finish();
        assert(!creates && !records);
        return 0;
    }
    if (!strcmp(mode, "existing-file")) {
        FILE *fp = fopen(path, "wx"); assert(fp);
        fputs("keep\n", fp); fclose(fp);
        ds4_rocm_trace_layer(0, 100);
        rocm_trace_finish();
        assert(!creates && !records);
        return 0;
    }
    create_failure = !strcmp(mode, "create-failure");
    record_failure = !strcmp(mode, "record-failure");
    query_status = !strcmp(mode, "not-ready") ? 1 : !strcmp(mode, "query-failure") ? 2 : 0;
    elapsed_failure = !strcmp(mode, "elapsed-failure") ? 1 : !strcmp(mode, "nonfinite") ? 2 : 0;
    ds4_rocm_trace_layer(0, 100);
    ds4_rocm_trace_stage("unrecorded");
    ds4_rocm_trace_layer(0, 101);
    if (!strcmp(mode, "capacity")) {
        for (unsigned i = 0; i < 100; i++) {
            uint32_t id = rocm_trace_cpu_begin("read", 7, 64);
            rocm_trace_cpu_end(id, 1);
        }
    } else if (!strcmp(mode, "threads")) {
        pthread_t workers[4];
        for (uintptr_t i = 0; i < 4; i++) assert(!pthread_create(&workers[i], NULL, cpu_worker, (void *)(i + 1)));
        for (unsigned i = 0; i < 4; i++) assert(!pthread_join(workers[i], NULL));
    } else {
        ds4_rocm_trace_stage("q_path");
        uint32_t read = rocm_trace_cpu_begin("read", 7, 64);
        uint32_t upload = rocm_trace_upload_begin((void *)7, 64);
        rocm_trace_upload_end(upload, (void *)7, strcmp(mode, "operation-failure") != 0);
        if (strcmp(mode, "unclosed")) rocm_trace_cpu_end(read, 1);
    }
    ds4_rocm_trace_stage("ffn_hc_post");
    assert(!rocm_trace_cpu_begin("outside-layer", 0, 0));
    ds4_rocm_trace_layer(0, 102);
    ds4_rocm_trace_stage("indexer_score");
    ds4_rocm_trace_stage("indexer_topk");
    ds4_rocm_trace_stage("ffn_hc_post");
    ds4_rocm_trace_layer(0, 103);
    ds4_rocm_trace_stage("outside-window");
    rocm_trace_finish();
    assert(live_events == 0);
    return 0;
}
