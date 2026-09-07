/* Exercise the production phase-boundary body with synchronous CPU mocks.
 * GPU integration tests separately check actual cache reuse and output. */
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../rocm/ds4_rocm_stream_policy.h"
#define DS4_GPU_LOG_PREFIX "test: "
static pthread_mutex_t g_stream_read_mutex = PTHREAD_MUTEX_INITIALIZER;
static void *g_stream_read_active_jobs;
static struct { int active; } g_stream_selected_pending, g_stream_batch_selected_pending;
static int g_ssd_streaming_mode, g_glm_model, hardware;
static uint64_t g_model_range_bytes;
static int synced, released, sync_error;
static int cuda_stream_r9700_profile(void) { return hardware; }
static int cudaDeviceSynchronize(void) { synced++; return sync_error; }
static int cuda_ok(int error, const char *what) { (void)what; return !error; }
static void cuda_model_range_release_ranges_only(void) {
    assert(synced == 1 && !sync_error);
    assert(!g_stream_read_active_jobs && !g_stream_selected_pending.active &&
           !g_stream_batch_selected_pending.active);
    released++;
    g_model_range_bytes = 0;
}
#include "ds4_decode_reset_under_test.h"
int main(void) {
    unsetenv("DS4_ROCM_STREAM_CACHE_STATS");
    const char *values[] = {NULL, "0", "1"};
    for (int stream = 0; stream <= 1; stream++)
    for (int glm = 0; glm <= 1; glm++)
    for (int hw = 0; hw <= 1; hw++)
    for (int active = 0; active <= 3; active++)
    for (unsigned en = 0; en < 3; en++)
    for (unsigned dis = 0; dis < 3; dis++)
    for (int error = 0; error <= 1; error++) {
        if (values[en]) setenv("DS4_ROCM_ENABLE_DECODE_MODEL_RESET", values[en], 1);
        else unsetenv("DS4_ROCM_ENABLE_DECODE_MODEL_RESET");
        if (values[dis]) setenv("DS4_ROCM_DISABLE_DECODE_MODEL_RESET", values[dis], 1);
        else unsetenv("DS4_ROCM_DISABLE_DECODE_MODEL_RESET");
        g_ssd_streaming_mode = stream; g_glm_model = glm; hardware = hw;
        g_stream_read_active_jobs = active == 1 ? &hardware : NULL;
        g_stream_selected_pending.active = active == 2;
        g_stream_batch_selected_pending.active = active == 3;
        g_model_range_bytes = 123;
        synced = released = 0; sync_error = error;
        int enabled = stream && !glm && !active && dis != 2 &&
                      (en == 2 || (en == 0 && hw));
        int result = ds4_rocm_release_prefill_model_ranges();
        assert(result == !(enabled && error));
        assert(synced == enabled);
        assert(released == (enabled && !error));
        assert(g_model_range_bytes == (released ? 0 : 123));
    }
    puts("Decode model reset policy/synchronization tests passed");
    return 0;
}
