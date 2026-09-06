/* Test production bulk-job partitioning and publication ordering. Worker SDMA
 * completion itself is exercised separately by test_rocm_stream_pipeline.c. */
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    char *dst;
    uint64_t offset, bytes;
    void *host_raw, *host_buf;
    int ok, uploaded, errnum, direct;
} cuda_stream_read_job;
static uint64_t g_model_registered_size, g_model_range_bytes;
static char *destination;
static uint64_t expected_offset, total, submitted, dropped, discarded, progress;
static unsigned batches, fail_batch;
static int missing_upload, complete;

static int cuda_stream_read_jobs_parallel(cuda_stream_read_job *jobs, uint32_t count) {
    assert(count && count <= 64);
    complete = 0;
    batches++;
    for (uint32_t i = 0; i < count; i++) {
        assert(jobs[i].dst == destination + submitted);
        assert(jobs[i].offset == expected_offset + submitted);
        assert(jobs[i].bytes && jobs[i].bytes <= 8ull * 1048576ull);
        assert(jobs[i].bytes <= total - submitted);
        submitted += jobs[i].bytes;
        jobs[i].uploaded = !missing_upload;
    }
    complete = 1;
    return batches != fail_batch;
}
static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
    assert(complete && offset == expected_offset + dropped);
    dropped += bytes;
}
static void cuda_model_discard_source_pages(const void *model, uint64_t size,
                                            uint64_t offset, uint64_t bytes) {
    assert(model == destination && size == g_model_registered_size);
    assert(complete && offset == expected_offset + discarded);
    discarded += bytes;
}
static void cuda_model_load_progress_note(uint64_t bytes) {
    assert(complete && submitted == dropped && dropped == discarded);
    assert(bytes == g_model_range_bytes + submitted);
    progress = bytes;
}
#include "ds4_model_upload_under_test.h"

int main(void) {
    /* Allocate address space only; mocks never touch the payload. */
    const uint64_t max_bytes = 1024ull * 1048576ull + 17;
    destination = malloc(max_bytes);
    assert(destination);
    const uint64_t sizes[] = {0, 1, 8388607, 8388608, 8388609,
                             16777216, 536870911, 536870912, 536870913, max_bytes};
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        for (unsigned failure = 0; failure < 5; failure++) {
            total = sizes[i];
            expected_offset = 731; /* Includes an unaligned source range. */
            g_model_registered_size = total + expected_offset;
            g_model_range_bytes = 1024;
            submitted = dropped = discarded = progress = batches = complete = 0;
            fail_batch = failure < 4 ? failure : 0;
            missing_upload = failure == 4;
            const unsigned wanted_batches = (unsigned)((total + 536870911) / 536870912);
            const int should_fail = (fail_batch && fail_batch <= wanted_batches) ||
                                    (missing_upload && total);
            assert(cuda_model_range_upload_parallel(destination, destination,
                       expected_offset, total) == !should_fail);
            if (!should_fail && total) {
                assert(submitted == total && dropped == total && discarded == total);
                assert(progress == g_model_range_bytes + total);
            }
            if (should_fail) assert(dropped < submitted);
        }
    }
    free(destination);
    puts("Bulk upload partitioning/failure-order tests passed");
    return 0;
}
