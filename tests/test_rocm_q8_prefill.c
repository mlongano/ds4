/* Exact comparison against the retained scalar Q8 kernel. Includes partial
 * token, output-row and K-block tiles, plus production projection shapes. */
#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void check(unsigned k, unsigned m, unsigned n) {
    const uint64_t wb = (uint64_t)(k / 32) * 34 * m;
    const uint64_t xb = (uint64_t)k * n * sizeof(float);
    const uint64_t yb = (uint64_t)m * n * sizeof(float);
    FILE *file = tmpfile();
    assert(file && ftruncate(fileno(file), wb) == 0);
    unsigned char *w = mmap(NULL, wb, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fileno(file), 0);
    assert(w != MAP_FAILED);
    for (uint64_t b = 0; b < wb / 34; b++) {
        const uint16_t scale = (uint16_t)(0x2000 + (b % 20) * 32);
        memcpy(w + b * 34, &scale, 2);
        for (unsigned j = 0; j < 32; j++)
            w[b * 34 + 2 + j] = (unsigned char)((int)((b * 19 + j * 31) % 256) - 128);
    }
    float *x = malloc(xb), *ref = malloc(yb), *candidate = malloc(yb);
    assert(x && ref && candidate);
    for (uint64_t i = 0; i < xb / sizeof(float); i++)
        x[i] = (float)((int)((i * 193 + 71) % 1001) - 500) * 0.00137f;
    for (uint64_t i = 0; i < yb / sizeof(float); i++) candidate[i] = NAN;

    assert(ds4_gpu_init());
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(true);
    const uint64_t zero = 0;
    assert(ds4_gpu_set_model_map(w, wb));
    assert(ds4_gpu_set_model_fd(fileno(file)));
    assert(ds4_gpu_set_model_map_spans(w, wb, &zero, &wb, 1, wb));
    ds4_gpu_tensor *xt = ds4_gpu_tensor_alloc(xb), *yt = ds4_gpu_tensor_alloc(yb);
    assert(xt && yt && ds4_gpu_tensor_write(xt, 0, x, xb));
    assert(ds4_gpu_tensor_write(yt, 0, candidate, yb));
    setenv("DS4_ROCM_DISABLE_Q8_PREFILL_PAIR", "1", 1);
    assert(ds4_gpu_matmul_q8_0_tensor(yt, w, wb, 0, k, m, xt, n));
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(yt, 0, ref, yb));
    assert(ds4_gpu_tensor_write(yt, 0, candidate, yb));
    unsetenv("DS4_ROCM_DISABLE_Q8_PREFILL_PAIR");
    setenv("DS4_ROCM_ENABLE_Q8_PREFILL_PAIR", "1", 1);
    assert(ds4_gpu_matmul_q8_0_tensor(yt, w, wb, 0, k, m, xt, n));
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(yt, 0, candidate, yb));
    int nonzero = 0;
    for (uint64_t i = 0; i < yb / sizeof(float); i++) {
        assert(isfinite(ref[i]) && isfinite(candidate[i]));
        nonzero |= ref[i] != 0;
    }
    assert(nonzero && memcmp(ref, candidate, yb) == 0);
    printf("Q8 exact: k=%u m=%u tokens=%u\n", k, m, n);
    ds4_gpu_tensor_free(xt);
    ds4_gpu_tensor_free(yt);
    ds4_gpu_cleanup();
    munmap(w, wb);
    fclose(file);
    free(x);
    free(ref);
    free(candidate);
}

int main(void) {
    const unsigned tails[] = {1, 17, 31, 32, 33, 255, 256, 257};
    for (unsigned i = 0; i < sizeof(tails) / sizeof(tails[0]); i++)
        check(1120, 65, tails[i]);
    check(32, 1, 33);
    check(4096, 4096, 4096);
    check(1024, 32768, 4096);
    check(8192, 4096, 4096);
    return 0;
}
