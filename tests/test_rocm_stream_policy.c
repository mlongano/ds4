#include "../rocm/ds4_rocm_stream_policy.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
    const uint64_t mem = 32ull * 1073741824ull;
#define POLICY(arch, integrated, gate, down, en, legacy, dis) \
    ds4_rocm_stream_direct_policy(1, 0, arch, integrated, mem, gate, down, en, legacy, dis)
    assert(POLICY("gfx1201", 0, 16, 10, NULL, NULL, NULL));
    assert(POLICY("gfx1201:xnack-", 0, 16, 10, NULL, NULL, NULL));
    assert(!POLICY("gfx12010", 0, 16, 10, NULL, NULL, NULL));
    assert(!POLICY("gfx1100", 0, 16, 10, NULL, NULL, NULL));
    assert(!POLICY("gfx1201", 1, 16, 10, NULL, NULL, NULL));
    assert(!POLICY("gfx1201", 0, 10, 10, "1", "1", NULL));
    assert(!POLICY("gfx1201", 0, 16, 16, "1", NULL, NULL));
    assert(!POLICY("gfx1201", 0, 16, 10, "1", "1", "1"));
    assert(!POLICY("gfx1201", 0, 16, 10, "0", NULL, NULL));
    assert(!POLICY("gfx1201", 0, 16, 10, NULL, "0", NULL));
    assert(POLICY("gfx1201", 0, 16, 10, NULL, NULL, "0"));
    assert(POLICY("gfx1100", 0, 16, 10, "1", NULL, NULL));
    assert(POLICY("gfx1100", 0, 16, 10, NULL, "1", NULL));
    assert(!ds4_rocm_stream_direct_policy(0, 0, "gfx1201", 0, mem, 16, 10, "1", NULL, NULL));
    assert(!ds4_rocm_stream_direct_policy(1, 1, "gfx1201", 0, mem, 16, 10, "1", NULL, NULL));
    assert(!ds4_rocm_stream_direct_policy(1, 0, "gfx1201", 0, mem / 2, 16, 10, NULL, NULL, NULL));
    assert(ds4_rocm_stream_option(1, NULL, NULL));
    assert(!ds4_rocm_stream_option(0, NULL, NULL));
    assert(!ds4_rocm_stream_option(1, "1", "1"));
    assert(!ds4_rocm_stream_option(1, "0", NULL));
    assert(ds4_rocm_stream_option(0, "1", "0"));
    puts("ROCm direct-pointer policy: PASS");
    return 0;
}
