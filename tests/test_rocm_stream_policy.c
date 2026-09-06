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
    for (int streaming = 0; streaming <= 1; streaming++) {
        for (int glm = 0; glm <= 1; glm++) {
            for (int hw = 0; hw <= 1; hw++) {
                for (int ram = 0; ram <= 1; ram++) {
                    for (int active = 0; active <= 1; active++) {
                        for (int small = 0; small <= 1; small++) {
                            uint64_t bytes = 16ull * 1048576ull - small;
                            int allowed = streaming && !glm && !active && !small;
                            assert(ds4_rocm_stream_bulk_policy(streaming, glm, hw, ram,
                                bytes, active, NULL, NULL) == (allowed && hw && ram));
                            assert(ds4_rocm_stream_bulk_policy(streaming, glm, hw, ram,
                                bytes, active, "1", NULL) == allowed);
                            assert(!ds4_rocm_stream_bulk_policy(streaming, glm, hw, ram,
                                bytes, active, "1", "1"));
                            assert(!ds4_rocm_stream_bulk_policy(streaming, glm, hw, ram,
                                bytes, active, "0", NULL));
                        }
                    }
                }
                const uint64_t tokens[] = {1, 31, 32, 255, 256, 257, 4096};
                for (unsigned i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
                    int allowed = streaming && !glm && tokens[i] >= 32;
                    assert(ds4_rocm_stream_pair_policy(streaming, glm, hw, tokens[i],
                        NULL, NULL) == (allowed && hw && tokens[i] >= 256));
                    assert(ds4_rocm_stream_pair_policy(streaming, glm, hw, tokens[i],
                        "1", NULL) == allowed);
                    assert(!ds4_rocm_stream_pair_policy(streaming, glm, hw, tokens[i],
                        "1", "1"));
                    assert(!ds4_rocm_stream_pair_policy(streaming, glm, hw, tokens[i],
                        "0", NULL));
                }
            }
        }
    }
    puts("ROCm direct-pointer policy: PASS");
    return 0;
}
