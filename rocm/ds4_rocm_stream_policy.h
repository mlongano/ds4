#ifndef DS4_ROCM_STREAM_POLICY_H
#define DS4_ROCM_STREAM_POLICY_H

#include <stdint.h>
#include <string.h>

static inline int ds4_rocm_stream_flag(const char *value) {
    return value && value[0] && strcmp(value, "0") != 0;
}

static inline int ds4_rocm_stream_option(int automatic, const char *enable,
                                         const char *disable) {
    if (ds4_rocm_stream_flag(disable)) return 0;
    return enable && enable[0] ? ds4_rocm_stream_flag(enable) : automatic;
}

static inline int ds4_rocm_stream_r9700_hardware(
        const char *arch, int integrated, uint64_t total_bytes) {
    const uint64_t gib = 1073741824ull;
    return arch && strncmp(arch, "gfx1201", 7) == 0 &&
           (arch[7] == '\0' || arch[7] == ':') && !integrated &&
           total_bytes >= 30ull * gib && total_bytes <= 36ull * gib;
}

/* Small prompts keep the original kernel unless explicitly exercising tails. */
static inline int ds4_rocm_stream_pair_policy(
        int streaming, int glm, int r9700, uint64_t tokens,
        const char *enable, const char *disable) {
    return streaming && !glm && tokens >= 32u &&
        ds4_rocm_stream_option(r9700 && tokens >= 256u, enable, disable);
}

/* Do not change disk I/O scheduling automatically. Active selected uploads own
 * the worker pool and retain their original uploader until their wait ends. */
static inline int ds4_rocm_stream_bulk_policy(
        int streaming, int glm, int r9700, int tmpfs, uint64_t bytes,
        int selected_active, const char *enable, const char *disable) {
    return streaming && !glm && bytes >= 16ull * 1048576ull && !selected_active &&
        ds4_rocm_stream_option(r9700 && tmpfs, enable, disable);
}

/* Keep automatic arithmetic changes within the measured hardware/model pair.
 * An explicit enable may select another device, never another quantization. */
static inline int ds4_rocm_stream_direct_policy(
        int streaming, int glm, const char *arch, int integrated,
        uint64_t total_bytes, unsigned gate_type, unsigned down_type,
        const char *enable, const char *legacy_enable, const char *disable) {
    if (!streaming || glm || gate_type != 16u || down_type != 10u ||
        ds4_rocm_stream_flag(disable)) return 0;
    if (ds4_rocm_stream_flag(enable) || ds4_rocm_stream_flag(legacy_enable)) return 1;
    /* An explicit zero also overrides automatic selection. */
    if ((enable && enable[0]) || (legacy_enable && legacy_enable[0])) return 0;
    return ds4_rocm_stream_r9700_hardware(arch, integrated, total_bytes);
}

#endif
