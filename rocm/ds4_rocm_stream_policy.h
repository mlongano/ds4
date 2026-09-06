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
