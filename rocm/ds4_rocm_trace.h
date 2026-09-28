#ifndef DS4_ROCM_TRACE_H
#define DS4_ROCM_TRACE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Diagnostic markers only. Neither entry point waits for GPU work. */
void ds4_rocm_trace_layer(uint32_t layer, uint32_t position);
void ds4_rocm_trace_stage(const char *name);
#ifdef __cplusplus
}
#endif
#endif
