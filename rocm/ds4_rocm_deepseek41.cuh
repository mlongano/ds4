/* V4.1 graph ops for the ROCm backend.
 *
 * The kernels and entry points are shared source with the CUDA backend
 * (../ds4_deepseek41_cuda.cuh) compiled through the compat layer in
 * ds4_rocm.h. This adapter supplies the CUDA-backend externals that file
 * expects and that the ROCm port names differently or does not have:
 *
 * - cuda_decode_stream(): the ROCm port runs decode work on the default
 *   stream, so this maps to stream 0.
 * - g_dsv41_shared and cuda_dsv41_shared_prepare(): the async shared-expert
 *   island. Upstream gates it behind ds4_gpu_device_is_spark(), so on this
 *   hardware ds4_gpu_dsv41_shared_start() returns 0 ("unsupported") and the
 *   engine keeps its sequential shared-expert path. The state stays dormant.
 * - g_decode_graph_capturing: this port has no capture-mode special case.
 * - ds4_tensor_device_idx, cuda_resolve_weight_ptr, cuda_tmp_alloc_on,
 *   cuda_cublas_for_tier: single-device versions of the CUDA backend's
 *   multi-GPU tier helpers.
 * - __shfl_sync/__shfl_down_sync: CUDA source passes 32-bit member masks
 *   where HIP wave64 wants 64-bit ones. The templates widen the mask; the
 *   object-like macros route the shared source's calls through them and are
 *   defined after the template bodies so the bodies call the real builtins.
 * - cublasGemmEx: hipBLAS wants hipblasComputeType_t where the CUDA source
 *   passes a hipDataType. The wrapper translates.
 *
 * Keep this file the only divergence point. When upstream extends
 * ds4_deepseek41_cuda.cuh, fix compile errors here rather than in the
 * shared source.
 */
#ifndef DS4_ROCM_DSV41_ADAPTER_H
#define DS4_ROCM_DSV41_ADAPTER_H

#ifndef cudaStreamWaitEvent
#define cudaStreamWaitEvent hipStreamWaitEvent
#endif

template <typename T>
__device__ static inline T ds41_shfl_sync(uint64_t mask, T val, int lane,
                                          int width = 32) {
    return __shfl_sync(mask, val, lane, width);
}

template <typename T>
__device__ static inline T ds41_shfl_down_sync(uint64_t mask, T val, int lane,
                                               int width = 32) {
    return __shfl_down_sync(mask, val, lane, width);
}

/* CUDA's __ballot_sync covers one 32-lane warp. On wave64 the logical group
 * is half a physical wave, so shift the upper half down; wave32 needs no
 * shift. */
__device__ static inline uint32_t ds41_ballot_sync(uint64_t mask, int pred) {
    (void)mask;
    const uint64_t bits = __ballot_sync(0xffffffffffffffffull, pred != 0);
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE == 64
    return (uint32_t)(bits >> ((threadIdx.x & 32u) ? 32u : 0u));
#else
    return (uint32_t)bits;
#endif
}

#define __shfl_sync ds41_shfl_sync
#define __shfl_down_sync ds41_shfl_down_sync
#define __ballot_sync ds41_ballot_sync

static inline cudaStream_t cuda_decode_stream(void) {
    return (cudaStream_t)0;
}

static bool g_decode_graph_capturing = false;

/* Declared in ds4_gpu.h. No DGX Spark on this port. */
int ds4_gpu_device_is_spark(void) {
    return 0;
}

/* Single device: every tensor lives on tier 0. */
static inline int ds4_tensor_device_idx(const ds4_gpu_tensor *t) {
    (void)t;
    return 0;
}

static inline const char *cuda_resolve_weight_ptr(const void *model_map,
                                                  uint64_t offset,
                                                  uint64_t bytes,
                                                  int logical_tier,
                                                  const char *label) {
    (void)logical_tier;
    return cuda_model_range_ptr(model_map, offset, bytes, label);
}

static inline void *cuda_tmp_alloc_on(int logical_tier, uint64_t bytes,
                                      const char *what) {
    (void)logical_tier;
    return cuda_tmp_alloc(bytes, what);
}

static inline cublasHandle_t cuda_cublas_for_tier(int logical_tier) {
    (void)logical_tier;
    return g_cublas;
}

static inline cublasStatus_t ds41_gemmex(
        cublasHandle_t handle, hipblasOperation_t ta, hipblasOperation_t tb,
        int m, int n, int k, const void *alpha,
        const void *A, hipDataType at, int lda,
        const void *B, hipDataType bt, int ldb,
        const void *beta, void *C, hipDataType ct, int ldc,
        hipDataType compute, hipblasGemmAlgo_t algo) {
    hipblasComputeType_t compute_type;
    switch (compute) {
    case HIP_R_16F: compute_type = HIPBLAS_COMPUTE_16F; break;
    case HIP_R_32F: compute_type = HIPBLAS_COMPUTE_32F; break;
    case HIP_R_64F: compute_type = HIPBLAS_COMPUTE_64F; break;
    default: return HIPBLAS_STATUS_INVALID_VALUE;
    }
    return hipblasGemmEx(handle, ta, tb, m, n, k,
                         alpha, A, at, lda, B, bt, ldb,
                         beta, C, ct, ldc, compute_type, algo);
}

#undef cublasGemmEx
#define cublasGemmEx ds41_gemmex

static struct {
    cudaStream_t stream;
    cudaEvent_t ready, done;
    void *scratch;
    bool active, pending, disabled;
} g_dsv41_shared;

static const uint64_t CUDA_DSV41_SHARED_SCRATCH = 65536u;

static inline bool cuda_dsv41_shared_prepare(void) {
    return false;
}

#include "../ds4_deepseek41_cuda.cuh"

#endif
