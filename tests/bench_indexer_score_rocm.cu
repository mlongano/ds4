/* ROCm decode-indexer scoring microbenchmark.
 *
 * Compares the production 128-thread kernel with wider blocks. Wider variants
 * retain the same four-head accumulation statements and order as production;
 * only independent head dot products execute concurrently. Every score must
 * match bit-for-bit before a timing result is accepted.
 *
 * Usage: tests/bench_indexer_score_rocm [n_comp] [warmup] [iters]
 */
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t check_err = (call); \
    if (check_err != hipSuccess) { \
        std::fprintf(stderr, "bench-indexer: %s failed: %s\n", #call, hipGetErrorString(check_err)); \
        std::exit(1); \
    } \
} while (0)

__device__ static float bench_warp_sum_f32(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down(value, offset, 32);
    return value;
}

__global__ static void score_reference(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, float scale) {
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= 128u) return;

    __shared__ float krow[128];
    __shared__ float partial[4];
    krow[tid] = index_comp[(uint64_t)c * 128u + tid];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = bench_warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) total += partial[0] + partial[1] + partial[2] + partial[3];
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

/* One wave owns one compressed row. k stays in registers and the four values
 * entering each production accumulation statement stay in registers too.
 * Dot-product lane assignment and the sequence of four-head additions are
 * unchanged; only cross-wave shared-memory barriers are removed. */
template <uint32_t WARPS_PER_BLOCK>
__global__ static void score_wave_rows(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, float scale) {
    static_assert(WARPS_PER_BLOCK >= 1u && WARPS_PER_BLOCK <= 8u, "invalid waves per block");
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t c = blockIdx.x * WARPS_PER_BLOCK + warp;
    if (c >= n_comp) return;

    __shared__ volatile float contribution[WARPS_PER_BLOCK * 4u];
    const float4 kv = ((const float4 *)(index_comp + (uint64_t)c * 128u))[lane];
    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += 4u) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t h = h0 + j;
            const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
            float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
            dot = bench_warp_sum_f32(dot);
            if (lane == 0)
                contribution[warp * 4u + j] = fmaxf(dot, 0.0f) * weights[h] * scale;
        }
        if (lane == 0) {
            const uint32_t base = warp * 4u;
            total += contribution[base] + contribution[base + 1u] +
                     contribution[base + 2u] + contribution[base + 3u];
        }
    }
    if (lane == 0) scores[c] = total;
}

template <uint32_t WARPS>
__global__ static void score_wide(
        float *scores, const float *q, const float *weights,
        const float *index_comp, uint32_t n_comp, float scale) {
    static_assert(WARPS >= 8u && WARPS <= 32u && WARPS % 4u == 0u, "invalid warp count");
    const uint32_t c = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    if (c >= n_comp || tid >= WARPS * 32u) return;

    __shared__ float krow[128];
    __shared__ float partial[WARPS];
    for (uint32_t d = tid; d < 128u; d += WARPS * 32u)
        krow[d] = index_comp[(uint64_t)c * 128u + d];
    __syncthreads();

    float total = 0.0f;
    for (uint32_t h0 = 0; h0 < 64u; h0 += WARPS) {
        const uint32_t h = h0 + warp;
        const float4 qv = ((const float4 *)(q + (uint64_t)h * 128u))[lane];
        const float4 kv = ((const float4 *)krow)[lane];
        float dot = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        dot = bench_warp_sum_f32(dot);
        if (lane == 0) partial[warp] = fmaxf(dot, 0.0f) * weights[h] * scale;
        __syncthreads();
        if (tid == 0) {
#pragma unroll
            for (uint32_t group = 0; group < WARPS; group += 4u)
                total += partial[group] + partial[group + 1u] + partial[group + 2u] + partial[group + 3u];
        }
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

static uint64_t next_random(uint64_t *state) {
    *state = *state * 6364136223846793005ull + 1442695040888963407ull;
    return *state;
}

static float random_float(uint64_t *state, float magnitude) {
    const int32_t signed_bits = (int32_t)(next_random(state) >> 32u);
    return ((float)signed_bits / 2147483648.0f) * magnitude;
}

template <uint32_t WARPS>
static void launch_wide(float *out, const float *q, const float *weights,
                        const float *cache, uint32_t n_comp, float scale) {
    score_wide<WARPS><<<n_comp, WARPS * 32u>>>(out, q, weights, cache, n_comp, scale);
}

template <uint32_t WARPS>
static void launch_wave_rows(float *out, const float *q, const float *weights,
                             const float *cache, uint32_t n_comp, float scale) {
    score_wave_rows<WARPS><<<(n_comp + WARPS - 1u) / WARPS, WARPS * 32u>>>(
            out, q, weights, cache, n_comp, scale);
}

typedef void (*launch_fn)(float *, const float *, const float *, const float *, uint32_t, float);

static void launch_reference(float *out, const float *q, const float *weights,
                             const float *cache, uint32_t n_comp, float scale) {
    score_reference<<<n_comp, 128u>>>(out, q, weights, cache, n_comp, scale);
}

static double percentile(std::vector<float> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = (size_t)std::floor((values.size() - 1u) * fraction);
    return values[index];
}

static std::vector<float> time_variant(launch_fn launch, float *out,
                                       const float *q, const float *weights,
                                       const float *cache, uint32_t n_comp,
                                       float scale, uint32_t warmup, uint32_t iters) {
    for (uint32_t i = 0; i < warmup; i++) launch(out, q, weights, cache, n_comp, scale);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t begin, end;
    HIP_CHECK(hipEventCreate(&begin));
    HIP_CHECK(hipEventCreate(&end));
    std::vector<float> times;
    times.reserve(iters);
    for (uint32_t i = 0; i < iters; i++) {
        HIP_CHECK(hipEventRecord(begin));
        launch(out, q, weights, cache, n_comp, scale);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(end));
        HIP_CHECK(hipEventSynchronize(end));
        float milliseconds = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
        times.push_back(milliseconds);
    }
    HIP_CHECK(hipEventDestroy(begin));
    HIP_CHECK(hipEventDestroy(end));
    return times;
}

int main(int argc, char **argv) {
    const uint32_t n_comp = argc > 1 ? (uint32_t)std::strtoul(argv[1], nullptr, 10) : 21876u;
    const uint32_t warmup = argc > 2 ? (uint32_t)std::strtoul(argv[2], nullptr, 10) : 10u;
    const uint32_t iters = argc > 3 ? (uint32_t)std::strtoul(argv[3], nullptr, 10) : 40u;
    if (!n_comp || !iters) return 2;

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, 0));
    if (properties.warpSize != 32) {
        std::fprintf(stderr, "bench-indexer: requires wavefront size 32, found %d\n", properties.warpSize);
        return 2;
    }
    std::fprintf(stderr, "bench-indexer: device=%s n_comp=%u warmup=%u iters=%u\n",
                 properties.name, n_comp, warmup, iters);

    const size_t q_count = 64u * 128u;
    const size_t cache_count = (size_t)n_comp * 128u;
    std::vector<float> q(q_count), weights(64u), cache(cache_count);
    uint64_t state = 0x243f6a8885a308d3ull;
    for (float &value : q) value = random_float(&state, 0.125f);
    for (float &value : weights) value = random_float(&state, 1.0f);
    for (float &value : cache) value = random_float(&state, 0.125f);
    const float scale = 1.0f / std::sqrt(8192.0f);

    float *d_q = nullptr, *d_weights = nullptr, *d_cache = nullptr;
    float *d_reference = nullptr, *d_candidate = nullptr;
    HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_weights, weights.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cache, cache.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_reference, (size_t)n_comp * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_candidate, (size_t)n_comp * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cache, cache.data(), cache.size() * sizeof(float), hipMemcpyHostToDevice));

    launch_reference(d_reference, d_q, d_weights, d_cache, n_comp, scale);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<uint8_t> reference((size_t)n_comp * sizeof(float));
    std::vector<uint8_t> candidate(reference.size());
    HIP_CHECK(hipMemcpy(reference.data(), d_reference, reference.size(), hipMemcpyDeviceToHost));

    struct variant { const char *name; uint32_t threads; launch_fn launch; } variants[] = {
        {"reference", 128u, launch_reference},
        {"wide-256", 256u, launch_wide<8u>},
        {"wide-512", 512u, launch_wide<16u>},
        {"wide-1024", 1024u, launch_wide<32u>},
        {"wave-32", 32u, launch_wave_rows<1u>},
        {"wave-64", 64u, launch_wave_rows<2u>},
        {"wave-128", 128u, launch_wave_rows<4u>},
        {"wave-256", 256u, launch_wave_rows<8u>},
    };
    for (const variant &item : variants) {
        item.launch(d_candidate, d_q, d_weights, d_cache, n_comp, scale);
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(candidate.data(), d_candidate, candidate.size(), hipMemcpyDeviceToHost));
        if (std::memcmp(reference.data(), candidate.data(), reference.size()) != 0) {
            uint32_t mismatches = 0;
            const uint32_t *expected = (const uint32_t *)reference.data();
            const uint32_t *actual = (const uint32_t *)candidate.data();
            for (uint32_t i = 0; i < n_comp; i++) {
                if (expected[i] != actual[i] && mismatches++ < 4u)
                    std::fprintf(stderr, "bench-indexer: %s mismatch[%u] %08x != %08x\n",
                                 item.name, i, expected[i], actual[i]);
            }
            std::fprintf(stderr, "bench-indexer: %s FAILED %u bitwise mismatches\n", item.name, mismatches);
            return 1;
        }
        const std::vector<float> times = time_variant(item.launch, d_candidate, d_q, d_weights,
                                                       d_cache, n_comp, scale, warmup, iters);
        double sum = 0.0;
        for (float value : times) sum += value;
        std::fprintf(stdout,
                     "{\"variant\":\"%s\",\"threads\":%u,\"n_comp\":%u,"
                     "\"iters\":%u,\"bit_exact\":true,\"min_ms\":%.6f,"
                     "\"median_ms\":%.6f,\"p90_ms\":%.6f,\"mean_ms\":%.6f}\n",
                     item.name, item.threads, n_comp, iters,
                     *std::min_element(times.begin(), times.end()), percentile(times, 0.5),
                     percentile(times, 0.9), sum / times.size());
    }

    HIP_CHECK(hipFree(d_candidate));
    HIP_CHECK(hipFree(d_reference));
    HIP_CHECK(hipFree(d_cache));
    HIP_CHECK(hipFree(d_weights));
    HIP_CHECK(hipFree(d_q));
    return 0;
}
