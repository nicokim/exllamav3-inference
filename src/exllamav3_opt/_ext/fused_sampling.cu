/*
 * Fused sampling kernel: temperature + top-k + Gumbel noise + argmax.
 *
 * Single kernel replaces 3-4 separate launches from ComboSampler:
 *   1. Temperature scaling (logits *= inv_temperature)
 *   2. Top-k filtering (find k-th largest, mask below)
 *   3. Gumbel noise (-log(-log(uniform)))
 *   4. Argmax reduction
 *
 * Grid: (1, 1, 1) — single sequence (batch=1)
 * Block: (1024) — each thread processes ~148 elements for vocab=152064
 *
 * Approach:
 *   Pass 1: Apply temperature, find top-k threshold via partial sort
 *   Pass 2: Mask below threshold, add Gumbel noise, block-reduce argmax
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#include <cfloat>

constexpr int SAMPLE_THREADS = 1024;

// Gumbel noise: -log(-log(u)) where u ~ Uniform(0,1)
__device__ __forceinline__ float gumbel(float u) {
    return -__logf(fmaxf(-__logf(fmaxf(u, 1e-20f)), 1e-20f));
}

// Value-index pair for argmax
struct ValIdx {
    float val;
    int idx;
};

// Warp-level argmax reduction
__device__ __forceinline__ ValIdx warp_reduce_argmax(ValIdx vi) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val = __shfl_xor_sync(0xffffffff, vi.val, offset);
        int other_idx = __shfl_xor_sync(0xffffffff, vi.idx, offset);
        if (other_val > vi.val) {
            vi.val = other_val;
            vi.idx = other_idx;
        }
    }
    return vi;
}

// Block-level argmax reduction
__device__ ValIdx block_reduce_argmax(ValIdx vi) {
    __shared__ float s_val[32];
    __shared__ int s_idx[32];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = SAMPLE_THREADS / 32;

    vi = warp_reduce_argmax(vi);

    if (lane_id == 0) {
        s_val[warp_id] = vi.val;
        s_idx[warp_id] = vi.idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        vi.val = (lane_id < num_warps) ? s_val[lane_id] : -FLT_MAX;
        vi.idx = (lane_id < num_warps) ? s_idx[lane_id] : 0;
        vi = warp_reduce_argmax(vi);
    }

    return vi;
}


__global__ __launch_bounds__(SAMPLE_THREADS)
void fused_sample_kernel(
    const half* __restrict__ logits,    // (1, vocab_size)
    int64_t* __restrict__ output_id,    // (1,) — sampled token ID
    const int vocab_size,
    const float inv_temperature,
    const int top_k,                    // 0 = disabled
    const uint32_t random_seed
) {
    const int t = threadIdx.x;

    // ===== Phase 1: Apply temperature and find top-k threshold =====

    // Each thread tracks its local top value for top-k
    // For top-k, we need the k-th largest value as threshold
    // Simple approach: each thread finds its local max, then we do
    // a cooperative search

    // First pass: apply temperature and find per-thread max
    ValIdx local_best = { -FLT_MAX, 0 };

    for (int i = t; i < vocab_size; i += SAMPLE_THREADS) {
        float val = __half2float(logits[i]) * inv_temperature;
        if (val > local_best.val) {
            local_best.val = val;
            local_best.idx = i;
        }
    }

    // If no top-k, skip threshold computation
    float threshold = -FLT_MAX;

    if (top_k > 0 && top_k < vocab_size) {
        // Approximate top-k: use iterative threshold estimation
        // Start from global max and binary search down

        // Get global max first
        __shared__ float s_global_max;
        ValIdx global_best = block_reduce_argmax(local_best);
        if (t == 0)
            s_global_max = global_best.val;
        __syncthreads();

        float hi = s_global_max;
        float lo = hi - 100.0f;  // reasonable range for logits

        // Binary search for threshold where count >= top_k
        for (int iter = 0; iter < 20; iter++) {
            float mid = (hi + lo) * 0.5f;

            // Count elements above mid
            int local_count = 0;
            for (int i = t; i < vocab_size; i += SAMPLE_THREADS) {
                float val = __half2float(logits[i]) * inv_temperature;
                if (val >= mid)
                    local_count++;
            }

            // Block reduction for count
            __shared__ int s_count[32];
            int warp_id = t / 32;
            int lane_id = t % 32;

            // Warp reduce
            for (int offset = 16; offset > 0; offset >>= 1)
                local_count += __shfl_xor_sync(0xffffffff, local_count, offset);

            if (lane_id == 0)
                s_count[warp_id] = local_count;
            __syncthreads();

            int total_count = 0;
            if (t == 0) {
                for (int w = 0; w < SAMPLE_THREADS / 32; w++)
                    total_count += s_count[w];
            }
            __shared__ int s_total;
            if (t == 0)
                s_total = total_count;
            __syncthreads();

            if (s_total >= top_k)
                lo = mid;
            else
                hi = mid;
        }

        threshold = lo;
    }

    // ===== Phase 2: Apply mask, Gumbel noise, argmax =====

    curandStatePhilox4_32_10_t rng;
    curand_init(random_seed, t, 0, &rng);

    ValIdx best = { -FLT_MAX, 0 };

    for (int i = t; i < vocab_size; i += SAMPLE_THREADS) {
        float val = __half2float(logits[i]) * inv_temperature;

        if (val < threshold) {
            continue;  // masked by top-k
        }

        // Add Gumbel noise for sampling
        float u = curand_uniform(&rng);
        val += gumbel(u);

        if (val > best.val) {
            best.val = val;
            best.idx = i;
        }
    }

    best = block_reduce_argmax(best);

    if (t == 0) {
        output_id[0] = best.idx;
    }
}

// Argmax-only variant (temperature=0 / greedy)
__global__ __launch_bounds__(SAMPLE_THREADS)
void argmax_kernel(
    const half* __restrict__ logits,
    int64_t* __restrict__ output_id,
    const int vocab_size
) {
    const int t = threadIdx.x;

    ValIdx best = { -FLT_MAX, 0 };
    for (int i = t; i < vocab_size; i += SAMPLE_THREADS) {
        float val = __half2float(logits[i]);
        if (val > best.val) {
            best.val = val;
            best.idx = i;
        }
    }

    best = block_reduce_argmax(best);

    if (t == 0)
        output_id[0] = best.idx;
}


// C++ wrapper
void fused_sample(
    torch::Tensor logits,       // (1, vocab_size) fp16
    torch::Tensor output_id,    // (1,) int64
    float temperature,
    int top_k,
    uint32_t random_seed
) {
    TORCH_CHECK(logits.dtype() == torch::kHalf, "logits must be fp16");
    TORCH_CHECK(output_id.dtype() == torch::kLong, "output_id must be int64");

    const int vocab_size = logits.size(-1);

    const at::cuda::OptionalCUDAGuard device_guard(logits.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (temperature == 0.0f) {
        argmax_kernel<<<1, SAMPLE_THREADS, 0, stream>>>(
            (const half*)logits.data_ptr(),
            (int64_t*)output_id.data_ptr(),
            vocab_size
        );
    } else {
        float inv_temperature = 1.0f / temperature;
        fused_sample_kernel<<<1, SAMPLE_THREADS, 0, stream>>>(
            (const half*)logits.data_ptr(),
            (int64_t*)output_id.data_ptr(),
            vocab_size,
            inv_temperature,
            top_k,
            random_seed
        );
    }
}
