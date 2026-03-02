/*
 * FP8 E4M3FN paged KV cache kernels.
 *
 * Quantization: FP16 -> FP8 E4M3FN with per-head-per-token scaling.
 *   scale = absmax(head_values) / 448.0  (448 = max representable in E4M3FN)
 *   fp8_val = fp16_val / scale
 *
 * Dequantization: FP8 -> FP16
 *   fp16_val = fp8_val * scale
 *
 * Memory layout:
 *   K/V storage: (num_pages, PAGE_SIZE, num_kv_heads, head_dim) as float8_e4m3fn
 *   Scales:      (num_pages, PAGE_SIZE, num_kv_heads)           as float16
 *
 * Grid for quant:
 *   blockIdx.x = head index / warps_per_block
 *   blockIdx.y = token offset within new tokens
 *   blockIdx.z = batch index
 *   blockDim.x = 128 (4 warps)
 *
 * Each warp handles one KV head: reads head_dim values, computes absmax,
 * computes scale, quantizes to FP8.
 */

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int FP8_PAGE_SIZE = 256;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE; // 128


// ===== Quantization kernel: FP16 -> FP8 =====

__global__ void quant_fp8_cache_paged_kernel(
    const half* __restrict__ k_in,          // (bsz, new_tokens, num_heads, head_dim)
    __nv_fp8_e4m3* __restrict__ k_out,      // (num_pages, PAGE_SIZE, num_heads, head_dim)
    half* __restrict__ k_scales,            // (num_pages, PAGE_SIZE, num_heads)
    const half* __restrict__ v_in,
    __nv_fp8_e4m3* __restrict__ v_out,
    half* __restrict__ v_scales,
    const int32_t* __restrict__ cache_seqlens,  // (bsz,) — position BEFORE these new tokens
    const int32_t* __restrict__ block_table,    // (bsz, max_pages_per_seq)
    const int max_pages_per_seq,
    const int num_heads,
    const int head_dim,
    const int new_tokens                    // number of new tokens per sequence
) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int head_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    const int token_offset = blockIdx.y;
    const int batch_idx = blockIdx.z;

    if (head_idx >= num_heads || token_offset >= new_tokens)
        return;

    // Absolute token position in cache
    const int seq_pos = cache_seqlens[batch_idx] + token_offset;
    const int page_idx = seq_pos / FP8_PAGE_SIZE;
    const int page_offset = seq_pos % FP8_PAGE_SIZE;
    const int physical_page = block_table[batch_idx * max_pages_per_seq + page_idx];

    // Input offset: (batch, token, head, dim)
    const int in_offset = ((batch_idx * new_tokens + token_offset) * num_heads + head_idx) * head_dim;

    // Output offset in paged storage: (physical_page, page_offset, head, dim)
    const int out_offset = ((physical_page * FP8_PAGE_SIZE + page_offset) * num_heads + head_idx) * head_dim;

    // Scale offset: (physical_page, page_offset, head)
    const int scale_offset = (physical_page * FP8_PAGE_SIZE + page_offset) * num_heads + head_idx;

    // --- Process K ---
    {
        // Pass 1: find absmax across head_dim (warp reduction)
        float local_max = 0.0f;
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float val = __half2float(k_in[in_offset + d]);
            local_max = fmaxf(local_max, fabsf(val));
        }

        // Warp-level reduction for absmax
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));

        float scale = local_max / FP8_E4M3_MAX;
        scale = fmaxf(scale, 1e-12f);  // avoid division by zero
        float inv_scale = 1.0f / scale;

        // Store scale
        if (lane_id == 0)
            k_scales[scale_offset] = __float2half(scale);

        // Pass 2: quantize
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float val = __half2float(k_in[in_offset + d]) * inv_scale;
            // Clamp to FP8 E4M3 range
            val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            k_out[out_offset + d] = __nv_fp8_e4m3(val);
        }
    }

    // --- Process V (same pattern) ---
    {
        float local_max = 0.0f;
        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float val = __half2float(v_in[in_offset + d]);
            local_max = fmaxf(local_max, fabsf(val));
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));

        float scale = local_max / FP8_E4M3_MAX;
        scale = fmaxf(scale, 1e-12f);
        float inv_scale = 1.0f / scale;

        if (lane_id == 0)
            v_scales[scale_offset] = __float2half(scale);

        for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
            float val = __half2float(v_in[in_offset + d]) * inv_scale;
            val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            v_out[out_offset + d] = __nv_fp8_e4m3(val);
        }
    }
}


// ===== Dequantization kernel: FP8 -> FP16 =====

__global__ void dequant_fp8_cache_paged_kernel(
    const __nv_fp8_e4m3* __restrict__ k_in, // (num_pages, PAGE_SIZE, num_heads, head_dim)
    const half* __restrict__ k_scales,       // (num_pages, PAGE_SIZE, num_heads)
    half* __restrict__ k_out,                // (num_pages, PAGE_SIZE, num_heads, head_dim)
    const __nv_fp8_e4m3* __restrict__ v_in,
    const half* __restrict__ v_scales,
    half* __restrict__ v_out,
    const int32_t* __restrict__ cache_seqlens,  // (bsz,)
    const int32_t* __restrict__ block_table,    // (bsz, max_pages_per_seq)
    const int max_pages_per_seq,
    const int num_heads,
    const int head_dim,
    const int page_size
) {
    // Grid: (num_blocks_per_token, total_tokens_across_batch, 1)
    // Each block handles a portion of heads for one token

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int head_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    if (head_idx >= num_heads)
        return;

    // Determine which batch and token this block processes
    // blockIdx.y iterates over all tokens across all sequences
    int remaining = blockIdx.y;
    int batch_idx = 0;

    // Linear scan to find batch (for batch=1, this is trivial)
    // For single-user this is always batch_idx=0
    int total_tokens = cache_seqlens[0];

    // Token position within this sequence
    int token_pos = remaining;
    if (token_pos >= total_tokens)
        return;

    int page_idx = token_pos / page_size;
    int page_offset = token_pos % page_size;
    int physical_page = block_table[batch_idx * max_pages_per_seq + page_idx];

    int paged_offset = ((physical_page * page_size + page_offset) * num_heads + head_idx) * head_dim;
    int scale_offset = (physical_page * page_size + page_offset) * num_heads + head_idx;

    // Dequantize K
    float k_scale = __half2float(k_scales[scale_offset]);
    for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
        float val = float(k_in[paged_offset + d]) * k_scale;
        k_out[paged_offset + d] = __float2half(val);
    }

    // Dequantize V
    float v_scale = __half2float(v_scales[scale_offset]);
    for (int d = lane_id; d < head_dim; d += WARP_SIZE) {
        float val = float(v_in[paged_offset + d]) * v_scale;
        v_out[paged_offset + d] = __float2half(val);
    }
}


// ===== C++ wrappers for pybind11 =====

void quant_fp8_cache_paged(
    torch::Tensor k_in,         // (bsz, new_tokens, num_heads, head_dim) fp16
    torch::Tensor k_out,        // paged fp8 storage
    torch::Tensor k_scales,     // paged fp16 scales
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    torch::Tensor cache_seqlens, // (bsz,) int32
    torch::Tensor block_table,   // (bsz, max_pages) int32
    int new_tokens
) {
    const int bsz = k_in.size(0);
    const int num_heads = k_in.size(2);
    const int head_dim = k_in.size(3);
    const int max_pages_per_seq = block_table.size(1);

    dim3 grid(
        (num_heads + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        new_tokens,
        bsz
    );
    dim3 block(BLOCK_SIZE);

    const c10::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    quant_fp8_cache_paged_kernel<<<grid, block, 0, stream>>>(
        (const half*)k_in.data_ptr(),
        (__nv_fp8_e4m3*)k_out.data_ptr(),
        (half*)k_scales.data_ptr(),
        (const half*)v_in.data_ptr(),
        (__nv_fp8_e4m3*)v_out.data_ptr(),
        (half*)v_scales.data_ptr(),
        (const int32_t*)cache_seqlens.data_ptr(),
        (const int32_t*)block_table.data_ptr(),
        max_pages_per_seq,
        num_heads,
        head_dim,
        new_tokens
    );
}

void dequant_fp8_cache_paged(
    torch::Tensor k_in,
    torch::Tensor k_scales,
    torch::Tensor k_out,
    torch::Tensor v_in,
    torch::Tensor v_scales,
    torch::Tensor v_out,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int page_size
) {
    const int num_heads = k_out.sizes()[2];
    const int head_dim = k_out.sizes()[3];
    const int max_pages_per_seq = block_table.size(1);

    // For batch=1, total_tokens = cache_seqlens[0]
    // We'll launch enough threads for max possible tokens
    int max_tokens = k_out.size(0) * k_out.size(1); // num_pages * page_size

    dim3 grid(
        (num_heads + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK,
        max_tokens,
        1
    );
    dim3 block(BLOCK_SIZE);

    const c10::cuda::OptionalCUDAGuard device_guard(k_in.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    dequant_fp8_cache_paged_kernel<<<grid, block, 0, stream>>>(
        (const __nv_fp8_e4m3*)k_in.data_ptr(),
        (const half*)k_scales.data_ptr(),
        (half*)k_out.data_ptr(),
        (const __nv_fp8_e4m3*)v_in.data_ptr(),
        (const half*)v_scales.data_ptr(),
        (half*)v_out.data_ptr(),
        (const int32_t*)cache_seqlens.data_ptr(),
        (const int32_t*)block_table.data_ptr(),
        max_pages_per_seq,
        num_heads,
        head_dim,
        page_size
    );
}
