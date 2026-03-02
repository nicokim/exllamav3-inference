/*
 * Fused RMSNorm + Residual kernel.
 *
 * Computes: x += attn_out; y = rmsnorm(x, w)
 * in a single kernel, eliminating one kernel launch and one full read of
 * the hidden state tensor (saves ~7KB/token for hidden_size=3584).
 *
 * Two-pass algorithm:
 *   Pass 1: load x + attn_out, add, store x, accumulate sum_sq (vectorized half4)
 *   Pass 2: load x + weight, normalize, store y
 *
 * Grid: (num_rows, 1, 1)  Block: (1024, 1, 1)
 * Same pattern as upstream norm.cu but with fused residual add.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

constexpr int NUM_THREADS = 1024;
constexpr int WARPS_PER_BLOCK_NORM = NUM_THREADS / 32;

// Vectorized half4 read (aligned 64-bit load)
union half4 {
    uint2 as_uint2;
    half data[4];
};

__device__ __forceinline__ void load_half4(float4& f4, const half* addr, int idx) {
    half4 h4;
    h4.as_uint2 = *reinterpret_cast<const uint2*>(addr + idx);
    f4.x = __half2float(h4.data[0]);
    f4.y = __half2float(h4.data[1]);
    f4.z = __half2float(h4.data[2]);
    f4.w = __half2float(h4.data[3]);
}

__device__ __forceinline__ void store_half4(half* addr, int idx, const float4& f4) {
    half4 h4;
    h4.data[0] = __float2half(f4.x);
    h4.data[1] = __float2half(f4.y);
    h4.data[2] = __float2half(f4.z);
    h4.data[3] = __float2half(f4.w);
    *reinterpret_cast<uint2*>(addr + idx) = h4.as_uint2;
}

// Warp reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// Block reduction for sum using shared memory
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[WARPS_PER_BLOCK_NORM];

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0)
        shared[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < WARPS_PER_BLOCK_NORM) ? shared[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }

    __syncthreads();
    // Broadcast from thread 0
    if (threadIdx.x == 0)
        shared[0] = val;
    __syncthreads();

    return shared[0];
}


__global__ __launch_bounds__(NUM_THREADS)
void fused_rmsnorm_residual_kernel(
    half* __restrict__ x,                 // (rows, dim) — residual, modified in place
    const half* __restrict__ attn_out,    // (rows, dim) — added to x
    const half* __restrict__ weight,      // (dim,) — rmsnorm weight
    half* __restrict__ y,                 // (rows, dim) — normalized output
    const float epsilon,
    const int dim
) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const int row_offset = row * dim;

    // Number of float4 (4 halfs) per row
    const int columns = dim / 4;

    // ===== Pass 1: residual add + accumulate sum of squares =====
    float sum_sq = 0.0f;

    for (int col = t; col < columns; col += NUM_THREADS) {
        int idx = row_offset + col * 4;

        float4 x4, a4;
        load_half4(x4, x, idx);
        load_half4(a4, attn_out, idx);

        // Residual add
        x4.x += a4.x;
        x4.y += a4.y;
        x4.z += a4.z;
        x4.w += a4.w;

        // Store back to x
        store_half4(x, idx, x4);

        // Accumulate sum of squares
        sum_sq += x4.x * x4.x + x4.y * x4.y + x4.z * x4.z + x4.w * x4.w;
    }

    // Handle remaining elements (dim not multiple of 4)
    for (int d = columns * 4 + t; d < dim; d += NUM_THREADS) {
        int idx = row_offset + d;
        float xv = __half2float(x[idx]);
        float av = __half2float(attn_out[idx]);
        xv += av;
        x[idx] = __float2half(xv);
        sum_sq += xv * xv;
    }

    sum_sq = block_reduce_sum(sum_sq);

    // ===== Pass 2: normalize and apply weight =====
    float rms = rsqrtf(sum_sq / (float)dim + epsilon);

    for (int col = t; col < columns; col += NUM_THREADS) {
        int idx = row_offset + col * 4;
        int w_idx = col * 4;

        float4 x4, w4;
        load_half4(x4, x, idx);
        load_half4(w4, weight, w_idx);

        float4 y4;
        y4.x = x4.x * rms * w4.x;
        y4.y = x4.y * rms * w4.y;
        y4.z = x4.z * rms * w4.z;
        y4.w = x4.w * rms * w4.w;

        store_half4(y, idx, y4);
    }

    for (int d = columns * 4 + t; d < dim; d += NUM_THREADS) {
        int idx = row_offset + d;
        float xv = __half2float(x[idx]);
        float wv = __half2float(weight[d]);
        y[idx] = __float2half(xv * rms * wv);
    }
}


// C++ wrapper
void fused_rmsnorm_residual(
    torch::Tensor x,          // (rows, dim) fp16, modified in-place
    torch::Tensor attn_out,   // (rows, dim) fp16
    torch::Tensor weight,     // (dim,) fp16
    torch::Tensor y,          // (rows, dim) fp16 output
    float epsilon
) {
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be fp16");
    TORCH_CHECK(attn_out.dtype() == torch::kHalf, "attn_out must be fp16");
    TORCH_CHECK(weight.dtype() == torch::kHalf, "weight must be fp16");
    TORCH_CHECK(y.dtype() == torch::kHalf, "y must be fp16");

    const int rows = x.size(0);
    const int dim = x.size(1);

    dim3 grid(rows);
    dim3 block(NUM_THREADS);

    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    fused_rmsnorm_residual_kernel<<<grid, block, 0, stream>>>(
        (half*)x.data_ptr(),
        (const half*)attn_out.data_ptr(),
        (const half*)weight.data_ptr(),
        (half*)y.data_ptr(),
        epsilon,
        dim
    );
}
