/*
 * pybind11 bindings for exllamav3_opt CUDA kernels.
 */

#include <torch/extension.h>

// FP8 cache kernels
void quant_fp8_cache_paged(
    torch::Tensor k_in,
    torch::Tensor k_out,
    torch::Tensor k_scales,
    torch::Tensor v_in,
    torch::Tensor v_out,
    torch::Tensor v_scales,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    int new_tokens
);

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
);

// Fused RMSNorm + Residual
void fused_rmsnorm_residual(
    torch::Tensor x,
    torch::Tensor attn_out,
    torch::Tensor weight,
    torch::Tensor y,
    float epsilon
);

// Fused Sampling (temperature + top-k + Gumbel + argmax)
void fused_sample(
    torch::Tensor logits,
    torch::Tensor output_id,
    float temperature,
    int top_k,
    uint32_t random_seed
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quant_fp8_cache_paged", &quant_fp8_cache_paged,
          "Quantize FP16 KV to FP8 E4M3FN paged cache");

    m.def("dequant_fp8_cache_paged", &dequant_fp8_cache_paged,
          "Dequantize FP8 E4M3FN paged cache to FP16");

    m.def("fused_rmsnorm_residual", &fused_rmsnorm_residual,
          "Fused residual add + RMSNorm: x += attn_out; y = rmsnorm(x, w)");

    m.def("fused_sample", &fused_sample,
          "Fused sampling: temperature + top-k + Gumbel noise + argmax");
}
