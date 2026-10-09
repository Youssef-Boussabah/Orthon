"""Private test/benchmark observations; no runtime owner or public root aliases.

Matmul and CUDA convolution callables are None in CPU-only builds. Forced
routes bypass ordinary dispatch and are diagnostic measurements only.
"""
from . import _C

memory_pool_set_enabled = getattr(_C, "_memory_pool_set_enabled", None)
memory_pool_enabled = getattr(_C, "_memory_pool_enabled", None)
memory_pool_set_max_cached_bytes = getattr(_C, "_memory_pool_set_max_cached_bytes", None)
memory_pool_max_cached_bytes = getattr(_C, "_memory_pool_max_cached_bytes", None)
memory_pool_scratch_allocate = getattr(_C, "_memory_pool_scratch_allocate", None)
memory_pool_reset_scratch = getattr(_C, "_memory_pool_reset_scratch", None)
memory_pool_scratch_stats = getattr(_C, "_memory_pool_scratch_stats", None)
memory_pool_reset_stats = getattr(_C, "_memory_pool_reset_stats", None)
memory_pool_clear_cache = getattr(_C, "_memory_pool_clear_cache", None)
memory_pool_stats = getattr(_C, "_memory_pool_stats", None)
matmul_naive = getattr(_C, "_matmul_naive", None)
matmul_custom = getattr(_C, "_matmul_tiled_custom", None)
matmul_cublas = getattr(_C, "_matmul_tiled_cublas", None)
batched_matmul_custom = getattr(_C, "_matmul_batched_tiled_custom", None)
batched_matmul_cublas = getattr(_C, "_matmul_batched_tiled_cublas", None)
matmul_dispatch_path = getattr(_C, "_matmul_dispatch_path", None)
matmul_guard_report = getattr(_C, "_matmul_cublas_guard_report", None)
conv2d_direct = getattr(_C, "_conv2d_direct", None)
conv2d_im2col = getattr(_C, "_conv2d_im2col", None)
conv2d_cuda_naive = getattr(_C, "_conv2d_cuda_naive", None)
conv2d_cuda_optimized = getattr(_C, "_conv2d_cuda_optimized", None)
conv2d_cuda_backward_input = getattr(_C, "_conv2d_cuda_backward_input", None)
conv2d_cuda_backward_input_naive = getattr(_C, "_conv2d_cuda_backward_input_naive", None)
conv2d_cuda_backward_input_optimized = getattr(_C, "_conv2d_cuda_backward_input_optimized", None)
conv2d_cuda_backward_input_optimized_v2 = getattr(_C, "_conv2d_cuda_backward_input_optimized_v2", None)
conv2d_cuda_backward_weight = getattr(_C, "_conv2d_cuda_backward_weight", None)
conv2d_cuda_backward_weight_naive = getattr(_C, "_conv2d_cuda_backward_weight_naive", None)
conv2d_cuda_backward_weight_optimized = getattr(_C, "_conv2d_cuda_backward_weight_optimized", None)
conv2d_cuda_backward_weight_optimized_v2 = getattr(_C, "_conv2d_cuda_backward_weight_optimized_v2", None)
conv2d_cuda_backward_bias = getattr(_C, "_conv2d_cuda_backward_bias", None)
conv2d_cuda_backward_bias_naive = getattr(_C, "_conv2d_cuda_backward_bias_naive", None)
conv2d_cuda_backward_bias_optimized = getattr(_C, "_conv2d_cuda_backward_bias_optimized", None)
