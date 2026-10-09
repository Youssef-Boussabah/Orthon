"""Current compatibility consumers and private native binding boundaries."""

NATIVE_CONSUMERS = {
    name: frozenset(paths)
    for names, paths in (
        (
            ('Storage',),
            (
                'include/orthon/core/storage.h', 'src/core/storage.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('allocator_for', 'get_allocator'),
            (
                'include/orthon/memory/allocator.h', 'src/memory/allocator.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('default_memory_pool',),
            (
                'include/orthon/memory/pool.h', 'src/memory/cpu_allocator.cpp',
                'src/memory/cuda_allocator.cpp', 'src/memory/pool.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('scratch_arena_for',),
            (
                'include/orthon/memory/scratch.h', 'src/memory/scratch.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('kernel_operand',),
            (
                'include/orthon/ops/shape_ops.h', 'src/ops/shape_ops.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('memory_pool_allocate', 'memory_pool_deallocate'),
            (
                'include/orthon/core/memory_pool.h', 'src/core/memory_pool.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('cuda_cross_entropy',),
            (
                'include/orthon/backends/cuda/cross_entropy_kernels.h',
                'src/backends/cuda/cross_entropy_kernels.cu',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('conv2d_direct', 'conv2d_im2col'),
            (
                'bindings/ops_bindings.cpp', 'include/orthon/ops/conv2d_ops.h',
                'src/ops/conv2d_ops.cpp'
            ),
        ),
        (
            (
                'cuda_batched_matmul_tiled_cublas', 'cuda_batched_matmul_tiled_custom',
                'cuda_matmul_cublas_guard_report', 'cuda_matmul_dispatch_path',
                'cuda_matmul_naive', 'cuda_matmul_tiled_cublas', 'cuda_matmul_tiled_custom'
            ),
            (
                'bindings/ops_bindings.cpp', 'include/orthon/backends/cuda/matmul_kernels.h',
                'src/diagnostics/matmul.cpp'
            ),
        ),
        (
            (
                'cuda_conv2d_backward_bias', 'cuda_conv2d_backward_bias_naive',
                'cuda_conv2d_backward_bias_optimized', 'cuda_conv2d_backward_input',
                'cuda_conv2d_backward_input_naive', 'cuda_conv2d_backward_input_optimized',
                'cuda_conv2d_backward_input_optimized_v2', 'cuda_conv2d_backward_weight',
                'cuda_conv2d_backward_weight_naive', 'cuda_conv2d_backward_weight_optimized',
                'cuda_conv2d_backward_weight_optimized_v2', 'cuda_conv2d_forward_naive',
                'cuda_conv2d_forward_optimized'
            ),
            (
                'bindings/ops_bindings.cpp', 'include/orthon/backends/cuda/conv2d_kernels.h',
                'src/backends/cuda/conv2d_cuda.cu'
            ),
        ),
        (
            (
                'memory_pool_set_enabled', 'memory_pool_enabled',
                'memory_pool_set_max_cached_bytes', 'memory_pool_max_cached_bytes',
                'memory_pool_stats', 'memory_pool_reset_stats', 'memory_pool_clear_cache',
                'memory_pool_scratch_allocate', 'memory_pool_reset_scratch',
                'memory_pool_scratch_stats'
            ),
            (
                'bindings/module.cpp', 'include/orthon/core/memory_pool.h',
                'src/core/memory_pool.cpp', 'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('legacy_allocate', 'legacy_reset'),
            (
                'include/orthon/memory/scratch.h', 'src/core/memory_pool.cpp',
                'src/memory/scratch.cpp', 'tests/cpp/test_allocator.cpp'
            ),
        ),
        (
            (
                'cuda_matmul_tiled', 'cuda_matmul_uses_cublas', 'cuda_matmul_cublas_elapsed_ms',
                'cuda_matmul_cublas_elapsed_ms_execute', 'cuda_matmul_cublas_profile_report',
                'cuda_matmul_cublas_profile_report_execute', 'Conv2DProfileStats',
                'conv2d_profile_enabled', 'conv2d_profile_now', 'conv2d_profile_elapsed',
                'conv2d_profile_key', 'conv2d_profile_record', 'cuda_conv2d_profile_reset',
                'cuda_conv2d_profile_summary'
            ),
            (),
        ),
    )
    for name in names
}

PYTHON_CONSUMERS = {
    name: frozenset(paths)
    for names, paths in (
        (
            ('cuda_synchronize',),
            ('python/orthon/__init__.py',),
        ),
        (
            ('_amp_copy_master_to_model_', '_amp_unscale_to_float32_and_check'),
            ('python/orthon/__init__.py', 'python/orthon/amp/optimizer.py'),
        ),
        (
            (
                '_copy_data_', '_rebind_model_value_', '_parameters', '_buffers',
                '_non_persistent_buffers', '_state_entries', '_data', '_set_member'
            ),
            ('python/orthon/nn/module.py',),
        ),
        (
            ('_copy_flat_range_from',),
            (
                'python/orthon/distributed/_reducer.py',
                'python/orthon/distributed/local_backend.py'
            ),
        ),
        (
            ('_copy_flat_range_to', '_impl_id', '_storage_version'),
            ('python/orthon/distributed/_reducer.py',),
        ),
        (
            ('_register_post_accumulate_grad_hook', '_register_backward_finalizer'),
            ('python/orthon/distributed/ddp.py',),
        ),
        (
            ('_tensor_to_bytes', '_tensor_from_bytes'),
            ('python/orthon/__init__.py', 'python/orthon/_serialization/tensor.py'),
        ),
        (
            (
                '_nccl_all_reduce_sum', '_nccl_broadcast', '_has_grad_fn', '_matmul_tiled',
                '_matmul_uses_cublas', '_matmul_cublas_elapsed_ms',
                '_matmul_cublas_profile_report', '_conv2d_cuda_profile_reset',
                '_conv2d_cuda_profile_summary'
            ),
            (),
        ),
        (
            (
                '_nccl_all_reduce_sum_async', '_nccl_barrier', '_nccl_broadcast_async',
                '_nccl_init', '_nccl_shutdown', '_nccl_unique_id'
            ),
            ('python/orthon/distributed/nccl_backend.py',),
        ),
        (
            (
                '_matmul_batched_tiled_cublas', '_matmul_batched_tiled_custom',
                '_matmul_cublas_guard_report', '_matmul_dispatch_path', '_matmul_naive',
                '_matmul_tiled_cublas', '_matmul_tiled_custom', '_conv2d_cuda_backward_bias',
                '_conv2d_cuda_backward_bias_naive', '_conv2d_cuda_backward_bias_optimized',
                '_conv2d_cuda_backward_input', '_conv2d_cuda_backward_input_naive',
                '_conv2d_cuda_backward_input_optimized',
                '_conv2d_cuda_backward_input_optimized_v2', '_conv2d_cuda_backward_weight',
                '_conv2d_cuda_backward_weight_naive', '_conv2d_cuda_backward_weight_optimized',
                '_conv2d_cuda_backward_weight_optimized_v2', '_conv2d_cuda_naive',
                '_conv2d_cuda_optimized', '_conv2d_direct', '_conv2d_im2col',
                '_memory_pool_clear_cache', '_memory_pool_enabled',
                '_memory_pool_max_cached_bytes', '_memory_pool_reset_scratch',
                '_memory_pool_reset_stats', '_memory_pool_scratch_allocate',
                '_memory_pool_scratch_stats', '_memory_pool_set_enabled',
                '_memory_pool_set_max_cached_bytes', '_memory_pool_stats'
            ),
            ('python/orthon/_diagnostics.py',),
        ),
        (
            ('_modules',),
            ('python/orthon/nn/module.py', 'python/orthon/nn/sequential.py'),
        ),
        (
            ('_bindings', '_capture_bindings'),
            ('python/orthon/optim/adam.py', 'python/orthon/optim/sgd.py'),
        ),
        (
            (
                '_rebind_parameters', '_states', '_inner', '_all_reduce_master_grads',
                '_reduce_master_gradients'
            ),
            ('python/orthon/amp/optimizer.py',),
        ),
        (
            ('_t',),
            ('python/orthon/optim/adam.py',),
        ),
        (
            ('_step_prepared',),
            ('python/orthon/amp/grad_scaler.py', 'python/orthon/amp/optimizer.py'),
        ),
        (
            ('_unscale_and_check', '_amp_collective_device'),
            ('python/orthon/amp/grad_scaler.py',),
        ),
        (
            ('_prepare_load_state_dict', '_commit_prepared'),
            (
                'python/orthon/_serialization/transaction.py', 'python/orthon/amp/optimizer.py',
                'python/orthon/nn/module.py', 'python/orthon/optim/adam.py',
                'python/orthon/optim/sgd.py'
            ),
        ),
        (
            ('_ProcessGroup',),
            ('python/orthon/distributed/__init__.py',),
        ),
        (
            ('_Reducer',),
            ('python/orthon/distributed/_reducer.py', 'python/orthon/distributed/ddp.py'),
        ),
    )
    for name in names
}

SIGNATURE_CONSUMERS = {
    name: frozenset(paths)
    for names, paths in (
        (
            ('CPUAllocator:0',),
            (
                'include/orthon/memory/cpu_allocator.h', 'src/memory/cpu_allocator.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('CUDAAllocator:1',),
            (
                'include/orthon/memory/cuda_allocator.h', 'src/memory/cuda_allocator.cpp',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('cuda_add:2', 'cuda_mul:2'),
            (
                'include/orthon/backends/cuda/binary_kernels.h',
                'src/backends/cuda/binary_kernels.cu',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
        (
            ('cuda_sum:4', 'cuda_mean:4', 'cuda_max_with_indices:4', 'cuda_max_scatter_grad:6'),
            (
                'include/orthon/backends/cuda/reduce_kernels.h',
                'src/backends/cuda/reduce_kernels.cu',
                'tests/cpp/test_compatibility_surface.cpp'
            ),
        ),
    )
    for name in names
}

PRIVATE_BINDINGS = frozenset(
    {
        '_amp_copy_master_to_model_',
        '_amp_unscale_to_float32_and_check',
        '_conv2d_cuda_backward_bias',
        '_conv2d_cuda_backward_bias_naive',
        '_conv2d_cuda_backward_bias_optimized',
        '_conv2d_cuda_backward_input',
        '_conv2d_cuda_backward_input_naive',
        '_conv2d_cuda_backward_input_optimized',
        '_conv2d_cuda_backward_input_optimized_v2',
        '_conv2d_cuda_backward_weight',
        '_conv2d_cuda_backward_weight_naive',
        '_conv2d_cuda_backward_weight_optimized',
        '_conv2d_cuda_backward_weight_optimized_v2',
        '_conv2d_cuda_naive',
        '_conv2d_cuda_optimized',
        '_conv2d_direct',
        '_conv2d_im2col',
        '_copy_data_',
        '_copy_flat_range_from',
        '_copy_flat_range_to',
        '_has_grad_fn',
        '_impl_id',
        '_matmul_batched_tiled_cublas',
        '_matmul_batched_tiled_custom',
        '_matmul_cublas_guard_report',
        '_matmul_dispatch_path',
        '_matmul_naive',
        '_matmul_tiled_cublas',
        '_matmul_tiled_custom',
        '_memory_pool_clear_cache',
        '_memory_pool_enabled',
        '_memory_pool_max_cached_bytes',
        '_memory_pool_reset_scratch',
        '_memory_pool_reset_stats',
        '_memory_pool_scratch_allocate',
        '_memory_pool_scratch_stats',
        '_memory_pool_set_enabled',
        '_memory_pool_set_max_cached_bytes',
        '_memory_pool_stats',
        '_model_value_reference',
        '_nccl_all_reduce_sum',
        '_nccl_all_reduce_sum_async',
        '_nccl_barrier',
        '_nccl_broadcast',
        '_nccl_broadcast_async',
        '_nccl_init',
        '_nccl_shutdown',
        '_nccl_unique_id',
        '_rebind_model_value_',
        '_register_backward_finalizer',
        '_register_post_accumulate_grad_hook',
        '_storage_version',
        '_tensor_from_bytes',
        '_tensor_to_bytes',
    }
)

RETIRED_BINDINGS = frozenset(
    {
        '_conv2d_cuda_profile_reset',
        '_conv2d_cuda_profile_summary',
        '_matmul_cublas_elapsed_ms',
        '_matmul_cublas_profile_report',
        '_matmul_tiled',
        '_matmul_uses_cublas',
    }
)
