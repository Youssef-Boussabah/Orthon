import os as _os
from contextlib import contextmanager as _contextmanager

# Windows: Python 3.8+ no longer searches PATH for DLL dependencies of
# extension modules. Register the CUDA runtime directory explicitly so
# cudart64_NN.dll is found when the CUDA build of _C is loaded.
if _os.name == 'nt':
    _cuda_path = _os.environ.get('CUDA_PATH', '')
    if _cuda_path:
        for _cuda_bin in (
            _os.path.join(_cuda_path, 'bin'),
            _os.path.join(_cuda_path, 'bin', 'x64'),
        ):
            if _os.path.isdir(_cuda_bin):
                _os.add_dll_directory(_cuda_bin)

from ._C import (  # noqa: F401
    DType,
    DeviceType,
    Device,
    Tensor,
    isfinite_all,
    empty,
    zeros,
    ones,
    full,
    from_numpy,
    fill_,
    add,
    mul,
    sub,
    div,
    _scalar_like,
    reshape,
    transpose,
    contiguous,
    sum,
    mean,
    max,
    argmax,
    neg,
    relu,
    sigmoid,
    tanh,
    leaky_relu,
    gelu,
    exp,
    dropout,
    embedding,
    sqrt,
    reciprocal,
    to,
    cast,
    matmul,
    fused_linear_relu,
    cuda_enabled,
    cuda_synchronize,
    is_grad_enabled,
    set_grad_enabled,
    softmax,
    log_softmax,
    cross_entropy,
    nll_loss,
    conv2d,
    maxpool2d,
    _memory_pool_set_enabled,
    _memory_pool_enabled,
    _memory_pool_set_max_cached_bytes,
    _memory_pool_max_cached_bytes,
    _memory_pool_scratch_allocate,
    _memory_pool_reset_scratch,
    _memory_pool_scratch_stats,
    _memory_pool_reset_stats,
    _memory_pool_clear_cache,
    _memory_pool_stats,
    _amp_unscale_to_float32_and_check,
    _amp_copy_master_to_model_,
    _tensor_to_bytes,
    _tensor_from_bytes,
)


def flatten(tensor, start_dim=0, end_dim=-1):
    return tensor.flatten(start_dim, end_dim)


def layer_norm(tensor, normalized_shape, weight=None, bias=None, eps=1e-5):
    from .functional import layer_norm as _layer_norm
    return _layer_norm(
        tensor,
        normalized_shape,
        weight=weight,
        bias=bias,
        eps=eps,
    )


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0, training=True):
    from .functional import scaled_dot_product_attention as _attention
    return _attention(
        q,
        k,
        v,
        mask=mask,
        dropout_p=dropout_p,
        training=training,
    )


@_contextmanager
def no_grad():
    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)

# CUDA-only benchmark helpers (None in CPU-only builds).
from . import _C as _Cmod  # noqa: F401
nccl_enabled = getattr(_Cmod, 'nccl_enabled', lambda: False)
_matmul_naive = getattr(_Cmod, '_matmul_naive', None)
_matmul_tiled = getattr(_Cmod, '_matmul_tiled', None)
_matmul_tiled_custom = getattr(_Cmod, '_matmul_tiled_custom', None)
_matmul_tiled_cublas = getattr(_Cmod, '_matmul_tiled_cublas', None)
_matmul_batched_tiled_custom = getattr(_Cmod, '_matmul_batched_tiled_custom', None)
_matmul_batched_tiled_cublas = getattr(_Cmod, '_matmul_batched_tiled_cublas', None)
_matmul_uses_cublas = getattr(_Cmod, '_matmul_uses_cublas', None)
_matmul_dispatch_path = getattr(_Cmod, '_matmul_dispatch_path', None)
_matmul_cublas_guard_report = getattr(_Cmod, '_matmul_cublas_guard_report', None)
_matmul_cublas_elapsed_ms = getattr(_Cmod, '_matmul_cublas_elapsed_ms', None)
_matmul_cublas_profile_report = getattr(_Cmod, '_matmul_cublas_profile_report', None)
cuda_conv2d_profile_reset = getattr(_Cmod, '_conv2d_cuda_profile_reset', None)
cuda_conv2d_profile_summary = getattr(_Cmod, '_conv2d_cuda_profile_summary', None)

from ._random import manual_seed  # noqa: F401
from . import nn, optim, training, amp  # noqa: F401
# All four file APIs come from the ONE codec in checkpoint.py. There is no
# second serialization path to fall out of step with it.
from .checkpoint import (  # noqa: F401
    save_state_dict,
    load_state_dict,
    save_checkpoint,
    load_checkpoint,
)
