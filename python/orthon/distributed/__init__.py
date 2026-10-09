from ._backend import (
    Backend,
    _SingleRankBackend,
    get_backend_factory,
    register_backend,
)
from . import _state
from ._env import apply_env_defaults, rank_config_from_env
from .local_backend import spawn
from .work import Work
from ._process_group import _ProcessGroup


def init_process_group(backend=None, **kwargs):
    """Initialize a single-node process group using local or NCCL transport.

    Rank settings come from arguments or launcher environment variables.
    NCCL supports contiguous CUDA Float32/Float16/BFloat16 tensors.
    DDP reduces gradients before backward returns; sync_grads=False selects
    explicit all_reduce_gradients() calls. See docs/distributed.md."""
    if _state.is_initialized():
        raise RuntimeError(
            "distributed process group is already initialized; "
            "call destroy_process_group() before reinitializing"
        )
    backend, kwargs = apply_env_defaults(backend, kwargs)
    factory = get_backend_factory(backend)
    instance = factory(**kwargs)
    _state.set_active(_ProcessGroup(instance))
    return instance


def destroy_process_group():
    if not _state.is_initialized():
        return None
    backend = _state.get_active()
    try:
        backend.shutdown()
    finally:
        _state.clear_active()
    return None


def is_initialized():
    return _state.is_initialized()


def get_rank():
    return _state.get_active().rank


def get_world_size():
    return _state.get_active().world_size


def barrier():
    return _state.get_active().barrier()


def all_reduce(tensor, op="sum", async_op=False):
    return _state.get_active().all_reduce(tensor, op=op, async_op=async_op)


def broadcast(tensor, src=0, async_op=False):
    return _state.get_active().broadcast(tensor, src=src, async_op=async_op)


from .ddp import DDP, DistributedDataParallel


__all__ = [
    "Backend",
    "_SingleRankBackend",
    "register_backend",
    "rank_config_from_env",
    "DistributedDataParallel",
    "DDP",
    "init_process_group",
    "destroy_process_group",
    "is_initialized",
    "get_rank",
    "get_world_size",
    "barrier",
    "all_reduce",
    "broadcast",
    "spawn",
    "Work",
]
