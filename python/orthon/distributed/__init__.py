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


def init_process_group(backend=None, **kwargs):
    """Initialize the correctness-first DDP-lite process group.

    Supported backends are:
    - local: CPU/local multiprocessing collectives for correctness tests.
    - nccl: single-node GPU-resident collectives for contiguous CUDA
      float32/float16/bfloat16 tensors.

    Rank settings may be passed explicitly or inferred from RANK, WORLD_SIZE,
    LOCAL_RANK, LOCAL_WORLD_SIZE, and ORTHON_DISTRIBUTED_BACKEND.

    DDP synchronizes gradients automatically by default (sync_grads=True
    registers post-accumulate hooks that all-reduce after backward()).
    Construct DDP with sync_grads=False and call
    ddp_model.all_reduce_gradients() for explicit manual synchronization.
    """
    if _state.is_initialized():
        raise RuntimeError(
            "distributed process group is already initialized; "
            "call destroy_process_group() before reinitializing"
        )
    backend, kwargs = apply_env_defaults(backend, kwargs)
    factory = get_backend_factory(backend)
    instance = factory(**kwargs)
    _state.set_active(instance)
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
