from abc import ABC, abstractmethod


class Backend(ABC):
    @property
    @abstractmethod
    def rank(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def world_size(self):
        raise NotImplementedError

    @abstractmethod
    def all_reduce(self, tensor, op="sum", async_op=False):
        raise NotImplementedError

    @abstractmethod
    def broadcast(self, tensor, src=0, async_op=False):
        raise NotImplementedError

    @abstractmethod
    def barrier(self):
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        raise NotImplementedError


_BACKENDS = {}


def register_backend(name, factory):
    if not isinstance(name, str) or not name:
        raise ValueError("backend name must be a non-empty string")
    if not callable(factory):
        raise TypeError("backend factory must be callable")
    _BACKENDS[name] = factory


def get_backend_factory(name):
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        raise ValueError(
            f"unknown distributed backend: {name!r}; available backends: {available}"
        ) from exc


class _SingleRankBackend(Backend):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def rank(self):
        return 0

    @property
    def world_size(self):
        return 1

    def all_reduce(self, tensor, op="sum", async_op=False):
        if async_op:
            from .work import _CompletedWork

            return _CompletedWork()
        return tensor

    def broadcast(self, tensor, src=0, async_op=False):
        if src != 0:
            raise ValueError("single-rank backend only supports src=0")
        if async_op:
            from .work import _CompletedWork

            return _CompletedWork()
        return tensor

    def barrier(self):
        return None

    def shutdown(self):
        return None


def _local_backend_factory(**kwargs):
    from .local_backend import LocalBackend

    return LocalBackend(**kwargs)


def _nccl_backend_factory(**kwargs):
    from .nccl_backend import NcclBackend

    return NcclBackend(**kwargs)


register_backend("local", _local_backend_factory)
register_backend("nccl", _nccl_backend_factory)


__all__ = [
    "Backend",
    "_SingleRankBackend",
    "_BACKENDS",
    "register_backend",
    "get_backend_factory",
]
