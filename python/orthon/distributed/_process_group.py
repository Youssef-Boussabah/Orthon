from ._backend import _TransportFailure
from .work import Work


class _TransportWork(Work):
    def __init__(self, work, group):
        self._work = work
        self._group = group

    def _call(self, name):
        try:
            return getattr(self._work, name)()
        except Exception as exc:
            self._group._failure = str(exc)
            raise

    def wait(self):
        return self._call("wait")

    def is_completed(self):
        return self._call("is_completed")


class _ProcessGroup:
    """Captured lifetime and ordering boundary for one backend instance."""
    def __init__(self, backend):
        self._backend = backend
        self.rank = backend.rank
        self.world_size = backend.world_size
        self._closed = False
        self._failure = None

    def ensure_open(self):
        if self._closed:
            raise RuntimeError("distributed process group has been shut down")
        if self._failure is not None:
            raise RuntimeError("distributed process group failed: " + self._failure)

    def _call(self, name, *args, **kwargs):
        self.ensure_open()
        try:
            result = getattr(self._backend, name)(*args, **kwargs)
        except _TransportFailure as exc:
            self._failure = str(exc)
            raise
        if kwargs.get("async_op", False):
            return _TransportWork(result, self)
        return result

    def all_reduce(self, tensor, op="sum", async_op=False):
        return self._call("all_reduce", tensor, op=op, async_op=async_op)

    def validate_gradient_dtype(self, dtype):
        validate = getattr(self._backend, "_validate_gradient_dtype", None)
        if validate is not None:
            validate(dtype)

    def broadcast(self, tensor, src=0, async_op=False):
        return self._call("broadcast", tensor, src=src, async_op=async_op)

    def barrier(self):
        return self._call("barrier")

    def shutdown(self):
        if self._closed:
            return None
        self._closed = True
        return self._backend.shutdown()
