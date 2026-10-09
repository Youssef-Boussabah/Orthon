import os as _os
import time as _time

import orthon as _ortho

from ._backend import Backend, _TransportFailure
from ._process_group import _TransportWork


_NCCL_INIT_TIMEOUT_SECONDS = 30.0
_NCCL_UNIQUE_ID_KEY = "nccl:unique_id"
_NCCL_FILE_STORE_ENV = "ORTHON_NCCL_STORE_DIR"
_NCCL_FILE_NAME = "nccl_unique_id.bin"


class NcclBackend(Backend):
    """Minimal single-node NCCL backend for correctness validation."""

    def __init__(self, rank: int, world_size: int, store=None, device=None):
        if not _ortho.cuda_enabled() or not _ortho.nccl_enabled():
            raise RuntimeError(
                "nccl backend requires a build with CUDA and NCCL enabled "
                "(-DWITH_CUDA=ON -DWITH_NCCL=ON)"
            )
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")

        self._rank = int(rank)
        self._world_size = int(world_size)
        self._device_index = self._resolve_device(device)
        self._condition = None if store is None else store["condition"]
        self._values = None if store is None else store["values"]
        self._file_store_dir = _os.environ.get(_NCCL_FILE_STORE_ENV)
        if store is None and not self._file_store_dir:
            raise ValueError(
                "nccl backend requires a shared multiprocessing store or "
                f"{_NCCL_FILE_STORE_ENV}"
            )
        self._closed = False
        self._failure = None

        unique_id = self._rendezvous_unique_id()
        self._comm = _ortho._Cmod._nccl_init(
            unique_id,
            self._rank,
            self._world_size,
            self._device_index,
        )

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def device_index(self):
        return self._device_index

    def all_reduce(self, tensor, op="sum", async_op=False):
        self._ensure_open()
        if op != "sum":
            raise NotImplementedError("nccl backend only supports all_reduce op='sum'")
        self._validate_tensor(tensor, "all_reduce")
        try:
            work = _TransportWork(_ortho._Cmod._nccl_all_reduce_sum_async(self._comm, tensor), self)
            if async_op:
                return work
            work.wait()
        except RuntimeError as exc:
            self._failure = str(exc)
            raise _TransportFailure(str(exc)) from exc
        return tensor

    def broadcast(self, tensor, src=0, async_op=False):
        self._ensure_open()
        if src < 0 or src >= self._world_size:
            raise ValueError("broadcast src must be in [0, world_size)")
        self._validate_tensor(tensor, "broadcast")
        try:
            work = _TransportWork(_ortho._Cmod._nccl_broadcast_async(self._comm, tensor, int(src)), self)
            if async_op:
                return work
            work.wait()
        except RuntimeError as exc:
            self._failure = str(exc)
            raise _TransportFailure(str(exc)) from exc
        return tensor

    def barrier(self):
        self._ensure_open()
        try:
            _ortho._Cmod._nccl_barrier(self._comm)
        except RuntimeError as exc:
            self._failure = str(exc)
            raise _TransportFailure(str(exc)) from exc
        return None

    def shutdown(self):
        if self._closed:
            return None
        try:
            if self._comm is not None:
                _ortho._Cmod._nccl_shutdown(self._comm)
        finally:
            self._comm = None
            self._closed = True
        return None

    def _resolve_device(self, device):
        if device is None:
            return self._rank
        if isinstance(device, int):
            if device < 0:
                raise ValueError("nccl device index must be >= 0")
            return int(device)
        if isinstance(device, str):
            if device == "cuda":
                return 0
            prefix = "cuda:"
            if device.startswith(prefix):
                index = int(device[len(prefix):])
                if index < 0:
                    raise ValueError("nccl device index must be >= 0")
                return index
        if hasattr(device, "is_cuda") and device.is_cuda():
            if device.index < 0:
                raise ValueError("nccl device index must be >= 0")
            return int(device.index)
        raise ValueError("nccl backend device must be an int, 'cuda[:N]', or CUDA Device")

    def _rendezvous_unique_id(self):
        if self._condition is None:
            return self._rendezvous_unique_id_file()
        with self._condition:
            if self._rank == 0 and _NCCL_UNIQUE_ID_KEY not in self._values:
                self._values[_NCCL_UNIQUE_ID_KEY] = _ortho._Cmod._nccl_unique_id()
                self._condition.notify_all()
            deadline = _time.monotonic() + _NCCL_INIT_TIMEOUT_SECONDS
            while _NCCL_UNIQUE_ID_KEY not in self._values:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "timed out waiting for NCCL unique id "
                        f"on rank {self._rank}/{self._world_size}"
                    )
                self._condition.wait(timeout=remaining)
            return self._values[_NCCL_UNIQUE_ID_KEY]

    def _rendezvous_unique_id_file(self):
        path = _os.path.join(self._file_store_dir, _NCCL_FILE_NAME)
        if self._rank == 0:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "wb") as handle:
                handle.write(_ortho._Cmod._nccl_unique_id())
            _os.replace(tmp_path, path)

        deadline = _time.monotonic() + _NCCL_INIT_TIMEOUT_SECONDS
        while True:
            try:
                with open(path, "rb") as handle:
                    data = handle.read()
                if data:
                    return data
            except FileNotFoundError:
                pass
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "timed out waiting for NCCL file rendezvous "
                    f"on rank {self._rank}/{self._world_size}"
                )
            _time.sleep(min(0.05, remaining))

    def _validate_tensor(self, tensor, op_name):
        if not tensor.device.is_cuda():
            raise RuntimeError(f"nccl {op_name}: tensor must be CUDA")
        if tensor.device.index != self._device_index:
            raise RuntimeError(
                f"nccl {op_name}: tensor CUDA device does not match communicator device"
            )
        if not tensor.is_contiguous:
            raise RuntimeError(f"nccl {op_name}: tensor must be contiguous")
        if tensor.dtype not in ("float32", "float16", "bfloat16"):
            raise RuntimeError(
                f"nccl {op_name}: only float32, float16, and bfloat16 tensors are supported"
            )

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("nccl distributed backend has been shut down")
        if self._failure is not None:
            raise _TransportFailure("nccl distributed backend failed: " + self._failure)

    def _validate_gradient_dtype(self, dtype):
        if dtype != "float32":
            raise RuntimeError("NCCL DDP gradient averaging requires float32 gradients")


__all__ = ["NcclBackend"]
