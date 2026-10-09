import multiprocessing as _mp
import time as _time
import traceback as _traceback

import numpy as _np
from . import _state
from ._backend import Backend, _TransportFailure
from .work import _CompletedWork

_COLLECTIVE_TIMEOUT_SECONDS = 30.0


class LocalBackend(Backend):
    def __init__(self, rank: int, world_size: int, store):
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must be in [0, world_size)")
        if store is None:
            raise ValueError("local backend requires a shared multiprocessing store")
        self._rank, self._world_size = int(rank), int(world_size)
        self._condition, self._values = store["condition"], store["values"]
        self._sequence, self._closed, self._failure = 0, False, None
        # Every rank advances its own generation. Old collective departure
        # cannot delete a newly initialized group's sequence-zero completion.
        with self._condition:
            key = f"generation:{rank}"
            self._generation = int(self._values.get(key, -1)) + 1
            self._values[key] = self._generation

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("local distributed backend has been shut down")
        if self._failure is not None:
            raise _TransportFailure("local distributed backend failed: " + self._failure)

    def _validate_tensor(self, tensor, reduction=False):
        if not tensor.is_contiguous:
            raise RuntimeError("local collectives require contiguous tensors")
        supported = {"float32", "float64", "float16", "int32", "int64", "uint8", "int8"}
        if not reduction:
            supported.add("bool")
        if tensor.dtype not in supported:
            raise RuntimeError(f"local collective does not support dtype {tensor.dtype}")

    def all_reduce(self, tensor, op="sum", async_op=False):
        self._ensure_open()
        if op != "sum":
            raise NotImplementedError("local backend only supports all_reduce op='sum'")
        self._validate_tensor(tensor, reduction=True)
        descriptor = ("all_reduce", tuple(tensor.shape), tensor.dtype, None)
        result = self._exchange(descriptor, _tensor_to_payload(tensor))
        _copy_payload_into_tensor(result, tensor)
        return _CompletedWork() if async_op else tensor

    def broadcast(self, tensor, src=0, async_op=False):
        self._ensure_open()
        if src < 0 or src >= self._world_size:
            raise ValueError("broadcast src must be in [0, world_size)")
        self._validate_tensor(tensor)
        descriptor = ("broadcast", tuple(tensor.shape), tensor.dtype, src)
        result = self._exchange(descriptor, _tensor_to_payload(tensor) if self.rank == src else None)
        _copy_payload_into_tensor(result, tensor)
        return _CompletedWork() if async_op else tensor

    def barrier(self):
        self._ensure_open()
        self._exchange(("barrier", (), None, None), None)

    def shutdown(self):
        self._closed = True

    def _exchange(self, descriptor, payload):
        prefix = f"group:{self._generation}"
        tag = f"{prefix}:{self._sequence}"
        self._sequence += 1
        entries_key, result_key, leave_key = (tag + suffix for suffix in (":entries", ":result", ":leave"))
        abort_key = prefix + ":abort"
        with self._condition:
            entries = dict(self._values.get(entries_key, {}))
            entries[self.rank] = (descriptor, payload)
            self._values[entries_key] = entries
            if len(entries) == self.world_size:
                try:
                    ordered = [entries[r] for r in range(self.world_size)]
                    if any(d != descriptor for d, _ in ordered):
                        raise RuntimeError(f"local collective mismatch at sequence {self._sequence - 1}: "
                                           f"{[d for d, _ in ordered]}")
                    kind, _, _, src = descriptor
                    result = (_sum_payloads([p for _, p in ordered]) if kind == "all_reduce" else
                              ordered[src][1] if kind == "broadcast" else None)
                    self._values[result_key] = result
                except Exception as exc:
                    self._values[abort_key] = str(exc)
                self._condition.notify_all()
            deadline = _time.monotonic() + _COLLECTIVE_TIMEOUT_SECONDS
            while True:
                failure = self._values.get("spawn:abort") or self._values.get(abort_key)
                if failure is not None:
                    self._failure = str(failure)
                    raise _TransportFailure(self._failure)
                if result_key in self._values:
                    break
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    self._values[abort_key] = f"timed out waiting for local collective {tag}"
                    self._condition.notify_all()
                    continue
                self._condition.wait(timeout=min(remaining, 0.1))
            result = self._values[result_key]
            leaves = int(self._values.get(leave_key, 0)) + 1
            self._values[leave_key] = leaves
            if leaves == self.world_size:
                for key in (entries_key, result_key, leave_key):
                    self._values.pop(key, None)
            return result


def _tensor_to_payload(tensor):
    # Collectives serialize live model parameters, which can require grad;
    # the host materialisation is an internal temporary, so make it an
    # explicit no_grad transfer rather than a transient ToNode.
    import orthon as _ortho
    with _ortho.no_grad():
        return (tuple(tensor.shape), str(tensor.dtype), tensor.to("cpu").tolist())


def _payload_to_numpy(payload):
    shape, dtype, flat_values = payload
    return _np.asarray(flat_values, dtype=_np.dtype(dtype)).reshape(shape)


def _sum_payloads(payloads):
    arrays = [_payload_to_numpy(payload) for payload in payloads]
    total = arrays[0].copy()
    for array in arrays[1:]:
        total += array
    return (tuple(total.shape), str(total.dtype), total.reshape(-1).tolist())


def _copy_payload_into_tensor(payload, tensor):
    import orthon as _ortho

    array = _np.ascontiguousarray(_payload_to_numpy(payload))
    source = _ortho.from_numpy(array)
    # The staging copy has to land on the destination tensor's exact device:
    # _copy_flat_range_from requires src.device == dst.device, and a bare
    # "cuda" spec resolves to cuda:0, so a collective over tensors on cuda:1
    # would fail with "copy_flat_range: device mismatch".
    if tensor.device.is_cuda():
        source = source.to(tensor.device)
    with _ortho.no_grad():
        tensor._copy_flat_range_from(0, source)


def create_store(ctx=None):
    if ctx is None:
        ctx = _mp.get_context("spawn")
    manager = ctx.Manager()
    return manager, {
        "condition": manager.Condition(),
        "values": manager.dict(),
    }


def _abort_store(store, message):
    with store["condition"]:
        store["values"]["spawn:abort"] = message
        store["condition"].notify_all()


def _spawn_worker_entry(rank, world_size, backend_name, store, fn, args, results, errors):
    import orthon.distributed as ddist
    try:
        kwargs = dict(backend=backend_name, rank=rank, world_size=world_size, store=store)
        if backend_name == "nccl":
            kwargs["device"] = rank
        ddist.init_process_group(**kwargs)
        results[rank] = fn(rank, *args)
    except BaseException:
        message = _traceback.format_exc()
        errors[rank] = message
        _abort_store(store, f"rank {rank} failed: {message}")
        raise
    finally:
        ddist.destroy_process_group()


def spawn(fn, args=(), nprocs=2, backend="local"):
    """Run local workers with bounded failure supervision and shared abort."""
    if nprocs < 1:
        raise ValueError("nprocs must be >= 1")
    if _state.is_initialized():
        raise RuntimeError("cannot spawn while distributed is already initialized")
    ctx = _mp.get_context("spawn")
    manager, store = create_store(ctx)
    processes = []
    try:
        results, errors = manager.list([None] * nprocs), manager.dict()
        for rank in range(nprocs):
            process = ctx.Process(target=_spawn_worker_entry,
                                 args=(rank, nprocs, backend, store, fn, args, results, errors))
            process.start()
            processes.append(process)
        while any(p.is_alive() for p in processes):
            if errors or any(p.exitcode not in (None, 0) for p in processes):
                break
            for process in processes:
                process.join(timeout=0.05)
        failures = dict(errors)
        bad_exits = [(r, p.exitcode) for r, p in enumerate(processes) if p.exitcode not in (None, 0)]
        if failures or bad_exits:
            details = [f"rank {r} failed:\n{tb}" for r, tb in sorted(failures.items())]
            details.extend(f"rank {r} exited with code {code}" for r, code in bad_exits)
            raise RuntimeError("local distributed spawn failed\n" + "\n".join(details))
        return list(results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        manager.shutdown()


__all__ = ["LocalBackend", "create_store", "spawn"]
