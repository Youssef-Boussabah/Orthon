import multiprocessing as _mp
import time as _time
import traceback as _traceback

import numpy as _np

from . import _state
from ._backend import Backend
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
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._condition = store["condition"]
        self._values = store["values"]
        self._sequence = 0
        self._closed = False

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def all_reduce(self, tensor, op="sum", async_op=False):
        self._ensure_open()
        if op != "sum":
            raise NotImplementedError("local backend only supports all_reduce op='sum'")
        tag = self._next_tag("all_reduce")
        payload = _tensor_to_payload(tensor)
        result = self._collect_payloads(tag, payload, _sum_payloads)
        _copy_payload_into_tensor(result, tensor)
        if async_op:
            return _CompletedWork()
        return tensor

    def broadcast(self, tensor, src=0, async_op=False):
        self._ensure_open()
        if src < 0 or src >= self._world_size:
            raise ValueError("broadcast src must be in [0, world_size)")
        tag = self._next_tag("broadcast")
        payload = _tensor_to_payload(tensor) if self._rank == src else None
        result = self._broadcast_payload(tag, payload, src)
        _copy_payload_into_tensor(result, tensor)
        if async_op:
            return _CompletedWork()
        return tensor

    def barrier(self):
        self._ensure_open()
        tag = self._next_tag("barrier")
        self._barrier(tag)
        return None

    def shutdown(self):
        self._closed = True
        return None

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("local distributed backend has been shut down")

    def _next_tag(self, kind):
        tag = f"{kind}:{self._sequence}"
        self._sequence += 1
        return tag

    def _collect_payloads(self, tag, payload, reducer):
        result_key = f"{tag}:result"
        values_key = f"{tag}:values"
        leave_key = f"{tag}:leave"

        with self._condition:
            entries = dict(self._values.get(values_key, {}))
            entries[self._rank] = payload
            self._values[values_key] = entries
            if len(entries) == self._world_size:
                self._values[result_key] = reducer(
                    [entries[idx] for idx in range(self._world_size)]
                )
                self._condition.notify_all()
            else:
                self._wait_for_key_locked(result_key, tag)

            result = self._values[result_key]
            self._leave_collective_locked(tag, [values_key, result_key, leave_key])
            return result

    def _broadcast_payload(self, tag, payload, src):
        result_key = f"{tag}:result"
        seen_key = f"{tag}:seen"
        leave_key = f"{tag}:leave"

        with self._condition:
            seen = dict(self._values.get(seen_key, {}))
            seen[self._rank] = True
            self._values[seen_key] = seen
            if self._rank == src:
                self._values[result_key] = payload
                self._condition.notify_all()
            else:
                self._wait_for_key_locked(result_key, tag)

            result = self._values[result_key]
            self._leave_collective_locked(tag, [seen_key, result_key, leave_key])
            return result

    def _barrier(self, tag):
        count_key = f"{tag}:count"
        done_key = f"{tag}:done"
        leave_key = f"{tag}:leave"

        with self._condition:
            count = int(self._values.get(count_key, 0)) + 1
            self._values[count_key] = count
            if count == self._world_size:
                self._values[done_key] = True
                self._condition.notify_all()
            else:
                self._wait_for_key_locked(done_key, tag)
            self._leave_collective_locked(tag, [count_key, done_key, leave_key])

    def _wait_for_key_locked(self, key, tag):
        deadline = _time.monotonic() + _COLLECTIVE_TIMEOUT_SECONDS
        while key not in self._values:
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "timed out waiting for local collective "
                    f"{tag} on rank {self._rank}/{self._world_size}; missing key {key!r}"
                )
            self._condition.wait(timeout=remaining)

    def _leave_collective_locked(self, tag, keys):
        leave_key = f"{tag}:leave"
        leave_count = int(self._values.get(leave_key, 0)) + 1
        self._values[leave_key] = leave_count
        if leave_count == self._world_size:
            for key in keys:
                self._values.pop(key, None)
            self._condition.notify_all()


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
    # The staging copy has to land on the DESTINATION tensor's exact device:
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


def _spawn_worker_entry(rank, world_size, backend_name, store, fn, args, results, errors):
    import orthon.distributed as _ddist

    try:
        kwargs = {
            "backend": backend_name,
            "rank": rank,
            "world_size": world_size,
            "store": store,
        }
        if backend_name == "nccl":
            kwargs["device"] = rank
        _ddist.init_process_group(**kwargs)
        results[rank] = fn(rank, *args)
    except BaseException:
        errors.put((rank, _traceback.format_exc()))
        raise
    finally:
        _ddist.destroy_process_group()


def spawn(fn, args=(), nprocs=2, backend="local"):
    """Run a function across local worker processes with DDP-lite initialized.

    This helper uses the local multiprocessing backend for correctness testing
    and small CPU demos. It is not an NCCL, multi-GPU, or production launcher.
    """
    if nprocs < 1:
        raise ValueError("nprocs must be >= 1")
    if _state.is_initialized():
        raise RuntimeError("cannot spawn while distributed is already initialized")

    ctx = _mp.get_context("spawn")
    manager, store = create_store(ctx)
    results = manager.list([None] * nprocs)
    errors = ctx.Queue()
    processes = []
    try:
        for rank in range(nprocs):
            process = ctx.Process(
                target=_spawn_worker_entry,
                args=(rank, nprocs, backend, store, fn, args, results, errors),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        failures = []
        while not errors.empty():
            failures.append(errors.get())
        bad_exits = [
            (rank, process.exitcode)
            for rank, process in enumerate(processes)
            if process.exitcode != 0
        ]
        if failures or bad_exits:
            details = [f"rank {rank} failed:\n{tb}" for rank, tb in failures]
            details.extend(f"rank {rank} exited with code {code}" for rank, code in bad_exits)
            raise RuntimeError("local distributed spawn failed\n" + "\n".join(details))
        return list(results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        manager.shutdown()


__all__ = [
    "LocalBackend",
    "create_store",
    "spawn",
]
