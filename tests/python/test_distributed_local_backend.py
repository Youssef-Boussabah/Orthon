import multiprocessing as mp
import traceback

import numpy as np
import pytest

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed.local_backend import create_store


def _make_filled_tensor(value, shape=(4,)):
    tensor = ortho.ones(list(shape))
    tensor.fill_(float(value))
    return tensor


def _local_backend_worker(rank, world_size, store, case_name, results, failures):
    try:
        ddist.init_process_group(
            backend="local",
            rank=rank,
            world_size=world_size,
            store=store,
        )
        if case_name == "all_reduce_2":
            tensor = _make_filled_tensor(rank + 1)
            ddist.all_reduce(tensor)
            results.put((rank, tensor.tolist()))
        elif case_name == "all_reduce_4":
            tensor = _make_filled_tensor(rank + 1)
            ddist.all_reduce(tensor)
            results.put((rank, tensor.tolist()))
        elif case_name == "all_reduce_async":
            tensor = _make_filled_tensor(rank + 1)
            work = ddist.all_reduce(tensor, async_op=True)
            first_wait = work.wait()
            second_wait = work.wait()
            results.put((rank, {
                "values": tensor.tolist(),
                "first_wait": first_wait,
                "second_wait": second_wait,
                "completed": work.is_completed(),
            }))
        elif case_name == "broadcast":
            if rank == 0:
                tensor = ortho.from_numpy(np.asarray([2.0, 4.0, 8.0], dtype=np.float32))
            else:
                tensor = _make_filled_tensor(-rank, shape=(3,))
            ddist.broadcast(tensor, src=0)
            results.put((rank, tensor.tolist()))
        elif case_name == "broadcast_async":
            if rank == 0:
                tensor = ortho.from_numpy(np.asarray([2.0, 4.0, 8.0], dtype=np.float32))
            else:
                tensor = _make_filled_tensor(-rank, shape=(3,))
            work = ddist.broadcast(tensor, src=0, async_op=True)
            first_wait = work.wait()
            second_wait = work.wait()
            results.put((rank, {
                "values": tensor.tolist(),
                "first_wait": first_wait,
                "second_wait": second_wait,
                "completed": work.is_completed(),
            }))
        elif case_name == "barrier":
            ddist.barrier()
            results.put((rank, "passed"))
        else:
            raise ValueError(f"unknown distributed test case: {case_name}")
    except BaseException:
        failures.put((rank, traceback.format_exc()))
        raise
    finally:
        ddist.destroy_process_group()


def _run_local_case(case_name, world_size):
    ctx = mp.get_context("spawn")
    manager, store = create_store(ctx)
    results = ctx.Queue()
    failures = ctx.Queue()
    processes = []
    try:
        for rank in range(world_size):
            process = ctx.Process(
                target=_local_backend_worker,
                args=(rank, world_size, store, case_name, results, failures),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()

        failure_messages = []
        while not failures.empty():
            rank, message = failures.get()
            failure_messages.append(f"rank {rank} failed:\n{message}")
        exit_failures = [
            f"rank {rank} exited with {process.exitcode}"
            for rank, process in enumerate(processes)
            if process.exitcode != 0
        ]
        if failure_messages or exit_failures:
            pytest.fail("\n".join(failure_messages + exit_failures))

        collected = {}
        for _ in range(world_size):
            rank, value = results.get(timeout=1)
            collected[rank] = value
        return [collected[rank] for rank in range(world_size)]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        manager.shutdown()


def test_local_all_reduce_two_ranks_sums_ones_and_twos():
    results = _run_local_case("all_reduce_2", world_size=2)
    assert results == [[3.0, 3.0, 3.0, 3.0]] * 2


def test_local_all_reduce_four_ranks_sums_rank_values():
    results = _run_local_case("all_reduce_4", world_size=4)
    assert results == [[10.0, 10.0, 10.0, 10.0]] * 4


def test_local_all_reduce_async_wait_completes_and_is_idempotent():
    results = _run_local_case("all_reduce_async", world_size=2)
    for result in results:
        assert result["values"] == [3.0, 3.0, 3.0, 3.0]
        assert result["first_wait"] is True
        assert result["second_wait"] is True
        assert result["completed"] is True


def test_local_broadcast_copies_rank_zero_tensor():
    results = _run_local_case("broadcast", world_size=3)
    assert results == [[2.0, 4.0, 8.0]] * 3


def test_local_broadcast_async_wait_completes_and_is_idempotent():
    results = _run_local_case("broadcast_async", world_size=3)
    for result in results:
        assert result["values"] == [2.0, 4.0, 8.0]
        assert result["first_wait"] is True
        assert result["second_wait"] is True
        assert result["completed"] is True


def test_local_barrier_all_ranks_enter_and_exit():
    results = _run_local_case("barrier", world_size=4)
    assert results == ["passed"] * 4
