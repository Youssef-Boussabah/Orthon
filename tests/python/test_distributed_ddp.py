import multiprocessing as mp
import traceback

import pytest

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed import DistributedDataParallel as DDP
from orthon.distributed.local_backend import create_store
from orthon.distributed._reducer import _Reducer
from orthon.nn import Linear


def _filled_input(value, shape):
    tensor = ortho.ones(list(shape))
    tensor.fill_(float(value))
    return tensor


def _fill_model_parameters(model, rank):
    for index, parameter in enumerate(model.parameters()):
        parameter.fill_(10.0 * rank + index + 1.0)


def _parameter_values(model):
    return [parameter.tolist() for parameter in model.parameters()]


def _gradient_values(model):
    return [
        None if parameter.grad is None else parameter.grad.tolist()
        for parameter in model.parameters()
    ]


def _flatten_nested(values):
    flat = []
    for value in values:
        if isinstance(value, list):
            flat.extend(_flatten_nested(value))
        else:
            flat.append(float(value))
    return flat


def _ddp_worker(rank, world_size, store, case_name, results, failures):
    try:
        ddist.init_process_group(
            backend="local",
            rank=rank,
            world_size=world_size,
            store=store,
        )

        if case_name == "broadcast":
            model = Linear(2, 1)
            _fill_model_parameters(model, rank)
            rank0_original = _parameter_values(model) if rank == 0 else None
            ddp_model = DDP(model)
            results.put((rank, {
                "rank0_original": rank0_original,
                "after": _parameter_values(ddp_model.module),
            }))
        elif case_name == "grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(model, sync_grads=False)
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            local_grad = _gradient_values(model)
            ddp_model.all_reduce_gradients()
            averaged_grad = _gradient_values(model)
            results.put((rank, {
                "local_grad": local_grad,
                "averaged_grad": averaged_grad,
            }))
        elif case_name == "auto_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(model, sync_grads=True)
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            results.put((rank, {
                "averaged_grad": _gradient_values(model),
            }))
        elif case_name == "bucket_auto_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(model, sync_grads=True, bucket_size_mb=0.000004)
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            results.put((rank, {
                "averaged_grad": _gradient_values(model),
                "bucket_count": ddp_model._reducer.diagnostics()["buckets"],
            }))
        elif case_name == "bucket_overlap_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(
                model,
                sync_grads=True,
                bucket_size_mb=0.000004,
                overlap_grad_sync=True,
            )
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            pending_after_backward = ddp_model._reducer.diagnostics()["outstanding"]
            launch_count = ddp_model._reducer.diagnostics()["async_launches"]
            ddp_model.wait_for_gradient_sync()
            first_wait_grad = _gradient_values(model)
            ddp_model.wait_for_gradient_sync()
            results.put((rank, {
                "averaged_grad": _gradient_values(model),
                "first_wait_grad": first_wait_grad,
                "bucket_count": ddp_model._reducer.diagnostics()["buckets"],
                "launch_count": launch_count,
                "pending_after_backward": pending_after_backward,
                "pending_after_wait": ddp_model._reducer.diagnostics()["outstanding"],
            }))
        elif case_name == "bucket_overlap_manual_flush":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(
                model,
                sync_grads=True,
                bucket_size_mb=0.000004,
                overlap_grad_sync=True,
            )
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            pending_after_backward = ddp_model._reducer.diagnostics()["outstanding"]
            launch_count = ddp_model._reducer.diagnostics()["async_launches"]
            ddp_model.all_reduce_gradients()
            results.put((rank, {
                "averaged_grad": _gradient_values(model),
                "launch_count": launch_count,
                "pending_after_backward": pending_after_backward,
                "pending_after_manual_sync": ddp_model._reducer.diagnostics()["outstanding"],
            }))
        elif case_name == "bucket_overlap_zero_grad_is_safe_after_backward":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(
                model,
                sync_grads=True,
                bucket_size_mb=0.000004,
                overlap_grad_sync=True,
            )
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            pending_after_backward = ddp_model._reducer.diagnostics()["outstanding"]
            ddp_model.zero_grad()
            results.put((rank, {
                "pending_after_backward": pending_after_backward,
                "grad_after_zero": _gradient_values(model),
            }))
        elif case_name == "disabled_auto_sync":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            ddp_model = DDP(model, sync_grads=False)
            x = _filled_input(rank + 1, [1, 2])
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            results.put((rank, {
                "local_grad": _gradient_values(model),
            }))
        elif case_name == "forward":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            x = _filled_input(3.0, [1, 2])
            expected = model(x).tolist()
            ddp_model = DDP(model)
            results.put((rank, {
                "expected": expected,
                "actual": ddp_model(x).tolist(),
            }))
        else:
            raise ValueError(f"unknown DDP test case: {case_name}")
    except BaseException:
        failures.put((rank, traceback.format_exc()))
        raise
    finally:
        ddist.destroy_process_group()


def _run_ddp_case(case_name, world_size=2):
    ctx = mp.get_context("spawn")
    manager, store = create_store(ctx)
    results = ctx.Queue()
    failures = ctx.Queue()
    processes = []
    try:
        for rank in range(world_size):
            process = ctx.Process(
                target=_ddp_worker,
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


def test_ddp_constructor_broadcasts_rank_zero_parameters():
    results = _run_ddp_case("broadcast", world_size=2)
    rank0_original = results[0]["rank0_original"]
    assert rank0_original is not None
    assert results[0]["after"] == rank0_original
    assert results[1]["after"] == rank0_original


def test_ddp_all_reduce_gradients_averages_across_ranks():
    results = _run_ddp_case("grad_average", world_size=2)
    rank0_local = results[0]["local_grad"]
    rank1_local = results[1]["local_grad"]
    expected = []
    for grad0, grad1 in zip(rank0_local, rank1_local):
        expected.append([(a + b) / 2.0 for a, b in zip(grad0, grad1)])

    for result in results:
        for averaged, expected_grad in zip(result["averaged_grad"], expected):
            assert averaged == pytest.approx(expected_grad)


def test_ddp_auto_sync_averages_gradients_after_backward():
    results = _run_ddp_case("auto_grad_average", world_size=2)
    expected = [1.5, 1.5, 1.0]
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)


def test_ddp_bucketed_auto_sync_averages_gradients_after_backward():
    results = _run_ddp_case("bucket_auto_grad_average", world_size=2)
    expected = [1.5, 1.5, 1.0]
    assert all(result["bucket_count"] > 1 for result in results)
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)


def test_ddp_bucketed_overlap_wait_averages_gradients_after_backward():
    results = _run_ddp_case("bucket_overlap_grad_average", world_size=2)
    expected = [1.5, 1.5, 1.0]
    assert all(result["bucket_count"] > 1 for result in results)
    assert all(result["launch_count"] > 0 for result in results)
    assert all(result["pending_after_backward"] == 0 for result in results)
    assert all(result["pending_after_wait"] == 0 for result in results)
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)
        assert result["averaged_grad"] == result["first_wait_grad"]


def test_ddp_bucketed_overlap_manual_all_reduce_flushes_pending_work():
    results = _run_ddp_case("bucket_overlap_manual_flush", world_size=2)
    expected = [1.5, 1.5, 1.0]
    assert all(result["launch_count"] > 0 for result in results)
    assert all(result["pending_after_backward"] == 0 for result in results)
    assert all(result["pending_after_manual_sync"] == 0 for result in results)
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)


def test_ddp_bucketed_overlap_zero_grad_is_safe_after_backward():
    results = _run_ddp_case("bucket_overlap_zero_grad_is_safe_after_backward", world_size=2)
    for result in results:
        assert result["pending_after_backward"] == 0
        assert result["grad_after_zero"] == [None, None]


def test_ddp_auto_sync_can_be_disabled():
    results = _run_ddp_case("disabled_auto_sync", world_size=2)
    assert _flatten_nested(results[0]["local_grad"]) != pytest.approx(
        _flatten_nested(results[1]["local_grad"])
    )


def test_ddp_forward_delegates_to_wrapped_module():
    results = _run_ddp_case("forward", world_size=1)
    assert results[0]["actual"] == pytest.approx(results[0]["expected"])


def test_tensor_post_accumulate_grad_hook_fires_once_per_backward():
    tensor = ortho.ones([2])
    tensor.requires_grad_(True)
    calls = []
    tensor._register_post_accumulate_grad_hook(lambda: calls.append(tensor.grad.tolist()))

    loss = ortho.sum(tensor * tensor)
    loss.backward()

    assert calls == [[2.0, 2.0]]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ddp_rejects_low_precision_gradient_sync_dtype(dtype):
    grad = ortho.cast(ortho.ones([2]), dtype)
    with pytest.raises(RuntimeError, match="only supports float32/float64"):
        _Reducer.validate_gradient(grad)
