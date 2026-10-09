import multiprocessing as mp
import subprocess
import sys
import textwrap
import traceback

import pytest

import orthon as ortho
import orthon.distributed as ddist
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.distributed import DistributedDataParallel as DDP
from orthon.distributed.local_backend import create_store
from orthon.nn import Linear
from orthon.optim import SGD


def _visible_gpu_count():
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
    except Exception:
        return 0
    if result.returncode != 0:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.startswith("GPU "))


def _nccl_available(min_gpus=2):
    if not ortho.cuda_enabled() or not ortho.nccl_enabled():
        return False
    if _visible_gpu_count() < min_gpus:
        return False
    try:
        for index in range(min_gpus):
            ortho.zeros([1]).to(f"cuda:{index}")
        return True
    except Exception:
        return False


requires_nccl_2gpu = pytest.mark.skipif(
    not _nccl_available(),
    reason="CUDA+NCCL build with at least 2 GPUs is required",
)

requires_nccl_4gpu = pytest.mark.skipif(
    not _nccl_available(min_gpus=4),
    reason="CUDA+NCCL build with at least 4 GPUs is required",
)

requires_nccl_8gpu = pytest.mark.skipif(
    not _nccl_available(min_gpus=8),
    reason="CUDA+NCCL build with at least 8 GPUs is required",
)


def _tensor_on_rank(rank, value, shape=(4,), dtype="float32"):
    tensor = ortho.full(list(shape), float(value), dtype=dtype).to(f"cuda:{rank}")
    return tensor


def _grad_values(model):
    return [
        None if parameter.grad is None else parameter.grad.to("cpu").tolist()
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


def _flatten_parameter_values(values):
    return _flatten_nested(values)


def _expect_runtime_error(fn):
    try:
        fn()
    except (RuntimeError, NotImplementedError) as exc:
        return str(exc)
    raise AssertionError("expected NCCL operation to fail")


def _nccl_worker(rank, world_size, store, case_name, results, failures):
    try:
        ddist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            store=store,
            device=rank,
        )

        if case_name == "init":
            ddist.barrier()
            results.put((rank, {"rank": ddist.get_rank(), "world_size": ddist.get_world_size()}))
        elif case_name == "broadcast":
            tensor = _tensor_on_rank(rank, 7.0 if rank == 0 else -1.0, shape=(3,))
            ddist.broadcast(tensor, src=0)
            results.put((rank, tensor.to("cpu").tolist()))
        elif case_name == "broadcast_float16":
            tensor = _tensor_on_rank(
                rank,
                7.0 if rank == 0 else -1.0,
                shape=(3,),
                dtype="float16",
            )
            ddist.broadcast(tensor, src=0)
            results.put((rank, tensor.to("cpu").tolist()))
        elif case_name == "broadcast_bfloat16":
            tensor = _tensor_on_rank(
                rank,
                7.0 if rank == 0 else -1.0,
                shape=(3,),
                dtype="bfloat16",
            )
            ddist.broadcast(tensor, src=0)
            results.put((rank, tensor.to("cpu").tolist()))
        elif case_name == "broadcast_async":
            tensor = _tensor_on_rank(rank, 7.0 if rank == 0 else -1.0, shape=(3,))
            work = ddist.broadcast(tensor, src=0, async_op=True)
            before_wait_completed = work.is_completed()
            first_wait = work.wait()
            second_wait = work.wait()
            results.put((rank, {
                "values": tensor.to("cpu").tolist(),
                "before_wait_completed": before_wait_completed,
                "first_wait": first_wait,
                "second_wait": second_wait,
                "after_wait_completed": work.is_completed(),
            }))
        elif case_name == "all_reduce":
            tensor = _tensor_on_rank(rank, rank + 1, shape=(4,))
            ddist.all_reduce(tensor, op="sum")
            results.put((rank, tensor.to("cpu").tolist()))
        elif case_name == "all_reduce_async":
            tensor = _tensor_on_rank(rank, rank + 1, shape=(4,))
            work = ddist.all_reduce(tensor, op="sum", async_op=True)
            before_wait_completed = work.is_completed()
            first_wait = work.wait()
            second_wait = work.wait()
            results.put((rank, {
                "values": tensor.to("cpu").tolist(),
                "before_wait_completed": before_wait_completed,
                "first_wait": first_wait,
                "second_wait": second_wait,
                "after_wait_completed": work.is_completed(),
            }))
        elif case_name == "all_reduce_float16":
            tensor = _tensor_on_rank(rank, rank + 1, shape=(4,), dtype="float16")
            ddist.all_reduce(tensor, op="sum")
            results.put((rank, tensor.to("cpu").tolist()))
        elif case_name == "all_reduce_bfloat16":
            tensor = _tensor_on_rank(rank, rank + 1, shape=(4,), dtype="bfloat16")
            ddist.all_reduce(tensor, op="sum")
            results.put((rank, tensor.to("cpu").tolist()))
        elif case_name == "barrier":
            ddist.barrier()
            results.put((rank, "passed"))
        elif case_name == "ddp_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            model.to(f"cuda:{rank}")
            ddp_model = DDP(model, sync_grads=False)
            x = _tensor_on_rank(rank, rank + 1, shape=(1, 2))
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            local_grad = _grad_values(model)
            ddp_model.all_reduce_gradients()
            averaged_grad = _grad_values(model)
            results.put((rank, {
                "local_grad": local_grad,
                "averaged_grad": averaged_grad,
            }))
        elif case_name == "ddp_auto_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            model.to(f"cuda:{rank}")
            ddp_model = DDP(model, sync_grads=True)
            x = _tensor_on_rank(rank, rank + 1, shape=(1, 2))
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            results.put((rank, {
                "averaged_grad": _grad_values(model),
            }))
        elif case_name == "ddp_bucket_auto_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            model.to(f"cuda:{rank}")
            ddp_model = DDP(model, sync_grads=True, bucket_size_mb=0.000004)
            x = _tensor_on_rank(rank, rank + 1, shape=(1, 2))
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            results.put((rank, {
                "averaged_grad": _grad_values(model),
                "bucket_count": ddp_model._reducer.diagnostics()["buckets"],
            }))
        elif case_name == "ddp_bucket_overlap_grad_average":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            model.to(f"cuda:{rank}")
            ddp_model = DDP(
                model,
                sync_grads=True,
                bucket_size_mb=0.000004,
                overlap_grad_sync=True,
            )
            x = _tensor_on_rank(rank, rank + 1, shape=(1, 2))
            loss = ortho.sum(ddp_model(x))
            loss.backward()
            pending_after_backward = ddp_model._reducer.diagnostics()["outstanding"]
            launch_count = ddp_model._reducer.diagnostics()["async_launches"]
            ddp_model.wait_for_gradient_sync()
            first_wait_grad = _grad_values(model)
            ddp_model.wait_for_gradient_sync()
            results.put((rank, {
                "averaged_grad": _grad_values(model),
                "first_wait_grad": first_wait_grad,
                "bucket_count": ddp_model._reducer.diagnostics()["buckets"],
                "launch_count": launch_count,
                "pending_after_backward": pending_after_backward,
                "pending_after_wait": ddp_model._reducer.diagnostics()["outstanding"],
            }))
        elif case_name == "ddp_overlap_multi_step":
            # Multi-STEP overlap training with the documented
            # optimizer.zero_grad() sequence. One backward is not enough to
            # catch stale per-backward overlap state
            # made every backward after the first launch zero collectives, so
            # the ranks silently trained on their different local gradients.
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(1.0)
            model.to(f"cuda:{rank}")
            ddp_model = DDP(
                model,
                sync_grads=True,
                bucket_size_mb=0.000004,
                overlap_grad_sync=True,
            )
            optimizer = SGD(model.parameters(), lr=0.01)
            x = _tensor_on_rank(rank, rank + 1, shape=(1, 2))
            launch_counts = []
            params_per_step = []
            for _ in range(3):
                loss = ortho.sum(ddp_model(x))
                loss.backward()
                launch_counts.append(ddp_model._reducer.diagnostics()["async_launches"])
                optimizer.step()
                optimizer.zero_grad()
                params_per_step.append([
                    _flatten_nested(parameter.to("cpu").tolist())
                    for parameter in model.parameters()
                ])
            results.put((rank, {
                "bucket_count": ddp_model._reducer.diagnostics()["buckets"],
                "launch_counts": launch_counts,
                "params_per_step": params_per_step,
            }))
        elif case_name == "amp_ddp_finite":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(0.25)
            model.to(f"cuda:{rank}")
            model.half()
            ddp_model = DDP(model, sync_grads=False)
            optimizer = MixedPrecisionOptimizer(model, SGD, lr=0.01)
            scaler = GradScaler(init_scale=8.0, growth_interval=1)
            x = _tensor_on_rank(rank, rank + 1, shape=(1, 2), dtype="float16")

            optimizer.zero_grad()
            loss = ortho.sum(ddp_model(x))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            results.put((rank, {
                "params": [
                    _flatten_nested(ortho.cast(parameter.to("cpu"), "float32").tolist())
                    for parameter in model.parameters()
                ],
                "scale": scaler.scale_factor,
                "found_inf": scaler.found_inf,
            }))
        elif case_name == "amp_ddp_nonfinite":
            model = Linear(2, 1)
            for parameter in model.parameters():
                parameter.fill_(0.25)
            model.to(f"cuda:{rank}")
            model.half()
            ddp_model = DDP(model, sync_grads=False)
            optimizer = MixedPrecisionOptimizer(model, SGD, lr=0.01)
            scaler = GradScaler(init_scale=8.0, backoff_factor=0.25)
            x = _tensor_on_rank(rank, 1.0, shape=(1, 2), dtype="float16")

            optimizer.zero_grad()
            loss = ortho.sum(ddp_model(x))
            scaler.scale(loss).backward()
            if rank == 1:
                first_param = next(model.parameters())
                first_param.set_grad(
                    ortho.cast(
                        ortho.full(list(first_param.shape), float("inf")).to(f"cuda:{rank}"),
                        "float16",
                    )
                )
            before = [
                _flatten_nested(ortho.cast(parameter.to("cpu"), "float32").tolist())
                for parameter in model.parameters()
            ]
            scaler.step(optimizer)
            scaler.update()
            after = [
                _flatten_nested(ortho.cast(parameter.to("cpu"), "float32").tolist())
                for parameter in model.parameters()
            ]

            results.put((rank, {
                "before": before,
                "after": after,
                "scale": scaler.scale_factor,
                "found_inf": scaler.found_inf,
            }))
        elif case_name == "reject_cpu_tensor":
            tensor = ortho.full([2], rank + 1)
            message = _expect_runtime_error(lambda: ddist.all_reduce(tensor, op="sum"))
            results.put((rank, message))
        elif case_name == "reject_non_contiguous":
            tensor = ortho.full([2, 2], rank + 1).to(f"cuda:{rank}").transpose(0, 1)
            message = _expect_runtime_error(lambda: ddist.all_reduce(tensor, op="sum"))
            results.put((rank, message))
        elif case_name == "reject_unsupported_dtype":
            tensor = ortho.full([2], rank + 1, dtype="int32").to(f"cuda:{rank}")
            message = _expect_runtime_error(lambda: ddist.all_reduce(tensor, op="sum"))
            results.put((rank, message))
        elif case_name == "reject_wrong_cuda_device":
            wrong_device = 1 - rank
            tensor = ortho.full([2], rank + 1).to(f"cuda:{wrong_device}")
            message = _expect_runtime_error(lambda: ddist.all_reduce(tensor, op="sum"))
            results.put((rank, message))
        elif case_name == "reject_non_sum_op":
            tensor = _tensor_on_rank(rank, rank + 1, shape=(2,))
            message = _expect_runtime_error(lambda: ddist.all_reduce(tensor, op="max"))
            results.put((rank, message))
        else:
            raise ValueError(f"unknown NCCL test case: {case_name}")
    except BaseException:
        failures.put((rank, traceback.format_exc()))
        raise
    finally:
        ddist.destroy_process_group()


def _run_nccl_case(case_name, world_size=2):
    ctx = mp.get_context("spawn")
    manager, store = create_store(ctx)
    results = ctx.Queue()
    failures = ctx.Queue()
    processes = []
    try:
        for rank in range(world_size):
            process = ctx.Process(
                target=_nccl_worker,
                args=(rank, world_size, store, case_name, results, failures),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join(timeout=30)
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


@requires_nccl_2gpu
def test_nccl_init_shutdown_two_ranks():
    results = _run_nccl_case("init")
    assert results == [{"rank": 0, "world_size": 2}, {"rank": 1, "world_size": 2}]


@requires_nccl_2gpu
def test_nccl_broadcast_two_ranks():
    results = _run_nccl_case("broadcast")
    assert results == [[7.0, 7.0, 7.0]] * 2


@requires_nccl_2gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nccl_broadcast_two_ranks_supported_low_precision_dtypes(dtype):
    results = _run_nccl_case(f"broadcast_{dtype}")
    assert results == [[7.0, 7.0, 7.0]] * 2


@requires_nccl_2gpu
def test_nccl_broadcast_async_two_ranks_wait_completes_and_is_idempotent():
    results = _run_nccl_case("broadcast_async")
    for result in results:
        assert isinstance(result["before_wait_completed"], bool)
        assert result["values"] == [7.0, 7.0, 7.0]
        assert result["first_wait"] is True
        assert result["second_wait"] is True
        assert result["after_wait_completed"] is True


@requires_nccl_2gpu
def test_nccl_all_reduce_two_ranks_sums_on_gpu():
    results = _run_nccl_case("all_reduce")
    assert results == [[3.0, 3.0, 3.0, 3.0]] * 2


@requires_nccl_2gpu
def test_nccl_all_reduce_async_two_ranks_wait_completes_and_is_idempotent():
    results = _run_nccl_case("all_reduce_async")
    for result in results:
        assert isinstance(result["before_wait_completed"], bool)
        assert result["values"] == [3.0, 3.0, 3.0, 3.0]
        assert result["first_wait"] is True
        assert result["second_wait"] is True
        assert result["after_wait_completed"] is True


@requires_nccl_2gpu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nccl_all_reduce_two_ranks_sums_supported_low_precision_dtypes(dtype):
    results = _run_nccl_case(f"all_reduce_{dtype}")
    assert results == [[3.0, 3.0, 3.0, 3.0]] * 2


@requires_nccl_2gpu
def test_nccl_barrier_two_ranks():
    results = _run_nccl_case("barrier")
    assert results == ["passed"] * 2


@requires_nccl_2gpu
def test_nccl_ddp_explicit_gradient_average_smoke():
    results = _run_nccl_case("ddp_grad_average")
    rank0_local = results[0]["local_grad"]
    rank1_local = results[1]["local_grad"]
    expected = []
    for grad0, grad1 in zip(rank0_local, rank1_local):
        expected.append([(a + b) / 2.0 for a, b in zip(grad0, grad1)])

    for result in results:
        for averaged, expected_grad in zip(result["averaged_grad"], expected):
            assert averaged == pytest.approx(expected_grad)


@requires_nccl_2gpu
def test_nccl_ddp_auto_gradient_average_smoke():
    results = _run_nccl_case("ddp_auto_grad_average")
    expected = [1.5, 1.5, 1.0]
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)


@requires_nccl_2gpu
def test_nccl_ddp_bucketed_auto_gradient_average_smoke():
    results = _run_nccl_case("ddp_bucket_auto_grad_average")
    expected = [1.5, 1.5, 1.0]
    assert all(result["bucket_count"] > 1 for result in results)
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)


@requires_nccl_2gpu
def test_nccl_ddp_bucketed_overlap_gradient_average_smoke():
    results = _run_nccl_case("ddp_bucket_overlap_grad_average")
    expected = [1.5, 1.5, 1.0]
    assert all(result["bucket_count"] > 1 for result in results)
    assert all(result["launch_count"] > 0 for result in results)
    assert all(result["pending_after_backward"] == 0 for result in results)
    assert all(result["pending_after_wait"] == 0 for result in results)
    for result in results:
        assert _flatten_nested(result["averaged_grad"]) == pytest.approx(expected)
        assert result["averaged_grad"] == result["first_wait_grad"]


@requires_nccl_2gpu
def test_nccl_ddp_overlap_multi_step_launches_and_stays_synchronized():
    """Three optimizer steps of overlap DDP on real GPUs, one rank per device.

    Regression for the multi-GPU validation finding: with the documented
    optimizer.zero_grad() loop, stale per-backward overlap bookkeeping meant
    backward 2+ launched no collectives and ranks silently diverged. Every
    backward must launch NEW bucket all-reduces, and both ranks must hold
    identical, correctly-averaged parameters after every step.
    """
    results = _run_nccl_case("ddp_overlap_multi_step")
    steps = 3

    assert all(result["bucket_count"] > 1 for result in results)
    for result in results:
        counts = result["launch_counts"]
        assert len(counts) == steps
        assert counts[0] > 0
        for earlier, later in zip(counts, counts[1:]):
            assert later > earlier, (
                f"no new overlap collectives launched: {counts}"
            )

    # The loss sum(Wx + b) is linear, so the averaged gradients are the
    # constants dW = mean of the rank inputs = (1.5, 1.5) and db = 1.0, and
    # SGD(lr=0.01) from parameters filled with 1.0 walks a closed-form line.
    for step in range(steps):
        k = step + 1
        expected = [1.0 - 0.015 * k, 1.0 - 0.015 * k, 1.0 - 0.01 * k]
        for result in results:
            flat = _flatten_nested(result["params_per_step"][step])
            assert flat == pytest.approx(expected, rel=1e-5, abs=1e-6), (
                f"step {k}: rank parameters diverged from the averaged-"
                f"gradient trajectory: {flat} != {expected}"
            )


@requires_nccl_2gpu
def test_nccl_amp_ddp_finite_master_gradients_sync_two_ranks():
    results = _run_nccl_case("amp_ddp_finite")
    assert _flatten_parameter_values(results[0]["params"]) == pytest.approx(
        _flatten_parameter_values(results[1]["params"])
    )
    assert [result["scale"] for result in results] == [16.0, 16.0]
    assert [result["found_inf"] for result in results] == [False, False]


@requires_nccl_2gpu
def test_nccl_amp_ddp_global_nonfinite_skips_two_ranks():
    results = _run_nccl_case("amp_ddp_nonfinite")
    for result in results:
        assert _flatten_parameter_values(result["after"]) == pytest.approx(
            _flatten_parameter_values(result["before"])
        )
        assert result["scale"] == pytest.approx(2.0)
        assert result["found_inf"] is False


@requires_nccl_2gpu
def test_nccl_launcher_smoke_uses_env_inference(tmp_path):
    script = tmp_path / "nccl_launcher_smoke.py"
    script.write_text(textwrap.dedent(
        """
        import os

        import orthon as ortho
        import orthon.distributed as ddist

        local_rank = int(os.environ["LOCAL_RANK"])
        ddist.init_process_group()
        try:
            tensor = ortho.full([4], ddist.get_rank() + 1).to(f"cuda:{local_rank}")
            ddist.all_reduce(tensor, op="sum")
            values = tensor.to("cpu").tolist()
            if values != [3.0, 3.0, 3.0, 3.0]:
                raise RuntimeError(f"unexpected all_reduce result: {values}")
        finally:
            ddist.destroy_process_group()
        """
    ))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "orthon.distributed.run",
            "--nproc_per_node",
            "2",
            "--backend",
            "nccl",
            str(script),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr


@requires_nccl_2gpu
def test_nccl_rejects_cpu_tensors():
    results = _run_nccl_case("reject_cpu_tensor")
    assert all("tensor must be CUDA" in message for message in results)


@requires_nccl_2gpu
def test_nccl_rejects_non_contiguous_tensors():
    results = _run_nccl_case("reject_non_contiguous")
    assert all("tensor must be contiguous" in message for message in results)


@requires_nccl_2gpu
def test_nccl_rejects_unsupported_dtype_tensors():
    results = _run_nccl_case("reject_unsupported_dtype")
    assert all(
        "only float32, float16, and bfloat16 tensors are supported" in message
        for message in results
    )


@requires_nccl_2gpu
def test_nccl_rejects_wrong_cuda_device_for_rank():
    results = _run_nccl_case("reject_wrong_cuda_device")
    assert all("tensor CUDA device does not match communicator device" in message for message in results)


@requires_nccl_2gpu
def test_nccl_rejects_non_sum_all_reduce_op():
    results = _run_nccl_case("reject_non_sum_op")
    assert all("only supports all_reduce op='sum'" in message for message in results)


@requires_nccl_4gpu
def test_nccl_init_shutdown_four_ranks():
    results = _run_nccl_case("init", world_size=4)
    assert results == [
        {"rank": 0, "world_size": 4},
        {"rank": 1, "world_size": 4},
        {"rank": 2, "world_size": 4},
        {"rank": 3, "world_size": 4},
    ]


@requires_nccl_4gpu
def test_nccl_broadcast_four_ranks():
    results = _run_nccl_case("broadcast", world_size=4)
    assert results == [[7.0, 7.0, 7.0]] * 4


@requires_nccl_4gpu
def test_nccl_all_reduce_four_ranks_sums_on_gpu():
    results = _run_nccl_case("all_reduce", world_size=4)
    assert results == [[10.0, 10.0, 10.0, 10.0]] * 4


@requires_nccl_4gpu
def test_nccl_barrier_four_ranks():
    results = _run_nccl_case("barrier", world_size=4)
    assert results == ["passed"] * 4


@requires_nccl_4gpu
def test_nccl_ddp_explicit_gradient_average_four_ranks_smoke():
    results = _run_nccl_case("ddp_grad_average", world_size=4)
    local_grads = [result["local_grad"] for result in results]
    expected = []
    for parameter_grads in zip(*local_grads):
        averaged = [
            sum(values) / 4.0
            for values in zip(*parameter_grads)
        ]
        expected.append(averaged)

    for result in results:
        for averaged, expected_grad in zip(result["averaged_grad"], expected):
            assert averaged == pytest.approx(expected_grad)


@requires_nccl_4gpu
def test_nccl_amp_ddp_finite_master_gradients_sync_four_ranks():
    results = _run_nccl_case("amp_ddp_finite", world_size=4)
    for result in results[1:]:
        assert _flatten_parameter_values(result["params"]) == pytest.approx(
            _flatten_parameter_values(results[0]["params"])
        )
    assert [result["scale"] for result in results] == [16.0] * 4
    assert [result["found_inf"] for result in results] == [False] * 4


@requires_nccl_4gpu
def test_nccl_amp_ddp_global_nonfinite_skips_four_ranks():
    results = _run_nccl_case("amp_ddp_nonfinite", world_size=4)
    for result in results:
        assert _flatten_parameter_values(result["after"]) == pytest.approx(
            _flatten_parameter_values(result["before"])
        )
        assert result["scale"] == pytest.approx(2.0)
        assert result["found_inf"] is False


@requires_nccl_8gpu
def test_nccl_init_shutdown_eight_ranks():
    results = _run_nccl_case("init", world_size=8)
    assert results == [{"rank": rank, "world_size": 8} for rank in range(8)]


@requires_nccl_8gpu
def test_nccl_broadcast_eight_ranks():
    results = _run_nccl_case("broadcast", world_size=8)
    assert results == [[7.0, 7.0, 7.0]] * 8


@requires_nccl_8gpu
def test_nccl_all_reduce_eight_ranks_sums_on_gpu():
    results = _run_nccl_case("all_reduce", world_size=8)
    assert results == [[36.0, 36.0, 36.0, 36.0]] * 8


@requires_nccl_8gpu
def test_nccl_barrier_eight_ranks():
    results = _run_nccl_case("barrier", world_size=8)
    assert results == ["passed"] * 8


@requires_nccl_8gpu
def test_nccl_ddp_explicit_gradient_average_eight_ranks_smoke():
    results = _run_nccl_case("ddp_grad_average", world_size=8)
    local_grads = [result["local_grad"] for result in results]
    expected = []
    for parameter_grads in zip(*local_grads):
        averaged = [
            sum(values) / 8.0
            for values in zip(*parameter_grads)
        ]
        expected.append(averaged)

    for result in results:
        for averaged, expected_grad in zip(result["averaged_grad"], expected):
            assert averaged == pytest.approx(expected_grad)


@requires_nccl_8gpu
def test_nccl_amp_ddp_finite_master_gradients_sync_eight_ranks():
    results = _run_nccl_case("amp_ddp_finite", world_size=8)
    for result in results[1:]:
        assert _flatten_parameter_values(result["params"]) == pytest.approx(
            _flatten_parameter_values(results[0]["params"])
        )
    assert [result["scale"] for result in results] == [16.0] * 8
    assert [result["found_inf"] for result in results] == [False] * 8


@requires_nccl_8gpu
def test_nccl_amp_ddp_global_nonfinite_skips_eight_ranks():
    results = _run_nccl_case("amp_ddp_nonfinite", world_size=8)
    for result in results:
        assert _flatten_parameter_values(result["after"]) == pytest.approx(
            _flatten_parameter_values(result["before"])
        )
        assert result["scale"] == pytest.approx(2.0)
        assert result["found_inf"] is False
