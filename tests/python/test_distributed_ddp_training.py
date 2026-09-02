import multiprocessing as mp
import traceback

import numpy as np
import pytest

import orthon as ortho
import orthon.distributed as ddist
from orthon.distributed import DistributedDataParallel as DDP
from orthon.distributed.local_backend import create_store
from orthon.nn import Linear
from orthon.optim import Adam
from orthon.optim import SGD


_STEPS = 3
_LR = 0.05
_ADAM_LR = 0.01


def _fixed_model():
    model = Linear(2, 1)
    params = list(model.parameters())
    params[0].fill_(0.25)
    params[1].fill_(0.1)
    return model


def _shard_arrays(rank):
    xs = [
        np.asarray([[1.0, 2.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float32),
    ]
    target_negs = [
        np.asarray([[-1.5]], dtype=np.float32),
        np.asarray([[-3.5]], dtype=np.float32),
    ]
    return xs[rank], target_negs[rank]


def _full_batch_arrays():
    return (
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[-1.5], [-3.5]], dtype=np.float32),
    )


def _mse_loss(model, x_array, target_neg_array):
    x = ortho.from_numpy(x_array)
    target_neg = ortho.from_numpy(target_neg_array)
    pred = model(x)
    diff = pred + target_neg
    return ortho.mean(diff * diff)


def _flatten_nested(values):
    flat = []
    for value in values:
        if isinstance(value, list):
            flat.extend(_flatten_nested(value))
        else:
            flat.append(float(value))
    return flat


def _flatten_model_params(model):
    flat = []
    for parameter in model.parameters():
        flat.extend(_flatten_nested(parameter.tolist()))
    return flat


def _make_optimizer(name, params):
    if name == "sgd":
        return SGD(params, lr=_LR)
    if name == "adam":
        return Adam(params, lr=_ADAM_LR)
    raise ValueError(f"unknown optimizer: {name}")


def _single_process_reference(optimizer_name):
    model = _fixed_model()
    optimizer = _make_optimizer(optimizer_name, model.parameters())
    x, target_neg = _full_batch_arrays()
    losses = []
    for _ in range(_STEPS):
        loss = _mse_loss(model, x, target_neg)
        losses.append(float(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return {
        "params": _flatten_model_params(model),
        "losses": losses,
    }


def _ddp_training_worker(
    rank,
    world_size,
    store,
    optimizer_name,
    sync_grads,
    manual_sync,
    bucket_size_mb,
    results,
    failures,
    overlap_grad_sync=False,
):
    try:
        ddist.init_process_group(
            backend="local",
            rank=rank,
            world_size=world_size,
            store=store,
        )
        model = _fixed_model()
        ddp_model = DDP(
            model,
            sync_grads=sync_grads,
            bucket_size_mb=bucket_size_mb,
            overlap_grad_sync=overlap_grad_sync,
        )
        optimizer = _make_optimizer(optimizer_name, ddp_model.parameters())
        x, target_neg = _shard_arrays(rank)
        losses = []
        params_per_step = []
        launch_counts = []
        for _ in range(_STEPS):
            loss = _mse_loss(ddp_model, x, target_neg)
            losses.append(float(loss.item()))
            loss.backward()
            # Cumulative overlap bucket launches observed AFTER each backward.
            # A stale-state regression shows up here as a step where the count
            # does not advance - the exact signature of the bug where step 2+
            # silently stopped launching collectives.
            launch_counts.append(ddp_model._overlap_bucket_launch_count)
            if manual_sync:
                ddp_model.all_reduce_gradients()
            optimizer.step()
            optimizer.zero_grad()
            params_per_step.append(_flatten_model_params(model))
        results.put((rank, {
            "params": _flatten_model_params(model),
            "params_per_step": params_per_step,
            "launch_counts": launch_counts,
            "losses": losses,
        }))
    except BaseException:
        failures.put((rank, traceback.format_exc()))
        raise
    finally:
        ddist.destroy_process_group()


def _run_ddp_training(
    optimizer_name,
    world_size=2,
    sync_grads=False,
    manual_sync=True,
    bucket_size_mb=None,
    overlap_grad_sync=False,
):
    ctx = mp.get_context("spawn")
    manager, store = create_store(ctx)
    results = ctx.Queue()
    failures = ctx.Queue()
    processes = []
    try:
        for rank in range(world_size):
            process = ctx.Process(
                target=_ddp_training_worker,
                args=(
                    rank,
                    world_size,
                    store,
                    optimizer_name,
                    sync_grads,
                    manual_sync,
                    bucket_size_mb,
                    results,
                    failures,
                    overlap_grad_sync,
                ),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join(timeout=15)
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


def test_local_ddp_sgd_matches_single_process_full_batch():
    reference = _single_process_reference("sgd")
    rank_results = _run_ddp_training("sgd", world_size=2)

    assert rank_results[0]["params"] == pytest.approx(
        rank_results[1]["params"],
        rel=1e-5,
        abs=1e-6,
    )
    for result in rank_results:
        assert result["params"] == pytest.approx(
            reference["params"],
            rel=1e-5,
            abs=1e-6,
        )


def test_local_ddp_adam_matches_single_process_full_batch():
    reference = _single_process_reference("adam")
    rank_results = _run_ddp_training("adam", world_size=2)

    assert rank_results[0]["params"] == pytest.approx(
        rank_results[1]["params"],
        rel=1e-5,
        abs=1e-6,
    )
    for result in rank_results:
        assert result["params"] == pytest.approx(
            reference["params"],
            rel=1e-5,
            abs=1e-6,
        )


def test_local_ddp_auto_sync_sgd_matches_single_process_full_batch():
    reference = _single_process_reference("sgd")
    rank_results = _run_ddp_training(
        "sgd",
        world_size=2,
        sync_grads=True,
        manual_sync=False,
    )

    assert rank_results[0]["params"] == pytest.approx(
        rank_results[1]["params"],
        rel=1e-5,
        abs=1e-6,
    )
    for result in rank_results:
        assert result["params"] == pytest.approx(
            reference["params"],
            rel=1e-5,
            abs=1e-6,
        )


def test_local_ddp_bucketed_auto_sync_sgd_matches_single_process_full_batch():
    reference = _single_process_reference("sgd")
    rank_results = _run_ddp_training(
        "sgd",
        world_size=2,
        sync_grads=True,
        manual_sync=False,
        bucket_size_mb=0.000004,
    )

    assert rank_results[0]["params"] == pytest.approx(
        rank_results[1]["params"],
        rel=1e-5,
        abs=1e-6,
    )
    for result in rank_results:
        assert result["params"] == pytest.approx(
            reference["params"],
            rel=1e-5,
            abs=1e-6,
        )


def test_local_ddp_overlap_multi_step_stays_synchronized_and_matches_reference():
    """Multi-step overlap DDP with the documented optimizer.zero_grad() loop.

    This is the CPU-CI regression for the 2x A100 finding: overlap mode's
    per-backward bookkeeping was reset only by ddp_model.zero_grad(), so with
    the documented `optimizer.zero_grad()` sequence the second and later
    backwards launched no bucket collectives at all and ranks silently trained
    on local gradients. The shards differ per rank on purpose - a skipped
    all-reduce makes the ranks' parameters visibly diverge here.
    """
    reference = _single_process_reference("sgd")
    rank_results = _run_ddp_training(
        "sgd",
        world_size=2,
        sync_grads=True,
        manual_sync=False,
        bucket_size_mb=0.000004,     # 4 bytes: forces one bucket per parameter
        overlap_grad_sync=True,
    )

    # Every backward must launch NEW overlap collectives: the cumulative
    # launch count has to strictly increase step over step on every rank.
    for result in rank_results:
        counts = result["launch_counts"]
        assert len(counts) == _STEPS
        assert counts[0] > 0
        for earlier, later in zip(counts, counts[1:]):
            assert later > earlier, (
                f"no new overlap collectives launched: {counts}"
            )

    # Ranks must agree after EVERY optimizer step, not just at the end.
    for step in range(_STEPS):
        assert rank_results[0]["params_per_step"][step] == pytest.approx(
            rank_results[1]["params_per_step"][step],
            rel=1e-5,
            abs=1e-6,
        )

    # And the result is the same full-batch training the blocking modes prove.
    for result in rank_results:
        assert result["params"] == pytest.approx(
            reference["params"],
            rel=1e-5,
            abs=1e-6,
        )
