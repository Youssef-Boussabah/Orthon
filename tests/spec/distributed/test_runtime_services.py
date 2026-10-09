import sys
import time

import pytest

from .subprocesses import run_bounded


@pytest.mark.parametrize("kind", ["collective", "barrier", "shape", "dtype", "source"])
def test_rank_descriptor_mismatch_fails_every_rank_without_payload_write(distributed_job, kind):
    assert distributed_job("runtime-mismatch", kind=kind) == ["failed-and-unchanged"] * 2


@pytest.mark.parametrize("world", [2, 3, 4])
def test_uniform_preflight_failure_leaves_next_collective_usable(distributed_job, world):
    assert distributed_job("runtime-preflight", nprocs=world) == [[world * (world + 1) / 2, 2 * world]] * world


@pytest.mark.parametrize("world", [2, 3, 4])
def test_worker_exception_wakes_waiting_peers_and_reaps_processes(distributed_job, world):
    start = time.monotonic()
    result = distributed_job("runtime-abort", nprocs=world)
    assert result == {"error": "RuntimeError", "worker_exception": "ValueError"}
    assert time.monotonic() - start < 20


@pytest.mark.parametrize("options", [{}, {"manual": True}, {"bucket": .000001},
                                     {"bucket": .000001, "overlap": True},
                                     {"bucket": 1., "overlap": True}])
@pytest.mark.parametrize("world", [2, 3, 4])
def test_rank_divergent_unused_and_input_only_outputs_finalize_each_backward(distributed_job, options, world):
    results = distributed_job("runtime-unused", nprocs=world, **options)
    a, b = 2. / world, sum(2. + 4 * r for r in range(1, world)) / world
    mean = sum(2. + 4 * r for r in range(world)) / world
    for rounds in results:
        for iteration, gradients in enumerate(rounds):
            assert gradients[0]["values"] == pytest.approx([a * min(iteration + 1, 2) + (mean if iteration >= 2 else 0)])
            assert gradients[1]["values"] == pytest.approx([b * min(iteration + 1, 2)])
            assert gradients[2] is None


def test_manual_gradient_identity_and_mutation_are_consumed_once(distributed_job):
    for first, second, unused in distributed_job("runtime-manual-set", manual=True):
        assert first["values"] == [4.]
        assert second["values"] == [8.]
        assert unused is None


def test_distinct_parameter_owners_sharing_storage_receive_separate_reductions(distributed_job):
    assert distributed_job("runtime-distinct-owners") == [[[4., 4.], [12., 12.]]] * 2


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_rank_local_strided_gradient_forces_agreed_contiguous_fallback(distributed_job, dtype):
    for result in distributed_job("runtime-strided", dtype=dtype):
        assert result["values"] == [4., 5., 5., 8.]
        assert result["dtype"] == dtype and result["shape"] == [2, 2]


def test_all_unused_parameter_keeps_none_gradient_and_adam_history_with_weight_decay(distributed_job):
    results = distributed_job("runtime-unused-state")
    for rank, result in enumerate(results):
        assert result["unused"] == [3.]
        assert result["gradient"] is None
        assert result["moment"] is None and result["variance"] is None
        assert result["buffer"] == [rank]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("mode", ["direct", "disabled", "enabled"])
def test_amp_master_reducer_handles_optional_gradients_and_disabled_scaler(distributed_job, dtype, mode):
    results = distributed_job("runtime-amp", dtype=dtype, mode=mode)
    updates = 0
    for iteration in range(3):
        if not (mode == "enabled" and iteration == 1):
            updates += 1
        for rank in range(2):
            record = results[rank][iteration]
            assert record["model"] == results[1-rank][iteration]["model"]
            assert record["masters"][0]["values"] == [10. - .5 * updates]
            assert record["masters"][1]["values"] == [20. - updates]
            assert record["masters"][2]["values"] == [30.]
            assert record["grads"][2] is None
            if not (mode == "enabled" and iteration == 1):
                assert record["grads"][0]["values"] == [2.]
                assert record["grads"][1]["values"] == [4.]


def test_launcher_terminates_slow_peer_after_rank_failure(tmp_path):
    script = tmp_path / "slow peer.py"
    script.write_text("import os,time\nif os.environ['RANK']=='1': raise SystemExit(9)\ntime.sleep(90)\n")
    start = time.monotonic()
    result = run_bounded([sys.executable, "-m", "orthon.distributed.run", "--nproc_per_node", "2", str(script)], timeout=15)
    assert result.returncode == 9 and "rank 1" in result.stderr
    assert time.monotonic() - start < 15


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("kind", ["low-precision", "shape", "dtype-mismatch", "backend-dtype"])
def test_rank_local_invalid_gradient_profile_agrees_before_data_and_preserves_transport(distributed_job, world, kind):
    results = distributed_job("runtime-invalid-gradient", nprocs=world, kind=kind,
                              observe=True, float32_gradient_backend=kind == "backend-dtype")
    assert len({r["message"] for r in results}) == 1
    for result in results:
        assert result["control_calls"] == 1 and result["data_calls"] == 0
        assert result["value"] == [world * (world + 1) / 2, 2 * world]


def test_rank_local_wrong_gradient_device_agrees_before_data_and_preserves_transport(distributed_job):
    import orthon as ortho
    if not ortho.cuda_enabled():
        pytest.skip("requires one physical CUDA device for a wrong-device gradient")
    try:
        ortho.zeros([1]).to("cuda:0")
    except RuntimeError as error:
        if any(reason in str(error).lower() for reason in ("no cuda-capable device", "driver version is insufficient",
                   "initialization error", "unsupported display driver", "invalid device ordinal")):
            pytest.skip(f"requires one usable CUDA device: {error}")
        raise
    results = distributed_job("runtime-invalid-gradient", kind="device", observe=True)
    assert len({r["message"] for r in results}) == 1
    assert all(r["data_calls"] == 0 and r["value"] == [3., 4.] for r in results)
