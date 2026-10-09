import numpy as np
import pytest


from .cases import MODES, COEFFICIENTS, INITIAL


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_each_backward_and_immediate_optimizer_step_use_one_current_rank_average(
    distributed_job, mode, dtype,
):
    results = distributed_job("training", dtype=dtype, observe=True, **mode)
    expected_parameters = {name: values.copy() for name, values in INITIAL.items()}
    for iteration in range(3):
        factor = 4 * (iteration + 1)
        for name, coefficients in COEFFICIENTS.items():
            for rank, result in enumerate(results):
                step = result[iteration]
                assert step["reduced"][name]["values"] == pytest.approx(coefficients * factor)
                assert step["reduced"][name]["dtype"] == dtype
                assert step["reduced"][name]["requires_grad"] is False
                assert step["before"][name]["values"] == pytest.approx(expected_parameters[name])
                if mode.get("manual"):
                    assert step["local"][name]["values"] == pytest.approx(coefficients * (2 + 4 * rank) * (iteration + 1))
            expected_parameters[name] -= .125 * factor * coefficients
            for result in results:
                assert result[iteration]["after"][name]["values"] == pytest.approx(expected_parameters[name])
        for result in results:
            events = result[iteration]["events"]
            assert events
            assert all(event["waited"] for event in events if event["async_op"])
            if mode.get("overlap") and dtype == "float32":
                assert any(event["async_op"] for event in events)
                if mode["bucket"] < .01:
                    assert all(event["missing_model_gradients"] == 0 for event in events)


@pytest.mark.parametrize("mode", MODES[:2])
def test_single_rank_ddp_preserves_the_unaveraged_gradient_and_update(distributed_job, mode):
    result = distributed_job("training", nprocs=1, **mode)[0]
    for iteration, step in enumerate(result):
        for name, coefficient in COEFFICIENTS.items():
            assert step["reduced"][name]["values"] == pytest.approx(coefficient * 2 * (iteration + 1))


@pytest.mark.parametrize("bucket", [None, .000001, 1.])
def test_manual_sync_handles_uniform_unused_parameters_and_changing_usage(distributed_job, bucket):
    results = distributed_job("training", manual=True, bucket=bucket, unused=True, usage=["a", "b", "all"])
    for result in results:
        for iteration, step in enumerate(result):
            assert step["local"]["unused"] is None and step["reduced"]["unused"] is None
            assert step["after"]["unused"]["values"] == [7.]
            names = ["a"] if iteration == 0 else ["b"] if iteration == 1 else ["a", "b", "c"]
            for name in COEFFICIENTS:
                if name in names:
                    assert step["reduced"][name]["values"] == pytest.approx(COEFFICIENTS[name] * 4 * (iteration + 1))
                else:
                    assert step["reduced"][name] is None
