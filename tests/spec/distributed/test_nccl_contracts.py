import pytest

from .cases import MODES, COEFFICIENTS, INITIAL


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_two_device_nccl_raw_sum_broadcast_and_async_work(nccl_two_devices, distributed_job, dtype):
    results = distributed_job("collectives", nccl=True, dtype=dtype)
    for rank, result in enumerate(results):
        for step in result:
            assert step["summed"]["values"] == [3., 6., 9., 12.]
            assert step["summed"]["dtype"] == dtype
            assert step["summed"]["device"] == nccl_two_devices[rank]
            assert step["broadcasts"][0]["values"] == [1., 2., 3., 4.]
            assert step["broadcasts"][1]["values"] == [11., 12., 13., 14.]
            if step["work"]:
                assert isinstance(step["work"]["initial"], bool)
                assert step["work"]["waits"] == [True, True] and step["work"]["done"] is True


def test_nccl_input_validation_preserves_values_and_group_remains_usable(nccl_two_devices, distributed_job):
    results = distributed_job("nccl-validation", nccl=True)
    for result in results:
        assert all(result[name] == "RuntimeError" for name in ("cpu", "strided", "dtype", "wrong-device"))
        assert result["NotImplementedError"] is True and result["ValueError"] is True


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nccl_amp_fp32_masters_and_enabled_finite_consensus_recover_together(
    nccl_two_devices, distributed_job, dtype,
):
    results = distributed_job("amp", nccl=True, dtype=dtype)
    for iteration in range(4):
        first, second = results[0][iteration], results[1][iteration]
        assert first["scale"] == second["scale"] == [8., 2., 2., 8.][iteration]
        assert first["found_inf"] is second["found_inf"] is (iteration == 1)
        assert first["steps"] == second["steps"] == [1, 1, 2, 3][iteration]
        for left, right in zip(first["masters"], second["masters"]):
            assert left["values"] == pytest.approx(right["values"], rel=2e-6, abs=2e-6)
            assert left["dtype"] == right["dtype"] == "float32"


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_nccl_raw_low_precision_support_does_not_enable_direct_ddp_averaging(
    nccl_two_devices, distributed_job, dtype,
):
    assert distributed_job("low-precision-ddp", nccl=True, dtype=dtype) == ["RuntimeError", "RuntimeError"]


@pytest.mark.parametrize("mode", MODES)
def test_nccl_fp32_ddp_each_backward_completes_before_immediate_multistep_update(
    nccl_two_devices, distributed_job, mode,
):
    results = distributed_job("training", nccl=True, **mode)
    for rank, result in enumerate(results):
        for iteration, step in enumerate(result):
            factor = 4 * (iteration + 1)
            for name, coefficient in COEFFICIENTS.items():
                assert step["reduced"][name]["values"] == pytest.approx(coefficient * factor)
                assert step["reduced"][name]["device"] == nccl_two_devices[rank]
                cumulative = 4 * (iteration + 1) * (iteration + 2) / 2
                assert step["after"][name]["values"] == pytest.approx(INITIAL[name] - .125 * coefficient * cumulative)
