import pytest

from .cases import COEFFICIENTS, MODES


@pytest.mark.parametrize("mode", [MODES[0], MODES[1], MODES[4]])
def test_retained_and_fresh_backward_accumulate_already_averaged_leaf_gradients(distributed_job, mode):
    results = distributed_job("accumulation", **mode)
    for result in results:
        for pass_number, gradients in enumerate(result, 1):
            for name, coefficient in COEFFICIENTS.items():
                assert gradients[name]["values"] == pytest.approx(coefficient * 20 * pass_number)


@pytest.mark.parametrize("manual", [False, True])
def test_one_tied_leaf_combines_all_paths_and_is_reduced_and_stepped_once(distributed_job, manual):
    results = distributed_job("tied", observe=True, manual=manual)
    for result in results:
        assert result["parameters"] == 1 and result["names"] == ["first", "second"]
        assert result["reductions"] == 1
        assert result["gradient"]["values"] == [20., 20.]
        assert result["values"] == [-1.5, -.5]
