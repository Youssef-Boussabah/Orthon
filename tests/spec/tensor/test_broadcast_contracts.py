import numpy as np
import pytest


@pytest.mark.parametrize("left_shape,right_shape", [
    ((2, 3), (2, 3)), ((), (2, 3)), ((2, 3), ()), ((2, 3), (3,)),
    ((2, 1), (1, 3)), ((2, 1, 3), (1, 4, 1)), ((0, 3), (1, 3)),
])
def test_broadcasting_aligns_trailing_dimensions_and_expands_only_size_one(
    operator_profile, typed_tensor, binary_contract, assert_output, left_shape, right_shape,
):
    device, dtype = operator_profile
    public, syntax = binary_contract
    left = np.arange(np.prod(left_shape), dtype=np.float64).reshape(left_shape) + 1
    right = np.arange(np.prod(right_shape), dtype=np.float64).reshape(right_shape) + 2
    output = public(typed_tensor(left, dtype=dtype, device=device),
                    typed_tensor(right, dtype=dtype, device=device))
    assert_output(output, syntax(left, right), dtype=dtype, device=device)


@pytest.mark.parametrize("left_shape,right_shape", [
    ((2, 3), (2,)), ((2, 3), (4, 3)), ((2, 1, 3), (4, 2)), ((0, 3), (2, 3)),
])
def test_broadcasting_rejects_any_aligned_pair_that_is_neither_equal_nor_one(
    operator_profile, typed_tensor, binary_contract, left_shape, right_shape,
):
    device, dtype = operator_profile
    public, _ = binary_contract
    left = typed_tensor(np.ones(left_shape), dtype=dtype, device=device)
    right = typed_tensor(np.ones(right_shape), dtype=dtype, device=device)
    with pytest.raises(RuntimeError):
        public(left, right)
