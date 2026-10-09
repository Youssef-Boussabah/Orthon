import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("batched", [False, True])
def test_matmul_requires_matching_contraction_and_batch_dimensions(
    operator_profile, typed_tensor, assert_output, batched,
):
    device, dtype = operator_profile
    left_values = np.array([[1, 2, 3], [4, 0, -1]])
    right_values = np.array([[2, 1], [0, 3], [-2, 4]])
    if batched:
        left_values = np.stack([left_values, left_values + 1])
        right_values = np.stack([right_values, right_values - 1])
    left = typed_tensor(left_values, dtype=dtype, device=device)
    right = typed_tensor(right_values, dtype=dtype, device=device)
    expected = left_values @ right_values
    output = ortho.matmul(left, right)
    for actual in (output, left @ right, left.matmul(right)):
        assert_output(actual, expected, dtype=dtype, device=device)
    if dtype in ("float32", "float64"):
        left.mul_(2)
        right.mul_(3)
        assert_output(output, expected, dtype=dtype, device=device)


@pytest.mark.parametrize("left_shape,right_shape", [
    ((3,), (3,)), ((2, 3), (3,)), ((2, 3), (4, 2)),
    ((2, 2, 3), (3, 2)), ((1, 2, 3), (2, 3, 2)),
    ((2, 2, 2, 3), (2, 2, 3, 2)),
])
def test_matmul_rejects_vector_rank_mismatch_batch_broadcast_and_bad_contraction(
    operator_profile, typed_tensor, left_shape, right_shape,
):
    device, dtype = operator_profile
    left = typed_tensor(np.ones(left_shape), dtype=dtype, device=device)
    right = typed_tensor(np.ones(right_shape), dtype=dtype, device=device)
    with pytest.raises(RuntimeError):
        ortho.matmul(left, right)
