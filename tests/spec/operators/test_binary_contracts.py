import numpy as np
import pytest


def test_binary_entry_points_preserve_properties_and_own_the_result(
    operator_profile, typed_tensor, binary_contract, assert_output,
):
    device, dtype = operator_profile
    public, syntax = binary_contract
    left = typed_tensor([[2, 4], [8, 16]], dtype=dtype, device=device)
    right = typed_tensor([[1, 2], [4, 8]], dtype=dtype, device=device)
    expected = syntax(np.array([[2, 4], [8, 16]]), np.array([[1, 2], [4, 8]]))
    output = public(left, right)
    assert_output(output, expected, dtype=dtype, device=device)
    assert_output(syntax(left, right), expected, dtype=dtype, device=device)
    if dtype in ("float32", "float64"):
        left.mul_(2)
        right.mul_(3)
        assert_output(output, expected, dtype=dtype, device=device)


@pytest.mark.parametrize("scalar", [2, 2.5, True, np.float64(2.5), np.array(2.5)])
@pytest.mark.parametrize("reverse", [False, True])
def test_python_numbers_use_the_tensor_dtype_device_and_shape(
    operator_profile, typed_tensor, binary_contract, assert_output, scalar, reverse,
):
    device, dtype = operator_profile
    _, syntax = binary_contract
    values = np.array([[2, 4], [8, 16]], dtype=np.float64)
    source = typed_tensor(values, dtype=dtype, device=device)
    output = syntax(scalar, source) if reverse else syntax(source, scalar)
    expected = syntax(float(scalar), values) if reverse else syntax(values, float(scalar))
    assert_output(output, expected, dtype=dtype, device=device)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_python_float_is_narrowed_directly_to_the_tensor_dtype(typed_tensor, dtype):
    source = typed_tensor([0], dtype=dtype)
    assert (source + 1 / 3).item() == np.array(1 / 3, dtype=dtype).item()


@pytest.mark.parametrize("other", [[1], "2", 1 + 2j, np.array([2.0])])
def test_non_scalar_python_operands_are_not_implicitly_tensors(
    typed_tensor, binary_contract, other,
):
    _, syntax = binary_contract
    with pytest.raises(TypeError):
        syntax(typed_tensor([2]), other)


def test_top_level_binary_functions_require_two_tensors(typed_tensor, binary_contract):
    public, _ = binary_contract
    with pytest.raises(TypeError):
        public(typed_tensor([2]), 3)
