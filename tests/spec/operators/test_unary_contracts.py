import math

import numpy as np
import pytest

import orthon as ortho


UNARY = ["neg", "relu", "sigmoid", "tanh", "leaky_relu", "gelu",
         "exp", "sqrt", "reciprocal"]


def reference(name, values):
    if name == "leaky_relu":
        return np.where(values >= 0, values, values / 8)
    if name == "gelu":
        return values / 2 * (1 + np.vectorize(math.erf, otypes=[float])(values / np.sqrt(2)))
    return {
        "neg": np.negative, "relu": lambda x: np.maximum(x, 0),
        "sigmoid": lambda x: 1 / (1 + np.exp(-x)), "tanh": np.tanh,
        "exp": np.exp, "sqrt": np.sqrt, "reciprocal": lambda x: 1 / x,
    }[name](values)


def apply(name, source):
    return (ortho.leaky_relu(source, negative_slope=.125) if name == "leaky_relu"
            else getattr(ortho, name)(source))


@pytest.mark.parametrize("name", UNARY)
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "slice"])
def test_unary_values_and_properties_follow_logical_coordinates(
    operator_profile, typed_tensor, assert_output, name, layout,
):
    device, dtype = operator_profile
    values = np.array([[1, 4, 9], [16, 25, 36]], dtype=np.float64)
    if name not in ("sqrt", "reciprocal"):
        values = np.array([[-2, -1, 0], [1, 2, 3]], dtype=np.float64)
    base = source = typed_tensor(values, dtype=dtype, device=device)
    if layout == "transpose":
        source, values = source.transpose(0, 1), values.T
    elif layout == "slice":
        source, values = source.slice(1, 1, 3), values[:, 1:3]
    expected = reference(name, values)
    output = apply(name, source)
    assert_output(output, expected, dtype=dtype, device=device)
    if name not in ("sqrt", "reciprocal"):
        method_output = (source.leaky_relu(.125) if name == "leaky_relu"
                         else getattr(source, name)())
        assert_output(method_output, expected, dtype=dtype, device=device)
    if dtype in ("float32", "float64"):
        base.mul_(2)
        assert_output(output, expected, dtype=dtype, device=device)


@pytest.mark.parametrize("name", UNARY)
@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
def test_general_unary_operators_accept_scalars_and_empty_shapes(
    operator_profile, typed_tensor, assert_output, name, shape,
):
    device, dtype = operator_profile
    values = np.full(shape, 4.0)
    output = apply(name, typed_tensor(values, dtype=dtype, device=device))
    assert_output(output, reference(name, values), dtype=dtype, device=device)


@pytest.mark.parametrize("name", ["softmax", "log_softmax"])
@pytest.mark.parametrize("dim", [0, -1])
def test_normalized_exponentials_preserve_shape_device_and_float32_dtype(
    operator_device, typed_tensor, assert_output, name, dim,
):
    device = operator_device
    values = np.array([[1, 2, 4], [3, 0, -1]], dtype=np.float64)
    source = typed_tensor(values, device=device)
    shifted = values - values.max(axis=dim, keepdims=True)
    log_result = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
    expected = np.exp(log_result) if name == "softmax" else log_result
    assert_output(getattr(ortho, name)(source, dim), expected, dtype="float32", device=device)


@pytest.mark.parametrize("name", ["softmax", "log_softmax"])
@pytest.mark.parametrize("invalid", ["scalar", "empty", "strided", "axis"])
def test_normalized_exponentials_validate_shape_layout_and_axis(operator_device, typed_tensor, name, invalid):
    source, dim, error = {
        "scalar": lambda: (typed_tensor(1), 0, RuntimeError),
        "empty": lambda: (typed_tensor(np.empty((2, 0))), 1, RuntimeError),
        "strided": lambda: (typed_tensor([[1, 2], [3, 4]]).transpose(0, 1), 1, RuntimeError),
        "axis": lambda: (typed_tensor([[1, 2], [3, 4]]), 2, IndexError),
    }[invalid]()
    if operator_device != "cpu":
        # Transfer packs a strided input, so construct the view after moving.
        source = (typed_tensor([[1, 2], [3, 4]], device=operator_device).transpose(0, 1)
                  if invalid == "strided" else source.to(operator_device))
    with pytest.raises(error):
        getattr(ortho, name)(source, dim)


@pytest.mark.parametrize("name", ["softmax", "log_softmax"])
def test_normalized_exponentials_require_float32(operator_device, storage_dtype, typed_tensor, name):
    source = typed_tensor([[1, 2]], dtype=storage_dtype, device=operator_device)
    if storage_dtype == "float32":
        assert getattr(ortho, name)(source, 1).dtype == "float32"
    else:
        with pytest.raises(RuntimeError):
            getattr(ortho, name)(source, 1)
