import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("name", ["sum", "mean", "max", "argmax"])
@pytest.mark.parametrize("dim", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reductions_have_explicit_axis_shape_and_result_dtype(
    operator_profile, typed_tensor, assert_output, name, dim, keepdim,
):
    device, dtype = operator_profile
    values = np.array([[[1, 7], [2, 6], [3, 5]], [[9, 0], [4, 8], [2, 3]]], dtype=np.float64)
    source = typed_tensor(values, dtype=dtype, device=device)
    if name == "argmax":
        expected = np.argmax(values, axis=dim)
        if keepdim:
            expected = (np.reshape(expected, (1,) * values.ndim) if dim is None
                        else np.expand_dims(expected, axis=dim))
    else:
        expected = getattr(np, name)(values, axis=dim, keepdims=keepdim)
    result_dtype = "int64" if name == "argmax" else dtype
    output = getattr(ortho, name)(source, dim=dim, keepdim=keepdim)
    assert_output(output, expected,
                  dtype=result_dtype, device=device)
    assert_output(getattr(source, name)(dim=dim, keepdim=keepdim), expected,
                  dtype=result_dtype, device=device)
    if dtype in ("float32", "float64"):
        source.mul_(2)
        assert_output(output, expected, dtype=result_dtype, device=device)


@pytest.mark.parametrize("name", ["sum", "mean", "max", "argmax"])
def test_global_reduction_of_a_scalar_has_scalar_shape(
    operator_profile, typed_tensor, assert_output, name,
):
    device, dtype = operator_profile
    source = typed_tensor(4, dtype=dtype, device=device)
    expected = 0 if name == "argmax" else 4
    assert_output(getattr(ortho, name)(source), expected,
                  dtype="int64" if name == "argmax" else dtype, device=device)
    with pytest.raises(IndexError):
        getattr(ortho, name)(source, dim=0)


@pytest.mark.parametrize("name", ["sum", "mean", "max", "argmax"])
@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_empty_reductions_raise_instead_of_inventing_an_identity(
    operator_profile, typed_tensor, name, shape,
):
    device, dtype = operator_profile
    source = typed_tensor(np.empty(shape), dtype=dtype, device=device)
    with pytest.raises(RuntimeError):
        getattr(ortho, name)(source)


@pytest.mark.parametrize("name", ["sum", "mean", "max", "argmax"])
@pytest.mark.parametrize("dim", [-3, 2])
def test_reduction_axes_must_address_an_existing_dimension(typed_tensor, name, dim):
    with pytest.raises(IndexError):
        getattr(ortho, name)(typed_tensor([[1, 2], [3, 4]]), dim=dim)


@pytest.mark.parametrize("dim", [None, 0, 1])
def test_argmax_uses_the_first_tied_index_in_logical_order(
    operator_profile, typed_tensor, assert_output, dim,
):
    device, dtype = operator_profile
    values = np.array([[1, 5, 5], [5, 2, 5]])
    source = typed_tensor(values, dtype=dtype, device=device).transpose(0, 1)
    assert_output(ortho.argmax(source, dim=dim), np.argmax(values.T, axis=dim),
                  dtype="int64", device=device)
