import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (2, 3, 4), (2, 0, 3)])
def test_rank_and_element_count_include_scalar_and_zero_dimensions(typed_tensor, shape):
    source = typed_tensor(np.zeros(shape))
    assert source.shape == shape
    assert source.ndim == len(shape)
    assert source.numel == int(np.prod(shape))
    assert len(source.tolist()) == source.numel


@pytest.mark.parametrize("factory", ["empty", "zeros", "ones", "full"])
def test_negative_factory_dimensions_are_invalid(factory):
    with pytest.raises(ValueError):
        getattr(ortho, factory)([2, -1], **({"value": 1} if factory == "full" else {}))


@pytest.mark.parametrize("shape", [(), (1,), (1, 1, 1)])
def test_item_accepts_exactly_one_element_regardless_of_rank(typed_tensor, shape):
    assert typed_tensor(np.full(shape, 3)).item() == 3


@pytest.mark.parametrize("shape", [(0,), (2,), (1, 2)])
def test_item_rejects_empty_or_multiple_elements(typed_tensor, shape):
    with pytest.raises(RuntimeError):
        typed_tensor(np.zeros(shape)).item()


def test_explicit_reshape_and_partial_flatten_preserve_logical_order(tensor, assert_tensor):
    values = np.arange(24).reshape(2, 3, 4)
    source = tensor(values)
    assert_tensor(source.reshape([4, 6]), values.reshape(4, 6))
    assert_tensor(source.flatten(1, -1), values.reshape(2, 12))
    assert_tensor(source.flatten(0, 1), values.reshape(6, 4))


def test_view_shape_transforms_do_not_require_an_arithmetic_dtype(
    storage_dtype, typed_tensor, assert_output,
):
    source = typed_tensor([[0, 1, 0], [1, 0, 1]], dtype=storage_dtype)
    assert_output(source.reshape([3, 2]), [[0, 1], [0, 1], [0, 1]],
                  dtype=storage_dtype, device="cpu")
    assert_output(source.transpose(0, 1).slice(0, 1, 3)[0], [1, 0],
                  dtype=storage_dtype, device="cpu")


@pytest.mark.parametrize("start,end", [(-4, 1), (2, 1), (0, 4)])
def test_explicit_slice_bounds_must_form_a_valid_interval(typed_tensor, start, end):
    with pytest.raises(IndexError):
        typed_tensor([[1, 2, 3], [4, 5, 6]]).slice(1, start, end)


@pytest.mark.parametrize("invalid", ["scalar", "reversed_axes", "strided"])
def test_flatten_requires_non_scalar_ordered_dimensions_and_contiguity(typed_tensor, invalid):
    with pytest.raises(RuntimeError):
        if invalid == "scalar":
            typed_tensor(1).flatten()
        elif invalid == "reversed_axes":
            typed_tensor([[1, 2], [3, 4]]).flatten(1, 0)
        else:
            typed_tensor([[1, 2], [3, 4]]).transpose(0, 1).flatten()


@pytest.mark.parametrize("operation", ["transpose", "slice", "flatten"])
@pytest.mark.parametrize("dim", [-3, 2])
def test_shape_operation_axes_must_address_existing_dimensions(typed_tensor, operation, dim):
    source = typed_tensor([[1, 2], [3, 4]])
    with pytest.raises(IndexError):
        if operation == "transpose":
            source.transpose(dim, 0)
        elif operation == "slice":
            source.slice(dim, 0, 1)
        else:
            source.flatten(dim, dim)


@pytest.mark.parametrize("index", [2, -3, (0, 0, 0)])
def test_integer_indexing_rejects_out_of_range_coordinates_or_too_many_axes(typed_tensor, index):
    with pytest.raises(IndexError):
        typed_tensor([[1, 2], [3, 4]])[index]


@pytest.mark.parametrize("index", [slice(None), (slice(None), 0), Ellipsis])
def test_python_slice_syntax_is_not_an_alias_for_the_explicit_slice_api(typed_tensor, index):
    # The rejection is promised; the binding's exception category is unsettled.
    with pytest.raises((TypeError, RuntimeError)):
        typed_tensor([[1, 2], [3, 4]])[index]
