import numpy as np
import pytest


def test_chained_views_keep_offsets_relative_to_the_original_values(tensor, assert_tensor):
    base = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    view = base.slice(0, 1, 3)[-1].slice(0, 1, 4).reshape([3, 1]).transpose(0, 1)
    assert_tensor(view, [[10, 11, 12]])
    view[0, 1].mul_(2)
    assert_tensor(base, [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 22, 12]])


def test_integer_indexing_drops_dimensions_and_can_return_a_scalar_view(tensor, assert_tensor):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    assert_tensor(base[-1], [4, 5, 6])
    scalar = base[-1, -2]
    assert scalar.shape == ()
    assert scalar.numel == 1
    scalar.mul_(2)
    assert_tensor(base, [[1, 2, 3], [4, 10, 6]])


def test_an_empty_slice_has_no_logical_elements(tensor, assert_tensor):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    empty = base.slice(1, 2, 2)
    assert_tensor(empty, np.empty((2, 0)))
    assert empty.numel == 0
    assert_tensor(base, [[1, 2, 3], [4, 5, 6]])


def test_reshape_requires_contiguity_instead_of_silently_copying(tensor, assert_tensor):
    source = tensor([[1, 2, 3], [4, 5, 6]]).transpose(0, 1)
    with pytest.raises(RuntimeError):
        source.reshape([6])
    assert_tensor(source, [[1, 4], [2, 5], [3, 6]])
    assert_tensor(source.contiguous().reshape([6]), [1, 4, 2, 5, 3, 6])


@pytest.mark.parametrize("shape", [[5], [-1]])
def test_reshape_rejects_element_count_changes_and_inferred_dimensions(tensor, assert_tensor, shape):
    source = tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        source.reshape(shape)
    assert_tensor(source, [[1, 2, 3], [4, 5, 6]])


def test_slice_normalizes_negative_bounds_but_rejects_out_of_range_bounds(tensor, assert_tensor):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    assert_tensor(base.slice(-1, -3, -1), [[1, 2], [4, 5]])
    with pytest.raises(IndexError):
        base.slice(1, 0, 4)
    assert_tensor(base, [[1, 2, 3], [4, 5, 6]])
