import gc

import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("family", ["reshape", "transpose", "slice", "index", "flatten"])
def test_views_share_writes_in_both_directions(tensor, assert_tensor, family):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    view, coordinate = {
        "reshape": lambda: (base.reshape([3, 2]), (2, 0)),
        "transpose": lambda: (base.transpose(0, 1), (1, 1)),
        "slice": lambda: (base.slice(1, 1, 3), (1, 0)),
        "index": lambda: (base[1], (1,)),
        "flatten": lambda: (base.flatten(), (4,)),
    }[family]()

    view[coordinate].mul_(10)
    assert_tensor(base, [[1, 2, 3], [4, 50, 6]])
    base[1, 1].add_(tensor(2))
    assert view[coordinate].item() == 52


def test_sibling_views_observe_only_their_overlapping_coordinates(tensor, assert_tensor):
    base = tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    left = base.slice(1, 0, 3)
    right = base.slice(1, 1, 4)

    left[1, 1].mul_(10)
    left[0, 0].mul_(100)

    assert_tensor(left, [[100, 2, 3], [5, 60, 7]])
    assert_tensor(right, [[2, 3, 4], [60, 7, 8]])
    assert_tensor(base, [[100, 2, 3, 4], [5, 60, 7, 8]])


def test_a_view_keeps_its_values_alive_after_the_base_reference_is_deleted(tensor, assert_tensor):
    base = tensor([[1, 2, 3], [4, 5, 6]])
    view = base.transpose(0, 1).slice(0, 1, 3)
    del base
    gc.collect()

    assert_tensor(view, [[2, 5], [3, 6]])
    view[0, 1].mul_(3)
    assert_tensor(view, [[2, 15], [3, 6]])


def test_equal_values_do_not_make_separate_factory_results_alias(tensor, assert_tensor):
    first = tensor([2, 3])
    second = tensor([2, 3])
    first.mul_(4)
    assert_tensor(first, [8, 12])
    assert_tensor(second, [2, 3])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_numpy_import_owns_a_snapshot_instead_of_borrowing_array_memory(assert_tensor, dtype):
    array = np.array([[1, 2], [3, 4]], dtype=dtype)
    imported = ortho.from_numpy(array)
    array[0, 0] = 99
    assert_tensor(imported, [[1, 2], [3, 4]])
    imported.mul_(2)
    assert array.tolist() == [[99, 2], [3, 4]]


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("alpha", [0., 1., -.5])
def test_r2_partial_add_aliases_reject_without_mutation(tensor, reverse, alpha):
    base = tensor([1, 2, 3, 4], requires_grad=True)
    saved = (base * base).sum()
    destination, source = base.slice(0, 1, 4), base.slice(0, 0, 3)
    if reverse:
        destination, source = source, destination
    version = base._storage_version
    with pytest.raises(RuntimeError, match="partially overlapping"):
        destination.add_(source, alpha)
    assert base.tolist() == [1, 2, 3, 4]
    assert base._storage_version == version
    saved.backward()
    assert base.grad.tolist() == [2, 4, 6, 8]


@pytest.mark.parametrize("family", ["self", "exact-view", "siblings", "different-buffer", "empty", "endpoint"])
@pytest.mark.parametrize("alpha", [0., 1., -.5])
def test_r2_valid_add_aliases_preserve_values_and_version(tensor, family, alpha):
    base = tensor([1, 2, 3, 4, 5, 6])
    if family == "self":
        destination = source = base
    elif family == "exact-view":
        destination, source = base.slice(0, 1, 5), base.slice(0, 1, 5)
        assert destination is not source
    elif family == "siblings":
        destination, source = base.slice(0, 0, 3), base.slice(0, 3, 6)
    elif family == "different-buffer":
        destination, source = base, tensor([7, 8, 9, 10, 11, 12])
    elif family == "empty":
        destination, source = base.slice(0, 0, 0), base.slice(0, 3, 3)
    else:
        destination = source = base.slice(0, 6, 6)
    before = np.array(destination.tolist())
    expected = before + float(np.float32(alpha)) * np.array(source.tolist())
    version = base._storage_version
    destination.add_(source, alpha)
    assert destination.tolist() == pytest.approx(expected.tolist(), abs=1e-6, rel=1e-6)
    assert base._storage_version == version + 1


@pytest.mark.parametrize("alpha", [0., 1., -1.])
def test_r2_exact_add_alias_keeps_nonfinite_expression(tensor, alpha):
    values = np.array([np.inf, -np.inf, np.nan, 0., -0., 2.])
    target = tensor(values)
    with np.errstate(invalid="ignore"):
        expected = values + float(np.float32(alpha)) * values
    target.add_(target, alpha)
    np.testing.assert_allclose(target.tolist(), expected, equal_nan=True)


def test_r2_successful_exact_alias_still_invalidates_saved_values(tensor):
    leaf = tensor([1, 2], requires_grad=True)
    saved = (leaf * leaf).sum()
    version = leaf._storage_version
    leaf.add_(leaf.reshape([2]), 0.)
    assert leaf._storage_version == version + 1
    with pytest.raises(RuntimeError, match="modified in-place"):
        saved.backward()
    assert leaf.grad is None
