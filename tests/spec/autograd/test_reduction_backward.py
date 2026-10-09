import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("name", ["sum", "mean"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reduction_in_a_squared_chain_expands_gradients_over_the_reduced_axes(
    tensor, assert_grad, name, axis, keepdim,
):
    raw = np.arange(1, 13).reshape(2, 2, 3) / 4
    leaf = tensor(raw, requires_grad=True)
    transformed = leaf * leaf + 1
    reduced = getattr(ortho, name)(transformed, dim=axis, keepdim=keepdim)
    ((reduced * reduced) + reduced * 3).sum().backward()
    factor = raw.size if axis is None else raw.shape[axis]
    kept = getattr(np, name)(raw * raw + 1, axis=axis, keepdims=True)
    expanded = np.broadcast_to(2 * kept + 3, raw.shape)
    assert_grad(leaf, expanded * 2 * raw / (factor if name == "mean" else 1))
    assert reduced.grad is None


def test_chained_reductions_of_a_view_reach_the_original_shape(tensor, assert_grad):
    raw = np.arange(1, 13).reshape(2, 2, 3) / 4
    leaf = tensor(raw, requires_grad=True)
    first = (leaf * leaf).transpose(0, 2).mean(dim=0)
    second = first.sum(dim=-1, keepdim=True)
    (second * second).sum().backward()
    sums = (raw * raw).mean(axis=2).sum(axis=0)
    assert_grad(leaf, 4 * raw * sums[None, :, None] / 3)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_max_scatter_composes_with_a_nonuniform_upstream_chain(tensor, assert_grad, axis, keepdim):
    raw = np.array([[1., 7., 3.], [6., 2., 9.]])
    leaf = tensor(raw, requires_grad=True)
    reduced = ortho.max(leaf, dim=axis, keepdim=keepdim)
    ((reduced * reduced) + reduced * 3).sum().backward()
    expected = np.zeros_like(raw)
    if axis is None:
        expected[1, 2] = 21
    elif axis == 0:
        expected[1, 0], expected[0, 1], expected[1, 2] = 15, 17, 21
    else:
        expected[0, 1], expected[1, 2] = 17, 21
    assert_grad(leaf, expected)


def test_rank_zero_reductions_can_share_a_scalar_nonlinear_graph(tensor, assert_grad):
    leaf = tensor(np.array(2.), requires_grad=True)
    squared = leaf * leaf
    reduced = squared.sum() + squared.mean()
    (reduced * reduced).backward()
    assert_grad(leaf, np.array(128.))


@pytest.mark.parametrize("name", ["sum", "mean", "max"])
def test_rejected_empty_reduction_does_not_destroy_an_existing_empty_graph(tensor, assert_grad, name):
    leaf = tensor(np.empty((0, 3)), requires_grad=True)
    output = leaf * leaf + leaf * 2
    with pytest.raises(RuntimeError):
        getattr(ortho, name)(output)
    assert leaf.grad is None
    output.backward()
    assert_grad(leaf, np.empty((0, 3)))
