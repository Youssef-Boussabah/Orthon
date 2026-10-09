import numpy as np


def test_a_weighted_view_and_copy_chain_scatters_to_the_original_leaf(tensor, assert_grad):
    leaf = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], requires_grad=True)
    selected = leaf.slice(0, 1, 3).transpose(0, 1).slice(0, 1, 3)
    flattened = selected.contiguous().reshape([4])
    (flattened * tensor([2, 3, 5, 7])).backward()
    assert_grad(leaf, [[0, 0, 0, 0], [0, 2, 5, 0], [0, 3, 7, 0]])
    assert selected.grad is None
    assert flattened.grad is None


def test_broadcast_backward_restores_the_original_shapes_after_a_transpose(tensor, assert_grad):
    matrix = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    bias = tensor([2, 3], requires_grad=True)
    weights = tensor([[1, 2], [3, 5], [7, 11]])
    ((matrix.transpose(0, 1) + bias) * weights).backward()
    assert_grad(matrix, [[1, 3, 7], [2, 5, 11]])
    assert_grad(bias, [11, 18])


def test_a_scalar_view_scatters_its_seed_to_one_original_coordinate(tensor, assert_grad):
    leaf = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    selected = leaf[-1, -2]
    selected.backward()
    assert_grad(leaf, [[0, 0, 0], [0, 1, 0]])
    assert selected.grad is None


def test_an_empty_view_backward_contributes_a_zero_parent_gradient(tensor, assert_grad):
    leaf = tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    leaf.slice(1, 1, 1).backward()
    assert_grad(leaf, np.zeros((2, 3)))


def test_a_differentiable_copy_uses_its_snapshot_even_if_the_source_is_mutated(tensor, assert_grad, assert_tensor):
    leaf = tensor([2, -3], requires_grad=True)
    copied = leaf.to(leaf.device)
    output = copied * copied
    leaf.mul_(7)
    assert_tensor(copied, [2, -3])
    output.backward()
    assert_grad(leaf, [4, -6])
    assert copied.grad is None
