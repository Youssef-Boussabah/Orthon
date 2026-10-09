import numpy as np

import orthon as ortho


def test_nested_view_copy_nonlinearity_and_direct_branch_scatter_to_base(tensor, assert_grad):
    raw = np.arange(1, 13).reshape(3, 4) / 4
    leaf = tensor(raw, requires_grad=True)
    shared = leaf * leaf + 1
    selected = shared.slice(0, 1, 3).transpose(0, 1).slice(0, 1, 4)
    flattened = selected.contiguous().reshape([6])
    weights = np.arange(1, 7)
    ((flattened * flattened * tensor(weights)).mean() + (shared * .25).mean()).backward()
    expected = raw / 24
    expected[1:3, 1:4] += 4 * raw[1:3, 1:4] * (raw[1:3, 1:4]**2 + 1) * weights.reshape(3, 2).T / 6
    assert_grad(leaf, expected)
    assert shared.grad is None and selected.grad is None and flattened.grad is None


def test_two_views_of_one_base_feed_both_sides_of_a_gram_matrix(tensor, assert_grad):
    raw = np.array([[1., 2., 3.], [4., 5., 6.]]) / 4
    leaf = tensor(raw, requires_grad=True)
    gram = leaf.transpose(0, 1) @ leaf
    weights = np.arange(1, 10).reshape(3, 3) / 2
    ((gram * gram) * tensor(weights)).mean().backward()
    upstream = 2 * (raw.T @ raw) * weights / 9
    assert_grad(leaf, (upstream @ raw.T).T + raw @ upstream)


def test_scalar_index_of_a_computed_transposed_view_routes_a_deep_derivative(tensor, assert_grad):
    leaf = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    shared = leaf * leaf + 1
    scalar = shared.transpose(0, 1)[-1, -1]
    ((scalar * 2 + 3) * scalar).backward()
    expected = np.zeros((2, 3))
    expected[1, 2] = (4 * 37 + 3) * 12
    assert_grad(leaf, expected)
    assert scalar.grad is None


def test_a_strided_leaf_receives_gradient_in_its_own_logical_coordinates(tensor, assert_grad):
    owner = tensor([[.25, .5, .75], [1., 1.25, 1.5]])
    leaf = owner.transpose(0, 1).requires_grad_(True)
    raw = np.array([[.25, 1.], [.5, 1.25], [.75, 1.5]])
    activated = ortho.tanh(leaf * 2)
    ((activated * activated).sum() + (leaf * 3).mean()).backward()
    assert_grad(leaf, 4 * np.tanh(raw * 2) * (1 - np.tanh(raw * 2)**2) + .5)
    assert owner.grad is None


def test_empty_view_can_flow_through_shared_arithmetic_and_transpose(tensor, assert_grad):
    leaf = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    empty = leaf.slice(1, 1, 1)
    output = (empty * empty + empty * 3).transpose(0, 1)
    output.backward()
    assert output.shape == (0, 2)
    assert_grad(leaf, np.zeros((2, 3)))
