import numpy as np
import pytest


@pytest.mark.parametrize("layout", ["plain", "lhs-transposed", "rhs-transposed", "both-transposed"])
@pytest.mark.parametrize("batched", [False, True])
def test_weighted_shared_matmul_output_reaches_both_logical_operand_owners(
    tensor, assert_grad, layout, batched,
):
    a = np.array([[1., 2., -1.], [.5, -1., 3.]])
    b = np.array([[2., 1.], [-1., 3.], [.5, -2.]])
    if batched:
        a, b = np.stack((a, a / 2 + 1)), np.stack((b, b / 2 - 1))
    a_transposed = layout in ("lhs-transposed", "both-transposed")
    b_transposed = layout in ("rhs-transposed", "both-transposed")
    lhs = tensor(np.swapaxes(a, -1, -2) if a_transposed else a, requires_grad=True)
    rhs = tensor(np.swapaxes(b, -1, -2) if b_transposed else b, requires_grad=True)
    left = lhs.transpose(-1, -2) if a_transposed else lhs
    right = rhs.transpose(-1, -2) if b_transposed else rhs
    product = left @ right
    expected_product = a @ b
    weights = np.arange(1, product.numel() + 1).reshape(expected_product.shape) / 4
    root = (product.transpose(-1, -2) * tensor(np.swapaxes(weights, -1, -2))).sum() + (product * product).mean()
    root.backward()
    upstream = weights + 2 * expected_product / expected_product.size
    a_grad = upstream @ np.swapaxes(b, -1, -2)
    b_grad = np.swapaxes(a, -1, -2) @ upstream
    assert_grad(lhs, np.swapaxes(a_grad, -1, -2) if a_transposed else a_grad)
    assert_grad(rhs, np.swapaxes(b_grad, -1, -2) if b_transposed else b_grad)
    assert product.grad is None


def test_one_matrix_used_twice_in_a_product_receives_both_partial_derivatives(tensor, assert_grad):
    raw = np.array([[1., 2.], [-1., 3.]])
    leaf = tensor(raw, requires_grad=True)
    product = leaf @ leaf
    weights = np.array([[1., 2.], [3., 5.]])
    ((product + 1) * tensor(weights)).sum().backward()
    assert_grad(leaf, weights @ raw.T + raw.T @ weights)


def test_two_matmuls_with_a_nonlinear_bridge_obey_the_matrix_chain_rule(tensor, assert_grad):
    a = np.array([[1., 2.], [-1., .5]])
    b = np.array([[.5, 1.], [1., -.5]])
    c = np.array([[1.], [2.]])
    first, middle, last = [tensor(v, requires_grad=True) for v in (a, b, c)]
    bridge = first @ middle
    activated = bridge * bridge + 1
    output = activated @ last
    (output * output).mean().backward()
    v = a @ b
    h = v * v + 1
    g_output = h @ c
    g_bridge = (g_output @ c.T) * 2 * v
    assert_grad(first, g_bridge @ b.T)
    assert_grad(middle, a.T @ g_bridge)
    assert_grad(last, h.T @ g_output)


def test_singleton_matrix_dimensions_preserve_rank_and_gradient_shape(tensor, assert_grad):
    lhs = tensor([[2.]], requires_grad=True)
    rhs = tensor([[3.]], requires_grad=True)
    product = lhs @ rhs
    ((product * product) + product * 2).mean().backward()
    assert product.shape == (1, 1)
    assert_grad(lhs, [[42.]])
    assert_grad(rhs, [[28.]])
