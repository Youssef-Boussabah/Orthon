import numpy as np
import pytest


@pytest.mark.parametrize("shape", [(), (1,), (1, 1), (2, 3)])
def test_nonscalar_deep_output_uses_an_implicit_one_at_every_output_coordinate(tensor, assert_grad, shape):
    raw = np.arange(1, np.prod(shape, dtype=int) + 1).reshape(shape) / 2
    leaf = tensor(raw, requires_grad=True)
    shared = leaf * leaf + 1
    output = shared * shared + leaf * 3
    output.backward()
    assert_grad(leaf, 4 * raw * (raw**2 + 1) + 3)


def test_weighting_public_outputs_expresses_a_nonuniform_vector_jacobian_product(tensor, assert_grad):
    raw = np.array([[1., 2., 3.], [4., 5., 6.]]) / 2
    weights = np.array([[1., -2.], [3., 5.], [-1., 7.]])
    leaf = tensor(raw, requires_grad=True)
    shifted = leaf * 2 + 1
    output = (shifted * shifted).transpose(0, 1)
    (output * tensor(weights)).sum().backward()
    assert_grad(leaf, 4 * (raw * 2 + 1) * weights.T)


def test_combining_outputs_before_backward_matches_the_sum_of_their_chain_rules(tensor, assert_grad):
    raw = np.array([1., 2., 3.])
    leaf = tensor(raw, requires_grad=True)
    shared = leaf * leaf + 1
    vector = shared * 3
    scalar = (shared * shared).mean()
    root = (vector * tensor([2., 3., 5.])).sum() + scalar * 7
    root.backward()
    assert_grad(leaf, 6 * raw * [2., 3., 5.] + 28 * raw * (raw**2 + 1) / 3)
