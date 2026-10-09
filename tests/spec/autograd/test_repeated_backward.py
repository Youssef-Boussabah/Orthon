import numpy as np
import pytest


def test_zeroing_a_leaf_between_retained_traversals_preserves_the_graph(tensor, assert_grad):
    raw = np.array([1., -2., 3.])
    leaf = tensor(raw, requires_grad=True)
    shared = leaf * leaf + 1
    root = (shared * shared + leaf * 5).mean()
    expected = (4 * raw * (raw**2 + 1) + 5) / 3
    root.backward(retain_graph=True)
    assert_grad(leaf, expected)
    leaf.zero_grad()
    assert leaf.grad is None
    root.backward(retain_graph=True)
    assert_grad(leaf, expected)
    root.backward()
    assert_grad(leaf, 2 * expected)
    leaf.zero_grad()
    with pytest.raises(RuntimeError, match="freed graph"):
        root.backward(retain_graph=True)
    assert leaf.grad is None


def test_fresh_deep_graphs_accumulate_even_after_previous_graphs_are_released(tensor, assert_grad):
    raw = np.array([1., -2.])
    leaf = tensor(raw, requires_grad=True)
    expected = np.zeros(2)
    for scale in (2, 3, 5):
        transformed = leaf * scale + 1
        (transformed * transformed).sum().backward()
        expected += 2 * scale * (raw * scale + 1)
        assert_grad(leaf, expected)
    leaf.zero_grad()
    transformed = leaf * 7 + 1
    (transformed * transformed).sum().backward()
    assert_grad(leaf, 14 * (raw * 7 + 1))


def test_clearing_one_leaf_does_not_reset_other_leaf_accumulation_on_a_retained_graph(tensor, assert_grad):
    a, b = tensor([1., 2.], requires_grad=True), tensor([3., 4.], requires_grad=True)
    shared = a * b + a
    (shared * shared).sum().backward(retain_graph=True)
    assert_grad(a, [32., 100.])
    assert_grad(b, [8., 40.])
    a.zero_grad()
    (shared * 3).sum().backward()
    assert_grad(a, [12., 15.])
    assert_grad(b, [11., 46.])
