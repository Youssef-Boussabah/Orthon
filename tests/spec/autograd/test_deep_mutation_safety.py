import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("writer", ["leaf", "intermediate", "intermediate-view", "disjoint-sibling"])
def test_stale_deep_dependency_rejects_before_any_reachable_leaf_gradient_changes(
    tensor, assert_grad, writer,
):
    first = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    old = tensor([[2., 3., 4.], [5., 6., 7.]], requires_grad=True)
    fresh = tensor([[3., 4., 5.], [6., 7., 8.]], requires_grad=True)
    (first * 2).backward()
    (old * 3).backward()
    shared = first * first + 1
    selected = shared.slice(0, 0, 1)
    sensitive = (selected * selected).sum()
    combined = sensitive + (old + fresh).mean()
    root = (combined * 2 + 1) * (combined * 2 + 1)
    target = {"leaf": lambda: first[0, 0], "intermediate": lambda: shared[0, 0],
              "intermediate-view": lambda: shared.transpose(0, 1)[0, 0],
              "disjoint-sibling": lambda: shared[1, 2]}[writer]()
    with ortho.no_grad():
        target.mul_(2)
    previous = ortho.is_grad_enabled()
    with pytest.raises(RuntimeError, match="modified in-place"):
        root.backward()
    assert ortho.is_grad_enabled() is previous
    assert_grad(first, np.full((2, 3), 2.))
    assert_grad(old, np.full((2, 3), 3.))
    assert fresh.grad is None
    assert shared.grad is None and sensitive.grad is None


def test_a_nontrainable_saved_constant_is_checked_through_a_longer_graph(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    coefficient = tensor([3., 5.])
    (leaf * 7).backward()
    root = ortho.tanh(leaf * coefficient + 2).mean()
    final = root * root + root * 3
    coefficient[1].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        final.backward()
    assert_grad(leaf, [7., 7.])
    assert coefficient.grad is None


@pytest.mark.parametrize("writer", ["leaf", "intermediate"])
def test_deep_derivative_allows_changes_to_values_it_does_not_need(tensor, assert_grad, writer):
    leaf = tensor([[1., 2.], [3., 4.]], requires_grad=True)
    intermediate = leaf * 2 + 3
    summary = intermediate.mean()
    root = summary * summary
    (leaf if writer == "leaf" else intermediate).mul_(5)
    root.backward()
    assert_grad(leaf, np.full((2, 2), 8.))


def test_retained_deep_graph_rejects_stale_values_but_a_fresh_graph_uses_new_values(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    shared = leaf * leaf + 1
    root = (shared * shared).sum()
    root.backward(retain_graph=True)
    assert_grad(leaf, [8., 40.])
    leaf.mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        root.backward()
    assert_grad(leaf, [8., 40.])
    fresh = leaf * leaf + 1
    (fresh * fresh).sum().backward()
    assert_grad(leaf, [48., 312.])


def test_rejected_mutation_preserves_every_path_in_a_deep_retained_graph(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    shared = leaf * leaf + 1
    root = (shared * shared + shared * 3).sum()
    root.backward(retain_graph=True)
    with pytest.raises(RuntimeError):
        shared.add_(tensor([1., 2., 3.]))
    root.backward()
    assert_grad(leaf, [28., 104.])
