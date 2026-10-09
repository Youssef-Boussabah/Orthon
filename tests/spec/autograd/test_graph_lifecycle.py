import gc

import numpy as np
import pytest

import orthon as ortho


def test_releasing_a_deep_root_also_prevents_reuse_of_reached_intermediates(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    middle = leaf * leaf + 1
    root = (middle * middle).mean()
    root.backward()
    assert_grad(leaf, [4., 20.])
    for released in (root, middle):
        with pytest.raises(RuntimeError, match="freed graph"):
            released.backward()
        assert_grad(leaf, [4., 20.])
        assert released.requires_grad and not released.is_leaf
        assert released.grad is None
    assert root.item() == 14.5
    assert middle.tolist() == [2., 5.]


def test_disjoint_graphs_sharing_only_a_leaf_have_independent_lifetimes(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    first = ((leaf * leaf + 1) * 3).sum()
    second = ((leaf * 2 + 1) * (leaf * 2 + 1)).sum()
    first.backward()
    assert_grad(leaf, [6., 12.])
    second.backward()
    assert_grad(leaf, [18., 32.])
    assert first.grad is None and second.grad is None


def test_forward_locals_can_be_discarded_while_the_output_remains_executable(tensor, assert_grad):
    leaf = tensor([.25, .5, .75], requires_grad=True)

    def build():
        coefficient = tensor([1., 2., 3.])
        interior = leaf * coefficient + 1
        return ortho.tanh(interior).mean()

    root = build()
    gc.collect()
    root.backward()
    raw = np.array([.25, .5, .75]) * [1., 2., 3.] + 1
    assert_grad(leaf, (1 - np.tanh(raw)**2) * [1., 2., 3.] / 3)


def test_dropping_an_unused_output_does_not_consume_a_shared_graph(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    shared = leaf * leaf + 1
    abandoned = (shared * 7).sum()
    retained = (shared * shared).sum()
    del abandoned
    gc.collect()
    retained.backward()
    assert_grad(leaf, [8., 40.])


@pytest.mark.parametrize("caller_grad_mode", [True, False])
def test_retained_backward_computes_without_recording_and_restores_caller_mode(
    tensor, assert_grad, caller_grad_mode,
):
    leaf = tensor([1., 2.], requires_grad=True)
    root = ((leaf * leaf + 1) * (leaf * 3 + 2)).sum()
    previous = ortho.is_grad_enabled()
    try:
        ortho.set_grad_enabled(caller_grad_mode)
        root.backward(retain_graph=True)
        assert_grad(leaf, [16., 47.])
        assert ortho.is_grad_enabled() is caller_grad_mode
        with pytest.raises(RuntimeError):
            leaf.grad.backward()
        assert ortho.is_grad_enabled() is caller_grad_mode
        root.backward()
        assert_grad(leaf, [32., 94.])
    finally:
        ortho.set_grad_enabled(previous)
