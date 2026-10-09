import numpy as np
import pytest


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_nonlinear_intermediate_collects_direct_and_indirect_paths(
    tensor, assert_grad, reverse,
):
    raw = np.array([1., -2., 3.])
    leaf = tensor(raw, requires_grad=True)
    shared = leaf * leaf + 2
    branches = [shared * shared, shared * 3, leaf * 5]
    if reverse:
        branches.reverse()
    ((branches[0] + branches[1]) + branches[2]).mean().backward()
    assert_grad(leaf, (4 * raw * (raw**2 + 2) + 6 * raw + 5) / 3)
    assert shared.grad is None
    assert all(branch.grad is None for branch in branches)


def test_reused_intermediates_in_a_recurrent_dag_count_every_path(tensor, assert_grad):
    leaf = tensor([1., 2.], requires_grad=True)
    shared = leaf
    for _ in range(6):
        shared = shared * 2 + shared + leaf
    shared.backward()
    assert_grad(leaf, [1093., 1093.])
    assert shared.grad is None


def test_path_accumulation_and_repeated_traversal_are_separate_contributions(tensor, assert_grad):
    leaf = tensor([1., -2.], requires_grad=True)
    shared = leaf * leaf
    consumers = [shared * coefficient for coefficient in (2, 3, -5, 7, 11)]
    total = consumers[0]
    for consumer in consumers[1:]:
        total = total + consumer
    total.backward(retain_graph=True)
    assert_grad(leaf, [36., -72.])
    total.backward()
    assert_grad(leaf, [72., -144.])
    assert all(consumer.grad is None for consumer in consumers)


@pytest.mark.parametrize("reverse", [False, True])
def test_retained_separate_outputs_can_contribute_through_one_shared_ancestor(
    tensor, assert_grad, reverse,
):
    raw = np.array([1., -2., 3.])
    leaf = tensor(raw, requires_grad=True)
    shared = leaf * leaf + 1
    first, second = (shared * 2).sum(), (shared * shared).mean()
    outputs = [first, second]
    if reverse:
        outputs.reverse()
    for output in outputs:
        output.backward(retain_graph=True)
    assert_grad(leaf, 4 * raw + 4 * raw * (raw**2 + 1) / 3)
    assert shared.grad is None and first.grad is None and second.grad is None


def test_disconnected_stale_branch_is_not_executed_or_accumulated(tensor, assert_grad):
    active = tensor([1., 2.], requires_grad=True)
    unused = tensor([3., 4.], requires_grad=True)
    (unused * 7).backward()
    abandoned = (unused * unused + 2).mean()
    unused.mul_(2)
    root = ((active * active + 1) * 3).mean()
    root.backward()
    assert_grad(active, [3., 6.])
    assert_grad(unused, [7., 7.])
    assert abandoned.grad is None
