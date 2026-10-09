import pytest


def test_new_forward_passes_accumulate_until_zero_grad_clears_the_leaf(tensor, assert_grad):
    leaf = tensor([2, 4], requires_grad=True)
    (leaf * 2).backward()
    assert_grad(leaf, [2, 2])
    (leaf * 3).backward()
    assert_grad(leaf, [5, 5])
    leaf.zero_grad()
    assert leaf.grad is None
    (leaf * 7).backward()
    assert_grad(leaf, [7, 7])


def test_a_retained_graph_adds_another_contribution_and_can_then_be_released(tensor, assert_grad):
    leaf = tensor([2, -3], requires_grad=True)
    output = leaf * leaf
    output.backward(retain_graph=True)
    assert_grad(leaf, [4, -6])
    output.backward()
    assert_grad(leaf, [8, -12])
    with pytest.raises(RuntimeError):
        output.backward()
    assert_grad(leaf, [8, -12])
    assert output.grad is None


def test_overlapping_branches_of_a_shared_intermediate_are_summed_in_leaf_coordinates(tensor, assert_grad):
    leaf = tensor([[1, 2, 3, 4], [5, 6, 7, 8]], requires_grad=True)
    shared = leaf * leaf
    left = shared.slice(1, 0, 3) * 2
    right = shared.slice(1, 1, 4) * 5
    (left.sum() + right.sum()).backward()
    assert_grad(leaf, [[4, 28, 42, 40], [20, 84, 98, 80]])
    assert shared.grad is None
    assert left.grad is None
    assert right.grad is None


def test_backward_on_a_leaf_seeds_every_element_and_can_be_repeated(tensor, assert_grad):
    leaf = tensor([[2, 3], [5, 7]], requires_grad=True)
    leaf.backward()
    assert_grad(leaf, [[1, 1], [1, 1]])
    leaf.backward()
    assert_grad(leaf, [[2, 2], [2, 2]])


def test_a_shared_leaf_is_counted_for_every_use_in_a_single_expression(tensor, assert_grad):
    leaf = tensor([2, 3], requires_grad=True)
    ((leaf + leaf) + leaf * 3).backward()
    assert_grad(leaf, [5, 5])
