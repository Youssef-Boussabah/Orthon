import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("operation", ["add_", "mul_"])
def test_in_place_arithmetic_returns_the_target_and_updates_existing_aliases(tensor, assert_tensor, operation):
    target = tensor([1, 2, 3])
    alias = target.reshape([3, 1])
    if operation == "add_":
        returned = target.add_(tensor([2, 3, 4]), alpha=2)
        expected = [[5], [8], [11]]
    else:
        returned = target.mul_(3)
        expected = [[3], [6], [9]]
    assert returned is target
    assert_tensor(alias, expected)


def test_a_leaf_mutation_changes_data_without_becoming_a_differentiable_step(tensor, assert_grad):
    leaf = tensor([2, 3], requires_grad=True)
    leaf.mul_(4)
    assert leaf.requires_grad is True
    assert leaf.is_leaf is True
    assert leaf.grad is None
    (leaf * 2).backward()
    assert_grad(leaf, [2, 2])


def test_in_place_add_can_use_the_same_values_as_its_rhs(tensor, assert_tensor):
    target = tensor([1, 2, 3])
    target.add_(target, alpha=2)
    assert_tensor(target, [3, 6, 9])


def test_rejected_in_place_arithmetic_does_not_partially_change_data(tensor, assert_tensor):
    target = tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(RuntimeError):
        target.add_(tensor([1, 2, 3]))
    with pytest.raises(RuntimeError):
        target.transpose(0, 1).mul_(2)
    assert_tensor(target, [[1, 2, 3], [4, 5, 6]])


def test_cpu_float32_fill_writes_only_the_selected_strided_region(assert_tensor):
    base = ortho.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"))
    view = base.transpose(0, 1).slice(0, 1, 3)
    ortho.fill_(view, 9)
    assert_tensor(base, [[1, 9, 9], [4, 9, 9]])


def test_cpu_float32_indexed_assignment_changes_the_existing_view(assert_tensor):
    base = ortho.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"))
    view = base.transpose(0, 1)
    view[1, 0] = 17
    assert_tensor(base, [[1, 17, 3], [4, 5, 6]])
    assert_tensor(view, [[1, 4], [17, 5], [3, 6]])
