import orthon as ortho


def test_sibling_leaf_gradients_can_be_mutated_and_cleared_independently(tensor, assert_grad, assert_tensor):
    first = tensor([2, 3], requires_grad=True)
    second = tensor([5, 7], requires_grad=True)
    (first + second).backward()
    first.grad.mul_(5)
    assert_grad(first, [5, 5])
    assert_grad(second, [1, 1])
    first.zero_grad()
    assert first.grad is None
    assert_grad(second, [1, 1])
    assert_tensor(first, [2, 3])
    assert_tensor(second, [5, 7])


def test_leaves_sharing_data_still_own_separate_gradient_values(tensor, assert_grad, assert_tensor):
    base = tensor([[1, 2], [3, 4]], requires_grad=True)
    with ortho.no_grad():
        alias_leaf = base.reshape([4])
    alias_leaf.requires_grad_(True)
    (base.reshape([4]) + alias_leaf).backward()
    base.grad.mul_(2)
    assert_grad(base, [[2, 2], [2, 2]])
    assert_grad(alias_leaf, [1, 1, 1, 1])
    assert_tensor(base, [[1, 2], [3, 4]])
    assert_tensor(alias_leaf, [1, 2, 3, 4])


def test_a_gradient_does_not_alias_the_parameter_or_a_forward_constant(tensor, assert_grad, assert_tensor):
    parameter = tensor([1, 2], requires_grad=True)
    coefficient = tensor([3, 5])
    (parameter * coefficient).backward()
    parameter.grad.mul_(10)
    assert_tensor(parameter, [1, 2])
    assert_tensor(coefficient, [3, 5])
    coefficient.mul_(2)
    assert_grad(parameter, [30, 50])


def test_ownership_is_preserved_when_both_leaves_receive_strided_contributions(tensor, assert_grad):
    first = tensor([[1, 2], [3, 4]], requires_grad=True)
    second = tensor([[5, 6], [7, 8]], requires_grad=True)
    weights = tensor([[1, 3], [5, 7]])
    ((first + second).transpose(0, 1) * weights).backward()
    first.grad[0, 1].mul_(10)
    assert_grad(first, [[1, 50], [3, 7]])
    assert_grad(second, [[1, 5], [3, 7]])
