import gc

import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import SGD


def test_parameter_registration_marks_the_original_leaf_tensor(tensor, assert_grad):
    source = tensor([2, 3])
    assert source.requires_grad is False
    marked = Parameter(source)
    assert source.requires_grad is True
    assert source.is_leaf is True
    assert source.grad is None
    model = Module()
    model.weight = marked
    assert model.weight is source
    assert list(model.parameters()) == [source]
    (model.weight * 3).sum().backward()
    assert_grad(source, [3, 3])


@pytest.mark.parametrize("value", [None, 2, [1, 2], {"weight": 1}, Module()])
def test_parameter_requires_a_tensor(value):
    with pytest.raises(TypeError):
        Parameter(value)


def test_registration_keeps_parameter_and_child_values_alive(tensor, assert_tensor):
    source = tensor([2, 3])
    child = Module()
    child.weight = Parameter(source)
    parent = Module()
    parent.child = child
    del source, child
    gc.collect()
    assert_tensor(parent.child.weight, [2, 3])
    parent.child.weight.mul_(2)
    assert_tensor(next(parent.parameters()), [4, 6])


def test_shared_parameter_has_one_unnamed_entry_but_every_registered_name(tensor):
    source = tensor([2, 3])
    shared = Parameter(source)
    model = Module()
    model.a = shared
    model.b = shared
    model.branch = Module()
    model.branch.c = shared
    assert list(model.parameters()) == [source]
    assert [name for name, _ in model.named_parameters()] == ["a", "b", "branch.c"]
    assert all(value is source for _, value in model.named_parameters())
    assert list(model.state_dict()) == ["a", "b", "branch.c"]
    assert all(value is source for value in model.state_dict().values())


def test_shared_uses_accumulate_one_gradient_and_one_optimizer_update(tensor, assert_grad, assert_tensor):
    source = tensor([2, 3])
    shared = Parameter(source)
    model = Module()
    model.a = shared
    model.b = shared
    optimizer = SGD(model.parameters(), lr=.25)
    (model.a * 2 + model.b * 3).sum().backward()
    assert_grad(source, [5, 5])
    assert model.a is model.b
    optimizer.step()
    assert_tensor(model.a, [.75, 1.75])
    assert_tensor(model.b, [.75, 1.75])
    optimizer.zero_grad()
    assert model.a.grad is None
    assert model.b.grad is None


def test_freezing_a_registered_parameter_does_not_remove_its_ownership(tensor):
    source = tensor([2, 3])
    model = Module()
    model.weight = Parameter(source)
    source.requires_grad_(False)
    assert list(model.parameters()) == [source]
    assert dict(model.named_parameters()) == {"weight": source}
    assert model.state_dict()["weight"] is source
    assert source.is_leaf is True


def test_distinct_leaves_sharing_values_remain_separately_discovered_parameters(tensor, assert_grad):
    first = tensor([2, 3])
    with ortho.no_grad():
        second = first.reshape([2])
    model = Module()
    model.a, model.b = Parameter(first), Parameter(second)
    assert list(model.parameters()) == [first, second]
    assert model.a is not model.b
    (model.a * 2 + model.b * 3).sum().backward()
    assert_grad(first, [2, 2])
    assert_grad(second, [3, 3])
    first.grad.mul_(2)
    assert_grad(second, [3, 3])
    first.mul_(2)
    assert second.tolist() == [4, 6]
