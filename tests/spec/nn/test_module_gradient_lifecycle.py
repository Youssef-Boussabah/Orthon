import pytest

from orthon.nn import Module, Parameter
from orthon import optim


def test_zero_grad_before_backward_and_after_accumulation_is_recursive(module_factory, assert_grad):
    model = module_factory()
    weight, bias = model.weight, model.branch.bias
    assert model.zero_grad() is None
    assert all(p.grad is None for p in model.parameters())
    for scale in (2, 3):
        (model.weight.sum() * scale + model.branch.bias.sum() * (scale + 1)).backward()
    assert_grad(weight, [[5, 5], [5, 5]])
    assert_grad(bias, [7, 7])
    for _ in range(2):
        model.zero_grad()
        assert model.weight is weight
        assert model.branch.bias is bias
        assert weight.grad is bias.grad is None
    (model.weight.sum() * 4 + model.branch.bias.sum() * 6).backward()
    assert_grad(weight, [[4, 4], [4, 4]])
    assert_grad(bias, [6, 6])


def test_zero_grad_clears_shared_parameters_but_leaves_unregistered_and_buffer_gradients(tensor, assert_grad):
    source = tensor([2, 3])
    shared = Parameter(source)
    model = Module()
    model.a = shared
    model.child = Module()
    model.child.b = shared
    model.register_buffer("state", tensor([4, 5], requires_grad=True))
    model.external = tensor([6, 7], requires_grad=True)
    (model.a * 2 + model.child.b * 3 + model.state * 4 + model.external * 5).sum().backward()
    assert_grad(source, [5, 5])
    model.zero_grad()
    assert source.grad is None
    assert model.child.b.grad is None
    assert_grad(model.state, [4, 4])
    assert_grad(model.external, [5, 5])


def test_unused_registered_parameters_do_not_receive_gradients(module_factory, assert_grad):
    model = module_factory()
    model.weight.sum().backward()
    assert_grad(model.weight, [[1, 1], [1, 1]])
    assert model.branch.bias.grad is None
    assert all(b.grad is None for b in model.buffers())


@pytest.mark.parametrize("optimizer_name", ["SGD", "Adam"])
def test_float32_optimizer_steps_preserve_registered_parameter_objects_and_gradients(
    operator_device, typed_tensor, optimizer_name,
):
    model = Module()
    model.weight = Parameter(typed_tensor([[1, 2], [3, 4]], device=operator_device))
    model.branch = Module()
    model.branch.bias = Parameter(typed_tensor([5, 6], device=operator_device))
    originals = list(model.parameters())
    values = [p.tolist() for p in originals]
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=.25)
    (model.weight.sum() + model.branch.bias.sum() * 2).backward()
    gradients = [p.grad.tolist() for p in originals]
    optimizer.step()
    assert all(current is original for current, original in zip(model.parameters(), originals))
    assert all(p.is_leaf and p.requires_grad for p in model.parameters())
    assert [p.grad.tolist() for p in originals] == gradients
    assert all(p.tolist() != before for p, before in zip(originals, values))
    optimizer.zero_grad()
    assert all(p.grad is None for p in originals)


def test_replacing_registration_does_not_retarget_an_existing_optimizer(tensor, assert_tensor):
    model = Module()
    original = tensor([2, 3])
    model.weight = Parameter(original)
    optimizer = optim.SGD(model.parameters(), lr=.25)
    (model.weight * 2).sum().backward()
    replacement = tensor([10, 20])
    model.weight = Parameter(replacement)
    model.weight.sum().backward()
    optimizer.step()
    assert_tensor(original, [1.5, 2.5])
    assert_tensor(model.weight, [10, 20])
    model.zero_grad()
    assert replacement.grad is None
    assert original.grad.tolist() == [2, 2]
