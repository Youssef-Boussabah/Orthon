import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


def test_step_before_backward_and_after_zero_grad_leaves_values_unchanged(
    parameter, optimizer, assert_tensor,
):
    p = parameter([[3., -2.]])
    opt = optimizer([p], weight_decay=0.5)
    assert opt.zero_grad() is None
    opt.step()
    assert_tensor(p, [[3., -2.]])
    p.sum().backward()
    opt.zero_grad()
    opt.step()
    assert_tensor(p, [[3., -2.]])
    assert p.grad is None


def test_step_preserves_gradient_values_and_ownership(
    parameter, optimizer, assign_gradient, assert_tensor,
):
    p = parameter([3., -2.])
    assign_gradient(p, [1., -1.])
    captured = p.grad
    opt = optimizer([p])
    opt.step()
    assert_tensor(p, [2.75, -1.75])
    assert_tensor(captured, [1., -1.])
    assert_tensor(p.grad, [1., -1.])
    captured.mul_(2.)
    assert_tensor(p.grad, [2., -2.])
    assert captured.requires_grad is False
    assert captured.dtype == p.dtype
    assert captured.device == p.device


def test_repeated_step_reuses_existing_gradient(
    parameter, optimizer, assign_gradient, assert_tensor,
):
    p = parameter([3., -2.])
    assign_gradient(p, [1., -1.])
    opt = optimizer([p])
    for expected in ([2.75, -1.75], [2.5, -1.5], [2.25, -1.25]):
        opt.step()
        assert_tensor(p, expected)
        assert_tensor(p.grad, [1., -1.])


def test_backward_accumulation_is_used_by_the_next_update(
    parameter, optimizer, optimizer_case, assert_grad, assert_tensor,
):
    kind, _, _ = optimizer_case
    p = parameter([3., -2.])
    opt = optimizer([p], **({"betas": (0.5, 0.75), "eps": 1.} if kind is Adam else {}))
    (p * 2).sum().backward()
    (p * -1).sum().backward()
    assert_grad(p, [1., 1.])
    opt.step()
    amount = 0.25 if kind is SGD else 0.125
    assert_tensor(p, [3. - amount, -2. - amount])
    assert_grad(p, [1., 1.])


@pytest.mark.parametrize("clear_with", ["optimizer", "module"])
def test_zero_grad_clears_owner_but_keeps_captured_gradient_alive(
    parameter, optimizer, assert_tensor, assert_grad, clear_with,
):
    p = parameter([3., -2.])
    model = Module()
    model.weight = Parameter(p)
    opt = optimizer(model.parameters())
    (p * 2).sum().backward()
    old_gradient = p.grad
    opt.step()
    receiver = opt if clear_with == "optimizer" else model
    assert receiver.zero_grad() is None
    assert receiver.zero_grad() is None
    assert p.grad is None
    assert_tensor(old_gradient, [2., 2.])
    p.sum().backward()
    assert_grad(p, [1., 1.])
    old_gradient.mul_(3.)
    assert_grad(p, [1., 1.])


def test_optimizer_zero_grad_is_limited_to_supplied_tensors(
    parameter, optimizer, assert_grad,
):
    selected, omitted = parameter([3.]), parameter([4.])
    (selected + omitted).sum().backward()
    opt = optimizer([selected])
    opt.zero_grad()
    assert selected.grad is None
    assert_grad(omitted, [1.])


def test_step_restores_callers_gradient_mode(parameter, optimizer, assign_gradient):
    p = parameter([3.])
    assign_gradient(p, [1.])
    opt = optimizer([p])
    assert ortho.is_grad_enabled()
    opt.step()
    assert ortho.is_grad_enabled()
    with ortho.no_grad():
        opt.step()
        assert ortho.is_grad_enabled() is False
    assert ortho.is_grad_enabled()
    assert p.requires_grad and p.is_leaf


def test_step_invalidates_a_graph_that_saved_parameter_values(
    parameter, optimizer, assert_grad,
):
    p = parameter([3.])
    opt = optimizer([p])
    saved = (p * p).sum()
    p.sum().backward()
    opt.step()
    with pytest.raises(RuntimeError):
        saved.backward()
    assert_grad(p, [1.])


def test_intermediate_without_owned_grad_is_not_updated(
    parameter, optimizer, assert_tensor, assert_grad,
):
    p = parameter([3.])
    intermediate = p * 2
    opt = optimizer([intermediate])
    intermediate.sum().backward()
    assert intermediate.grad is None
    opt.step()
    assert_tensor(intermediate, [6.])
    assert_grad(p, [2.])
