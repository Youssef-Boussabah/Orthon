import pytest

from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


@pytest.mark.parametrize("kind", [SGD, Adam])
def test_defaults_are_available_through_public_state(kind):
    opt = kind([], lr=0.125) if kind is SGD else kind([])
    state = opt.state_dict()
    assert state["lr"] == opt.get_lr() == (0.125 if kind is SGD else 0.001)
    assert state["weight_decay"] == 0
    assert state["decay_bias"] is False
    if kind is Adam:
        assert state["betas"] == (0.9, 0.999)
        assert state["eps"] == 1e-8
        assert state["t"] == 0
        assert state["m"] == state["v"] == []


def test_sgd_requires_an_explicit_learning_rate():
    with pytest.raises(TypeError):
        SGD([])


@pytest.mark.parametrize("kind", [SGD, Adam])
@pytest.mark.parametrize("params", [None, 3, 0.5])
def test_noniterable_parameter_input_is_rejected(kind, params):
    with pytest.raises(TypeError):
        kind(params, lr=0.125)


@pytest.mark.parametrize("betas", [(), (0.5,), (0.5, 0.75, 0.9)])
def test_adam_requires_two_beta_values(betas):
    with pytest.raises(ValueError):
        Adam([], betas=betas)


@pytest.mark.parametrize("kind", [SGD, Adam])
def test_empty_optimizer_can_step_and_clear_repeatedly(kind):
    opt = kind([], lr=0.125)
    for _ in range(3):
        assert opt.step() is None
        assert opt.zero_grad() is None
    assert opt.get_lr() == 0.125
    if kind is Adam:
        assert opt.state_dict()["t"] == 3
        assert opt.state_dict()["m"] == opt.state_dict()["v"] == []


@pytest.mark.parametrize("input_kind", ["list", "tuple", "generator"])
def test_optimizer_captures_tensor_iterable_at_construction(
    parameter, optimizer, assign_gradient, assert_tensor, input_kind,
):
    p, later = parameter([2.]), parameter([4.])
    inputs = [p]
    supplied = {"list": inputs, "tuple": tuple(inputs),
                "generator": (item for item in inputs)}[input_kind]
    opt = optimizer(supplied)
    inputs.append(later)
    assign_gradient(p, [1.])
    assign_gradient(later, [1.])
    assert opt.step() is None
    assert_tensor(p, [1.75])
    assert_tensor(later, [4.])


def test_learning_rate_change_applies_to_next_step(
    parameter, optimizer, assign_gradient, assert_tensor,
):
    p = parameter([2., -3.])
    assign_gradient(p, [1., -1.])
    opt = optimizer([p])
    opt.step()
    assert opt.set_lr(0.5) is None
    assert opt.get_lr() == opt.state_dict()["lr"] == 0.5
    opt.step()
    assert_tensor(p, [1.25, -2.25])


def test_raw_tensor_need_not_be_marked_or_require_grad_if_unused(
    parameter, optimizer, assert_tensor,
):
    p = parameter([[2., -3.]], requires_grad=False)
    opt = optimizer([p], weight_decay=0.5, decay_bias=True)
    assert opt.step() is None
    assert opt.zero_grad() is None
    assert_tensor(p, [[2., -3.]])
    assert p.requires_grad is False
    assert p.grad is None


def test_shared_module_discovery_produces_one_numerical_update(
    parameter, optimizer, optimizer_case, assert_tensor, assert_grad,
):
    kind, _, _ = optimizer_case
    p = parameter([2., 3.])
    model = Module()
    child = Module()
    shared = Parameter(p)
    model.left = shared
    child.right = shared
    model.child = child
    opt = optimizer(model.parameters())
    (model.left + model.child.right).sum().backward()
    opt.step()
    assert model.left is model.child.right is p
    assert list(model.parameters()) == [p]
    assert_tensor(p, [1.5, 2.5] if kind is SGD else [1.75, 2.75])
    assert_grad(p, [2., 2.])
    opt.zero_grad()
    assert p.grad is None
