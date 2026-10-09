import numpy as np
import pytest

from orthon.optim import SGD


@pytest.mark.parametrize("values,gradient", [
    (3., -2.),
    ([3., -2., 1.], [2., -4., 0.]),
    ([[3., -2.], [1., 4.]], [[2., -4.], [0., 1.]]),
])
@pytest.mark.parametrize("lr", [0., 0.125, 0.5])
def test_plain_update_preserves_parameter_properties(
    tensor, tensor_case, assert_output, values, gradient, lr,
):
    device, dtype = tensor_case
    p = tensor(values, requires_grad=True)
    p.set_grad(tensor(gradient))
    opt = SGD([p], lr=lr)
    assert opt.step() is None
    assert_output(p, np.asarray(values) - lr * np.asarray(gradient), dtype=dtype, device=device)
    assert p.requires_grad is True
    assert p.is_leaf is True
    assert p.grad.requires_grad is False


def test_float64_update_with_decimal_learning_rate(
    typed_tensor,
):
    p = typed_tensor([2., -3.], dtype="float64").requires_grad_(True)
    p.set_grad(typed_tensor([1., -2.], dtype="float64"))
    SGD([p], lr=0.1).step()
    assert p.dtype == "float64"
    assert p.tolist() == pytest.approx([1.9, -2.8], rel=0, abs=1e-15)


@pytest.mark.parametrize("values", [2., [2., -4.], [[2., -4.]], [[[2., -4.]]]])
@pytest.mark.parametrize("decay_bias", [False, True])
def test_decay_is_decoupled_and_selected_by_rank(
    tensor, assert_tensor, values, decay_bias,
):
    initial = np.asarray(values)
    gradient = np.full(initial.shape, 2.)
    p = tensor(values, requires_grad=True)
    p.set_grad(tensor(gradient))
    SGD([p], lr=0.25, weight_decay=0.5, decay_bias=decay_bias).step()
    scale = 0.875 if decay_bias or initial.ndim > 1 else 1.
    assert_tensor(p, scale * initial - 0.5)


def test_zero_gradient_decays_but_missing_gradient_does_not(tensor, assert_tensor):
    zero, missing = tensor([[4.]], requires_grad=True), tensor([[4.]], requires_grad=True)
    zero.set_grad(tensor([[0.]]))
    opt = SGD([zero, missing], lr=0.25, weight_decay=0.5)
    for expected in (3.5, 3.0625):
        opt.step()
        assert_tensor(zero, [[expected]])
        assert_tensor(missing, [[4.]])
    assert missing.grad is None


def test_multiple_parameters_have_independent_update_trajectories(tensor, assert_tensor):
    a, b = tensor([3., -2.], requires_grad=True), tensor([[4.]], requires_grad=True)
    opt = SGD([a, b], lr=0.25)
    for ga, gb, ea, eb in [([2., -4.], [[1.]], [2.5, -1.], [[3.75]]),
                           ([-2., 2.], [[0.]], [3., -1.5], [[3.75]])]:
        a.set_grad(tensor(ga))
        b.set_grad(tensor(gb))
        opt.step()
        assert_tensor(a, ea)
        assert_tensor(b, eb)


def test_weight_decay_repeats_on_current_parameter_values(tensor, assert_tensor):
    p = tensor([[4.]], requires_grad=True)
    p.set_grad(tensor([[2.]]))
    opt = SGD([p], lr=0.25, weight_decay=0.5)
    for expected in (3., 2.125, 1.359375):
        opt.step()
        assert_tensor(p, [[expected]])
