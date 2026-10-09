import numpy as np
import pytest


def sum_replicated_contributions(values, shape):
    padded = (1,) * (values.ndim - len(shape)) + shape
    axes = tuple(i for i, (before, after) in enumerate(zip(padded, values.shape))
                 if before == 1 and after != 1)
    return values.sum(axis=axes, keepdims=True).reshape(shape)


@pytest.mark.parametrize("shapes", [((), (2, 3)), ((3,), (2, 3)),
                                   ((2, 1), (1, 3)), ((2, 1, 3), (1, 4, 1)),
                                   ((1, 2, 1), (3, 1, 4))])
@pytest.mark.parametrize("name", ["add", "sub", "mul", "div"])
def test_weighted_nonlinear_broadcast_reduces_each_leaf_to_its_original_shape(
    tensor, assert_grad, shapes, name,
):
    first_shape, second_shape = shapes
    a = np.arange(np.prod(first_shape, dtype=int)).reshape(first_shape) / 4 + 1
    b = np.arange(np.prod(second_shape, dtype=int)).reshape(second_shape) / 8 + 2
    first, second = tensor(a, requires_grad=True), tensor(b, requires_grad=True)
    operation = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y,
                 "mul": lambda x, y: x * y, "div": lambda x, y: x / y}[name]
    output = operation(first, second)
    forward = operation(a, b)
    weights = np.arange(1, forward.size + 1).reshape(forward.shape) / 2
    ((output * output).transpose(0, -1) * tensor(weights).transpose(0, -1)).mean().backward()
    upstream = 2 * forward * weights / forward.size
    a_partial, b_partial = {
        "add": (np.ones_like(forward), np.ones_like(forward)),
        "sub": (np.ones_like(forward), -np.ones_like(forward)),
        "mul": (np.broadcast_to(b, forward.shape), np.broadcast_to(a, forward.shape)),
        "div": (np.broadcast_to(1 / b, forward.shape), -a / b**2),
    }[name]
    assert_grad(first, sum_replicated_contributions(upstream * a_partial, first_shape))
    assert_grad(second, sum_replicated_contributions(upstream * b_partial, second_shape))
    assert output.grad is None


def test_one_broadcast_leaf_combines_consumers_with_different_output_ranks(tensor, assert_grad):
    bias = tensor([1., 2., 3.], requires_grad=True)
    matrix = tensor(np.arange(1, 7).reshape(2, 3))
    cube = tensor(np.arange(1, 13).reshape(2, 2, 3))
    ((matrix * bias).sum() + (cube + bias).mean() + (bias * bias).sum()).backward()
    assert_grad(bias, np.array([5., 7., 9.]) + 1 / 3 + 2 * np.array([1., 2., 3.]))
    assert matrix.grad is None and cube.grad is None
