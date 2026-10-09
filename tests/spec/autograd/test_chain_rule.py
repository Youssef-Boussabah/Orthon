import math

import numpy as np
import pytest

import orthon as ortho


@pytest.mark.parametrize("depth", [32, 129])
def test_long_affine_chain_preserves_every_derivative(tensor, assert_grad, depth):
    leaf = tensor([1., -2., 4.], requires_grad=True)
    output = leaf
    for _ in range(depth):
        output = -output + .25
    squared = output * output
    squared.mean().backward()
    final = np.array([1., -2., 4.]) if depth % 2 == 0 else .25 - np.array([1., -2., 4.])
    assert_grad(leaf, 2 * final * (-1)**depth / 3)
    assert output.grad is None and squared.grad is None


@pytest.mark.parametrize("name", ["neg", "relu", "sigmoid", "tanh", "leaky_relu",
                                  "gelu", "exp", "sqrt", "reciprocal"])
def test_unary_derivative_composes_with_affine_square_and_weighted_mean(
    tensor, assert_grad, name,
):
    raw = np.array([.5, 1.5, 3.]) if name in ("sqrt", "reciprocal") else np.array([-.8, .3, 1.2])
    leaf = tensor(raw, requires_grad=True)
    weights = np.array([1., 2., 4.])
    operation = (lambda x: ortho.leaky_relu(x, .125)) if name == "leaky_relu" else getattr(ortho, name)
    if name == "neg":
        forward, derivative = -raw, -np.ones(3)
    elif name == "relu":
        forward, derivative = np.maximum(raw, 0), (raw > 0).astype(float)
    elif name == "sigmoid":
        forward = 1 / (1 + np.exp(-raw))
        derivative = forward * (1 - forward)
    elif name == "tanh":
        forward = np.tanh(raw)
        derivative = 1 - forward**2
    elif name == "leaky_relu":
        forward, derivative = np.where(raw > 0, raw, raw / 8), np.where(raw > 0, 1., .125)
    elif name == "gelu":
        cdf = (1 + np.vectorize(math.erf)(raw / np.sqrt(2))) / 2
        forward, derivative = raw * cdf, cdf + raw * np.exp(-raw**2 / 2) / np.sqrt(2 * np.pi)
    elif name == "exp":
        forward = derivative = np.exp(raw)
    elif name == "sqrt":
        forward, derivative = np.sqrt(raw), .5 / np.sqrt(raw)
    else:
        forward, derivative = 1 / raw, -1 / raw**2
    transformed = operation(leaf) * 2 + 1
    ((transformed * transformed) * tensor(weights)).mean().backward()
    assert_grad(leaf, 4 * (2 * forward + 1) * derivative * weights / 3)


def test_mixed_tensor_division_reaches_both_operand_leaves(tensor, assert_grad):
    a_values, b_values = np.array([1., 2., -3.]), np.array([2., 4., 5.])
    a, b = tensor(a_values, requires_grad=True), tensor(b_values, requires_grad=True)
    numerator, denominator = a * b + a, b + 2
    quotient = numerator / denominator
    (quotient * quotient).sum().backward()
    q = a_values * (b_values + 1) / (b_values + 2)
    assert_grad(a, 2 * q * (b_values + 1) / (b_values + 2))
    assert_grad(b, 2 * q * a_values / (b_values + 2)**2)
    assert numerator.grad is None and denominator.grad is None and quotient.grad is None


def test_mean_followed_by_a_nonlinear_scalar_chain_broadcasts_its_derivative(tensor, assert_grad):
    leaf = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    mean = leaf.mean()
    shifted = mean * 2 + 1
    (shifted * shifted + mean * 3).backward()
    assert_grad(leaf, np.full((2, 3), (4 * 8 + 3) / 6))
    assert mean.grad is None


def test_cross_entropy_inside_a_scalar_chain_propagates_its_upstream_scale(
    operator_device, typed_tensor, assert_output,
):
    raw = np.array([[.25, -.5, 1.], [1., .5, -.25]])
    leaf = typed_tensor(raw, device=operator_device).requires_grad_(True)
    logits = leaf * 2 + .5
    loss = ortho.cross_entropy(logits, [0, 2])
    (loss * loss + loss * 3).backward()
    exp = np.exp(raw * 2 - np.max(raw * 2, axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    value = -np.log(probabilities[[0, 1], [0, 2]]).mean()
    probabilities[[0, 1], [0, 2]] -= 1
    assert_output(leaf.grad, (2 * value + 3) * probabilities,
                  dtype="float32", device=operator_device)
    assert loss.grad is None and logits.grad is None


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_log_softmax_and_nll_loss_compose_with_a_squared_loss(
    reduction, operator_device, typed_tensor, assert_output,
):
    raw = np.array([[.25, -.5, 1.], [1., .5, -.25]])
    leaf = typed_tensor(raw, device=operator_device).requires_grad_(True)
    targets = typed_tensor([0, 2], dtype="int64", device=operator_device)
    losses = ortho.nll_loss(ortho.log_softmax(leaf), targets, reduction=reduction)
    (losses * losses).sum().backward()
    exp = np.exp(raw - raw.max(axis=1, keepdims=True))
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    nll = -np.log(probabilities[[0, 1], [0, 2]])
    if reduction == "none":
        scale = 2 * nll[:, None]
    else:
        scale = 2 * nll.sum() if reduction == "sum" else nll.mean()
    probabilities[[0, 1], [0, 2]] -= 1
    assert_output(leaf.grad, probabilities * scale, dtype="float32", device=operator_device)
    assert targets.grad is None
