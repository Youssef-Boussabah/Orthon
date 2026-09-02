"""Larger synthetic gradient checks against PyTorch."""

import pytest
import orthon
from orthon.nn import Linear, ReLU, Sequential

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _values(count, scale=0.01, offset=0):
    return [((i + offset) % 23 - 11) * scale for i in range(count)]


def _unravel(flat_index, shape):
    idx = []
    for size in reversed(shape):
        idx.append(flat_index % size)
        flat_index //= size
    return list(reversed(idx))


def _tensor(shape, values, requires_grad=False):
    t = orthon.zeros(shape)
    for i, value in enumerate(values):
        t.set_item(_unravel(i, shape), float(value))
    if requires_grad:
        t.requires_grad_(True)
    return t


def _set_tensor(tensor, shape, values):
    for i, value in enumerate(values):
        tensor.set_item(_unravel(i, shape), float(value))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_scaled_mlp_linear_gradients_match_torch():
    batch, in_dim, hidden = 24, 128, 64
    x_vals = _values(batch * in_dim, scale=0.005, offset=1)
    w1_vals = _values(in_dim * hidden, scale=0.004, offset=2)
    b1_vals = _values(hidden, scale=0.003, offset=3)

    layer = Linear(in_dim, hidden)
    _set_tensor(layer.weight, [in_dim, hidden], w1_vals)
    _set_tensor(layer.bias, [hidden], b1_vals)

    x_d = _tensor([batch, in_dim], x_vals, requires_grad=True)
    out_d = layer(x_d)
    out_d.backward()

    x_t = torch.tensor(x_vals, dtype=torch.float32).reshape(batch, in_dim).clone().requires_grad_(True)
    w_t = torch.tensor(w1_vals, dtype=torch.float32).reshape(in_dim, hidden).clone().requires_grad_(True)
    b_t = torch.tensor(b1_vals, dtype=torch.float32).clone().requires_grad_(True)
    out_t = x_t @ w_t + b_t
    out_t.sum().backward()

    assert x_d.grad.tolist() == pytest.approx(x_t.grad.flatten().tolist(), abs=1e-4)
    assert layer.weight.grad.tolist() == pytest.approx(w_t.grad.flatten().tolist(), abs=1e-4)
    assert layer.bias.grad.tolist() == pytest.approx(b_t.grad.tolist(), abs=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_scaled_conv2d_gradients_match_torch():
    batch, cin, cout, h, w, kernel = 3, 4, 5, 5, 5, 3
    x_vals = _values(batch * cin * h * w, scale=0.01, offset=4)
    weight_vals = _values(cout * cin * kernel * kernel, scale=0.006, offset=5)
    bias_vals = _values(cout, scale=0.004, offset=6)

    x_d = _tensor([batch, cin, h, w], x_vals, requires_grad=True)
    weight_d = _tensor([cout, cin, kernel, kernel], weight_vals, requires_grad=True)
    bias_d = _tensor([cout], bias_vals, requires_grad=True)
    out_d = orthon.conv2d(x_d, weight_d, bias_d, 1, 1, 1, 1)
    out_d.backward()

    x_t = torch.tensor(x_vals, dtype=torch.float32).reshape(batch, cin, h, w).clone().requires_grad_(True)
    weight_t = torch.tensor(weight_vals, dtype=torch.float32).reshape(cout, cin, kernel, kernel).clone().requires_grad_(True)
    bias_t = torch.tensor(bias_vals, dtype=torch.float32).clone().requires_grad_(True)
    out_t = F.conv2d(x_t, weight_t, bias_t, padding=1)
    out_t.sum().backward()

    assert x_d.grad.tolist() == pytest.approx(x_t.grad.flatten().tolist(), abs=1e-4)
    assert weight_d.grad.tolist() == pytest.approx(weight_t.grad.flatten().tolist(), abs=1e-4)
    assert bias_d.grad.tolist() == pytest.approx(bias_t.grad.tolist(), abs=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_scaled_linear_relu_linear_chain_gradients_match_torch():
    batch, in_dim, hidden, out_dim = 20, 128, 64, 10
    x_vals = _values(batch * in_dim, scale=0.006, offset=7)
    w1_vals = _values(in_dim * hidden, scale=0.004, offset=8)
    b1_vals = _values(hidden, scale=0.003, offset=9)
    w2_vals = _values(hidden * out_dim, scale=0.005, offset=10)
    b2_vals = _values(out_dim, scale=0.002, offset=11)

    model = Sequential(Linear(in_dim, hidden), ReLU(), Linear(hidden, out_dim))
    _set_tensor(model._modules["0"].weight, [in_dim, hidden], w1_vals)
    _set_tensor(model._modules["0"].bias, [hidden], b1_vals)
    _set_tensor(model._modules["2"].weight, [hidden, out_dim], w2_vals)
    _set_tensor(model._modules["2"].bias, [out_dim], b2_vals)

    x_d = _tensor([batch, in_dim], x_vals, requires_grad=True)
    out_d = model(x_d)
    out_d.backward()

    x_t = torch.tensor(x_vals, dtype=torch.float32).reshape(batch, in_dim).clone().requires_grad_(True)
    w1_t = torch.tensor(w1_vals, dtype=torch.float32).reshape(in_dim, hidden).clone().requires_grad_(True)
    b1_t = torch.tensor(b1_vals, dtype=torch.float32).clone().requires_grad_(True)
    w2_t = torch.tensor(w2_vals, dtype=torch.float32).reshape(hidden, out_dim).clone().requires_grad_(True)
    b2_t = torch.tensor(b2_vals, dtype=torch.float32).clone().requires_grad_(True)
    hidden_t = torch.relu(x_t @ w1_t + b1_t)
    out_t = hidden_t @ w2_t + b2_t
    out_t.sum().backward()

    assert x_d.grad.tolist() == pytest.approx(x_t.grad.flatten().tolist(), abs=1e-4)
    assert model._modules["0"].weight.grad.tolist() == pytest.approx(w1_t.grad.flatten().tolist(), abs=1e-4)
    assert model._modules["0"].bias.grad.tolist() == pytest.approx(b1_t.grad.tolist(), abs=1e-4)
    assert model._modules["2"].weight.grad.tolist() == pytest.approx(w2_t.grad.flatten().tolist(), abs=1e-4)
    assert model._modules["2"].bias.grad.tolist() == pytest.approx(b2_t.grad.tolist(), abs=1e-4)
