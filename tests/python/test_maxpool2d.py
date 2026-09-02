"""Tests for MaxPool2d."""

import pytest

import orthon
from orthon.nn import MaxPool2d

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _unravel_index(flat_index, shape):
    idx = []
    for size in reversed(shape):
        idx.append(flat_index % size)
        flat_index //= size
    return list(reversed(idx))


def _make_tensor(shape, values, requires_grad=False):
    tensor = orthon.zeros(shape)
    for k, value in enumerate(values):
        tensor.set_item(_unravel_index(k, shape), float(value))
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def test_maxpool2d_output_shape():
    layer = MaxPool2d(kernel_size=2)
    x = orthon.full([2, 3, 8, 8], 1.0)
    out = layer(x)
    assert out.shape == (2, 3, 4, 4)


def test_maxpool2d_known_forward_values():
    x = _make_tensor(
        [1, 1, 4, 4],
        [
            1, 2, 5, 4,
            3, 4, 1, 2,
            0, 7, 8, 1,
            6, 5, 2, 9,
        ],
    )
    out = MaxPool2d(kernel_size=2)(x)
    assert out.shape == (1, 1, 2, 2)
    assert out.tolist() == pytest.approx([4, 5, 7, 9])


def test_maxpool2d_backward_routes_gradients_to_max_locations():
    x = _make_tensor(
        [1, 1, 4, 4],
        [
            1, 2, 5, 4,
            3, 4, 1, 2,
            0, 7, 8, 1,
            6, 5, 2, 9,
        ],
        requires_grad=True,
    )
    out = MaxPool2d(kernel_size=2)(x)
    out.backward()
    assert x.grad.tolist() == pytest.approx([
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1,
    ])


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_maxpool2d_forward_matches_torch():
    import torch

    shape = [2, 2, 5, 5]
    values = [((i % 19) - 9) * 0.1 for i in range(2 * 2 * 5 * 5)]
    x_d = _make_tensor(shape, values)
    out_d = MaxPool2d(kernel_size=3, stride=2, padding=1)(x_d)

    x_t = torch.tensor(values, dtype=torch.float32).reshape(shape)
    out_t = torch.nn.functional.max_pool2d(x_t, kernel_size=3, stride=2, padding=1)

    assert out_d.shape == tuple(out_t.shape)
    assert out_d.tolist() == pytest.approx(out_t.flatten().tolist(), abs=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_maxpool2d_backward_matches_torch():
    import torch

    shape = [2, 2, 5, 5]
    values = [((i % 23) - 11) * 0.07 for i in range(2 * 2 * 5 * 5)]
    x_d = _make_tensor(shape, values, requires_grad=True)
    out_d = MaxPool2d(kernel_size=3, stride=2, padding=1)(x_d)
    out_d.backward()

    x_t = torch.tensor(values, dtype=torch.float32).reshape(shape).clone().requires_grad_(True)
    out_t = torch.nn.functional.max_pool2d(x_t, kernel_size=3, stride=2, padding=1)
    out_t.sum().backward()

    assert x_d.grad.tolist() == pytest.approx(x_t.grad.flatten().tolist(), abs=1e-6)
