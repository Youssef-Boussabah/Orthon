"""Tests for MaxPool2d."""

import pytest
import numpy as np
from tests.support.capabilities import orthon_cuda_available

import orthon
from orthon.nn import MaxPool2d

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


R2_POOL_CASES = [
    pytest.param(np.full((2, 2), -np.inf), 2, 0, id="negative-infinity"),
    pytest.param([[-np.inf, 2], [1, -np.inf]], 2, 0, id="mixed-negative-infinity"),
    pytest.param([[3, 3], [3, 3]], 2, 0, id="equal-finite"),
    pytest.param([[1, 4], [4, 2]], 2, 0, id="equal-maxima"),
    pytest.param([[np.inf, 1], [np.inf, 2]], 2, 0, id="positive-infinity-tie"),
    pytest.param(np.full((2, 2), np.nan), 2, 0, id="all-nan"),
    pytest.param([[-0., 0.], [0., -0.]], 2, 0, id="signed-zero-tie"),
    pytest.param(np.full((3, 3), -np.inf), 1, 0, id="overlapping-infinity"),
    pytest.param([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 1, 0, id="overlapping-center"),
    pytest.param([[np.nan, 2, 1], [2, np.nan, 2], [1, 2, 1]], 1, 0, id="overlapping-nans"),
    pytest.param([[-np.inf]], 1, 1, id="padding-intersection"),
    pytest.param([[np.nan]], 1, 1, id="padding-nan"),
    pytest.param(np.empty((0, 0)), 1, 1, id="padding-only-both-axes"),
    pytest.param(np.empty((0, 2)), 1, 1, id="padding-only-height"),
    pytest.param(np.empty((2, 0)), 1, 1, id="padding-only-width"),
]
for _position in range(4):
    _values = np.array([1., 4., 3., 2.])
    _values[_position] = np.nan
    R2_POOL_CASES.append(pytest.param(_values.reshape(2, 2), 2, 0, id="nan-at-" + str(_position)))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not orthon_cuda_available(), reason="CUDA not available"))])
@pytest.mark.parametrize("values,stride,padding", R2_POOL_CASES)
def test_r2_pool_winners_nonfinite_and_padding_contract(device, values, stride, padding):
    raw = np.asarray(values, dtype=np.float32)
    h, w = raw.shape
    ho, wo = (h + 2 * padding - 2) // stride + 1, (w + 2 * padding - 2) // stride + 1
    expected = np.empty((ho, wo), dtype=np.float64)
    gradient = np.zeros(raw.shape, dtype=np.float64)
    for oh in range(ho):
        for ow in range(wo):
            coordinates = [(ih, iw) for ih in range(max(0, oh * stride - padding), min(h, oh * stride - padding + 2))
                           for iw in range(max(0, ow * stride - padding), min(w, ow * stride - padding + 2))]
            numeric = [ij for ij in coordinates if not np.isnan(raw[ij])]
            if not coordinates:
                expected[oh, ow] = -np.inf
                continue
            candidates = numeric or coordinates[:1]
            winner = max(candidates, key=lambda ij: raw[ij])
            expected[oh, ow] = raw[winner]
            gradient[winner] += 1.
    x = (orthon.from_numpy(raw.reshape(1, 1, h, w)) if raw.size else orthon.empty([1, 1, h, w])).to(device).requires_grad_(True)
    result = orthon.maxpool2d(x, 2, 2, stride, stride, padding, padding)
    assert result.shape == (1, 1, ho, wo)
    np.testing.assert_allclose(result.to("cpu").tolist(), expected.ravel(), equal_nan=True)
    result.backward()
    np.testing.assert_array_equal(x.grad.to("cpu").tolist(), gradient.ravel())


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
