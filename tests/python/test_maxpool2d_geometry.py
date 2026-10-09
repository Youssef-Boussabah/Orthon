"""Checked pooling admission and independent geometry/winner gradients."""
import numpy as np
import pytest
import orthon as o
from tests.support.capabilities import orthon_cuda_available


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not orthon_cuda_available(), reason="CUDA not available"))]


def test_r51_historical_cpu_overflow_is_managed():
    x = o.ones([1, 1, 2, 2]).requires_grad_(True)
    before = x.tolist()
    m = (1 << 63) - 1
    with pytest.raises(OverflowError, match="padded spatial extent overflows"):
        o.maxpool2d(x, m, m, m, m, m // 2, m // 2)
    assert x.tolist() == before
    assert x.grad is None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kernel,stride", [(2, 2), (3, 8)])
def test_r51_negative_numerator_rejects_before_division(device, kernel, stride):
    x = o.ones([1, 1, 1, 1]).to(device)
    with pytest.raises(RuntimeError, match="output spatial size is non-positive"):
        o.maxpool2d(x, kernel, kernel, stride, stride)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(1, 1, 0, 0), (1, 1, 0, 2), (1, 1, 2, 0),
                                  (0, 1, 2, 2), (1, 0, 2, 2)])
def test_r51_padding_only_and_zero_batch_channel(device, shape):
    x = o.empty(list(shape)).to(device).requires_grad_(True)
    y = o.maxpool2d(x, 2, 2, 1, 1, 1, 1)
    assert y.shape == (shape[0], shape[1], shape[2] + 1, shape[3] + 1)
    assert all(v == -np.inf for v in y.to("cpu").tolist())
    y.backward()
    assert x.grad.shape == shape
    assert x.grad.to("cpu").tolist() == []


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape,args", [
    ((2, 2, 4, 6), (2, 2, 2, 2, 0, 0)),
    ((2, 3, 5, 7), (3, 3, 1, 2, 1, 1)),
    ((1, 2, 4, 7), (2, 3, 3, 2, 1, 0)),
    ((1, 1, 2, 2), (2, 2, 9, 7, 0, 0)),
])
def test_r51_asymmetric_forward_backward_independent(device, shape, args):
    kh, kw, sh, sw, ph, pw = args
    raw = ((np.arange(np.prod(shape)) * 17) % 101 - 50).astype(np.float32).reshape(shape)
    ho, wo = (shape[2] + 2 * ph - kh) // sh + 1, (shape[3] + 2 * pw - kw) // sw + 1
    expected = np.empty((shape[0], shape[1], ho, wo), np.float64)
    gradient = np.zeros(shape, np.float64)
    seed = np.arange(expected.size, dtype=np.float32).reshape(expected.shape) + 1
    for n, c, h, w in np.ndindex(expected.shape):
        coordinates = [(ih, iw) for ih in range(max(0, h * sh - ph), min(shape[2], h * sh - ph + kh))
                       for iw in range(max(0, w * sw - pw), min(shape[3], w * sw - pw + kw))]
        if not coordinates:
            expected[n, c, h, w] = -np.inf
            continue
        winner = max(coordinates, key=lambda ij: float(raw[(n, c) + ij]))
        expected[n, c, h, w] = raw[(n, c) + winner]
        gradient[(n, c) + winner] += float(seed[n, c, h, w])
    x = o.from_numpy(raw).to(device).requires_grad_(True)
    y = o.maxpool2d(x, *args)
    np.testing.assert_array_equal(np.asarray(y.to("cpu").tolist()).reshape(y.shape), expected)
    # Public backward accepts no seed argument; a weighted scalar loss provides it.
    upstream = o.from_numpy(np.ascontiguousarray(seed.swapaxes(2, 3))).to(device).transpose(2, 3)
    (y * upstream).sum().backward()
    np.testing.assert_array_equal(np.asarray(x.grad.to("cpu").tolist()).reshape(shape), gradient)


@pytest.mark.skipif(not orthon_cuda_available(), reason="CUDA not available")
@pytest.mark.parametrize("args", [(1, 1, 1 << 31, 1, 0, 0),
                                  (1 << 31, 1, 1, 1, 1 << 30, 0)])
def test_r51_cuda_scalar_narrowing_rejects_before_execution(args):
    # Post-fix source guard reviewed before this bounded tiny-input rejection.
    x = o.ones([1, 1, 2, 2]).to("cuda")
    before = x.to("cpu").tolist()
    with pytest.raises(OverflowError, match="CUDA geometry exceeds int32 range"):
        o.maxpool2d(x, *args)
    assert x.to("cpu").tolist() == before
