"""Tests for the Conv2d layer."""
import pytest
import orthon
import orthon._C as orthon_c
from orthon.nn import Conv2d, ReLU, Sequential

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


def _set_flat_values(tensor, shape, values):
    for k, value in enumerate(values):
        tensor.set_item(_unravel_index(k, shape), float(value))


def _make_tensor(shape, values, requires_grad=False):
    tensor = orthon.zeros(shape)
    _set_flat_values(tensor, shape, values)
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _deterministic_values(count, scale=0.1, offset=0):
    return [((i + offset) % 17 - 8) * scale for i in range(count)]


def _assert_conv2d_backward_matches_torch(
    input_shape,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    bias=True,
):
    import torch
    import torch.nn.functional as F

    N, C_in, H, W = input_shape
    if isinstance(kernel_size, int):
        K_h, K_w = kernel_size, kernel_size
    else:
        K_h, K_w = kernel_size

    weight_shape = [out_channels, C_in, K_h, K_w]
    bias_shape = [out_channels]

    x_values = _deterministic_values(N * C_in * H * W, scale=0.07, offset=1)
    w_values = _deterministic_values(out_channels * C_in * K_h * K_w, scale=0.05, offset=3)
    b_values = _deterministic_values(out_channels, scale=0.03, offset=5)

    layer = Conv2d(C_in, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
    _set_flat_values(layer.weight, weight_shape, w_values)
    if bias:
        _set_flat_values(layer.bias, bias_shape, b_values)

    x_d = _make_tensor(input_shape, x_values, requires_grad=True)
    out_d = layer(x_d)
    out_d.backward()

    x_t = torch.tensor(x_values, dtype=torch.float32).reshape(input_shape).clone().requires_grad_(True)
    w_t = torch.tensor(w_values, dtype=torch.float32).reshape(weight_shape).clone().requires_grad_(True)
    b_t = None
    if bias:
        b_t = torch.tensor(b_values, dtype=torch.float32).clone().requires_grad_(True)

    out_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding)
    out_t.sum().backward()

    assert out_d.tolist() == pytest.approx(out_t.flatten().tolist(), abs=1e-5)
    assert x_d.grad.tolist() == pytest.approx(x_t.grad.flatten().tolist(), abs=1e-4)
    assert layer.weight.grad.tolist() == pytest.approx(w_t.grad.flatten().tolist(), abs=1e-4)
    if bias:
        assert layer.bias.grad.tolist() == pytest.approx(b_t.grad.tolist(), abs=1e-4)


# Shape.

def test_conv2d_output_shape_basic():
    layer = Conv2d(1, 4, kernel_size=3)
    x = orthon.full([2, 1, 8, 8], 1.0)
    out = layer(x)
    # H_out = W_out = (8-3)/1 + 1 = 6
    assert out.shape == (2, 4, 6, 6)


def test_conv2d_output_shape_with_padding():
    layer = Conv2d(3, 8, kernel_size=3, padding=1)
    x = orthon.full([1, 3, 16, 16], 0.5)
    out = layer(x)
    assert out.shape == (1, 8, 16, 16)   # same-padding


def test_conv2d_output_shape_with_stride():
    layer = Conv2d(1, 2, kernel_size=3, stride=2, padding=1)
    x = orthon.full([1, 1, 8, 8], 1.0)
    out = layer(x)
    # H_out = (8 + 2 - 3)/2 + 1 = 4
    assert out.shape == (1, 2, 4, 4)


def test_conv2d_output_shape_tuple_kernel():
    layer = Conv2d(1, 2, kernel_size=(3, 5), padding=(1, 2))
    x = orthon.full([1, 1, 10, 10], 1.0)
    out = layer(x)
    assert out.shape == (1, 2, 10, 10)   # same-padding for both dims


# Parameter registration.

def test_conv2d_param_count():
    layer = Conv2d(3, 8, kernel_size=3)
    params = list(layer.parameters())
    assert len(params) == 2


def test_conv2d_weight_shape():
    layer = Conv2d(3, 8, kernel_size=3)
    assert layer.weight.shape == (8, 3, 3, 3)


def test_conv2d_bias_shape():
    layer = Conv2d(3, 8, kernel_size=3)
    assert layer.bias.shape == (8,)


def test_conv2d_params_require_grad():
    layer = Conv2d(1, 4, kernel_size=3)
    for p in layer.parameters():
        assert p.requires_grad


def test_conv2d_no_bias():
    layer = Conv2d(1, 4, kernel_size=3, bias=False)
    params = list(layer.parameters())
    assert len(params) == 1          # weight only
    assert params[0].shape == (4, 1, 3, 3)


# Forward values.

def test_conv2d_known_forward():
    """1×1×3×3 input, 1×1×2×2 weight [[1,0],[0,1]], no bias → output == trace pairs."""
    layer = Conv2d(1, 1, kernel_size=2, bias=False)
    # Override weight in-place to [[1,0],[0,1]]
    w = layer.weight
    for f in range(1):
        for c in range(1):
            for i in range(2):
                for j in range(2):
                    w.set_item([f, c, i, j], 1.0 if i == j else 0.0)

    x = orthon.zeros([1, 1, 3, 3])
    vals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(3):
        for j in range(3):
            x.set_item([0, 0, i, j], float(vals[i*3+j]))

    out = layer(x)
    assert out.shape == (1, 1, 2, 2)
    flat = out.tolist()
    # output[0,0,0,0] = 1*1+0*2+0*4+1*5 = 6
    # output[0,0,0,1] = 1*2+0*3+0*5+1*6 = 8
    # output[0,0,1,0] = 1*4+0*5+0*7+1*8 = 12
    # output[0,0,1,1] = 1*5+0*6+0*8+1*9 = 14
    assert flat == pytest.approx([6.0, 8.0, 12.0, 14.0])


def test_conv2d_bias_shifts_output():
    layer = Conv2d(1, 1, kernel_size=1)    # 1×1 kernel = pointwise
    w = layer.weight
    w.set_item([0, 0, 0, 0], 1.0)          # identity weight
    b = layer.bias
    b.set_item([0], 5.0)

    x = orthon.full([1, 1, 3, 3], 2.0)
    out = layer(x)
    assert out.tolist() == pytest.approx([7.0] * 9)


# Autograd.

def test_conv2d_output_requires_grad():
    layer = Conv2d(1, 2, kernel_size=3)
    x = orthon.full([1, 1, 5, 5], 1.0)
    out = layer(x)
    assert out.requires_grad
    assert out._has_grad_fn


def test_conv2d_backward_populates_gradients():
    layer = Conv2d(1, 2, kernel_size=3)
    x = orthon.full([1, 1, 5, 5], 1.0)
    out = layer(x)
    out.backward()   # seeds with ones

    for i, p in enumerate(layer.parameters()):
        assert p.grad is not None, f"param {i} has no gradient"
        assert p.grad.shape == p.shape, f"param {i} grad shape mismatch"


def test_conv2d_grad_input_shape():
    """Verify grad w.r.t. input has same shape as input."""
    x = orthon.full([2, 1, 5, 5], 1.0)
    x.requires_grad_(True)
    w = orthon.full([1, 1, 3, 3], 0.1)
    w.requires_grad_(True)
    b = orthon.zeros([1])
    b.requires_grad_(True)

    out = orthon.conv2d(x, w, b)
    out.backward()

    assert x.grad is not None
    assert x.grad.shape == (2, 1, 5, 5)
    assert w.grad.shape == (1, 1, 3, 3)
    assert b.grad.shape == (1,)


# PyTorch comparison.

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_values_match_torch():
    """Forward output matches torch.nn.functional.conv2d on a small case."""
    import torch
    import torch.nn.functional as F

    N, C_in, H, W = 1, 1, 5, 5
    C_out, K = 1, 3

    # Build identical weight and input
    w_flat = [float(k + 1) * 0.1 for k in range(C_out * C_in * K * K)]
    x_flat = [float(k + 1)       for k in range(N * C_in * H * W)]

    x_d = orthon.zeros([N, C_in, H, W])
    for k, v in enumerate(x_flat):
        i, j = k // W, k % W
        x_d.set_item([0, 0, i, j], v)

    w_d = orthon.zeros([C_out, C_in, K, K])
    for k, v in enumerate(w_flat):
        f = k // (C_in * K * K)
        rem = k % (C_in * K * K)
        c = rem // (K * K)
        rem = rem % (K * K)
        i = rem // K
        j = rem % K
        w_d.set_item([f, c, i, j], v)

    b_d = orthon.full([C_out], 0.5)

    out_d = orthon.conv2d(x_d, w_d, b_d).tolist()

    x_t = torch.tensor(x_flat).reshape(N, C_in, H, W)
    w_t = torch.tensor(w_flat).reshape(C_out, C_in, K, K)
    b_t = torch.tensor([0.5])
    out_t = F.conv2d(x_t, w_t, b_t).flatten().tolist()

    assert out_d == pytest.approx(out_t, rel=1e-5)


def _assert_direct_im2col_match(input_shape, out_channels, kernel_size, stride=1, padding=0):
    N, C_in, H, W = input_shape
    if isinstance(kernel_size, int):
        K_h, K_w = kernel_size, kernel_size
    else:
        K_h, K_w = kernel_size

    weight_shape = [out_channels, C_in, K_h, K_w]
    x_values = _deterministic_values(N * C_in * H * W, scale=0.07, offset=2)
    w_values = _deterministic_values(out_channels * C_in * K_h * K_w, scale=0.05, offset=4)
    b_values = _deterministic_values(out_channels, scale=0.03, offset=6)

    x = _make_tensor(input_shape, x_values)
    w = _make_tensor(weight_shape, w_values)
    b = _make_tensor([out_channels], b_values)

    direct = orthon_c._conv2d_direct(x, w, b, stride, stride, padding, padding)
    im2col = orthon_c._conv2d_im2col(x, w, b, stride, stride, padding, padding)

    assert im2col.shape == direct.shape
    assert im2col.tolist() == pytest.approx(direct.tolist(), abs=1e-6)


def test_conv2d_im2col_matches_direct_basic():
    _assert_direct_im2col_match(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )


def test_conv2d_im2col_matches_direct_with_padding():
    _assert_direct_im2col_match(
        input_shape=[2, 3, 6, 6],
        out_channels=4,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_im2col_matches_direct_with_stride():
    _assert_direct_im2col_match(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch():
    """Backward gradients match torch on a tiny 1×1×4×4 case."""
    import torch
    import torch.nn.functional as F

    N, C_in, H, W = 1, 1, 4, 4
    C_out, K = 1, 2

    x_flat = [float(k + 1) for k in range(N * C_in * H * W)]
    w_flat = [0.1, 0.2, 0.3, 0.4]

    x_d = orthon.zeros([N, C_in, H, W])
    for k, v in enumerate(x_flat):
        x_d.set_item([0, 0, k // W, k % W], v)
    x_d.requires_grad_(True)

    w_d = orthon.zeros([C_out, C_in, K, K])
    for k, v in enumerate(w_flat):
        w_d.set_item([0, 0, k // K, k % K], v)
    w_d.requires_grad_(True)

    b_d = orthon.full([C_out], 0.0)
    b_d.requires_grad_(True)

    out_d = orthon.conv2d(x_d, w_d, b_d)
    out_d.backward()   # seed = ones

    # PyTorch equivalent
    x_t = torch.tensor(x_flat, dtype=torch.float32).reshape(N, C_in, H, W).clone().requires_grad_(True)
    w_t = torch.tensor(w_flat, dtype=torch.float32).reshape(C_out, C_in, K, K).clone().requires_grad_(True)
    b_t = torch.zeros(C_out, dtype=torch.float32).requires_grad_(True)
    out_t = F.conv2d(x_t, w_t, b_t)
    out_t.sum().backward()

    assert x_d.grad.tolist() == pytest.approx(x_t.grad.flatten().tolist(), rel=1e-4)
    assert w_d.grad.tolist() == pytest.approx(w_t.grad.flatten().tolist(), rel=1e-4)
    assert b_d.grad.tolist() == pytest.approx(b_t.grad.tolist(), rel=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch_stride_gt_one():
    _assert_conv2d_backward_matches_torch(
        input_shape=[1, 1, 6, 6],
        out_channels=1,
        kernel_size=3,
        stride=2,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch_with_padding():
    _assert_conv2d_backward_matches_torch(
        input_shape=[1, 1, 5, 5],
        out_channels=1,
        kernel_size=3,
        padding=1,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch_multi_channel_input():
    _assert_conv2d_backward_matches_torch(
        input_shape=[1, 2, 5, 5],
        out_channels=1,
        kernel_size=3,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch_multiple_output_channels():
    _assert_conv2d_backward_matches_torch(
        input_shape=[1, 1, 5, 5],
        out_channels=3,
        kernel_size=3,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch_batch_size_gt_one():
    _assert_conv2d_backward_matches_torch(
        input_shape=[2, 1, 5, 5],
        out_channels=1,
        kernel_size=3,
    )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_conv2d_gradients_match_torch_bias_false():
    _assert_conv2d_backward_matches_torch(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
        bias=False,
    )


def test_conv2d_weight_grad_matches_finite_difference():
    input_shape = [1, 1, 3, 3]
    weight_shape = [1, 1, 2, 2]
    x_values = [0.2, -0.1, 0.3, 0.4, 0.0, -0.2, 0.1, 0.5, -0.3]
    w_values = [0.1, -0.2, 0.3, -0.4]
    eps = 1e-3

    layer = Conv2d(1, 1, kernel_size=2, bias=False)
    _set_flat_values(layer.weight, weight_shape, w_values)

    x = _make_tensor(input_shape, x_values, requires_grad=True)
    out = layer(x)
    out.backward()
    analytical = layer.weight.grad.tolist()

    numerical = []
    for k, original in enumerate(w_values):
        idx = _unravel_index(k, weight_shape)

        layer.weight.set_item(idx, original + eps)
        plus = sum(layer(x).tolist())

        layer.weight.set_item(idx, original - eps)
        minus = sum(layer(x).tolist())

        layer.weight.set_item(idx, original)
        numerical.append((plus - minus) / (2.0 * eps))

    assert analytical == pytest.approx(numerical, abs=1e-3)


# Integration.

def test_conv2d_in_sequential():
    model = Sequential(
        Conv2d(1, 4, kernel_size=3, padding=1),
        ReLU(),
    )
    x = orthon.full([2, 1, 8, 8], 0.5)
    out = model(x)
    assert out.shape == (2, 4, 8, 8)
    assert out.requires_grad


@pytest.mark.parametrize("route", ["conv2d", "_conv2d_direct", "_conv2d_im2col"])
@pytest.mark.parametrize("spatial", [(0, 1, 0, 0), (1, 0, 0, 0), (-1, 1, 0, 0),
                                    (1, -1, 0, 0), (1, 1, -1, 0), (1, 1, 0, -1)])
def test_r1_conv2d_rejects_invalid_spatial_arguments(route, spatial):
    with pytest.raises(ValueError):
        getattr(orthon_c, route)(orthon.ones([1, 1, 3, 3]),
                               orthon.ones([1, 1, 1, 1]), orthon.zeros([1]), *spatial)


@pytest.mark.parametrize("route", ["conv2d", "_conv2d_direct", "_conv2d_im2col"])
@pytest.mark.parametrize("stride", [1, 2])
def test_r1_conv2d_rejects_kernel_larger_than_padded_input(route, stride):
    with pytest.raises(ValueError, match="kernel exceeds"):
        getattr(orthon_c, route)(orthon.ones([1, 1, 1, 1]),
                               orthon.ones([1, 1, 2, 2]), orthon.zeros([1]), stride, stride)


@pytest.mark.parametrize("route", ["conv2d", "_conv2d_direct", "_conv2d_im2col"])
@pytest.mark.parametrize("weight_shape", [[1, 1, 0, 2], [1, 1, 2, 0]])
def test_r1_conv2d_rejects_empty_kernel(route, weight_shape):
    with pytest.raises(ValueError, match="positive"):
        getattr(orthon_c, route)(orthon.ones([1, 1, 3, 3]),
                               orthon.ones(weight_shape), orthon.zeros([1]))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.parametrize("route", ["conv2d", "_conv2d_direct", "_conv2d_im2col"])
@pytest.mark.parametrize("spatial", [(2, 3, 0, 1), (1, 2, 2, 0)])
def test_r1_conv2d_asymmetric_geometry_matches_reference(route, spatial):
    import torch.nn.functional as F
    sh, sw, ph, pw = spatial
    x = _make_tensor([1, 2, 7, 8], _deterministic_values(112, .07, 1), True)
    w = _make_tensor([3, 2, 2, 3], _deterministic_values(36, .05, 3), True)
    b = _make_tensor([3], [.2, -.1, .3], True)
    tx = torch.tensor(x.tolist(), dtype=torch.float64).reshape(x.shape).requires_grad_(True)
    tw = torch.tensor(w.tolist(), dtype=torch.float64).reshape(w.shape).requires_grad_(True)
    tb = torch.tensor(b.tolist(), dtype=torch.float64).requires_grad_(True)
    actual = getattr(orthon_c, route)(x, w, b, *spatial)
    expected = F.conv2d(tx, tw, tb, stride=(sh, sw), padding=(ph, pw))
    assert actual.shape == tuple(expected.shape)
    assert actual.tolist() == pytest.approx(expected.detach().flatten().tolist(), abs=2e-6, rel=2e-5)
    actual.sum().backward()
    expected.sum().backward()
    for value, reference in [(x, tx), (w, tw), (b, tb)]:
        assert value.grad.tolist() == pytest.approx(reference.grad.flatten().tolist(), abs=2e-5, rel=2e-5)


@pytest.mark.parametrize("stride", [(0, 1), (1, 0)])
def test_r1_conv2d_zero_stride_is_a_managed_subprocess_error(stride):
    import subprocess
    import sys
    code = f"""import orthon as o
try:
    o.conv2d(o.ones([1,1,2,2]), o.ones([1,1,1,1]), o.zeros([1]), {stride[0]}, {stride[1]})
except ValueError:
    print('REJECTED')
else:
    raise AssertionError('invalid stride admitted')
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "REJECTED"
