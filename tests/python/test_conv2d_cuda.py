"""CUDA Conv2d forward/backward parity tests."""

from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest

import orthon
import orthon._C as orthon_c

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False




def _make_values(count, scale=0.05, offset=0):
    return [((i + offset) % 23 - 11) * scale for i in range(count)]


def _make_varied_values(count, scale=0.01, offset=0):
    values = []
    for i in range(count):
        a = ((i * 37 + offset * 11) % 101) - 50
        b = ((i * 17 + offset * 7) % 29) - 14
        sign = -1.0 if ((i + offset) % 3 == 0) else 1.0
        values.append(sign * (a * scale + b * scale * 0.13))
    return values


def _make_tensor(shape, values):
    tensor = orthon.zeros(shape)
    for flat, value in enumerate(values):
        idx = []
        k = flat
        for size in reversed(shape):
            idx.append(k % size)
            k //= size
        tensor.set_item(list(reversed(idx)), float(value))
    return tensor


def _torch_device():
    return "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"


def _assert_cuda_conv2d_forward_matches_torch(
    input_shape, out_channels, kernel_size, stride=1, padding=0
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    x_values = _make_values(N * C_in * H * W, scale=0.04, offset=1)
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=5)
    b_values = _make_values(out_channels, scale=0.02, offset=7)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    b_d = _make_tensor([out_channels], b_values).to("cuda")
    out_d = orthon.conv2d(x_d, w_d, b_d, stride, stride, padding, padding).to("cpu")

    device = _torch_device()
    x_t = torch.tensor(x_values, dtype=torch.float32, device=device).reshape(input_shape)
    w_t = torch.tensor(w_values, dtype=torch.float32, device=device).reshape(weight_shape)
    b_t = torch.tensor(b_values, dtype=torch.float32, device=device)
    out_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding).detach().cpu()

    assert out_d.shape == tuple(out_t.shape)
    assert out_d.tolist() == pytest.approx(out_t.flatten().tolist(), abs=1e-4)


def _assert_cuda_optimized_matches_naive(
    input_shape, out_channels, kernel_size, stride=1, padding=0
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_naive"):
        pytest.skip("CUDA Conv2d benchmark helpers are not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    x_values = _make_values(N * C_in * H * W, scale=0.04, offset=11)
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=13)
    b_values = _make_values(out_channels, scale=0.02, offset=17)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    b_d = _make_tensor([out_channels], b_values).to("cuda")

    naive = orthon_c._conv2d_cuda_naive(
        x_d, w_d, b_d, stride, stride, padding, padding).to("cpu")
    optimized = orthon_c._conv2d_cuda_optimized(
        x_d, w_d, b_d, stride, stride, padding, padding).to("cpu")

    assert optimized.shape == naive.shape
    assert optimized.tolist() == pytest.approx(naive.tolist(), abs=1e-5)


def _assert_cuda_grad_weight_optimized_matches_naive(
    input_shape, out_channels, kernel_size, stride=1, padding=0
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_weight_optimized"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    x_values = _make_values(N * C_in * H * W, scale=0.04, offset=19)
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=23)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = orthon.ones([N, out_channels, h_out, w_out]).to("cuda")

    naive = orthon_c._conv2d_cuda_backward_weight_naive(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")
    optimized = orthon_c._conv2d_cuda_backward_weight_optimized(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")

    assert optimized.shape == naive.shape
    assert optimized.tolist() == pytest.approx(naive.tolist(), abs=1e-4)


def _assert_cuda_grad_weight_optimized_v2_matches_naive(
    input_shape, out_channels, kernel_size=3, stride=1, padding=1
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_weight_optimized_v2"):
        pytest.skip("CUDA Conv2d optimized_v2 helper is not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    x_values = _make_values(N * C_in * H * W, scale=0.04, offset=19)
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=23)
    grad_values = _make_values(N * out_channels * h_out * w_out, scale=0.02, offset=31)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = _make_tensor([N, out_channels, h_out, w_out], grad_values).to("cuda")

    naive = orthon_c._conv2d_cuda_backward_weight_naive(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")
    optimized = orthon_c._conv2d_cuda_backward_weight_optimized_v2(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")

    assert optimized.shape == naive.shape
    assert optimized.tolist() == pytest.approx(naive.tolist(), abs=2e-4)


def _assert_cuda_grad_input_optimized_matches_naive(
    input_shape, out_channels, kernel_size, stride=1, padding=0
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_input_optimized"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=27)
    grad_values = _make_values(N * out_channels * h_out * w_out, scale=0.04, offset=31)

    x_d = orthon.zeros(input_shape).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = _make_tensor([N, out_channels, h_out, w_out], grad_values).to("cuda")

    naive = orthon_c._conv2d_cuda_backward_input_naive(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")
    optimized = orthon_c._conv2d_cuda_backward_input_optimized(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")

    assert optimized.shape == naive.shape
    assert optimized.tolist() == pytest.approx(naive.tolist(), abs=1e-4)


def _assert_cuda_grad_input_optimized_v2_matches_naive(
    input_shape, out_channels, kernel_size=3, stride=1, padding=1
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_input_optimized_v2"):
        pytest.skip("CUDA Conv2d grad-input optimized_v2 helper is not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=27)
    grad_values = _make_values(N * out_channels * h_out * w_out, scale=0.04, offset=31)

    x_d = orthon.zeros(input_shape).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = _make_tensor([N, out_channels, h_out, w_out], grad_values).to("cuda")

    naive = orthon_c._conv2d_cuda_backward_input_naive(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")
    optimized = orthon_c._conv2d_cuda_backward_input_optimized_v2(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")

    assert optimized.shape == naive.shape
    assert optimized.tolist() == pytest.approx(naive.tolist(), abs=1e-4)


def _max_diff_with_index(actual, expected):
    max_diff = -1.0
    max_idx = -1
    for idx, (a, e) in enumerate(zip(actual, expected)):
        diff = abs(float(a) - float(e))
        if diff > max_diff:
            max_diff = diff
            max_idx = idx
    return max_diff, max_idx, actual[max_idx], expected[max_idx]


def _unravel_index(flat, shape):
    idx = []
    for size in reversed(shape):
        idx.append(flat % size)
        flat //= size
    return list(reversed(idx))


def _assert_cuda_grad_input_v2_matches_torch(
    input_shape, out_channels, *, atol=2e-4
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_input_optimized_v2"):
        pytest.skip("CUDA Conv2d grad-input optimized_v2 helper is not available")

    kernel_size = 3
    stride = 1
    padding = 1
    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    h_out = H
    w_out = W
    x_values = _make_varied_values(N * C_in * H * W, scale=0.003, offset=5)
    w_values = _make_varied_values(out_channels * C_in * kernel_size * kernel_size, scale=0.004, offset=13)
    grad_values = _make_varied_values(N * out_channels * h_out * w_out, scale=0.002, offset=23)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = _make_tensor([N, out_channels, h_out, w_out], grad_values).to("cuda")

    optimized_v2 = orthon_c._conv2d_cuda_backward_input_optimized_v2(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")
    old_optimized = orthon_c._conv2d_cuda_backward_input_optimized(
        x_d, w_d, grad_output, stride, stride, padding, padding).to("cpu")

    device = _torch_device()
    x_t = torch.tensor(x_values, dtype=torch.float32, device=device).reshape(input_shape)
    x_t.requires_grad_(True)
    w_t = torch.tensor(w_values, dtype=torch.float32, device=device).reshape(weight_shape)
    go_t = torch.tensor(grad_values, dtype=torch.float32, device=device).reshape([N, out_channels, h_out, w_out])
    out_t = F.conv2d(x_t, w_t, None, stride=stride, padding=padding)
    out_t.backward(go_t)
    expected = x_t.grad.detach().cpu().flatten().tolist()

    actual = optimized_v2.tolist()
    max_diff, max_idx, actual_value, expected_value = _max_diff_with_index(actual, expected)
    old_max_diff, _, _, _ = _max_diff_with_index(old_optimized.tolist(), expected)
    assert max_diff <= atol, (
        f"max_diff={max_diff} at {_unravel_index(max_idx, input_shape)} "
        f"actual={actual_value} expected={expected_value} "
        f"old_optimized_max_diff={old_max_diff}"
    )


def _assert_cuda_grad_bias_optimized_matches_naive(
    input_shape, out_channels, kernel_size, stride=1, padding=0
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_bias_optimized"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    N, _, H, W = input_shape
    h_out = (H + 2 * padding - kernel_size) // stride + 1
    w_out = (W + 2 * padding - kernel_size) // stride + 1
    grad_values = _make_values(N * out_channels * h_out * w_out, scale=0.04, offset=29)
    grad_output = _make_tensor([N, out_channels, h_out, w_out], grad_values).to("cuda")
    bias = orthon.zeros([out_channels]).to("cuda")

    naive = orthon_c._conv2d_cuda_backward_bias_naive(bias, grad_output).to("cpu")
    optimized = orthon_c._conv2d_cuda_backward_bias_optimized(bias, grad_output).to("cpu")

    assert optimized.shape == naive.shape
    assert optimized.tolist() == pytest.approx(naive.tolist(), abs=1e-4)


def _assert_cuda_conv2d_backward_matches_torch(
    input_shape, out_channels, kernel_size, stride=1, padding=0
):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not available")

    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    x_values = _make_values(N * C_in * H * W, scale=0.04, offset=2)
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=6)
    b_values = _make_values(out_channels, scale=0.02, offset=9)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    b_d = _make_tensor([out_channels], b_values).to("cuda")
    x_d.requires_grad_(True)
    w_d.requires_grad_(True)
    b_d.requires_grad_(True)

    out_d = orthon.conv2d(x_d, w_d, b_d, stride, stride, padding, padding)
    out_d.backward()

    device = _torch_device()
    x_t = torch.tensor(x_values, dtype=torch.float32, device=device).reshape(input_shape)
    w_t = torch.tensor(w_values, dtype=torch.float32, device=device).reshape(weight_shape)
    b_t = torch.tensor(b_values, dtype=torch.float32, device=device)
    x_t = x_t.clone().requires_grad_(True)
    w_t = w_t.clone().requires_grad_(True)
    b_t = b_t.clone().requires_grad_(True)
    out_t = F.conv2d(x_t, w_t, b_t, stride=stride, padding=padding)
    out_t.sum().backward()

    assert out_d.to("cpu").tolist() == pytest.approx(
        out_t.detach().cpu().flatten().tolist(), abs=1e-4
    )
    assert x_d.grad.to("cpu").tolist() == pytest.approx(
        x_t.grad.detach().cpu().flatten().tolist(), abs=1e-4
    )
    assert w_d.grad.to("cpu").tolist() == pytest.approx(
        w_t.grad.detach().cpu().flatten().tolist(), abs=1e-4
    )
    assert b_d.grad.to("cpu").tolist() == pytest.approx(
        b_t.grad.detach().cpu().flatten().tolist(), abs=1e-4
    )


def test_conv2d_cuda_forward_basic_matches_torch():
    _assert_cuda_conv2d_forward_matches_torch(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )


def test_conv2d_cuda_forward_padding_matches_torch():
    _assert_cuda_conv2d_forward_matches_torch(
        input_shape=[2, 3, 6, 6],
        out_channels=4,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_forward_stride_matches_torch():
    _assert_cuda_conv2d_forward_matches_torch(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def test_conv2d_cuda_optimized_matches_naive_basic():
    _assert_cuda_optimized_matches_naive(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )


def test_conv2d_cuda_optimized_matches_naive_padding_stride():
    _assert_cuda_optimized_matches_naive(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def test_conv2d_cuda_optimized_matches_naive_multi_channel_multi_output():
    _assert_cuda_optimized_matches_naive(
        input_shape=[2, 4, 8, 8],
        out_channels=5,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_env_can_select_naive_forward(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_naive"):
        pytest.skip("CUDA Conv2d benchmark helpers are not available")

    input_shape = [1, 2, 5, 5]
    out_channels = 3
    kernel_size = 3
    padding = 1
    weight_shape = [out_channels, input_shape[1], kernel_size, kernel_size]

    x = _make_tensor(input_shape, _make_values(50, scale=0.04, offset=3)).to("cuda")
    w = _make_tensor(weight_shape, _make_values(54, scale=0.03, offset=7)).to("cuda")
    b = _make_tensor([out_channels], _make_values(out_channels, scale=0.02, offset=9)).to("cuda")

    monkeypatch.setenv("ORTHON_CUDA_CONV2D_FORWARD", "naive")
    selected = orthon.conv2d(x, w, b, 1, 1, padding, padding).to("cpu")
    direct_naive = orthon_c._conv2d_cuda_naive(x, w, b, 1, 1, padding, padding).to("cpu")

    assert selected.tolist() == pytest.approx(direct_naive.tolist(), abs=1e-5)


def test_conv2d_cuda_grad_weight_optimized_matches_naive_basic():
    _assert_cuda_grad_weight_optimized_matches_naive(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )


def test_conv2d_cuda_grad_weight_optimized_matches_naive_padding_stride():
    _assert_cuda_grad_weight_optimized_matches_naive(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def test_conv2d_cuda_grad_weight_optimized_matches_naive_multi_channel_multi_output():
    _assert_cuda_grad_weight_optimized_matches_naive(
        input_shape=[2, 4, 8, 8],
        out_channels=5,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_grad_weight_optimized_v2_matches_naive():
    _assert_cuda_grad_weight_optimized_v2_matches_naive(
        input_shape=[4, 4, 8, 8],
        out_channels=5,
    )


def test_conv2d_cuda_grad_weight_default_matches_grouped3x3(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")

    input_shape = [4, 4, 8, 8]
    out_channels = 5
    kernel_size = 3
    padding = 1
    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    x_values = _make_values(N * C_in * H * W, scale=0.04, offset=19)
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=23)
    grad_values = _make_values(N * out_channels * H * W, scale=0.02, offset=31)

    x_d = _make_tensor(input_shape, x_values).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = _make_tensor([N, out_channels, H, W], grad_values).to("cuda")

    monkeypatch.delenv("ORTHON_CUDA_CONV2D_GRAD_WEIGHT", raising=False)
    default = orthon_c._conv2d_cuda_backward_weight(
        x_d, w_d, grad_output, 1, 1, padding, padding).to("cpu")

    monkeypatch.setenv("ORTHON_CUDA_CONV2D_GRAD_WEIGHT", "grouped3x3")
    grouped = orthon_c._conv2d_cuda_backward_weight(
        x_d, w_d, grad_output, 1, 1, padding, padding).to("cpu")

    assert default.tolist() == pytest.approx(grouped.tolist(), abs=1e-6)


def test_conv2d_cuda_grad_input_optimized_matches_naive_basic():
    _assert_cuda_grad_input_optimized_matches_naive(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )


def test_conv2d_cuda_grad_input_optimized_matches_naive_padding_stride():
    _assert_cuda_grad_input_optimized_matches_naive(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def test_conv2d_cuda_grad_input_optimized_matches_naive_multi_channel_multi_output():
    _assert_cuda_grad_input_optimized_matches_naive(
        input_shape=[2, 4, 8, 8],
        out_channels=5,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_grad_input_optimized_v2_matches_naive():
    _assert_cuda_grad_input_optimized_v2_matches_naive(
        input_shape=[2, 4, 8, 8],
        out_channels=5,
    )


def test_conv2d_cuda_grad_input_default_matches_optimized_v2(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_input_optimized_v2"):
        pytest.skip("CUDA Conv2d grad-input optimized_v2 helper is not available")

    input_shape = [2, 4, 8, 8]
    out_channels = 5
    kernel_size = 3
    padding = 1
    N, C_in, H, W = input_shape
    weight_shape = [out_channels, C_in, kernel_size, kernel_size]
    w_values = _make_values(out_channels * C_in * kernel_size * kernel_size, scale=0.03, offset=27)
    grad_values = _make_values(N * out_channels * H * W, scale=0.04, offset=31)

    x_d = orthon.zeros(input_shape).to("cuda")
    w_d = _make_tensor(weight_shape, w_values).to("cuda")
    grad_output = _make_tensor([N, out_channels, H, W], grad_values).to("cuda")

    monkeypatch.delenv("ORTHON_CUDA_CONV2D_GRAD_INPUT", raising=False)
    default = orthon_c._conv2d_cuda_backward_input(
        x_d, w_d, grad_output, 1, 1, padding, padding).to("cpu")
    optimized_v2 = orthon_c._conv2d_cuda_backward_input_optimized_v2(
        x_d, w_d, grad_output, 1, 1, padding, padding).to("cpu")

    assert default.tolist() == pytest.approx(optimized_v2.tolist(), abs=1e-6)


@pytest.mark.parametrize(
    "input_shape,out_channels",
    [
        ([2, 3, 32, 32], 32),
        ([2, 32, 32, 32], 32),
        ([2, 32, 16, 16], 64),
        ([2, 64, 16, 16], 64),
        ([2, 64, 8, 8], 128),
        ([2, 128, 8, 8], 128),
    ],
)
def test_conv2d_cuda_grad_input_optimized_v2_minivgg_shapes_match_torch(
    input_shape, out_channels
):
    _assert_cuda_grad_input_v2_matches_torch(input_shape, out_channels)


def test_conv2d_cuda_grad_input_optimized_v2_boundary_case_matches_torch():
    _assert_cuda_grad_input_v2_matches_torch([1, 2, 4, 4], 3, atol=1e-5)


def test_conv2d_cuda_env_can_select_naive_grad_input(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_input_naive"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    input_shape = [1, 2, 5, 5]
    out_channels = 3
    kernel_size = 3
    padding = 1
    weight_shape = [out_channels, input_shape[1], kernel_size, kernel_size]

    x = orthon.zeros(input_shape).to("cuda")
    w = _make_tensor(weight_shape, _make_values(54, scale=0.03, offset=7)).to("cuda")
    grad_output = orthon.ones([1, out_channels, 5, 5]).to("cuda")

    monkeypatch.setenv("ORTHON_CUDA_CONV2D_GRAD_INPUT", "naive")
    selected = orthon_c._conv2d_cuda_backward_input(
        x, w, grad_output, 1, 1, padding, padding).to("cpu")
    direct_naive = orthon_c._conv2d_cuda_backward_input_naive(
        x, w, grad_output, 1, 1, padding, padding).to("cpu")

    assert selected.tolist() == pytest.approx(direct_naive.tolist(), abs=1e-4)


def test_conv2d_cuda_env_can_select_naive_grad_weight(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_weight_naive"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    input_shape = [1, 2, 5, 5]
    out_channels = 3
    kernel_size = 3
    padding = 1
    weight_shape = [out_channels, input_shape[1], kernel_size, kernel_size]

    x = _make_tensor(input_shape, _make_values(50, scale=0.04, offset=3)).to("cuda")
    w = _make_tensor(weight_shape, _make_values(54, scale=0.03, offset=7)).to("cuda")
    grad_output = orthon.ones([1, out_channels, 5, 5]).to("cuda")

    monkeypatch.setenv("ORTHON_CUDA_CONV2D_GRAD_WEIGHT", "naive")
    selected = orthon_c._conv2d_cuda_backward_weight(
        x, w, grad_output, 1, 1, padding, padding).to("cpu")
    direct_naive = orthon_c._conv2d_cuda_backward_weight_naive(
        x, w, grad_output, 1, 1, padding, padding).to("cpu")

    assert selected.tolist() == pytest.approx(direct_naive.tolist(), abs=1e-4)


def test_conv2d_cuda_grad_bias_optimized_matches_naive_basic():
    _assert_cuda_grad_bias_optimized_matches_naive(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )


def test_conv2d_cuda_grad_bias_optimized_matches_naive_multi_batch_multi_channel():
    _assert_cuda_grad_bias_optimized_matches_naive(
        input_shape=[4, 3, 8, 8],
        out_channels=5,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_grad_bias_optimized_matches_naive_padding_stride():
    _assert_cuda_grad_bias_optimized_matches_naive(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def test_conv2d_cuda_env_can_select_naive_grad_bias(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_bias_naive"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    grad_output = _make_tensor(
        [2, 3, 5, 5], _make_values(150, scale=0.04, offset=31)
    ).to("cuda")
    bias = orthon.zeros([3]).to("cuda")

    monkeypatch.setenv("ORTHON_CUDA_CONV2D_GRAD_BIAS", "naive")
    selected = orthon_c._conv2d_cuda_backward_bias(bias, grad_output).to("cpu")
    direct_naive = orthon_c._conv2d_cuda_backward_bias_naive(bias, grad_output).to("cpu")

    assert selected.tolist() == pytest.approx(direct_naive.tolist(), abs=1e-4)


def test_conv2d_cuda_bias_grad_env_alias_can_select_naive(monkeypatch):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not hasattr(orthon_c, "_conv2d_cuda_backward_bias_naive"):
        pytest.skip("CUDA Conv2d backward benchmark helpers are not available")

    grad_output = _make_tensor(
        [2, 3, 5, 5], _make_values(150, scale=0.04, offset=37)
    ).to("cuda")
    bias = orthon.zeros([3]).to("cuda")

    monkeypatch.setenv("ORTHON_CUDA_CONV2D_BIAS_GRAD", "naive")
    selected = orthon_c._conv2d_cuda_backward_bias(bias, grad_output).to("cpu")
    direct_naive = orthon_c._conv2d_cuda_backward_bias_naive(bias, grad_output).to("cpu")

    assert selected.tolist() == pytest.approx(direct_naive.tolist(), abs=1e-4)


def test_conv2d_cuda_backward_basic_matches_torch():
    _assert_cuda_conv2d_backward_matches_torch(
        input_shape=[1, 1, 5, 5],
        out_channels=2,
        kernel_size=3,
    )
    # Consecutive weight/bias reductions reuse shared storage across warps.
    _assert_cuda_conv2d_backward_matches_torch(
        input_shape=[2, 1, 6, 6],
        out_channels=2,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_backward_padding_matches_torch():
    _assert_cuda_conv2d_backward_matches_torch(
        input_shape=[2, 3, 6, 6],
        out_channels=4,
        kernel_size=3,
        padding=1,
    )


def test_conv2d_cuda_backward_stride_matches_torch():
    _assert_cuda_conv2d_backward_matches_torch(
        input_shape=[2, 2, 7, 7],
        out_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def test_conv2d_cuda_backward_multi_channel_multi_output_matches_torch():
    _assert_cuda_conv2d_backward_matches_torch(
        input_shape=[2, 4, 5, 5],
        out_channels=5,
        kernel_size=3,
        padding=1,
    )


@pytest.mark.skipif(not (_cuda_available() and HAS_TORCH), reason="CUDA/PyTorch not available")
@pytest.mark.parametrize("geometry", [
    (1, 1, 5, 7, 2, 2, 1, 1, 1, 1),
    (1, 1, 7, 9, 8, 3, 2, 1, 1, 1),
    (1, 1, 7, 9, 8, 3, 1, 1, 0, 2),
    (1, 2, 9, 17, 8, 3, 1, 1, 1, 1),
    (2, 1, 49, 43, 1, 3, 1, 1, 1, 1),
])
def test_r1_cuda_v2_direct_and_forced_gradients_match_reference(monkeypatch, geometry):
    import numpy as np
    n, ci, h, width, co, k, sh, sw, ph, pw = geometry
    x_values = np.asarray(_make_values(n * ci * h * width, .04, 19), dtype=np.float32)
    w_values = np.asarray(_make_values(co * ci * k * k, .03, 23), dtype=np.float32)
    x = orthon.from_numpy(x_values.reshape(n, ci, h, width)).to("cuda")
    w = orthon.from_numpy(w_values.reshape(co, ci, k, k)).to("cuda")
    b = orthon.zeros([co]).to("cuda")
    y = orthon.conv2d(x, w, b, sh, sw, ph, pw)
    g_values = np.asarray(_make_values(y.numel, .02, 31), dtype=np.float32).reshape(y.shape)
    g = orthon.from_numpy(g_values).to("cuda")
    tx = torch.tensor(x_values, dtype=torch.float64).reshape(x.shape).requires_grad_(True)
    tw = torch.tensor(w_values, dtype=torch.float64).reshape(w.shape).requires_grad_(True)
    reference = F.conv2d(tx, tw, stride=(sh, sw), padding=(ph, pw))
    reference.backward(torch.tensor(g_values, dtype=torch.float64))
    assert y.shape == tuple(reference.shape)
    assert y.to("cpu").tolist() == pytest.approx(reference.detach().flatten().tolist(), abs=3e-5, rel=1e-4)
    for component, expected in [("input", tx.grad), ("weight", tw.grad)]:
        monkeypatch.setenv("ORTHON_CUDA_CONV2D_GRAD_" + component.upper(), "optimized_v2")
        for suffix in ["", "_optimized_v2"]:
            fn = getattr(orthon_c, "_conv2d_cuda_backward_" + component + suffix)
            actual = fn(x, w, g, sh, sw, ph, pw).to("cpu")
            assert actual.tolist() == pytest.approx(expected.flatten().tolist(), abs=3e-5, rel=1e-4)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_r1_cuda_v2_low_precision_uses_typed_fallback(monkeypatch, dtype):
    x = orthon.full([1, 1, 5, 7], .25).cast(dtype).to("cuda")
    w = orthon.full([8, 1, 3, 3], .5).cast(dtype).to("cuda")
    g = orthon.full([1, 8, 5, 7], .125).cast(dtype).to("cuda")
    for component in ["input", "weight"]:
        ordinary = getattr(orthon_c, "_conv2d_cuda_backward_" + component + "_optimized")
        reference = ordinary(x, w, g, 1, 1, 1, 1).to("cpu").cast("float32").tolist()
        monkeypatch.setenv("ORTHON_CUDA_CONV2D_GRAD_" + component.upper(), "optimized_v2")
        for suffix in ["", "_optimized_v2"]:
            actual = getattr(orthon_c, "_conv2d_cuda_backward_" + component + suffix)(x, w, g, 1, 1, 1, 1)
            assert actual.dtype == dtype and actual.device == x.device
            assert actual.to("cpu").cast("float32").tolist() == reference


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_r1_cuda_private_v2_rejects_metadata_before_execution():
    x = orthon.ones([1, 1, 5, 5]).to("cpu")
    w = orthon.ones([8, 1, 3, 3]).to("cpu")
    g = orthon.ones([1, 8, 5, 5]).to("cpu")
    for component in ["input", "weight"]:
        fn = getattr(orthon_c, "_conv2d_cuda_backward_" + component + "_optimized_v2")
        with pytest.raises(ValueError, match="CUDA tensors"):
            fn(x, w, g, 1, 1, 1, 1)
        dx, dw, dg = x.to("cuda"), w.to("cuda"), g.to("cuda")
        with pytest.raises(ValueError, match="dtype"):
            fn(dx, dw, dg.cast("float16"), 1, 1, 1, 1)
        with pytest.raises(ValueError, match="shape"):
            fn(dx, dw, dg.slice(2, 0, 4).contiguous(), 1, 1, 1, 1)
        with pytest.raises(RuntimeError, match="contiguous"):
            fn(dx.transpose(2, 3), dw, dg, 1, 1, 1, 1)


@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
def test_r1_cuda_v2_ineligible_fallback_completes_in_subprocess():
    import os
    import subprocess
    import sys
    code = """import orthon as o
x=o.ones([1,1,5,7]).to('cuda'); w=o.ones([2,1,2,2]).to('cuda'); g=o.ones([1,2,6,8]).to('cuda')
r=o._C._conv2d_cuda_backward_weight(x,w,g,1,1,1,1).to('cpu')
assert r.tolist()==[35.0]*8
print('FALLBACK COMPLETE')
"""
    env = dict(os.environ, ORTHON_CUDA_CONV2D_GRAD_WEIGHT="optimized_v2")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "FALLBACK COMPLETE"
