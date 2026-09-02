"""CUDA MaxPool2d forward/backward parity tests."""

import pytest

import orthon
from orthon.nn import MaxPool2d

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _cuda_available():
    if not orthon.cuda_enabled():
        return False
    try:
        orthon.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


def _unravel_index(flat_index, shape):
    idx = []
    for size in reversed(shape):
        idx.append(flat_index % size)
        flat_index //= size
    return list(reversed(idx))


def _make_values(count, scale=0.07, offset=0):
    return [((i + offset) % 23 - 11) * scale for i in range(count)]


def _make_tensor(shape, values, requires_grad=False):
    tensor = orthon.zeros(shape)
    for k, value in enumerate(values):
        tensor.set_item(_unravel_index(k, shape), float(value))
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def _torch_device():
    return "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"


def _assert_cuda_maxpool_matches_torch(shape, kernel_size, stride=None, padding=0):
    if not _cuda_available():
        pytest.skip("Orthon CUDA is not available")
    if not HAS_TORCH:
        pytest.skip("PyTorch is not available")

    if stride is None:
        stride = kernel_size

    values = _make_values(shape[0] * shape[1] * shape[2] * shape[3], offset=3)
    # to() is differentiable, so moving a TRACKED CPU tensor would
    # make x_d a non-leaf ToNode output whose .grad lands on the CPU source.
    # This test wants the CUDA tensor itself to be the leaf: move first,
    # track after.
    x_d = _make_tensor(shape, values).to("cuda")
    x_d.requires_grad_(True)
    out_d = MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)(x_d)
    out_d.backward()

    device = _torch_device()
    x_t = torch.tensor(values, dtype=torch.float32, device=device).reshape(shape)
    x_t = x_t.clone().requires_grad_(True)
    out_t = torch.nn.functional.max_pool2d(
        x_t,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    out_t.sum().backward()

    assert out_d.to("cpu").tolist() == pytest.approx(
        out_t.detach().cpu().flatten().tolist(), abs=1e-6
    )
    assert x_d.grad.to("cpu").tolist() == pytest.approx(
        x_t.grad.detach().cpu().flatten().tolist(), abs=1e-6
    )


def test_maxpool2d_cuda_basic_matches_torch():
    _assert_cuda_maxpool_matches_torch(
        shape=[1, 1, 4, 4],
        kernel_size=2,
    )


def test_maxpool2d_cuda_stride_matches_torch():
    _assert_cuda_maxpool_matches_torch(
        shape=[2, 3, 6, 6],
        kernel_size=2,
        stride=2,
    )


def test_maxpool2d_cuda_padding_matches_torch():
    _assert_cuda_maxpool_matches_torch(
        shape=[2, 2, 5, 5],
        kernel_size=3,
        stride=2,
        padding=1,
    )
