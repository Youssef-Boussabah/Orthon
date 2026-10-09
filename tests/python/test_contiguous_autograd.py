from tests.support.capabilities import orthon_cuda_available as _cuda_available
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")




requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and _cuda_available()), reason="CUDA not available")


def _t(arr, device="cpu"):
    t = ortho.from_numpy(np.asarray(arr, dtype=np.float32))
    return t.to("cuda") if device == "cuda" else t


def _close(a, b, rel=1e-5, abs_=1e-5):
    if a.device.is_cuda():
        a = a.to("cpu")
    assert list(a.shape) == list(b.shape)
    assert a.tolist() == pytest.approx(
        b.detach().cpu().flatten().tolist(), rel=rel, abs=abs_
    )


# CPU forward behavior.

def test_contiguous_tensor_returns_self_values():
    x = _t([[1.0, 2.0], [3.0, 4.0]])
    y = x.contiguous()
    assert y.is_contiguous
    assert y.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])


def test_already_contiguous_preserves_grad_fn():
    x = _t([[1.0, 2.0], [3.0, 4.0]])
    x.requires_grad_(True)
    y = x.contiguous()
    # Already contiguous → return the same tensor; grad_fn is x's own (None for leaf)
    assert not y._has_grad_fn  # no new ContiguousNode wrapping a contiguous leaf
    assert y.is_contiguous


def test_transpose_then_contiguous_matches_torch():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(3, 4)).astype(np.float32)

    x = _t(arr)
    y = x.transpose(0, 1).contiguous()
    assert y.is_contiguous
    expected = torch.tensor(arr).t().contiguous()
    _close(y, expected)


def test_transpose_contiguous_reshape_works():
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(2, 4, 3, 5)).astype(np.float32)

    x = _t(arr)
    # [2,4,3,5] → transpose(1,2) → [2,3,4,5] → contiguous → reshape [6,4,5]
    y = x.transpose(1, 2).contiguous().reshape([6, 4, 5])
    assert list(y.shape) == [6, 4, 5]

    tx = torch.tensor(arr)
    ty = tx.transpose(1, 2).contiguous().reshape(6, 4, 5)
    _close(y, ty)


# CPU autograd backward.

def test_contiguous_backward_identity_cpu():
    """contiguous of a contiguous tensor: grad flows unchanged."""
    arr = np.ones((2, 3), dtype=np.float32)
    x = _t(arr)
    x.requires_grad_(True)
    y = x.contiguous()
    y.sum().backward()
    _close(x.grad, torch.ones(2, 3))


def test_transpose_contiguous_backward_cpu():
    """x → transpose → contiguous → sum: x.grad should be all ones."""
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(3, 4)).astype(np.float32)

    x = _t(arr)
    x.requires_grad_(True)
    y = x.transpose(0, 1).contiguous()
    y.sum().backward()

    tx = torch.tensor(arr, requires_grad=True)
    tx.t().contiguous().sum().backward()

    _close(x.grad, tx.grad)


def test_transpose_contiguous_backward_4d_cpu():
    """4-D transpose → contiguous → sum: grads match PyTorch."""
    rng = np.random.default_rng(3)
    arr = rng.normal(size=(2, 3, 4, 5)).astype(np.float32)

    x = _t(arr)
    x.requires_grad_(True)
    x.transpose(1, 2).contiguous().sum().backward()

    tx = torch.tensor(arr, requires_grad=True)
    tx.transpose(1, 2).contiguous().sum().backward()

    _close(x.grad, tx.grad)


def test_transpose_contiguous_reshape_backward_cpu():
    """Full multi-head-attention reshape chain: backward gives correct grads."""
    rng = np.random.default_rng(4)
    B, T, H, D = 2, 4, 2, 3
    arr = rng.normal(size=(B, T, H, D)).astype(np.float32)

    x = _t(arr)
    x.requires_grad_(True)
    y = x.transpose(1, 2).contiguous().reshape([B * H, T, D])
    y.sum().backward()

    tx = torch.tensor(arr, requires_grad=True)
    tx.transpose(1, 2).contiguous().reshape(B * H, T, D).sum().backward()

    _close(x.grad, tx.grad)


def test_orthon_contiguous_top_level():
    """ortho.contiguous(t) mirrors t.contiguous()."""
    rng = np.random.default_rng(5)
    arr = rng.normal(size=(3, 4)).astype(np.float32)

    x = _t(arr)
    x.requires_grad_(True)
    y1 = x.transpose(0, 1).contiguous()
    x2 = _t(arr)
    x2.requires_grad_(True)
    y2 = ortho.contiguous(x2.transpose(0, 1))

    assert y1.tolist() == pytest.approx(y2.tolist(), rel=1e-5)


# CUDA forward and backward.

@requires_cuda
def test_transpose_contiguous_values_cuda():
    rng = np.random.default_rng(6)
    arr = rng.normal(size=(3, 4)).astype(np.float32)

    x = _t(arr, "cuda")
    y = x.transpose(0, 1).contiguous()

    assert y.device.is_cuda()
    assert y.is_contiguous

    expected = torch.tensor(arr).t().contiguous()
    _close(y, expected)


@requires_cuda
def test_transpose_contiguous_backward_cuda():
    rng = np.random.default_rng(7)
    arr = rng.normal(size=(3, 4)).astype(np.float32)

    x = _t(arr, "cuda")
    x.requires_grad_(True)
    x.transpose(0, 1).contiguous().sum().backward()

    assert x.grad.device.is_cuda()

    tx = torch.tensor(arr, requires_grad=True)
    tx.t().contiguous().sum().backward()
    _close(x.grad, tx.grad, rel=1e-5, abs_=1e-5)


@requires_cuda
def test_transpose_contiguous_reshape_backward_cuda():
    rng = np.random.default_rng(8)
    B, T, H, D = 2, 4, 2, 3
    arr = rng.normal(size=(B, T, H, D)).astype(np.float32)

    x = _t(arr, "cuda")
    x.requires_grad_(True)
    x.transpose(1, 2).contiguous().reshape([B * H, T, D]).sum().backward()

    assert x.grad.device.is_cuda()

    tx = torch.tensor(arr, requires_grad=True)
    tx.transpose(1, 2).contiguous().reshape(B * H, T, D).sum().backward()

    _close(x.grad, tx.grad, rel=2e-5, abs_=2e-5)
