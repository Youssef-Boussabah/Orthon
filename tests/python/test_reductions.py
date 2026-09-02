import pytest

import orthon as ortho
import orthon.functional as F

torch = pytest.importorskip("torch")


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _tensor_from_values(values, shape, device="cpu", requires_grad=False):
    x = ortho.zeros(shape)
    cols = shape[-1] if shape else 1
    if len(shape) == 1:
        for i, value in enumerate(values):
            x[i] = float(value)
    elif len(shape) == 2:
        for i, value in enumerate(values):
            x[[i // cols, i % cols]] = float(value)
    elif len(shape) == 3:
        d1, d2 = shape[1], shape[2]
        for i, value in enumerate(values):
            n = i // (d1 * d2)
            rem = i % (d1 * d2)
            x[[n, rem // d2, rem % d2]] = float(value)
    else:
        raise ValueError("unsupported test shape")
    if device == "cuda":
        x = x.to("cuda")
    x.requires_grad_(requires_grad)
    return x


@pytest.mark.parametrize("op", ["sum", "mean"])
def test_full_reduction_cpu_forward_backward_matches_torch(op):
    vals = [float(i - 4) for i in range(12)]
    x = _tensor_from_values(vals, [3, 4], requires_grad=True)
    y = getattr(x, op)()
    assert isinstance(y, ortho.Tensor)
    assert list(y.shape) == []
    y.backward()

    tx = torch.tensor(vals, dtype=torch.float32).reshape(3, 4).requires_grad_(True)
    ty = getattr(tx, op)()
    ty.backward()

    assert y.item() == pytest.approx(ty.item())
    assert x.grad.tolist() == pytest.approx(tx.grad.flatten().tolist())


@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_dim_reduction_cpu_forward_backward_matches_torch(op, dim, keepdim):
    vals = [float(i + 1) for i in range(24)]
    x = _tensor_from_values(vals, [2, 3, 4], requires_grad=True)
    y = getattr(x, op)(dim=dim, keepdim=keepdim)
    y.backward()

    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(True)
    ty = getattr(tx, op)(dim=dim, keepdim=keepdim)
    ty.sum().backward()

    assert list(y.shape) == list(ty.shape)
    assert y.tolist() == pytest.approx(ty.detach().flatten().tolist())
    assert x.grad.tolist() == pytest.approx(tx.grad.flatten().tolist())


def test_functional_sum_mean_return_tensors():
    x = ortho.ones([2, 3])
    assert isinstance(F.sum(x), ortho.Tensor)
    assert isinstance(F.mean(x, dim=1), ortho.Tensor)
    assert F.sum(x).item() == pytest.approx(6.0)
    assert F.mean(x, dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_cpu_non_contiguous_dim_reduction_matches_torch():
    vals = [float(i + 1) for i in range(6)]
    base = _tensor_from_values(vals, [2, 3])
    view = base.transpose(0, 1)
    y = view.sum(dim=1)

    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3).t()
    ty = tx.sum(dim=1)
    assert not view.is_contiguous
    assert y.tolist() == pytest.approx(ty.flatten().tolist())


@pytest.mark.parametrize("op", ["sum", "mean"])
@requires_cuda
def test_full_reduction_cuda_forward_backward_matches_torch(op):
    vals = [float(i - 7) for i in range(64 * 128)]
    x = _tensor_from_values(vals, [64, 128], device="cuda", requires_grad=True)
    y = getattr(x, op)()
    y.backward()

    tx = torch.tensor(vals, dtype=torch.float32).reshape(64, 128).requires_grad_(True)
    ty = getattr(tx, op)()
    ty.backward()

    assert y.device.is_cuda()
    assert list(y.shape) == []
    assert y.to("cpu").item() == pytest.approx(ty.item(), rel=1e-4, abs=1e-4)
    assert x.grad.device.is_cuda()
    assert x.grad.to("cpu").tolist() == pytest.approx(
        tx.grad.flatten().tolist(), rel=1e-5, abs=1e-5
    )


@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
@requires_cuda
def test_dim_reduction_cuda_forward_backward_matches_torch(op, dim, keepdim):
    vals = [float(i + 1) / 10.0 for i in range(2 * 3 * 4)]
    x = _tensor_from_values(vals, [2, 3, 4], device="cuda", requires_grad=True)
    y = getattr(x, op)(dim=dim, keepdim=keepdim)
    y.backward()

    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(True)
    ty = getattr(tx, op)(dim=dim, keepdim=keepdim)
    ty.sum().backward()

    assert y.device.is_cuda()
    assert list(y.shape) == list(ty.shape)
    assert y.to("cpu").tolist() == pytest.approx(
        ty.detach().flatten().tolist(), rel=1e-5, abs=1e-5
    )
    assert x.grad.device.is_cuda()
    assert x.grad.to("cpu").tolist() == pytest.approx(
        tx.grad.flatten().tolist(), rel=1e-5, abs=1e-5
    )


@requires_cuda
def test_cuda_non_contiguous_dim_reduction_matches_torch():
    vals = [float(i + 1) for i in range(6)]
    base = _tensor_from_values(vals, [2, 3], device="cuda")
    view = base.transpose(0, 1)
    y = view.sum(dim=1)

    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3).t()
    ty = tx.sum(dim=1)
    assert not view.is_contiguous
    assert y.device.is_cuda()
    assert y.to("cpu").tolist() == pytest.approx(ty.flatten().tolist())


@requires_cuda
def test_cuda_full_reductions_use_scratch_workspace_and_reuse_reserved_block():
    x = ortho.ones([4096]).to("cuda")
    ortho._memory_pool_reset_scratch("cuda")

    summed = ortho.sum(x)
    ortho.cuda_synchronize()
    assert summed.to("cpu").item() == pytest.approx(4096.0)
    sum_stats = ortho._memory_pool_scratch_stats("cuda")
    assert sum_stats["used_bytes"] > 0
    assert sum_stats["reserved_bytes"] >= sum_stats["used_bytes"]

    meaned = ortho.mean(x)
    ortho.cuda_synchronize()
    assert meaned.to("cpu").item() == pytest.approx(1.0)
    mean_stats = ortho._memory_pool_scratch_stats("cuda")
    assert mean_stats["reserved_bytes"] == sum_stats["reserved_bytes"]
    assert mean_stats["used_bytes"] == sum_stats["used_bytes"]

    maxed = ortho.max(x)
    ortho.cuda_synchronize()
    assert maxed.to("cpu").item() == pytest.approx(1.0)
    max_stats = ortho._memory_pool_scratch_stats("cuda")
    assert max_stats["reserved_bytes"] == sum_stats["reserved_bytes"]
    assert max_stats["used_bytes"] > sum_stats["used_bytes"]


def test_invalid_dim_raises():
    x = ortho.ones([2, 3])
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        x.sum(dim=2)
