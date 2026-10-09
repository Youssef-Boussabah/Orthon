from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest
import orthon as ortho
import orthon.functional as F

torch = pytest.importorskip("torch")




requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _make(values, shape, device="cpu", requires_grad=False):
    x = ortho.zeros(shape)
    cols = shape[-1] if shape else 1
    if len(shape) == 1:
        for i, v in enumerate(values):
            x[i] = float(v)
    elif len(shape) == 2:
        for i, v in enumerate(values):
            x[[i // cols, i % cols]] = float(v)
    elif len(shape) == 3:
        d1, d2 = shape[1], shape[2]
        for i, v in enumerate(values):
            n = i // (d1 * d2)
            rem = i % (d1 * d2)
            x[[n, rem // d2, rem % d2]] = float(v)
    else:
        raise ValueError("unsupported shape")
    if device == "cuda":
        x = x.to("cuda")
    x.requires_grad_(requires_grad)
    return x


# Full reduction.

def test_max_full_reduction_value_matches_torch():
    vals = [1.0, 5.0, 3.0, 2.0, 7.0, 4.0]
    x  = _make(vals, [2, 3], requires_grad=False)
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3)
    assert x.max().item() == pytest.approx(tx.max().item())


def test_max_full_reduction_backward():
    vals = [1.0, 5.0, 3.0, 2.0, 7.0, 4.0]
    x  = _make(vals, [2, 3], requires_grad=True)
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3).requires_grad_(True)

    x.max().backward()
    tx.max().backward()

    assert x.grad.tolist() == pytest.approx(tx.grad.flatten().tolist())


def test_argmax_full_reduction_index():
    vals = [1.0, 5.0, 3.0, 2.0, 7.0, 4.0]
    x  = _make(vals, [2, 3])
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3)
    assert x.argmax().item() == tx.argmax().item()


def test_argmax_full_reduction_dtype_is_int64():
    x = _make([1.0, 3.0, 2.0], [3])
    result = x.argmax()
    assert result.dtype == "int64"


def test_argmax_no_grad_fn():
    x = _make([1.0, 3.0, 2.0], [3], requires_grad=True)
    result = x.argmax()
    assert not result._has_grad_fn
    assert not result.requires_grad


# Dim reduction.

@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_max_dim_reduction_value_shape_matches_torch(dim, keepdim):
    vals = [float(i + 1) for i in range(24)]
    x  = _make(vals, [2, 3, 4])
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3, 4)

    y  = x.max(dim=dim, keepdim=keepdim)
    ty = tx.max(dim=dim, keepdim=keepdim).values

    assert list(y.shape) == list(ty.shape)
    assert y.tolist() == pytest.approx(ty.detach().flatten().tolist())


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_argmax_dim_reduction_index_shape_matches_torch(dim, keepdim):
    vals = [float(i + 1) for i in range(24)]
    x  = _make(vals, [2, 3, 4])
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3, 4)

    idx  = x.argmax(dim=dim, keepdim=keepdim)
    tidx = tx.argmax(dim=dim, keepdim=keepdim)

    assert list(idx.shape) == list(tidx.shape)
    assert idx.tolist() == tidx.detach().flatten().tolist()


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_max_dim_backward_matches_torch(dim, keepdim):
    vals = [float(i + 1) for i in range(24)]
    x  = _make(vals, [2, 3, 4], requires_grad=True)
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(True)

    y  = x.max(dim=dim, keepdim=keepdim)
    y.backward()

    ty = tx.max(dim=dim, keepdim=keepdim).values
    ty.sum().backward()

    assert x.grad.tolist() == pytest.approx(tx.grad.flatten().tolist())


# Non-contiguous input.

def test_max_noncontiguous_matches_torch():
    vals = [float(i + 1) for i in range(6)]
    base = _make(vals, [2, 3])
    view = base.transpose(0, 1)
    y = view.max(dim=1)

    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 3).t()
    ty = tx.max(dim=1).values

    assert y.tolist() == pytest.approx(ty.detach().flatten().tolist())


def test_argmax_noncontiguous_matches_torch():
    vals = [float(i + 1) for i in range(6)]
    base = _make(vals, [2, 3])
    view = base.transpose(0, 1)
    idx  = view.argmax(dim=1)

    tx   = torch.tensor(vals, dtype=torch.float32).reshape(2, 3).t()
    tidx = tx.argmax(dim=1)

    assert idx.tolist() == tidx.detach().flatten().tolist()


# Functional API.

def test_functional_max_argmax():
    vals = [1.0, 5.0, 3.0, 2.0, 7.0, 4.0]
    x = _make(vals, [2, 3])
    assert F.max(x).item()    == pytest.approx(7.0)
    assert F.argmax(x).item() == 4


def test_module_level_max_argmax():
    vals = [1.0, 5.0, 3.0, 2.0, 7.0, 4.0]
    x = _make(vals, [2, 3])
    assert ortho.max(x).item()    == pytest.approx(7.0)
    assert ortho.argmax(x).item() == 4


# item() for int64.

def test_item_int64_returns_python_int():
    x   = _make([1.0, 3.0, 2.0], [3])
    idx = x.argmax()
    v   = idx.item()
    assert isinstance(v, int)
    assert v == 1


# CUDA tests.

@requires_cuda
def test_cuda_max_full_reduction_matches_cpu():
    vals = [float(i - 5) for i in range(32)]
    x_cpu  = _make(vals, [4, 8])
    x_cuda = _make(vals, [4, 8], device="cuda")
    assert x_cuda.max().to("cpu").item() == pytest.approx(x_cpu.max().item())


@requires_cuda
def test_cuda_argmax_full_reduction_matches_cpu():
    vals = [float(i - 5) for i in range(32)]
    x_cpu  = _make(vals, [4, 8])
    x_cuda = _make(vals, [4, 8], device="cuda")
    assert x_cuda.argmax().to("cpu").item() == x_cpu.argmax().item()


@requires_cuda
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_cuda_max_dim_matches_cpu(dim, keepdim):
    vals = [float(i + 1) / 4.0 for i in range(24)]
    x_cpu  = _make(vals, [2, 3, 4])
    x_cuda = _make(vals, [2, 3, 4], device="cuda")

    y_cpu  = x_cpu.max(dim=dim, keepdim=keepdim)
    y_cuda = x_cuda.max(dim=dim, keepdim=keepdim)

    assert list(y_cuda.shape) == list(y_cpu.shape)
    assert y_cuda.to("cpu").tolist() == pytest.approx(y_cpu.tolist(), rel=1e-5)


@requires_cuda
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_cuda_argmax_dim_matches_cpu(dim, keepdim):
    vals = [float(i + 1) / 4.0 for i in range(24)]
    x_cpu  = _make(vals, [2, 3, 4])
    x_cuda = _make(vals, [2, 3, 4], device="cuda")

    idx_cpu  = x_cpu.argmax(dim=dim, keepdim=keepdim)
    idx_cuda = x_cuda.argmax(dim=dim, keepdim=keepdim)

    assert list(idx_cuda.shape) == list(idx_cpu.shape)
    assert idx_cuda.to("cpu").tolist() == idx_cpu.tolist()


@requires_cuda
@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_cuda_max_dim_backward_matches_cpu(dim, keepdim):
    vals = [float(i + 1) for i in range(24)]
    x_cpu  = _make(vals, [2, 3, 4], requires_grad=True)
    x_cuda = _make(vals, [2, 3, 4], device="cuda", requires_grad=True)

    x_cpu.max(dim=dim, keepdim=keepdim).backward()
    x_cuda.max(dim=dim, keepdim=keepdim).backward()

    assert x_cuda.grad.to("cpu").tolist() == pytest.approx(
        x_cpu.grad.tolist(), rel=1e-5, abs=1e-5
    )


# Reduction identity: -infinity, not a finite sentinel.
# Values in (-FLT_MAX, -1e38], -FLT_MAX itself, and -inf inputs must reduce
# correctly on both devices.

NEG_INF   = float("-inf")
FLT_LOWEST = -3.4028234663852886e38


def test_max_values_below_neg_1e38_matches_torch():
    vals = [-2e38, -3e38, -1.5e38]
    x  = _make(vals, [3])
    tx = torch.tensor(vals, dtype=torch.float32)
    assert x.max().item() == pytest.approx(tx.max().item(), rel=1e-6)
    assert x.argmax().item() == tx.argmax().item() == 2


def test_max_lowest_float_matches_torch():
    vals = [FLT_LOWEST, -1e38, FLT_LOWEST]
    x  = _make(vals, [3])
    tx = torch.tensor(vals, dtype=torch.float32)
    assert x.max().item() == pytest.approx(tx.max().item(), rel=1e-6)
    assert x.argmax().item() == tx.argmax().item() == 1


def test_max_all_negative_infinity():
    x = _make([NEG_INF] * 4, [4])
    assert x.max().item() == NEG_INF
    assert x.argmax().item() == 0


def test_max_mixed_negative_infinity_matches_torch():
    vals = [NEG_INF, -2.0, NEG_INF, -7.5]
    x  = _make(vals, [4])
    tx = torch.tensor(vals, dtype=torch.float32)
    assert x.max().item() == tx.max().item() == -2.0
    assert x.argmax().item() == tx.argmax().item() == 1


def test_max_dim_extreme_negative_matches_torch():
    vals = [NEG_INF, NEG_INF, -2e38, -3e38]
    x  = _make(vals, [2, 2])
    tx = torch.tensor(vals, dtype=torch.float32).reshape(2, 2)
    y  = x.max(dim=1)
    ty = tx.max(dim=1).values
    assert y.tolist() == pytest.approx(ty.tolist(), rel=1e-6)
    assert x.argmax(dim=1).tolist() == tx.argmax(dim=1).tolist()


@requires_cuda
def test_cuda_max_extreme_negative_matches_cpu():
    vals = [-2e38, -3e38, -1.5e38, NEG_INF]
    x_cpu  = _make(vals, [4])
    x_cuda = _make(vals, [4], device="cuda")
    assert x_cuda.max().to("cpu").item() == x_cpu.max().item()
    assert x_cuda.argmax().to("cpu").item() == x_cpu.argmax().item() == 2


@requires_cuda
def test_cuda_max_all_negative_infinity():
    x = _make([NEG_INF] * 4, [4], device="cuda")
    assert x.max().to("cpu").item() == NEG_INF
    assert x.argmax().to("cpu").item() == 0


@requires_cuda
def test_cuda_max_dim_extreme_negative_matches_cpu():
    vals = [NEG_INF, NEG_INF, -2e38, -3e38]
    x_cpu  = _make(vals, [2, 2])
    x_cuda = _make(vals, [2, 2], device="cuda")
    assert (x_cuda.max(dim=1).to("cpu").tolist()
            == pytest.approx(x_cpu.max(dim=1).tolist()))
    assert (x_cuda.argmax(dim=1).to("cpu").tolist()
            == x_cpu.argmax(dim=1).tolist())
