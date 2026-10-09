"""BatchNorm values, gradients and running-statistic parity against PyTorch."""
from tests.support.capabilities import orthon_cuda_available
import numpy as np
import pytest

import orthon as ortho
from orthon.nn import BatchNorm1d, BatchNorm2d

torch = pytest.importorskip("torch")

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)

EPS = 1e-5
MOM = 0.1

# Shapes exercised: BatchNorm1d [N, C] and BatchNorm2d [N, C, H, W].
SHAPES_1D = [(4, 3), (8, 4), (2, 5)]
SHAPES_2D = [(2, 3, 2, 2), (3, 2, 3, 4)]


def _fixtures(shape):
    """Deterministic, non-linear input and upstream gradient."""
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    x = (np.cos(i * 1.7) * 2.3 + np.sin(i * 0.43) * 1.1 + 0.4 * (i % 5)).reshape(shape)
    up = (np.sin(i * 2.1) * 1.4 + np.cos(i * 0.77) * 0.6
          + 0.15 * ((i * 7) % 3)).reshape(shape)
    return x, up


def _init_state(C):
    """Non-default initial running statistics and affine values."""
    rm = np.array([0.2 - 0.11 * k for k in range(C)], dtype=np.float64)
    rv = np.array([1.3 + 0.27 * k for k in range(C)], dtype=np.float64)
    w = np.array([1.0 + 0.25 * k for k in range(C)], dtype=np.float64)
    b = np.array([-0.5 + 0.3 * k for k in range(C)], dtype=np.float64)
    return rm, rv, w, b


def _dt(arr, device="cpu", dtype="float32"):
    t = ortho.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if device != "cpu":
        t = t.to(device)
    if dtype != "float32":
        t = ortho.cast(t, dtype)
    return t


def _npy(t):
    return np.array(t.tolist(), dtype=np.float64)


def _set_param(param, values):
    """Overwrite a Parameter in place without disturbing the graph identity."""
    param.mul_(0.0)
    param.add_(_dt(values), alpha=1.0)


def _build(kind, C, affine, device, dtype, rm, rv, w, b):
    Mod = BatchNorm1d if kind == "1d" else BatchNorm2d
    bn = Mod(C, eps=EPS, momentum=MOM, affine=affine)
    bn.running_mean = _dt(rm)
    bn.running_var = _dt(rv)
    if affine:
        _set_param(bn.weight, w)
        _set_param(bn.bias, b)
    if device != "cpu":
        bn.to(device)
    if dtype != "float32":
        bn.cast(dtype)
    return bn


def _torch_build(kind, C, affine, device, tdtype, rm, rv, w, b):
    Mod = torch.nn.BatchNorm1d if kind == "1d" else torch.nn.BatchNorm2d
    tm = Mod(C, eps=EPS, momentum=MOM, affine=affine)
    with torch.no_grad():
        tm.running_mean.copy_(torch.tensor(rm, dtype=torch.float32))
        tm.running_var.copy_(torch.tensor(rv, dtype=torch.float32))
        if affine:
            tm.weight.copy_(torch.tensor(w, dtype=torch.float32))
            tm.bias.copy_(torch.tensor(b, dtype=torch.float32))
    return tm.to(device).to(tdtype)


def _run_parity(kind, shape, affine, device="cpu", dtype="float32",
                tdtype=torch.float32, training=True, tol=1e-4):
    C = shape[1]
    x_np, up_np = _fixtures(shape)
    rm, rv, w, b = _init_state(C)

    bn = _build(kind, C, affine, device, dtype, rm, rv, w, b)
    bn.train() if training else bn.eval()
    x = _dt(x_np, device, dtype)
    x.requires_grad_(True)
    up = _dt(up_np, device, dtype)
    y = bn(x)
    ortho.sum(y * up).backward()

    tdev = "cuda" if device != "cpu" else "cpu"
    tm = _torch_build(kind, C, affine, tdev, tdtype, rm, rv, w, b)
    tm.train() if training else tm.eval()
    xt = torch.tensor(x_np, dtype=tdtype, device=tdev, requires_grad=True)
    upt = torch.tensor(up_np, dtype=tdtype, device=tdev)
    yt = tm(xt)
    (yt * upt).sum().backward()

    def tn(t):
        return t.detach().float().cpu().numpy().ravel().astype(np.float64)

    assert _npy(y) == pytest.approx(tn(yt), abs=tol), "forward output"
    assert _npy(x.grad) == pytest.approx(tn(xt.grad), abs=tol), "input gradient"
    if affine:
        assert _npy(bn.weight.grad) == pytest.approx(
            tn(tm.weight.grad), abs=tol), "weight gradient"
        assert _npy(bn.bias.grad) == pytest.approx(
            tn(tm.bias.grad), abs=tol), "bias gradient"
    assert _npy(bn.running_mean) == pytest.approx(
        tn(tm.running_mean), abs=tol), "running_mean"
    assert _npy(bn.running_var) == pytest.approx(
        tn(tm.running_var), abs=tol), "running_var"
    return bn, y, x


# CPU float32 parity
@pytest.mark.parametrize("shape", SHAPES_1D)
def test_bn1d_cpu_train_parity_affine(shape):
    _run_parity("1d", shape, affine=True)


@pytest.mark.parametrize("shape", SHAPES_1D)
def test_bn1d_cpu_train_parity_no_affine(shape):
    _run_parity("1d", shape, affine=False)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_bn2d_cpu_train_parity_affine(shape):
    _run_parity("2d", shape, affine=True)


@pytest.mark.parametrize("shape", SHAPES_2D)
def test_bn2d_cpu_train_parity_no_affine(shape):
    _run_parity("2d", shape, affine=False)


@pytest.mark.parametrize("kind,shape", [("1d", (4, 3)), ("2d", (2, 3, 2, 2))])
@pytest.mark.parametrize("affine", [True, False])
def test_cpu_eval_parity(kind, shape, affine):
    _run_parity(kind, shape, affine=affine, training=False)


def test_cpu_input_gradient_is_substantial():
    """Guards the oracle itself: a vanishing x.grad would make parity vacuous."""
    _, _, x = _run_parity("1d", (4, 3), affine=True)
    assert np.abs(_npy(x.grad)).max() > 0.1


def test_cpu_running_stats_differ_per_channel():
    """Guards the oracle: identical per-channel stats would hide indexing bugs."""
    bn, _, _ = _run_parity("1d", (4, 3), affine=True)
    rv = _npy(bn.running_var)
    assert len(set(np.round(rv, 6))) == len(rv)


# CUDA float32 parity
@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("shape", SHAPES_1D)
def test_bn1d_cuda_train_parity_affine(shape):
    _run_parity("1d", shape, affine=True, device="cuda")


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("shape", SHAPES_2D)
def test_bn2d_cuda_train_parity_affine(shape):
    _run_parity("2d", shape, affine=True, device="cuda")


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("kind,shape", [("1d", (4, 3)), ("2d", (2, 3, 2, 2))])
def test_cuda_train_parity_no_affine(kind, shape):
    _run_parity(kind, shape, affine=False, device="cuda")


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("kind,shape", [("1d", (4, 3)), ("2d", (2, 3, 2, 2))])
def test_cuda_eval_parity(kind, shape):
    _run_parity(kind, shape, affine=True, device="cuda", training=False)


@requires_cuda
def test_cuda_state_stays_on_device():
    C = 3
    rm, rv, w, b = _init_state(C)
    bn = _build("1d", C, True, "cuda", "float32", rm, rv, w, b)
    bn.train()
    x = _dt(_fixtures((4, 3))[0], "cuda")
    x.requires_grad_(True)
    y = bn(x)
    ortho.sum(y).backward()
    assert y.device.is_cuda()
    assert x.grad.device.is_cuda()
    assert bn.running_mean.device.is_cuda()
    assert bn.running_var.device.is_cuda()
    assert bn.weight.grad.device.is_cuda()


# low-precision CUDA
@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("kind,shape", [("1d", (4, 3)), ("2d", (2, 3, 2, 2))])
def test_cuda_float16_parity(kind, shape):
    _run_parity(kind, shape, affine=True, device="cuda", dtype="float16",
                tdtype=torch.float16, tol=3e-2)


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("kind,shape", [("1d", (4, 3)), ("2d", (2, 3, 2, 2))])
def test_cuda_bfloat16_parity(kind, shape):
    _run_parity(kind, shape, affine=True, device="cuda", dtype="bfloat16",
                tdtype=torch.bfloat16, tol=2e-1)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_low_precision_dtypes_and_devices(dtype):
    """Output follows the model dtype and buffers keep the module dtype."""
    C = 3
    rm, rv, w, b = _init_state(C)
    bn = _build("1d", C, True, "cuda", dtype, rm, rv, w, b)
    bn.train()
    x = _dt(_fixtures((4, 3))[0], "cuda", dtype)
    x.requires_grad_(True)
    y = bn(x)
    ortho.sum(y).backward()
    assert y.dtype == dtype
    assert bn.running_mean.dtype == dtype
    assert bn.running_var.dtype == dtype
    assert bn.running_mean.device.is_cuda()
    assert x.grad is not None and x.grad.device.is_cuda()


def test_cpu_float16_is_unsupported_by_reductions():
    """CPU half is not supported by the native mean kernel; BatchNorm raises."""
    bn = BatchNorm1d(3).half()
    bn.train()
    x = ortho.cast(_dt(_fixtures((4, 3))[0]), "float16")
    with pytest.raises(RuntimeError):
        bn(x)


# float64 CPU
def test_cpu_float64_forward_parity():
    """float64 forward matches PyTorch. Backward is blocked natively, see below."""
    shape = (4, 3)
    C = shape[1]
    x_np, _ = _fixtures(shape)
    rm, rv, w, b = _init_state(C)
    bn = BatchNorm1d(C, eps=EPS, momentum=MOM, affine=True)
    bn.running_mean = _dt(rm)
    bn.running_var = _dt(rv)
    _set_param(bn.weight, w)
    _set_param(bn.bias, b)
    bn.cast("float64")
    bn.train()
    x = ortho.from_numpy(np.ascontiguousarray(x_np, dtype=np.float64))
    y = bn(x)

    tm = _torch_build("1d", C, True, "cpu", torch.float64, rm, rv, w, b)
    tm.train()
    yt = tm(torch.tensor(x_np, dtype=torch.float64))

    assert y.dtype == "float64"
    assert _npy(y) == pytest.approx(
        yt.detach().numpy().ravel().astype(np.float64), abs=1e-10)


def test_cpu_float64_broadcast_backward():
    """float64 survives a broadcasting backward.

    unbroadcast() is typed on the gradient's own dtype. Hard-coding float32
    for the accumulator and for the read of the gradient's storage hands a
    float64 graph a float32 gradient, and the next accumulation raises a dtype
    mismatch.
    """
    x = ortho.from_numpy(np.ascontiguousarray(
        np.arange(6, dtype=np.float64).reshape(2, 3) + 1.0))
    x.requires_grad_(True)
    ortho.sum(x + x.mean(dim=0, keepdim=True)).backward()

    xt = torch.tensor(np.arange(6, dtype=np.float64).reshape(2, 3) + 1.0,
                      dtype=torch.float64, requires_grad=True)
    (xt + xt.mean(dim=0, keepdim=True)).sum().backward()

    assert x.grad.dtype == "float64"
    assert _npy(x.grad) == pytest.approx(xt.grad.numpy().ravel(), abs=1e-12)


def test_cpu_float64_backward_parity():
    """Full float64 BatchNorm backward matches PyTorch."""
    shape = (4, 3)
    C = shape[1]
    x_np, up_np = _fixtures(shape)
    rm, rv, w, b = _init_state(C)

    bn = BatchNorm1d(C, eps=EPS, momentum=MOM, affine=True)
    bn.running_mean = _dt(rm)
    bn.running_var = _dt(rv)
    _set_param(bn.weight, w)
    _set_param(bn.bias, b)
    bn.cast("float64")
    bn.train()
    x = ortho.from_numpy(np.ascontiguousarray(x_np, dtype=np.float64))
    x.requires_grad_(True)
    up = ortho.from_numpy(np.ascontiguousarray(up_np, dtype=np.float64))
    ortho.sum(bn(x) * up).backward()

    tm = _torch_build("1d", C, True, "cpu", torch.float64, rm, rv, w, b)
    tm.train()
    xt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    upt = torch.tensor(up_np, dtype=torch.float64)
    (tm(xt) * upt).sum().backward()

    assert x.grad.dtype == "float64"
    # The float32 running stats are cast to float64 on both sides, so the only
    # remaining difference is float32 representation of the initial state.
    assert _npy(x.grad) == pytest.approx(xt.grad.numpy().ravel(), abs=1e-6)
    assert np.abs(_npy(x.grad)).max() > 0.1, "weak oracle: gradient is ~zero"


# degenerate training shapes
@pytest.mark.parametrize("kind,shape", [("1d", (1, 3)), ("2d", (1, 3, 1, 1))])
def test_training_rejects_one_value_per_channel(kind, shape):
    Mod = BatchNorm1d if kind == "1d" else BatchNorm2d
    bn = Mod(shape[1])
    bn.train()
    x = _dt(np.ones(shape))
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        bn(x)


@pytest.mark.parametrize("kind,shape", [("1d", (2, 3)), ("2d", (1, 3, 2, 2))])
def test_training_accepts_two_or_more_values_per_channel(kind, shape):
    Mod = BatchNorm1d if kind == "1d" else BatchNorm2d
    bn = Mod(shape[1])
    bn.train()
    assert bn(_dt(np.ones(shape))) is not None


@pytest.mark.parametrize("kind,shape", [("1d", (1, 3)), ("2d", (1, 3, 1, 1))])
def test_eval_allows_one_value_per_channel(kind, shape):
    Mod = BatchNorm1d if kind == "1d" else BatchNorm2d
    bn = Mod(shape[1])
    bn.eval()
    assert bn(_dt(np.ones(shape))) is not None


def test_degenerate_message_matches_pytorch_wording():
    bn = BatchNorm1d(3)
    bn.train()
    with pytest.raises(ValueError) as exc:
        bn(_dt(np.ones((1, 3))))
    assert "Expected more than 1 value per channel when training" in str(exc.value)


# implementation properties
def test_batchnorm_module_does_not_import_numpy():
    """BatchNorm's numerical path is Tensor-native; NumPy is no longer needed."""
    import orthon.nn.batchnorm as bnmod
    assert not hasattr(bnmod, "np")
    assert not hasattr(bnmod, "numpy")


def test_batchnorm_source_has_no_tolist_or_numpy():
    import inspect
    import orthon.nn.batchnorm as bnmod
    src = inspect.getsource(bnmod)
    assert ".tolist()" not in src
    assert "import numpy" not in src


def test_batchnorm_source_does_not_hardcode_cuda_string():
    """Exact Device objects must be propagated, never collapsed to "cuda"."""
    import inspect
    import orthon.nn.batchnorm as bnmod
    src = inspect.getsource(bnmod)
    assert '.to("cuda")' not in src
    assert ".to('cuda')" not in src


@requires_cuda
def test_constants_land_on_the_inputs_exact_device():
    C = 3
    rm, rv, w, b = _init_state(C)
    bn = _build("1d", C, True, "cuda", "float32", rm, rv, w, b)
    bn.train()
    x = _dt(_fixtures((4, 3))[0], "cuda")
    y = bn(x)
    # Same Device object semantics: index and type both preserved.
    assert y.device == x.device
    assert bn.running_mean.device == x.device
