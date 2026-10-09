"""Unary backward preserves logical order for strided, nonuniform upstream gradients."""
from tests.support.capabilities import orthon_cuda_available
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)

X = np.array([[1.5, -2.0, 3.25], [0.5, -0.75, 4.0]], dtype=np.float64)      # [2, 3]
XP = np.array([[1.5, 2.0, 3.25], [0.5, 0.75, 4.0]], dtype=np.float64)       # strictly positive
UP = np.array([[0.7, -1.3], [2.1, -0.4], [1.9, 0.25]], dtype=np.float64)    # [3, 2] upstream

# (name, orthon fn, torch fn, input) - every ActivationKind plus ReLU.
UNARY_OPS = [
    ("neg", lambda t: ortho.neg(t), torch.neg, X),
    ("relu", lambda t: ortho.relu(t), torch.relu, X),
    ("sigmoid", lambda t: ortho.sigmoid(t), torch.sigmoid, X),
    ("tanh", lambda t: ortho.tanh(t), torch.tanh, X),
    ("leaky_relu", lambda t: ortho.leaky_relu(t, 0.1),
     lambda a: torch.nn.functional.leaky_relu(a, 0.1), X),
    ("gelu", lambda t: ortho.gelu(t), lambda a: torch.nn.functional.gelu(a), X),
    ("exp", lambda t: ortho.exp(t), torch.exp, X),
    ("sqrt", lambda t: ortho.sqrt(t), torch.sqrt, XP),
    ("reciprocal", lambda t: ortho.reciprocal(t), torch.reciprocal, XP),
]
OP_IDS = [n for n, _, _, _ in UNARY_OPS]


def _t(arr, device="cpu", dtype="float32", rg=False):
    npdt = np.float64 if dtype == "float64" else np.float32
    t = ortho.from_numpy(np.ascontiguousarray(arr, dtype=npdt))
    if device != "cpu":
        t = t.to(device)
    if dtype not in ("float32", "float64"):
        t = ortho.cast(t, dtype)
    if rg:
        t.requires_grad_(True)
    return t


def _npy(t):
    return np.array(t.tolist(), dtype=np.float64)


def _torch_grad(tfn, arr, device, tdtype):
    """Reference gradient for sum(op(x).T.contiguous() * upstream)."""
    tdev = "cuda" if device != "cpu" else "cpu"
    xt = torch.tensor(arr, dtype=tdtype, device=tdev, requires_grad=True)
    upt = torch.tensor(UP, dtype=tdtype, device=tdev)
    (tfn(xt).transpose(0, 1).contiguous() * upt).sum().backward()
    return xt.grad.double().cpu().numpy().ravel()


def _run(dfn, arr, device, dtype):
    """Graph whose forwards are all contiguous but whose backward is strided.

    ContiguousNode/TransposeNode hand a strided gradient to the unary backward.
    """
    x = _t(arr, device, dtype, rg=True)
    z = dfn(x).transpose(0, 1).contiguous()
    ortho.sum(z * _t(UP, device, dtype)).backward()
    return _npy(z), _npy(x.grad)


# the exact missed neg case
def test_neg_strided_grad_cpu_float32():
    _, got = _run(lambda t: ortho.neg(t), X, "cpu", "float32")
    assert got == pytest.approx(_torch_grad(torch.neg, X, "cpu", torch.float32),
                                abs=1e-6)


def test_neg_strided_grad_cpu_float32_closed_form():
    """x.grad must equal -upstream.T, in logical (not storage) order."""
    _, got = _run(lambda t: ortho.neg(t), X, "cpu", "float32")
    assert got == pytest.approx((-UP.T).ravel(), abs=1e-6)


def test_neg_strided_grad_cpu_float64():
    _, got = _run(lambda t: ortho.neg(t), X, "cpu", "float64")
    assert got == pytest.approx(_torch_grad(torch.neg, X, "cpu", torch.float64),
                                abs=1e-12)


def test_neg_strided_grad_ordering_is_not_storage_order():
    """Guards the oracle: the correct answer must differ from the buggy one."""
    correct = (-UP.T).ravel()
    storage_order = (-UP).ravel()          # what a linear read produced pre-fix
    assert not np.allclose(correct, storage_order)
    _, got = _run(lambda t: ortho.neg(t), X, "cpu", "float32")
    assert got == pytest.approx(correct, abs=1e-6)
    assert not np.allclose(got, storage_order)


@requires_cuda
@requires_torch_cuda
def test_neg_strided_grad_cuda_float32():
    _, got = _run(lambda t: ortho.neg(t), X, "cuda", "float32")
    assert got == pytest.approx(_torch_grad(torch.neg, X, "cuda", torch.float32),
                                abs=1e-6)


@requires_cuda
def test_neg_strided_grad_cuda_float32_closed_form():
    _, got = _run(lambda t: ortho.neg(t), X, "cuda", "float32")
    assert got == pytest.approx((-UP.T).ravel(), abs=1e-6)


@requires_cuda
@requires_torch_cuda
def test_neg_strided_grad_cuda_float16():
    _, got = _run(lambda t: ortho.neg(t), X, "cuda", "float16")
    assert got == pytest.approx(_torch_grad(torch.neg, X, "cuda", torch.float16),
                                abs=5e-3)


@requires_cuda
@requires_torch_cuda
def test_neg_strided_grad_cuda_bfloat16():
    _, got = _run(lambda t: ortho.neg(t), X, "cuda", "bfloat16")
    assert got == pytest.approx(_torch_grad(torch.neg, X, "cuda", torch.bfloat16),
                                abs=5e-2)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_neg_strided_grad_cuda_preserves_dtype_and_device(dtype):
    """The materialisation must not silently change dtype or device."""
    x = _t(X, "cuda", dtype, rg=True)
    ortho.sum(ortho.neg(x).transpose(0, 1).contiguous() * _t(UP, "cuda", dtype)).backward()
    assert x.grad is not None
    assert x.grad.dtype == dtype
    assert x.grad.device.is_cuda()
    assert x.grad.device == x.device


def test_neg_direct_transpose_form_cpu():
    """Transpose without an intervening contiguous."""
    x = _t(X, "cpu", "float32", rg=True)
    ortho.sum(ortho.neg(x).transpose(0, 1) * _t(UP)).backward()
    assert _npy(x.grad) == pytest.approx((-UP.T).ravel(), abs=1e-6)


# the whole unary family
@pytest.mark.parametrize("name,dfn,tfn,arr", UNARY_OPS, ids=OP_IDS)
def test_unary_family_strided_grad_cpu_float32(name, dfn, tfn, arr):
    _, got = _run(dfn, arr, "cpu", "float32")
    assert got == pytest.approx(_torch_grad(tfn, arr, "cpu", torch.float32),
                                abs=1e-5)


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("name,dfn,tfn,arr", UNARY_OPS, ids=OP_IDS)
def test_unary_family_strided_grad_cuda_float32(name, dfn, tfn, arr):
    _, got = _run(dfn, arr, "cuda", "float32")
    assert got == pytest.approx(_torch_grad(tfn, arr, "cuda", torch.float32),
                                abs=1e-5)


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("name,dfn,tfn,arr",
                         [op for op in UNARY_OPS if op[0] in ("relu", "sigmoid", "exp")],
                         ids=["relu", "sigmoid", "exp"])
def test_unary_family_strided_grad_cuda_float16(name, dfn, tfn, arr):
    """Representative low-precision coverage of the shared materialisation."""
    _, got = _run(dfn, arr, "cuda", "float16")
    assert got == pytest.approx(_torch_grad(tfn, arr, "cuda", torch.float16),
                                abs=5e-2)


@pytest.mark.parametrize("name,dfn,tfn,arr", UNARY_OPS, ids=OP_IDS)
def test_unary_family_forward_unchanged_by_fix(name, dfn, tfn, arr):
    """The correction is backward-only; forward values must be untouched."""
    x = _t(arr, "cpu", "float32", rg=True)
    y = dfn(x)
    tt = torch.tensor(arr, dtype=torch.float32)
    assert _npy(y) == pytest.approx(
        tfn(tt).detach().numpy().ravel().astype(np.float64), abs=1e-6)


# deeper compositions
def test_two_unaries_around_a_transpose():
    x = _t(X, "cpu", "float32", rg=True)
    ortho.sum(ortho.neg(ortho.sigmoid(x)).transpose(0, 1).contiguous() * _t(UP)).backward()
    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    (torch.neg(torch.sigmoid(xt)).transpose(0, 1).contiguous()
     * torch.tensor(UP, dtype=torch.float32)).sum().backward()
    assert _npy(x.grad) == pytest.approx(
        xt.grad.numpy().ravel().astype(np.float64), abs=1e-5)


def test_branched_graph_with_strided_and_contiguous_paths():
    """One branch delivers a strided gradient, the other a contiguous one."""
    x = _t(X, "cpu", "float32", rg=True)
    y = ortho.exp(x)
    strided = ortho.sum(y.transpose(0, 1).contiguous() * _t(UP))
    direct = ortho.sum(y * _t(X))
    ortho.sum(strided + direct).backward()

    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    yt = torch.exp(xt)
    ((yt.transpose(0, 1).contiguous() * torch.tensor(UP, dtype=torch.float32)).sum()
     + (yt * torch.tensor(X, dtype=torch.float32)).sum()).backward()
    assert _npy(x.grad) == pytest.approx(
        xt.grad.numpy().ravel().astype(np.float64), abs=1e-4)


@requires_cuda
def test_cuda_branched_graph_with_strided_path():
    """Two unaries feed x; one branch's backward is strided, the other is not.

    The branches use different unary outputs on purpose. Routing both through a
    single shared tensor would instead exercise gradient accumulation inside
    Tensor::backward, which is covered by test_grad_composition.py.
    """
    x = _t(X, "cuda", "float32", rg=True)
    strided = ortho.sum(ortho.tanh(x).transpose(0, 1).contiguous() * _t(UP, "cuda"))
    direct = ortho.sum(ortho.exp(x) * _t(X, "cuda"))
    ortho.sum(strided + direct).backward()

    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    ((torch.tanh(xt).transpose(0, 1).contiguous()
      * torch.tensor(UP, dtype=torch.float32)).sum()
     + (torch.exp(xt) * torch.tensor(X, dtype=torch.float32)).sum()).backward()
    assert _npy(x.grad.to("cpu")) == pytest.approx(
        xt.grad.numpy().ravel().astype(np.float64), abs=1e-4)


# The forward graph contract is unaffected
def test_no_grad_still_suppresses_graph():
    x = _t(X, rg=True)
    with ortho.no_grad():
        y = ortho.neg(x)
    assert y.requires_grad is False
    assert y._has_grad_fn is False


def test_contiguous_grad_path_still_attaches_normally():
    x = _t(X, rg=True)
    y = ortho.neg(x)
    assert y.requires_grad is True
    assert y._has_grad_fn is True
    ortho.sum(y * _t(X)).backward()
    assert _npy(x.grad) == pytest.approx((-X).ravel(), abs=1e-6)
