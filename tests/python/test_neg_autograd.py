"""Autograd correctness for unary negation.

neg() attaches an autograd node, so a graph containing a negation stays
connected to earlier operations. Producing correct values without a node is
not a wrong-looking number: backward raises "backward(): tensor does not
require grad", or, in a branched graph, simply drops the negated term's
contribution.

These tests therefore assert gradient VALUES against PyTorch and against closed
-form derivatives, not merely that a gradient exists.
"""
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = ortho.cuda_enabled()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)

X = np.array([[1.5, -2.0, 3.25], [0.5, -0.75, 4.0]], dtype=np.float64)
UP = np.array([[0.7, -1.3, 2.1], [-0.4, 1.9, 0.25]], dtype=np.float64)

ENTRY_POINTS = [
    ("ortho.neg", lambda t: ortho.neg(t)),
    ("method", lambda t: t.neg()),
    ("operator", lambda t: -t),
]


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


# ---- every entry point is graph-connected ----

@pytest.mark.parametrize("name,fn", ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_entry_point_attaches_graph(name, fn):
    x = _t(X, rg=True)
    y = fn(x)
    assert y.requires_grad is True
    assert y._has_grad_fn is True


@pytest.mark.parametrize("name,fn", ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_entry_point_forward_values_unchanged(name, fn):
    """Attaching a node must not disturb the forward values."""
    x = _t(X, rg=True)
    assert _npy(fn(x)) == pytest.approx((-X).ravel())


@pytest.mark.parametrize("name,fn", ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_entry_point_gradient_is_minus_upstream(name, fn):
    x = _t(X, rg=True)
    ortho.sum(fn(x) * _t(UP)).backward()
    assert _npy(x.grad) == pytest.approx((-UP).ravel(), abs=1e-5)


def test_entry_points_agree_with_each_other():
    grads = []
    for _, fn in ENTRY_POINTS:
        x = _t(X, rg=True)
        ortho.sum(fn(x) * _t(UP)).backward()
        grads.append(_npy(x.grad))
    assert grads[1] == pytest.approx(grads[0])
    assert grads[2] == pytest.approx(grads[0])


def test_no_graph_when_input_does_not_require_grad():
    y = ortho.neg(_t(X, rg=False))
    assert y.requires_grad is False
    assert y._has_grad_fn is False


# ---- grad mode ----

def test_no_grad_suppresses_graph():
    x = _t(X, rg=True)
    with ortho.no_grad():
        y = ortho.neg(x)
    assert y.requires_grad is False
    assert y._has_grad_fn is False


def test_no_grad_forward_values_still_correct():
    x = _t(X, rg=True)
    with ortho.no_grad():
        y = -x
    assert _npy(y) == pytest.approx((-X).ravel())


def test_set_grad_enabled_false_then_true():
    x = _t(X, rg=True)
    ortho.set_grad_enabled(False)
    try:
        assert ortho.neg(x)._has_grad_fn is False
    finally:
        ortho.set_grad_enabled(True)
    assert ortho.neg(x)._has_grad_fn is True


def test_grad_mode_restored_after_no_grad_block():
    x = _t(X, rg=True)
    with ortho.no_grad():
        ortho.neg(x)
    assert ortho.is_grad_enabled() is True
    assert ortho.neg(x)._has_grad_fn is True


# ---- CPU parity ----

def _parity(device="cpu", dtype="float32", tdtype=torch.float32, tol=1e-5,
            compose=False):
    x = _t(X, device, dtype, rg=True)
    up = _t(UP, device, dtype)
    y = ortho.neg(x * x) if compose else ortho.neg(x)
    ortho.sum(y * up).backward()

    tdev = "cuda" if device != "cpu" else "cpu"
    xt = torch.tensor(X, dtype=tdtype, device=tdev, requires_grad=True)
    upt = torch.tensor(UP, dtype=tdtype, device=tdev)
    yt = torch.neg(xt * xt) if compose else torch.neg(xt)
    (yt * upt).sum().backward()

    def tn(t):
        return t.detach().double().cpu().numpy().ravel()

    assert _npy(y) == pytest.approx(tn(yt), abs=tol), "forward"
    assert _npy(x.grad) == pytest.approx(tn(xt.grad), abs=tol), "gradient"


def test_cpu_float32_direct_parity():
    _parity()


def test_cpu_float32_composed_parity():
    """Proves the graph reconnects through an operation applied before neg."""
    _parity(compose=True)


def test_cpu_float64_direct_parity():
    _parity(dtype="float64", tdtype=torch.float64, tol=1e-12)


def test_cpu_float64_composed_parity():
    _parity(dtype="float64", tdtype=torch.float64, tol=1e-12, compose=True)


def test_cpu_composed_matches_closed_form():
    """d/dx sum(-(x*x) * up) = -2*x*up."""
    x = _t(X, rg=True)
    ortho.sum(ortho.neg(x * x) * _t(UP)).backward()
    assert _npy(x.grad) == pytest.approx((-2.0 * X * UP).ravel(), abs=1e-4)


def test_cpu_non_contiguous_input():
    """neg supports strided CPU input; the gradient must still be correct."""
    x = _t(X, rg=True)
    xt = x.transpose(0, 1)
    assert xt.is_contiguous is False
    y = ortho.neg(xt)
    ortho.sum(y).backward()
    assert _npy(y) == pytest.approx((-X).T.ravel())
    assert _npy(x.grad) == pytest.approx(np.full(X.size, -1.0), abs=1e-6)


def test_neg_after_another_op_parity():
    """neg placed after a differentiable op, rather than before one."""
    x = _t(X, rg=True)
    ortho.sum(ortho.neg(ortho.exp(x)) * _t(UP)).backward()
    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    (torch.neg(torch.exp(xt)) * torch.tensor(UP, dtype=torch.float32)).sum().backward()
    assert _npy(x.grad) == pytest.approx(
        xt.grad.numpy().ravel().astype(np.float64), abs=1e-4)


# ---- branched accumulation and double negation ----

def test_branched_accumulation():
    """a = -x ; b = x*x ; loss = sum(a + b) -> grad = -1 + 2x."""
    x = _t(X, rg=True)
    ortho.sum(ortho.neg(x) + (x * x)).backward()
    assert _npy(x.grad) == pytest.approx((-1.0 + 2.0 * X).ravel(), abs=1e-5)


def test_branched_accumulation_matches_torch():
    x = _t(X, rg=True)
    ortho.sum(ortho.neg(x) + (x * x)).backward()
    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    (torch.neg(xt) + xt * xt).sum().backward()
    assert _npy(x.grad) == pytest.approx(
        xt.grad.numpy().ravel().astype(np.float64), abs=1e-5)


def test_double_negation_gradient_is_plus_one():
    x = _t(X, rg=True)
    ortho.sum(ortho.neg(ortho.neg(x))).backward()
    assert _npy(x.grad) == pytest.approx(np.full(X.size, 1.0), abs=1e-6)


def test_double_negation_forward_is_identity():
    x = _t(X, rg=True)
    assert _npy(-(-x)) == pytest.approx(X.ravel())


def test_neg_used_twice_in_one_graph():
    """Same negation feeding two branches must accumulate both contributions."""
    x = _t(X, rg=True)
    n = ortho.neg(x)
    ortho.sum(n + n).backward()
    assert _npy(x.grad) == pytest.approx(np.full(X.size, -2.0), abs=1e-6)


def test_neg_inside_a_deeper_chain():
    x = _t(X, rg=True)
    ortho.sum(ortho.relu(ortho.neg(x)) * _t(UP)).backward()
    xt = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    (torch.relu(torch.neg(xt)) * torch.tensor(UP, dtype=torch.float32)).sum().backward()
    assert _npy(x.grad) == pytest.approx(
        xt.grad.numpy().ravel().astype(np.float64), abs=1e-5)


# ---- CUDA parity ----

@requires_cuda
@requires_torch_cuda
def test_cuda_float32_direct_parity():
    _parity(device="cuda")


@requires_cuda
@requires_torch_cuda
def test_cuda_float32_composed_parity():
    _parity(device="cuda", compose=True)


@requires_cuda
@requires_torch_cuda
def test_cuda_float16_parity():
    _parity(device="cuda", dtype="float16", tdtype=torch.float16, tol=5e-3)


@requires_cuda
@requires_torch_cuda
def test_cuda_bfloat16_parity():
    _parity(device="cuda", dtype="bfloat16", tdtype=torch.bfloat16, tol=5e-2)


@requires_cuda
@requires_torch_cuda
def test_cuda_float16_composed_parity():
    _parity(device="cuda", dtype="float16", tdtype=torch.float16, tol=5e-2,
            compose=True)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_cuda_output_dtype_and_device(dtype):
    x = _t(X, "cuda", dtype, rg=True)
    y = ortho.neg(x)
    ortho.sum(y).backward()
    assert y.dtype == dtype
    assert y.device.is_cuda()
    assert y.device == x.device
    assert x.grad is not None
    assert x.grad.dtype == dtype
    assert x.grad.device.is_cuda()


@requires_cuda
@pytest.mark.parametrize("name,fn", ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS])
def test_cuda_entry_points_graph_connected(name, fn):
    x = _t(X, "cuda", rg=True)
    y = fn(x)
    assert y.requires_grad is True
    assert y._has_grad_fn is True


@requires_cuda
def test_cuda_branched_accumulation():
    x = _t(X, "cuda", rg=True)
    ortho.sum(ortho.neg(x) + (x * x)).backward()
    assert _npy(x.grad.to("cpu")) == pytest.approx(
        (-1.0 + 2.0 * X).ravel(), abs=1e-4)


@requires_cuda
def test_cuda_no_grad_suppresses_graph():
    x = _t(X, "cuda", rg=True)
    with ortho.no_grad():
        y = ortho.neg(x)
    assert y._has_grad_fn is False


@requires_cuda
def test_cuda_neg_accepts_a_transposed_view():
    """CUDA neg accepts a view and returns LOGICAL values, matching CPU.

    Rejecting one would be a CUDA-only restriction with no CPU counterpart -
    CPU neg walks strides. The kernel operand is normalised instead, so the
    result is in logical rather than storage order.
    """
    x = _t(X, "cuda", rg=True)
    view = x.transpose(0, 1)
    y = ortho.neg(view)
    assert y.shape == view.shape
    assert _npy(y.to("cpu")) == pytest.approx((-X.T).ravel(), abs=1e-6)
    # The public node still hangs off the view, not off any internal copy.
    assert y._has_grad_fn and not y.is_leaf
