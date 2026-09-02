"""Public softmax autograd and dimension contract.

``orthon.softmax`` takes a ``dim``, works at any rank, and attaches graph
metadata. Without the metadata the severance is total::

    x.requires_grad_(True)
    y = ortho.softmax(x)      # y.requires_grad == False, no grad_fn

and the loss built from ``y`` does not require grad, so ``loss.backward()``
refuses to run.

An all-ones upstream is useless for testing softmax, because every row of the
Jacobian-vector product then cancels to zero legitimately. Every gradient
assertion here uses a non-uniform upstream and compares VALUES against PyTorch,
plus one PyTorch-independent closed-form oracle so a shared setup mistake
cannot make both sides agree on a wrong answer.
"""
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")
import torch.nn.functional as F

CUDA = ortho.cuda_enabled()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


def _vals(shape, ph, sc=1.0):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    return (np.cos(i * ph) * sc + np.sin(i * 0.37) * 0.5 + 0.2 * (i % 4)).reshape(shape)


X5 = _vals((5,), 0.9)
X34 = _vals((3, 4), 1.7)
X234 = _vals((2, 3, 4), 1.3)
UP5 = _vals((5,), 2.3, 1.4)
UP34 = _vals((3, 4), 2.3, 1.4)
UP234 = _vals((2, 3, 4), 2.3, 1.4)
UP34b = _vals((3, 4), 0.55, 0.8)
UP43 = _vals((4, 3), 0.6, 1.1)
# Values large enough that a naive exp() would overflow to inf.
BIG = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -999.0]])

# (label, array, dim, upstream) over every rank/dim combination in the contract.
DIM_CASES = [
    ("1d_d0", X5, 0, UP5),
    ("1d_dneg1", X5, -1, UP5),
    ("2d_d0", X34, 0, UP34),
    ("2d_d1", X34, 1, UP34),
    ("2d_dneg1", X34, -1, UP34),
    ("2d_dneg2", X34, -2, UP34),
    ("3d_d0", X234, 0, UP234),
    ("3d_d1", X234, 1, UP234),
    ("3d_d2", X234, 2, UP234),
    ("3d_dneg1", X234, -1, UP234),
    ("3d_dneg2", X234, -2, UP234),
]
CASE_IDS = [c[0] for c in DIM_CASES]


def _d(a, dev="cpu"):
    t = ortho.from_numpy(np.ascontiguousarray(np.array(a, dtype=np.float32)))
    return t.to(dev) if dev != "cpu" else t


def _t(a, rg=False, dtype=torch.float32):
    t = torch.tensor(np.array(a), dtype=dtype)
    if rg:
        t.requires_grad_(True)
    return t


def _np(t):
    """Flattened float64 view of an orthon tensor (tolist() flattens)."""
    return np.array(t.tolist(), dtype=np.float64)


def _ref(t):
    return t.grad.double().numpy().ravel()


# --------------------------------------------------------------------------
# Public API contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_default_dim_is_the_last_dimension(dev):
    """Existing callers wrote ortho.softmax(x) and meant row-wise. That must not
    have changed meaning."""
    plain = _np(ortho.softmax(_d(X34, dev)))
    neg1 = _np(ortho.softmax(_d(X34, dev), -1))
    one = _np(ortho.softmax(_d(X34, dev), 1))
    assert np.array_equal(plain, neg1)
    assert np.array_equal(plain, one)


@pytest.mark.parametrize("dev", DEVICES)
def test_dim_accepted_positionally_and_by_keyword(dev):
    a = _np(ortho.softmax(_d(X34, dev), 0))
    b = _np(ortho.softmax(_d(X34, dev), dim=0))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("dev", DEVICES)
def test_output_preserves_shape_dtype_and_exact_device(dev):
    x = _d(X234, dev)
    y = ortho.softmax(x, 1)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert y.device == x.device


# --------------------------------------------------------------------------
# Forward parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,dim,_up", DIM_CASES, ids=CASE_IDS)
def test_forward_matches_torch(dev, label, arr, dim, _up):
    y = ortho.softmax(_d(arr, dev), dim)
    expected = F.softmax(_t(arr), dim=dim).double().numpy().ravel()
    assert _np(y) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,dim,_up", DIM_CASES, ids=CASE_IDS)
def test_every_normalized_slice_sums_to_one(dev, label, arr, dim, _up):
    y = _np(ortho.softmax(_d(arr, dev), dim)).reshape(arr.shape)
    assert y.sum(axis=dim) == pytest.approx(1.0, abs=1e-5)
    assert np.all(y > 0.0)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1])
def test_numerical_stability_with_large_magnitudes(dev, dim):
    """Inputs near +/-1000 overflow a naive exp(); the shift must survive both
    the fast path (dim=1) and the general path (dim=0)."""
    y = _np(ortho.softmax(_d(BIG, dev), dim))
    assert np.all(np.isfinite(y))
    expected = F.softmax(_t(BIG), dim=dim).double().numpy().ravel()
    assert y == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------
# Graph metadata
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_softmax_attaches_grad_fn_when_input_requires_grad(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = ortho.softmax(x, -1)
    assert y.requires_grad is True
    assert y._has_grad_fn is True
    assert y.is_leaf is False


@pytest.mark.parametrize("dev", DEVICES)
def test_softmax_attaches_nothing_when_input_does_not_require_grad(dev):
    y = ortho.softmax(_d(X34, dev), -1)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True


@pytest.mark.parametrize("dev", DEVICES)
def test_no_grad_softmax_is_detached_but_correct(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with ortho.no_grad():
        y = ortho.softmax(x, -1)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    expected = F.softmax(_t(X34), dim=-1).double().numpy().ravel()
    assert _np(y) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_global_grad_disabled_then_restored(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.set_grad_enabled(False)
    try:
        off = ortho.softmax(x, -1)
    finally:
        ortho.set_grad_enabled(True)
    on = ortho.softmax(x, -1)
    assert (off.requires_grad, off._has_grad_fn) == (False, False)
    assert (on.requires_grad, on._has_grad_fn) == (True, True)


# --------------------------------------------------------------------------
# Backward parity
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,dim,up", DIM_CASES, ids=CASE_IDS)
def test_backward_matches_torch(dev, label, arr, dim, up):
    x = _d(arr, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, dim) * _d(up, dev)).backward()

    xt = _t(arr, rg=True)
    (F.softmax(xt, dim=dim) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_backward_gradient_is_not_trivially_zero(dev):
    """Oracle guard. sum(softmax(x)) has a genuinely zero gradient, so a
    broken backward returning zeros would pass a careless test."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, -1) * _d(UP34, dev)).backward()
    assert np.abs(_np(x.grad)).max() > 1e-3


@pytest.mark.parametrize("dev", DEVICES)
def test_uniform_upstream_really_does_cancel_to_zero(dev):
    """The flip side: with a constant upstream the true gradient IS zero, so
    this pins the cancellation rather than hiding it."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, -1)).backward()
    assert np.abs(_np(x.grad)).max() < 1e-6


@pytest.mark.parametrize("dev", DEVICES)
def test_backward_matches_closed_form_without_torch(dev):
    """dx_i = y_i * (g_i - sum_j g_j y_j), computed in numpy from scratch."""
    xs = np.array([0.5, -1.25, 2.0, 0.0])
    gs = np.array([1.5, -2.0, 0.25, 3.0])
    e = np.exp(xs - xs.max())
    ys = e / e.sum()
    expected = ys * (gs - float((gs * ys).sum()))

    x = _d(xs, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, 0) * _d(gs, dev)).backward()

    assert _np(x.grad) == pytest.approx(expected, abs=1e-6)
    # Independently pin the forward too, so the oracle stands alone.
    assert _np(ortho.softmax(_d(xs, dev), 0)) == pytest.approx(ys, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradient_preserves_dtype_and_exact_device(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, -1) * _d(UP34, dev)).backward()
    assert x.grad.dtype == x.dtype
    assert x.grad.device == x.device
    assert x.grad.shape == x.shape


# --------------------------------------------------------------------------
# First-order only: backward must not build another graph
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_softmax_backward_creates_no_higher_order_graph(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, -1) * _d(UP34, dev)).backward()
    g = x.grad
    assert g.requires_grad is False
    assert g._has_grad_fn is False
    assert g.is_leaf is True
    assert g.is_contiguous is True


def test_grad_mode_is_disabled_throughout_backward():
    """Grad mode is off for the whole traversal, hooks included.

    The guard lives in Tensor::backward rather than in each Node, so an
    unguarded public op inside a Node backward cannot attach a node. The local
    guards in SoftmaxNode/CrossEntropyNode are kept as defence in depth (see
    test_softmax_local_guard_is_now_defence_in_depth): redundant during an
    ordinary Tensor::backward, not wrong.
    """
    observed = []
    x = _d(X34)
    x.requires_grad_(True)
    x._register_post_accumulate_grad_hook(lambda: observed.append(ortho.is_grad_enabled()))
    ortho.sum(ortho.softmax(x, -1) * _d(UP34)).backward()
    assert observed == [False]
    # ...and the caller's own grad mode is untouched by that.
    assert ortho.is_grad_enabled() is True


@pytest.mark.parametrize("dev", DEVICES)
def test_softmax_attaches_a_node_in_ordinary_grad_mode(dev):
    """Companion to the test above.

    Outside backward, softmax on a requires-grad input attaches a node. It
    cannot happen INSIDE a backward, because the engine disables grad mode for
    the traversal. Both halves matter: forward builds a graph, backward never
    does.
    """
    logits = _d(X34, dev)
    logits.requires_grad_(True)
    probs = ortho.softmax(logits, -1)
    assert probs._has_grad_fn is True


# --------------------------------------------------------------------------
# Composition under strided and accumulated gradients
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [0, 1])
def test_strided_grad_output_into_softmax_node(dev, dim):
    x = _d(X34, dev)
    x.requires_grad_(True)
    z = ortho.softmax(x, dim).transpose(0, 1).contiguous()
    ortho.sum(z * _d(UP43, dev)).backward()

    xt = _t(X34, rg=True)
    (F.softmax(xt, dim=dim).transpose(0, 1).contiguous() * _t(UP43)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_branched_accumulation_through_softmax(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    s = ortho.softmax(x, -1)
    ortho.sum(ortho.sum(s * _d(UP34, dev)) + ortho.sum(s * _d(UP34b, dev))).backward()

    xt = _t(X34, rg=True)
    st = F.softmax(xt, dim=-1)
    ((st * _t(UP34)).sum() + (st * _t(UP34b)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


def test_the_two_upstreams_are_not_identical():
    assert not np.allclose(UP34, UP34b)


@pytest.mark.parametrize("dev", DEVICES)
def test_accumulates_across_separate_backward_calls(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(x, -1) * _d(UP34, dev)).backward()
    first = _np(x.grad).copy()
    ortho.sum(ortho.softmax(x, 0) * _d(UP34b, dev)).backward()
    second = _np(x.grad)

    xt = _t(X34, rg=True)
    (F.softmax(xt, dim=-1) * _t(UP34)).sum().backward()
    assert first == pytest.approx(_ref(xt), abs=1e-6)
    (F.softmax(xt, dim=0) * _t(UP34b)).sum().backward()
    assert second == pytest.approx(_ref(xt), abs=1e-6)
    assert not np.allclose(first, second)


@pytest.mark.parametrize("dev", DEVICES)
def test_softmax_composes_with_an_upstream_op(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.softmax(ortho.tanh(x), -1) * _d(UP34, dev)).backward()

    xt = _t(X34, rg=True)
    (F.softmax(torch.tanh(xt), dim=-1) * _t(UP34)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# --------------------------------------------------------------------------
# Cross-entropy interaction
# --------------------------------------------------------------------------

LOGITS = _vals((4, 5), 1.9, 2.0)
TARGETS = [2, 0, 4, 1]


@pytest.mark.parametrize("dev", DEVICES)
def test_cross_entropy_gradient_still_matches_torch(dev):
    x = _d(LOGITS, dev)
    x.requires_grad_(True)
    loss = ortho.cross_entropy(x, TARGETS)
    loss.backward()

    xt = _t(LOGITS, rg=True)
    F.cross_entropy(xt, torch.tensor(TARGETS)).backward()

    assert _np(loss)[0] == pytest.approx(
        F.cross_entropy(_t(LOGITS), torch.tensor(TARGETS)).item(), abs=1e-6)
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_cross_entropy_backward_creates_no_secondary_graph(dev):
    """CrossEntropyNode's CPU backward calls softmax() and returns the very
    tensor it computes. Without a NoGradGuard the returned gradient would carry
    a SoftmaxNode now that softmax attaches one."""
    x = _d(LOGITS, dev)
    x.requires_grad_(True)
    ortho.cross_entropy(x, TARGETS).backward()
    g = x.grad
    assert g.requires_grad is False
    assert g._has_grad_fn is False
    assert g.is_leaf is True


@pytest.mark.parametrize("dev", DEVICES)
def test_cross_entropy_gradient_survives_a_second_backward(dev):
    """A stale secondary graph would most likely surface on re-entry."""
    x = _d(LOGITS, dev)
    x.requires_grad_(True)
    ortho.cross_entropy(x, TARGETS).backward()
    ortho.cross_entropy(x, TARGETS).backward()

    xt = _t(LOGITS, rg=True)
    F.cross_entropy(xt, torch.tensor(TARGETS)).backward()
    F.cross_entropy(xt, torch.tensor(TARGETS)).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    assert x.grad._has_grad_fn is False


# --------------------------------------------------------------------------
# Fast path preservation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_2d_last_dim_still_uses_the_dedicated_fast_path(dev):
    """Runtime fingerprint rather than a source-string check.

    The dedicated row-wise implementation and the exp(log_softmax(...))
    general path sum in different orders, so they agree numerically but not
    bit-for-bit. A 2D last-dim call must therefore DIFFER from the general
    path, while a 2D dim=0 call - which has no fast path - must match it
    exactly. If the fast path were ever dropped, the first assertion fails.
    """
    x = _vals((4, 6), 1.7)
    fast = _np(ortho.softmax(_d(x, dev), -1))
    general = _np(ortho.exp(ortho.log_softmax(_d(x, dev), -1)))
    assert not np.array_equal(fast, general)
    assert fast == pytest.approx(general, abs=1e-6)

    dim0 = _np(ortho.softmax(_d(x, dev), 0))
    dim0_general = _np(ortho.exp(ortho.log_softmax(_d(x, dev), 0)))
    assert np.array_equal(dim0, dim0_general)


@pytest.mark.parametrize("dev", DEVICES)
def test_fast_path_output_still_attaches_autograd(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = ortho.softmax(x, -1)          # fast path
    assert y._has_grad_fn is True
    ortho.sum(y * _d(UP34, dev)).backward()
    xt = _t(X34, rg=True)
    (F.softmax(xt, dim=-1) * _t(UP34)).sum().backward()
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# --------------------------------------------------------------------------
# Dimension and dtype contract errors
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("dim", [2, 5, -3, -10])
def test_invalid_dim_raises(dev, dim):
    with pytest.raises(IndexError):
        ortho.softmax(_d(X34, dev), dim)


@pytest.mark.parametrize("dev", DEVICES)
def test_scalar_input_is_rejected(dev):
    scalar = ortho.sum(_d(X34, dev))
    assert scalar.ndim == 0
    with pytest.raises(RuntimeError):
        ortho.softmax(scalar)


def test_empty_input_is_rejected():
    with pytest.raises(RuntimeError):
        ortho.softmax(ortho.empty([0]))


def test_float64_input_is_still_rejected():
    """softmax is float32-only; dtype expansion is out of scope."""
    x = ortho.from_numpy(np.ascontiguousarray(X34.astype(np.float64)))
    with pytest.raises(RuntimeError):
        ortho.softmax(x)


@pytest.mark.parametrize("dev", DEVICES)
def test_non_contiguous_input_is_still_rejected(dev):
    with pytest.raises(RuntimeError):
        ortho.softmax(_d(X34, dev).transpose(0, 1))


# --------------------------------------------------------------------------
# log_softmax must be untouched
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,dim,up", DIM_CASES, ids=CASE_IDS)
def test_log_softmax_is_unchanged(dev, label, arr, dim, up):
    x = _d(arr, dev)
    x.requires_grad_(True)
    y = ortho.log_softmax(x, dim)
    expected = F.log_softmax(_t(arr), dim=dim).double().numpy().ravel()
    assert _np(y) == pytest.approx(expected, abs=1e-5)

    ortho.sum(y * _d(up, dev)).backward()
    xt = _t(arr, rg=True)
    (F.log_softmax(xt, dim=dim) * _t(up)).sum().backward()
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_softmax_and_log_softmax_agree(dev):
    s = _np(ortho.softmax(_d(X234, dev), 1))
    ls = _np(ortho.log_softmax(_d(X234, dev), 1))
    assert np.log(s) == pytest.approx(ls, abs=1e-5)
