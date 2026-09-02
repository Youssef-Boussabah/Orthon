"""Backward grad-mode structural safety.

``Tensor::backward`` holds a single ``NoGradGuard`` for the whole traversal,
so a gradient is always a plain, graph-free tensor.

The guard is load-bearing, not defensive. Node backwards legitimately build
gradients out of public differentiable ops applied to the inputs they hold --
``MulNode`` does ``mul(grad, inputs[1])``, ``MatMulNode`` transposes and
re-multiplies its operands -- and those inputs require grad by definition. Run
with grad mode enabled, those calls attach nodes and the returned gradient
carries a live graph::

    x.grad.requires_grad  -> True
    x.grad._has_grad_fn   -> True
    x.grad.is_leaf        -> False

on CPU **and** CUDA. A per-Node guard would leave the invariant depending on
every future Node remembering it; one central guard makes it structural.

The engine is strictly first-order. If ``create_graph`` is ever added, that
central guard is the thing that has to become conditional on it.
"""
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = ortho.cuda_enabled()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


def _vals(shape, ph, sc=1.0):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    return (np.cos(i * ph) * sc + np.sin(i * 0.37) * 0.5 + 0.2 * (i % 4)).reshape(shape)


X34 = _vals((3, 4), 1.7)
Y34 = _vals((3, 4), 0.9)
A34 = _vals((3, 4), 1.1)
B43 = _vals((4, 3), 0.6)
BIAS3 = _vals((3,), 0.4)
UP34 = _vals((3, 4), 2.3, 1.4)
UP33 = _vals((3, 3), 1.5, 1.2)
IMG = _vals((1, 1, 6, 6), 0.8)
KER = _vals((2, 1, 3, 3), 1.2)


def _d(a, dev="cpu"):
    t = ortho.from_numpy(np.ascontiguousarray(np.array(a, dtype=np.float32)))
    return t.to(dev) if dev != "cpu" else t


def _t(a, rg=False):
    t = torch.tensor(np.array(a), dtype=torch.float32)
    if rg:
        t.requires_grad_(True)
    return t


def _np(t):
    return np.array(t.tolist(), dtype=np.float64)


def _plain(g):
    """A first-order gradient: no requires_grad, no grad_fn, still a leaf."""
    return (g.requires_grad is False and g._has_grad_fn is False
            and g.is_leaf is True)


# --------------------------------------------------------------------------
# The families that actually regressed, pinned individually
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_mul_backward_produces_plain_gradients(dev):
    """Without the central guard this reports requires_grad=True and a
    grad_fn on BOTH devices."""
    x = _d(X34, dev)
    y = _d(Y34, dev)
    x.requires_grad_(True)
    y.requires_grad_(True)
    ortho.sum(ortho.mul(x, y) * _d(UP34, dev)).backward()
    assert _plain(x.grad)
    assert _plain(y.grad)


@pytest.mark.parametrize("dev", DEVICES)
def test_matmul_backward_produces_plain_gradients(dev):
    """MatMul gradients are plain on both devices.

    A CUDA-only result here would prove nothing: MatMulNode calls
    .to(same_device) to make the transposed operand contiguous, so a detaching
    to() would launder the graph away by accident rather than by guarantee."""
    a = _d(A34, dev)
    b = _d(B43, dev)
    a.requires_grad_(True)
    b.requires_grad_(True)
    ortho.sum(ortho.matmul(a, b) * _d(UP33, dev)).backward()
    assert _plain(a.grad)
    assert _plain(b.grad)


@pytest.mark.parametrize("dev", DEVICES)
def test_fused_linear_relu_backward_produces_plain_gradients(dev):
    """Same story as matmul: CPU carried a graph, CUDA was accidentally clean."""
    x = _d(A34, dev)
    w = _d(B43, dev)
    b = _d(BIAS3, dev)
    for t in (x, w, b):
        t.requires_grad_(True)
    ortho.sum(ortho.fused_linear_relu(x, w, b) * _d(UP33, dev)).backward()
    assert _plain(x.grad)
    assert _plain(w.grad)
    assert _plain(b.grad)


# --------------------------------------------------------------------------
# Node-family sweep: the central invariant, not one test per Node
# --------------------------------------------------------------------------

FAMILIES = {
    "AddNode": ({"x": X34, "y": Y34},
                lambda t, dv: ortho.sum(ortho.add(t["x"], t["y"]) * _d(UP34, dv))),
    "MulNode": ({"x": X34, "y": Y34},
                lambda t, dv: ortho.sum(ortho.mul(t["x"], t["y"]) * _d(UP34, dv))),
    "MatMulNode": ({"a": A34, "b": B43},
                   lambda t, dv: ortho.sum(ortho.matmul(t["a"], t["b"]) * _d(UP33, dv))),
    "FusedLinearReLUNode": ({"x": A34, "w": B43, "b": BIAS3},
                            lambda t, dv: ortho.sum(ortho.fused_linear_relu(
                                t["x"], t["w"], t["b"]) * _d(UP33, dv))),
    "ReduceNode_sum": ({"x": X34},
                       lambda t, dv: ortho.sum(ortho.sum(t["x"], 1, False)
                                            * _d(_vals((3,), 1.5), dv))),
    "ReduceNode_mean": ({"x": X34},
                        lambda t, dv: ortho.sum(ortho.mean(t["x"], 0, False)
                                             * _d(_vals((4,), 1.5), dv))),
    "MaxNode": ({"x": X34},
                lambda t, dv: ortho.sum(ortho.max(t["x"], 1, False)
                                     * _d(_vals((3,), 1.5), dv))),
    "ReluNode": ({"x": X34}, lambda t, dv: ortho.sum(ortho.relu(t["x"]) * _d(UP34, dv))),
    "ActivationNode": ({"x": X34}, lambda t, dv: ortho.sum(ortho.tanh(t["x"]) * _d(UP34, dv))),
    "ReshapeNode": ({"x": X34},
                    lambda t, dv: ortho.sum(ortho.reshape(t["x"], [12])
                                         * _d(_vals((12,), 1.0), dv))),
    "TransposeNode": ({"x": X34},
                      lambda t, dv: ortho.sum(ortho.contiguous(ortho.transpose(t["x"], 0, 1))
                                           * _d(_vals((4, 3), 1.0), dv))),
    "IndexNode": ({"x": X34},
                  lambda t, dv: ortho.sum(t["x"][1] * _d(_vals((4,), 1.0), dv))),
    "SoftmaxNode": ({"x": X34}, lambda t, dv: ortho.sum(ortho.softmax(t["x"], -1) * _d(UP34, dv))),
    "LogSoftmaxNode": ({"x": X34},
                       lambda t, dv: ortho.sum(ortho.log_softmax(t["x"], -1) * _d(UP34, dv))),
    "CrossEntropyNode": ({"x": _vals((4, 5), 1.9)},
                         lambda t, dv: ortho.cross_entropy(t["x"], [2, 0, 4, 1])),
    "Conv2DNode": ({"x": IMG, "w": KER},
                   lambda t, dv: ortho.sum(ortho.conv2d(t["x"], t["w"],
                                                  _d(np.zeros(2), dv), 1, 1, 0, 0))),
    "MaxPool2DNode": ({"x": IMG},
                      lambda t, dv: ortho.sum(ortho.maxpool2d(t["x"], 2, 2, 2, 2, 0, 0))),
}


def _build(dev, arrs, fn):
    ts = {}
    for k, a in arrs.items():
        t = _d(a, dev)
        t.requires_grad_(True)
        ts[k] = t
    return ts, fn(ts, dev)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_node_family_produces_plain_first_order_gradients(dev, family):
    arrs, fn = FAMILIES[family]
    ts, loss = _build(dev, arrs, fn)
    loss.backward()
    for name, t in sorted(ts.items()):
        assert _plain(t.grad), "%s: %s.grad carries graph metadata" % (family, name)


@pytest.mark.parametrize("dev", DEVICES)
def test_embedding_backward_produces_plain_gradients(dev):
    w = _d(_vals((6, 4), 1.3), dev)
    w.requires_grad_(True)
    idx = ortho.from_numpy(np.array([0, 3, 5, 1], dtype=np.int64))
    if dev != "cpu":
        idx = idx.to(dev)
    ortho.sum(ortho.embedding(idx, w) * _d(_vals((4, 4), 0.7), dev)).backward()
    assert _plain(w.grad)


# --------------------------------------------------------------------------
# Grad mode during backward, and caller-state restoration
# --------------------------------------------------------------------------

def _observe_during_backward(dev, disable_caller_grad=False):
    """Return (observed_inside_hook, grad_mode_after)."""
    observed = []
    x = _d(X34, dev)
    y = _d(Y34, dev)
    x.requires_grad_(True)
    y.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, y) * _d(UP34, dev))
    x._register_post_accumulate_grad_hook(
        lambda: observed.append(ortho.is_grad_enabled()))
    if disable_caller_grad:
        ortho.set_grad_enabled(False)
    try:
        loss.backward()
        after = ortho.is_grad_enabled()
    finally:
        ortho.set_grad_enabled(True)
    return observed, after


@pytest.mark.parametrize("dev", DEVICES)
def test_hooks_run_with_grad_mode_disabled(dev):
    observed, _ = _observe_during_backward(dev)
    assert observed == [False]


@pytest.mark.parametrize("dev", DEVICES)
def test_caller_grad_mode_true_is_restored(dev):
    assert ortho.is_grad_enabled() is True
    _, after = _observe_during_backward(dev)
    assert after is True
    assert ortho.is_grad_enabled() is True


@pytest.mark.parametrize("dev", DEVICES)
def test_caller_grad_mode_false_is_restored(dev):
    """Grad mode governs graph CONSTRUCTION, not whether an existing graph may
    run: backward must still compute, and must leave the caller disabled."""
    observed, after = _observe_during_backward(dev, disable_caller_grad=True)
    assert observed == [False]
    assert after is False
    assert ortho.is_grad_enabled() is True     # restored by the helper's finally


@pytest.mark.parametrize("dev", DEVICES)
def test_backward_computes_correctly_when_caller_grad_mode_is_off(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    ortho.set_grad_enabled(False)
    try:
        loss.backward()
        after = ortho.is_grad_enabled()
    finally:
        ortho.set_grad_enabled(True)
    assert after is False

    xt = _t(X34, rg=True)
    ((xt * _t(Y34)) * _t(UP34)).sum().backward()
    assert _np(x.grad) == pytest.approx(xt.grad.double().numpy().ravel(), abs=1e-6)
    assert _plain(x.grad)


@pytest.mark.parametrize("dev", DEVICES)
def test_forward_still_builds_a_graph_after_backward(dev):
    """The guard must not leak: ordinary graph construction still works."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.mul(x, _d(Y34, dev))).backward()
    assert ortho.mul(x, _d(Y34, dev))._has_grad_fn is True


# --------------------------------------------------------------------------
# Exception unwinding
# --------------------------------------------------------------------------

class _HookBoom(RuntimeError):
    pass


@pytest.mark.parametrize("dev", DEVICES)
def test_exception_from_a_hook_restores_caller_grad_mode_true(dev):
    """RAII, not a paired setter: an exception mid-backward must still restore.

    A post-accumulate hook is the safest deterministic failure point available -
    it needs no invalid memory access and no CUDA misuse.
    """
    seen = []

    def boom():
        seen.append(ortho.is_grad_enabled())
        raise _HookBoom("deliberate")

    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    x._register_post_accumulate_grad_hook(boom)

    assert ortho.is_grad_enabled() is True
    with pytest.raises(Exception):
        loss.backward()
    assert seen == [False]                  # disabled while the hook ran
    assert ortho.is_grad_enabled() is True     # ...and restored on the way out


@pytest.mark.parametrize("dev", DEVICES)
def test_exception_from_a_hook_restores_caller_grad_mode_false(dev):
    def boom():
        raise _HookBoom("deliberate")

    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    x._register_post_accumulate_grad_hook(boom)

    ortho.set_grad_enabled(False)
    try:
        with pytest.raises(Exception):
            loss.backward()
        after = ortho.is_grad_enabled()
    finally:
        ortho.set_grad_enabled(True)
    assert after is False


@pytest.mark.parametrize("dev", DEVICES)
def test_native_backward_failure_restores_caller_grad_mode(dev):
    """Second failure path, raised from inside the engine rather than a hook:
    calling backward twice on a freed graph."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    loss.backward()
    assert ortho.is_grad_enabled() is True
    with pytest.raises(RuntimeError):
        loss.backward()                     # graph already freed
    assert ortho.is_grad_enabled() is True


# --------------------------------------------------------------------------
# retain_graph and repeated backward
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_retain_graph_allows_a_second_backward_and_accumulates(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    loss.backward(True)
    first = _np(x.grad).copy()
    loss.backward(True)
    second = _np(x.grad)

    assert second == pytest.approx(first * 2.0, abs=1e-6)
    assert _plain(x.grad)
    assert ortho.is_grad_enabled() is True


@pytest.mark.parametrize("dev", DEVICES)
def test_graph_is_freed_without_retain_graph(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    loss.backward(True)
    loss.backward()                          # last call frees
    with pytest.raises(RuntimeError):
        loss.backward()
    assert ortho.is_grad_enabled() is True


@pytest.mark.parametrize("dev", DEVICES)
def test_hooks_see_grad_mode_disabled_on_every_traversal(dev):
    observed = []
    x = _d(X34, dev)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev))
    x._register_post_accumulate_grad_hook(
        lambda: observed.append(ortho.is_grad_enabled()))
    loss.backward(True)
    loss.backward(True)
    assert observed == [False, False]
    assert ortho.is_grad_enabled() is True


@pytest.mark.parametrize("dev", DEVICES)
def test_repeated_backward_across_separate_graphs_accumulates(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev)).backward()
    first = _np(x.grad).copy()
    ortho.sum(ortho.mul(x, _d(Y34, dev)) * _d(UP34, dev)).backward()
    second = _np(x.grad)
    assert second == pytest.approx(first * 2.0, abs=1e-6)
    assert _plain(x.grad)


# --------------------------------------------------------------------------
# Numerical parity is unaffected by the guard
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dev", DEVICES)
def test_mul_gradient_values_match_torch(dev):
    x = _d(X34, dev)
    y = _d(Y34, dev)
    x.requires_grad_(True)
    y.requires_grad_(True)
    ortho.sum(ortho.mul(x, y) * _d(UP34, dev)).backward()

    xt, yt = _t(X34, rg=True), _t(Y34, rg=True)
    ((xt * yt) * _t(UP34)).sum().backward()

    assert _np(x.grad) == pytest.approx(xt.grad.double().numpy().ravel(), abs=1e-6)
    assert _np(y.grad) == pytest.approx(yt.grad.double().numpy().ravel(), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_matmul_gradient_values_match_torch(dev):
    a = _d(A34, dev)
    b = _d(B43, dev)
    a.requires_grad_(True)
    b.requires_grad_(True)
    ortho.sum(ortho.matmul(a, b) * _d(UP33, dev)).backward()

    at, bt = _t(A34, rg=True), _t(B43, rg=True)
    ((at @ bt) * _t(UP33)).sum().backward()

    assert _np(a.grad) == pytest.approx(at.grad.double().numpy().ravel(), abs=1e-5)
    assert _np(b.grad) == pytest.approx(bt.grad.double().numpy().ravel(), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradient_dtype_and_exact_device_preserved(dev):
    x = _d(X34, dev)
    y = _d(Y34, dev)
    x.requires_grad_(True)
    y.requires_grad_(True)
    ortho.sum(ortho.mul(x, y) * _d(UP34, dev)).backward()
    assert x.grad.dtype == x.dtype and x.grad.device == x.device
    assert y.grad.dtype == y.dtype and y.grad.device == y.device
    assert x.grad.is_contiguous is True


# --------------------------------------------------------------------------
# Structural placement, and the deliberately-kept local guards
# --------------------------------------------------------------------------

def _tensor_cpp():
    import os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "src", "core", "tensor.cpp")
    with open(p, encoding="utf-8") as f:
        return f.read()


def test_backward_holds_a_single_central_nograd_guard():
    """Source-level: RAII, once, in the engine -- not paired setters and not
    per-Node."""
    src = _tensor_cpp()
    body = src[src.index("void Tensor::backward(bool retain_graph)"):]
    assert body.count("NoGradGuard") == 1
    # The guard must precede the traversal it protects.
    assert body.index("NoGradGuard") < body.index("grad_fn()->backward(")
    # RAII only: no manual pairing anywhere in the engine.
    assert "set_grad_enabled(" not in body


def test_softmax_and_cross_entropy_local_guards_are_retained():
    """The per-Node guards are deliberately KEPT.

    They are redundant during an ordinary Tensor::backward, but they still
    protect these Node backwards if ever invoked directly.
    """
    import os
    base = os.path.join(os.path.dirname(__file__), "..", "..",
                        "include", "orthon", "autograd")
    for name in ("softmax_node.h", "cross_entropy_node.h"):
        with open(os.path.join(base, name), encoding="utf-8") as f:
            assert "NoGradGuard" in f.read(), name


def test_engine_exposes_no_higher_order_autograd_api():
    """The central guard makes the engine explicitly first-order."""
    assert not hasattr(ortho.Tensor, "create_graph")
    assert not hasattr(ortho, "create_graph")
    x = _d(X34)
    x.requires_grad_(True)
    loss = ortho.sum(ortho.mul(x, _d(Y34)))
    with pytest.raises(TypeError):
        loss.backward(create_graph=True)


# --------------------------------------------------------------------------
# DDP post-accumulate hooks
# --------------------------------------------------------------------------

def _ddp_hook_worker(rank, _unused):
    """Runs in a spawned worker with distributed already initialised.

    The spawn helper invokes fn(rank, *args), hence the leading rank argument.

    DDP drives gradient synchronization from post-accumulate hooks, which now
    fire with grad mode disabled. Returns what the hooks observed plus the
    metadata of the synchronized gradients.
    """
    import orthon as ortho
    from orthon.distributed import DistributedDataParallel as DDP_
    from orthon.nn import Linear as Linear_

    model = Linear_(4, 3)
    for p in model.parameters():
        p.fill_(1.0)
    ddp_model = DDP_(model)

    observed = []
    for p in model.parameters():
        p._register_post_accumulate_grad_hook(
            lambda: observed.append(ortho.is_grad_enabled()))

    x = ortho.ones([2, 4])
    x.fill_(0.5)
    loss = ortho.sum(ddp_model(x))
    loss.backward()

    return {
        "hooks_fired": len(observed),
        "observed": observed,
        "grad_mode_after": ortho.is_grad_enabled(),
        "grads_plain": [
            (p.grad.requires_grad, p.grad._has_grad_fn, p.grad.is_leaf)
            for p in model.parameters()
        ],
        "grad_values": [p.grad.tolist() for p in model.parameters()],
    }


def test_ddp_hooks_fire_with_grad_mode_disabled_and_produce_plain_grads():
    """Hooks power DDP's automatic gradient sync, and they run under the
    engine's no-grad context. Uses the same spawn path as the DDP suite
    rather than an in-process shortcut."""
    from orthon.distributed.local_backend import spawn

    results = spawn(_ddp_hook_worker, args=(None,), nprocs=2)
    assert len(results) == 2
    for r in results:
        assert r["hooks_fired"] == 2                  # weight + bias
        assert set(r["observed"]) == {False}          # disabled throughout
        assert r["grad_mode_after"] is True           # caller state restored
        for rg, fn, leaf in r["grads_plain"]:
            assert (rg, fn, leaf) == (False, False, True)
    # Both ranks see identical averaged gradients.
    assert results[0]["grad_values"] == results[1]["grad_values"]
