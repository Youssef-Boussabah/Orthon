"""Node-family composition and accumulation with strided upstream gradients.

Nonuniform values expose logical-order errors that all-ones gradients hide."""
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

DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _vals(shape, ph, sc=1.0):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    return (np.cos(i * ph) * sc + np.sin(i * 0.37) * 0.5 + 0.2 * (i % 4)).reshape(shape)


X23 = _vals((2, 3), 1.7)
UP32 = _vals((3, 2), 2.1)
OTHER23 = _vals((2, 3), 0.55)
X234 = _vals((2, 3, 4), 1.3)
UP32b = _vals((3, 2), 0.9)


def _d(a, dev="cpu", dtype="float32"):
    t = ortho.from_numpy(np.ascontiguousarray(np.array(a, dtype=np.float32)))
    if dev != "cpu":
        t = t.to(dev)
    if dtype != "float32":
        t = ortho.cast(t, dtype)
    return t


def _t(a, rg=False, dtype=torch.float32, dev="cpu"):
    t = torch.tensor(np.array(a, dtype=np.float32), dtype=dtype, device=dev)
    if rg:
        t.requires_grad_(True)
    return t


def _npy(t):
    return np.array(t.tolist(), dtype=np.float64)


def _tn(t):
    return t.detach().double().cpu().numpy().ravel()


# Accumulation across a strided and a contiguous branch
@pytest.mark.parametrize("dev", DEVICES)
def test_branched_accumulation_strided_plus_contiguous(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    branch_strided = ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32, dev))
    branch_direct = ortho.sum(y * _d(OTHER23, dev))
    ortho.sum(branch_strided + branch_direct).backward()

    xt = _t(X23, rg=True)
    yt = torch.tanh(xt)
    ((yt.transpose(0, 1).contiguous() * _t(UP32)).sum()
     + (yt * _t(OTHER23)).sum()).backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


def test_the_two_branch_gradients_are_not_identical():
    """Guards the oracle: if both branches gave the same gradient, the test
    would pass even if accumulation dropped one of them."""
    x1 = _d(X23)
    x1.requires_grad_(True)
    ortho.sum(ortho.tanh(x1).transpose(0, 1).contiguous() * _d(UP32)).backward()
    x2 = _d(X23)
    x2.requires_grad_(True)
    ortho.sum(ortho.tanh(x2) * _d(OTHER23)).backward()
    assert not np.allclose(_npy(x1.grad), _npy(x2.grad))


# accumulation orderings
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("order", ["strided_first", "contiguous_first"])
def test_accumulation_both_branch_orders(dev, order):
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    a = ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32, dev))
    b = ortho.sum(y * _d(OTHER23, dev))
    ortho.sum(a + b if order == "strided_first" else b + a).backward()

    xt = _t(X23, rg=True)
    yt = torch.tanh(xt)
    ta = (yt.transpose(0, 1).contiguous() * _t(UP32)).sum()
    tb = (yt * _t(OTHER23)).sum()
    (ta + tb if order == "strided_first" else tb + ta).backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_accumulation_strided_plus_strided(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    a = ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32, dev))
    b = ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32 * 0.5, dev))
    ortho.sum(a + b).backward()

    xt = _t(X23, rg=True)
    yt = torch.tanh(xt)
    ((yt.transpose(0, 1).contiguous() * _t(UP32)).sum()
     + (yt.transpose(0, 1).contiguous() * _t(UP32 * 0.5)).sum()).backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_accumulation_contiguous_plus_contiguous_control(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    ortho.sum(ortho.sum(y * _d(OTHER23, dev)) + ortho.sum(y * _d(X23, dev))).backward()
    xt = _t(X23, rg=True)
    yt = torch.tanh(xt)
    ((yt * _t(OTHER23)).sum() + (yt * _t(X23)).sum()).backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_three_way_accumulation(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    total = ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32, dev))
    total = total + ortho.sum(y * _d(OTHER23, dev))
    total = total + ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32 * 0.25, dev))
    ortho.sum(total).backward()

    xt = _t(X23, rg=True)
    yt = torch.tanh(xt)
    ((yt.transpose(0, 1).contiguous() * _t(UP32)).sum()
     + (yt * _t(OTHER23)).sum()
     + (yt.transpose(0, 1).contiguous() * _t(UP32 * 0.25)).sum()).backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


# leaf accumulation across separate backward calls
@pytest.mark.parametrize("dev", DEVICES)
def test_leaf_accumulation_across_backward_calls(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.tanh(x).transpose(0, 1).contiguous() * _d(UP32, dev)).backward()
    first = _npy(x.grad).copy()
    ortho.sum(ortho.tanh(x) * _d(OTHER23, dev)).backward()
    second = _npy(x.grad)

    xt = _t(X23, rg=True)
    (torch.tanh(xt).transpose(0, 1).contiguous() * _t(UP32)).sum().backward()
    e1 = _tn(xt.grad).copy()
    (torch.tanh(xt) * _t(OTHER23)).sum().backward()
    e2 = _tn(xt.grad)

    assert first == pytest.approx(e1, abs=1e-5)
    assert second == pytest.approx(e2, abs=1e-5)
    assert not np.allclose(first, second)      # the second call really added


@pytest.mark.parametrize("dev", DEVICES)
def test_stored_leaf_gradient_is_contiguous(dev):
    """A strided gradient must not be stored as .grad in strided form."""
    x = _d(X23, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.tanh(x).transpose(0, 1).contiguous() * _d(UP32, dev)).backward()
    assert x.grad.is_contiguous is True


# no higher-order graph metadata
@pytest.mark.parametrize("dev", DEVICES)
def test_accumulated_gradient_has_no_graph_metadata(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    ortho.sum(ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32, dev))
           + ortho.sum(y * _d(OTHER23, dev))).backward()
    g = x.grad
    assert g.requires_grad is False
    assert g._has_grad_fn is False
    assert g.is_leaf is True


# Node-family composition
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("broadcast", [False, True], ids=["same_shape", "broadcast"])
def test_add_node_strided_grad(dev, broadcast):
    B = _vals((1, 3), 0.8) if broadcast else X23 * 0.5 + 0.3
    a = _d(X23, dev)
    a.requires_grad_(True)
    b = _d(B, dev)
    b.requires_grad_(True)
    ortho.sum((a + b).transpose(0, 1).contiguous() * _d(UP32, dev)).backward()
    at, bt = _t(X23, rg=True), _t(B, rg=True)
    ((at + bt).transpose(0, 1).contiguous() * _t(UP32)).sum().backward()
    assert _npy(a.grad) == pytest.approx(_tn(at.grad), abs=1e-5)
    assert _npy(b.grad) == pytest.approx(_tn(bt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("broadcast", [False, True], ids=["same_shape", "broadcast"])
def test_mul_node_strided_grad(dev, broadcast):
    B = _vals((1, 3), 0.8) if broadcast else X23 * 0.5 + 0.3
    a = _d(X23, dev)
    a.requires_grad_(True)
    b = _d(B, dev)
    b.requires_grad_(True)
    ortho.sum((a * b).transpose(0, 1).contiguous() * _d(UP32, dev)).backward()
    at, bt = _t(X23, rg=True), _t(B, rg=True)
    ((at * bt).transpose(0, 1).contiguous() * _t(UP32)).sum().backward()
    assert _npy(a.grad) == pytest.approx(_tn(at.grad), abs=1e-5)
    assert _npy(b.grad) == pytest.approx(_tn(bt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_matmul_node_strided_grad(dev):
    A, B, U = _vals((2, 3), 1.2), _vals((3, 4), 0.6), _vals((4, 2), 1.9)
    a = _d(A, dev)
    a.requires_grad_(True)
    b = _d(B, dev)
    b.requires_grad_(True)
    ortho.sum(ortho.matmul(a, b).transpose(0, 1).contiguous() * _d(U, dev)).backward()
    at, bt = _t(A, rg=True), _t(B, rg=True)
    ((at @ bt).transpose(0, 1).contiguous() * _t(U)).sum().backward()
    assert _npy(a.grad) == pytest.approx(_tn(at.grad), abs=1e-4)
    assert _npy(b.grad) == pytest.approx(_tn(bt.grad), abs=1e-4)


@pytest.mark.parametrize("dev", DEVICES)
def test_fused_linear_relu_node_strided_grad(dev):
    A, W, B, U = _vals((2, 3), 1.2), _vals((3, 4), 0.6), _vals((4,), 0.4), _vals((4, 2), 1.9)
    a = _d(A, dev)
    a.requires_grad_(True)
    w = _d(W, dev)
    w.requires_grad_(True)
    bb = _d(B, dev)
    bb.requires_grad_(True)
    ortho.sum(ortho.fused_linear_relu(a, w, bb).transpose(0, 1).contiguous() * _d(U, dev)).backward()
    at, wt, bt = _t(A, rg=True), _t(W, rg=True), _t(B, rg=True)
    (torch.relu(at @ wt + bt).transpose(0, 1).contiguous() * _t(U)).sum().backward()
    assert _npy(a.grad) == pytest.approx(_tn(at.grad), abs=1e-4)
    assert _npy(w.grad) == pytest.approx(_tn(wt.grad), abs=1e-4)
    assert _npy(bb.grad) == pytest.approx(_tn(bt.grad), abs=1e-4)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("kind", ["sum", "mean"])
def test_reduce_node_strided_grad(dev, kind):
    x = _d(X234, dev)
    x.requires_grad_(True)
    red = ortho.sum(x, dim=2) if kind == "sum" else ortho.mean(x, dim=2)
    ortho.sum(red.transpose(0, 1).contiguous() * _d(UP32b, dev)).backward()
    xt = _t(X234, rg=True)
    tred = xt.sum(2) if kind == "sum" else xt.mean(2)
    (tred.transpose(0, 1).contiguous() * _t(UP32b)).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_max_node_strided_grad(dev):
    x = _d(X234, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.max(x, dim=2).transpose(0, 1).contiguous() * _d(UP32b, dev)).backward()
    xt = _t(X234, rg=True)
    (xt.max(2).values.transpose(0, 1).contiguous() * _t(UP32b)).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_reshape_node_strided_grad(dev):
    U = _vals((2, 3), 1.4)
    x = _d(X23, dev)
    x.requires_grad_(True)
    ortho.sum(x.reshape([3, 2]).transpose(0, 1).contiguous() * _d(U, dev)).backward()
    xt = _t(X23, rg=True)
    (xt.reshape(3, 2).transpose(0, 1).contiguous() * _t(U)).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_transpose_node_strided_grad(dev):
    U = X23 * 0.7
    x = _d(X23, dev)
    x.requires_grad_(True)
    ortho.sum(x.transpose(0, 1).transpose(0, 1).contiguous() * _d(U, dev)).backward()
    xt = _t(X23, rg=True)
    (xt.transpose(0, 1).transpose(0, 1).contiguous() * _t(U)).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_contiguous_node_strided_grad(dev):
    U = X23 * 0.7
    x = _d(X23, dev)
    x.requires_grad_(True)
    y = x.transpose(0, 1).contiguous()
    ortho.sum(y.transpose(0, 1).contiguous() * _d(U, dev)).backward()
    xt = _t(X23, rg=True)
    (xt.transpose(0, 1).contiguous().transpose(0, 1).contiguous() * _t(U)).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_log_softmax_node_strided_grad(dev):
    x = _d(X23, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.log_softmax(x, dim=1).transpose(0, 1).contiguous() * _d(UP32, dev)).backward()
    xt = _t(X23, rg=True)
    (torch.log_softmax(xt, dim=1).transpose(0, 1).contiguous() * _t(UP32)).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_embedding_node_strided_grad(dev):
    W, U = _vals((4, 3), 0.9), _vals((3, 2), 1.6)
    w = _d(W, dev)
    w.requires_grad_(True)
    idx = ortho.from_numpy(np.ascontiguousarray(np.array([0, 2], dtype=np.int64)))
    if dev != "cpu":
        idx = idx.to(dev)
    ortho.sum(ortho.embedding(idx, w, None).transpose(0, 1).contiguous() * _d(U, dev)).backward()
    wt = _t(W, rg=True)
    (torch.nn.functional.embedding(torch.tensor([0, 2]), wt).transpose(0, 1).contiguous()
     * _t(U)).sum().backward()
    assert _npy(w.grad) == pytest.approx(_tn(wt.grad), abs=1e-5)


@pytest.mark.parametrize("dev", DEVICES)
def test_conv2d_node_strided_grad(dev):
    I, K, B, U = (_vals((1, 1, 4, 4), 1.1), _vals((2, 1, 2, 2), 0.7),
                  _vals((2,), 0.5), _vals((1, 2, 3, 3), 1.5))
    i = _d(I, dev)
    i.requires_grad_(True)
    k = _d(K, dev)
    k.requires_grad_(True)
    b = _d(B, dev)
    b.requires_grad_(True)
    ortho.sum(ortho.conv2d(i, k, b, 1, 1, 0, 0).transpose(2, 3).contiguous() * _d(U, dev)).backward()
    it, kt, bt = _t(I, rg=True), _t(K, rg=True), _t(B, rg=True)
    (torch.nn.functional.conv2d(it, kt, bt).transpose(2, 3).contiguous() * _t(U)).sum().backward()
    assert _npy(i.grad) == pytest.approx(_tn(it.grad), abs=1e-4)
    assert _npy(k.grad) == pytest.approx(_tn(kt.grad), abs=1e-4)
    assert _npy(b.grad) == pytest.approx(_tn(bt.grad), abs=1e-4)


@pytest.mark.parametrize("dev", DEVICES)
def test_maxpool2d_node_strided_grad(dev):
    I, U = _vals((1, 1, 4, 4), 1.1), _vals((1, 1, 2, 2), 1.5)
    i = _d(I, dev)
    i.requires_grad_(True)
    ortho.sum(ortho.maxpool2d(i, 2, 2, 2, 2, 0, 0).transpose(2, 3).contiguous() * _d(U, dev)).backward()
    it = _t(I, rg=True)
    (torch.nn.functional.max_pool2d(it, 2, 2).transpose(2, 3).contiguous() * _t(U)).sum().backward()
    assert _npy(i.grad) == pytest.approx(_tn(it.grad), abs=1e-5)


def test_conv2d_biasless_strided_grad_still_works():
    """bias=False semantics survive a strided upstream gradient."""
    from orthon.nn import Conv2d
    layer = Conv2d(1, 2, kernel_size=2, bias=False)
    x = _d(_vals((1, 1, 4, 4), 1.1))
    x.requires_grad_(True)
    y = layer(x)
    ortho.sum(y.transpose(2, 3).contiguous() * _d(_vals(tuple(int(v) for v in y.shape), 1.5))).backward()
    assert x.grad is not None
    assert layer.weight.grad is not None
    assert sorted(layer.state_dict().keys()) == ["weight"]


# representative low-precision CUDA coverage
@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("dtype,tdtype,tol",
                         [("float16", torch.float16, 5e-2),
                          ("bfloat16", torch.bfloat16, 2e-1)])
def test_accumulation_low_precision_cuda(dtype, tdtype, tol):
    x = _d(X23, "cuda", dtype)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    ortho.sum(ortho.sum(y.transpose(0, 1).contiguous() * _d(UP32, "cuda", dtype))
           + ortho.sum(y * _d(OTHER23, "cuda", dtype))).backward()

    xt = _t(X23, rg=True, dtype=tdtype, dev="cuda")
    yt = torch.tanh(xt)
    ((yt.transpose(0, 1).contiguous() * _t(UP32, dtype=tdtype, dev="cuda")).sum()
     + (yt * _t(OTHER23, dtype=tdtype, dev="cuda")).sum()).backward()
    assert x.grad.dtype == dtype
    assert x.grad.device.is_cuda()
    assert _npy(x.grad) == pytest.approx(_tn(xt.grad), abs=tol)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mul_node_strided_grad_low_precision_cuda(dtype):
    a = _d(X23, "cuda", dtype)
    a.requires_grad_(True)
    b = _d(X23 * 0.5 + 0.3, "cuda", dtype)
    b.requires_grad_(True)
    ortho.sum((a * b).transpose(0, 1).contiguous() * _d(UP32, "cuda", dtype)).backward()
    assert a.grad is not None and a.grad.dtype == dtype
    assert b.grad is not None and b.grad.dtype == dtype
    assert a.grad.device.is_cuda()
