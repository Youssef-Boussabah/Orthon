"""Slice views scatter nonuniform gradients into each parent prefix block.

Untouched regions remain zero. Python slice-key syntax is unsupported."""
from tests.support.capabilities import orthon_cuda_available
import os

import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = orthon_cuda_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


def _vals(shape, ph, sc=1.0):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    return (np.cos(i * ph) * sc + np.sin(i * 0.37) * 0.5 + 0.2 * (i % 4)).reshape(shape)


X5 = _vals((5,), 0.9)
X34 = _vals((3, 4), 1.7)
X43 = _vals((4, 3), 1.1)
X45 = _vals((4, 5), 1.45)
X253 = _vals((2, 5, 3), 1.3)
X236 = _vals((2, 3, 6), 0.85)
X345 = _vals((3, 4, 5), 0.55)
X35 = _vals((3, 5), 2.05)
X263 = _vals((2, 6, 3), 1.05)

UP233 = _vals((2, 3, 3), 2.1, 1.4)
UP233b = _vals((2, 3, 3), 0.8, 1.2)
UP232 = _vals((2, 3, 2), 1.9, 0.7)


def _d(a, dev="cpu", dtype="float32"):
    t = ortho.from_numpy(np.ascontiguousarray(np.array(a, dtype=np.float32)))
    if dev != "cpu":
        t = t.to(dev)
    if dtype != "float32":
        t = ortho.cast(t, dtype)
    return t


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


def _mat(t):
    """Materialise a strided view.

    cuda_mul rejects strided operands - a pre-existing forward limitation that
    fires with no slicing involved - so every slice that feeds a multiply is
    passed through here. It is a no-op for an already-contiguous view, and
    ContiguousNode passes the gradient straight through, so the graph shape
    under test is unchanged.
    """
    return t if t.is_contiguous else t.contiguous()


def _norm(shape, dim, start, end):
    """Orthon slice normalisation, reproduced so the PyTorch reference can be
    written with explicit positive bounds."""
    nd = len(shape)
    if dim < 0:
        dim += nd
    if start < 0:
        start += shape[dim]
    if end < 0:
        end += shape[dim]
    return dim, start, end


def _narrow(t, shape, dim, start, end):
    """PyTorch equivalent of orthon x.slice(dim, start, end)."""
    d, s, e = _norm(shape, dim, start, end)
    return t.narrow(d, s, e - s)


# forward view contract, unchanged by autograd attachment
VIEW_CASES = [
    # (array, (dim, start, end), shape, strides, storage_offset)
    (X5, (0, 1, 4), (3,), (1,), 1),
    (X34, (0, 1, 3), (2, 4), (4, 1), 4),
    (X34, (1, 1, 3), (3, 2), (4, 1), 1),
    (X253, (1, 1, 4), (2, 3, 3), (15, 3, 1), 3),
    (X253, (2, 1, 3), (2, 5, 2), (15, 3, 1), 1),
    (X35, (-1, -4, -1), (3, 3), (5, 1), 1),
]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("arr,args,shape,strides,offset", VIEW_CASES)
def test_slice_view_metadata(dev, arr, args, shape, strides, offset):
    x = _d(arr, dev)
    s = x.slice(*args)
    assert s.shape == shape
    assert s.strides == strides
    assert s.storage_offset == offset
    assert s.ndim == x.ndim


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("arr,args,shape,strides,offset", VIEW_CASES)
def test_tracked_slice_has_the_same_view_metadata_as_an_untracked_one(
        dev, arr, args, shape, strides, offset):
    """Attaching a node must not change shape/strides/offset of the view."""
    plain = _d(arr, dev).slice(*args)
    tracked = _d(arr, dev)
    tracked.requires_grad_(True)
    tracked = tracked.slice(*args)
    assert (tracked.shape, tracked.strides, tracked.storage_offset) == (
        plain.shape, plain.strides, plain.storage_offset)
    assert (tracked.shape, tracked.strides, tracked.storage_offset) == (
        shape, strides, offset)


def test_cpu_slice_view_shares_storage_with_its_parent():
    x = _d(X34)
    s = x.slice(1, 1, 3)                # columns 1..2 - a strided view
    assert s.is_contiguous is False
    s[0, 0] = 99.0                      # __setitem__ on the view
    assert x.get_item([0, 1]) == 99.0   # ...lands in the parent's storage


def test_cpu_slice_view_shares_storage_with_a_grad_tracking_parent():
    x = _d(X34)
    x.requires_grad_(True)
    s = x.slice(0, 1, 3)
    s[1, 2] = 42.0
    assert x.get_item([2, 2]) == 42.0


@requires_cuda
def test_cuda_slice_view_shares_storage_with_its_parent():
    """mul_ is the CUDA in-place mutation the current API supports, and it
    requires a contiguous operand, so use a leading-dimension slice (which is
    contiguous) and check the write lands in the parent's storage."""
    x = _d(X34, "cuda")
    s = x.slice(0, 1, 3)
    assert s.is_contiguous is True
    s.mul_(0.0)
    g = _np(x).reshape(3, 4)
    assert np.all(g[1:3] == 0.0)
    assert g[0] == pytest.approx(X34[0], abs=1e-6)


# forward parity with PyTorch
FORWARD_CASES = [
    ("1d", X5, (0, 1, 4)),
    ("2d-dim0", X34, (0, 1, 3)),
    ("2d-dim1", X45, (1, 1, 4)),
    ("3d-dim0", X345, (0, 1, 3)),
    ("3d-dim1", X345, (1, 1, 4)),
    ("3d-dim2", X345, (2, 1, 4)),
    ("neg-dim", X345, (-1, 1, 4)),
    ("neg-bounds", X45, (0, -4, -1)),
    ("neg-dim-and-bounds", X45, (-1, -4, -1)),
    ("full-range", X34, (1, 0, 4)),
    ("single-element", X34, (0, 1, 2)),
]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,args", FORWARD_CASES)
def test_slice_forward_matches_torch(dev, label, arr, args):
    got = _d(arr, dev).slice(*args)
    want = _narrow(_t(arr), arr.shape, *args)
    assert got.shape == tuple(want.shape)
    assert _np(got) == pytest.approx(want.double().numpy().ravel(), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_negative_bounds_use_orthon_semantics_not_python_clamping(dev):
    """x.slice(-1, -4, -1) on shape [3, 5] resolves to dim=1, [1, 4) - the same
    region as PyTorch's narrow(1, 1, 3). Pin it explicitly so a future move to
    Python's clamping slice-object rules is loud."""
    assert _norm((3, 5), -1, -4, -1) == (1, 1, 4)
    got = _d(X35, dev).slice(-1, -4, -1)
    want = _t(X35).narrow(1, 1, 3)
    assert got.shape == (3, 3)
    assert _np(got) == pytest.approx(want.double().numpy().ravel(), abs=1e-6)


# graph metadata
@pytest.mark.parametrize("dev", DEVICES)
def test_slice_attaches_grad_fn_when_parent_requires_grad(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = x.slice(1, 1, 3)
    assert y.requires_grad is True
    assert y._has_grad_fn is True
    assert y.is_leaf is False


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_attaches_nothing_when_parent_does_not_require_grad(dev):
    y = _d(X34, dev).slice(1, 1, 3)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True


@pytest.mark.parametrize("dev", DEVICES)
def test_no_grad_slice_returns_a_detached_view(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with ortho.no_grad():
        y = x.slice(1, 1, 3)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True
    # ...but still exactly the same view.
    assert (y.shape, y.strides, y.storage_offset) == ((3, 2), (4, 1), 1)
    assert _np(y) == pytest.approx(X34[:, 1:3].ravel(), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_global_grad_disabled_slice_returns_a_detached_view(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.set_grad_enabled(False)
    try:
        y = x.slice(1, 1, 3)
    finally:
        ortho.set_grad_enabled(True)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True
    assert y.storage_offset == 1


@pytest.mark.parametrize("dev", DEVICES)
def test_tracking_resumes_after_leaving_the_guard(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with ortho.no_grad():
        x.slice(1, 1, 3)
    y = x.slice(1, 1, 3)
    assert y.requires_grad is True and y._has_grad_fn is True
    ortho.sum(_mat(y) * _d(X34[:, 1:3], dev)).backward()
    assert x.grad is not None


@pytest.mark.parametrize("dev", DEVICES)
def test_no_grad_slice_still_shares_storage(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with ortho.no_grad():
        s = x.slice(0, 1, 3)
    assert s.storage_offset == 4
    assert _np(s) == pytest.approx(X34[1:3].ravel(), abs=1e-6)


# basic backward parity
BACKWARD_CASES = [
    ("1d", X5, (0, 1, 4)),
    ("2d-dim0", X34, (0, 1, 3)),
    ("2d-dim1", X45, (1, 1, 4)),
    ("3d-dim0", X345, (0, 1, 3)),
    ("3d-dim1", X345, (1, 1, 4)),
    ("3d-dim2", X345, (2, 1, 4)),
    ("neg-dim", X345, (-1, 1, 4)),
    ("neg-bounds", X45, (0, -4, -1)),
    ("neg-dim-and-bounds", X45, (-1, -4, -1)),
    ("full-range", X34, (1, 0, 4)),
    ("single-element", X34, (0, 1, 2)),
]


def _upstream(shape, ph=1.23):
    return _vals(shape, ph, 1.3)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,args", BACKWARD_CASES)
def test_slice_backward_matches_torch(dev, label, arr, args):
    x = _d(arr, dev)
    x.requires_grad_(True)
    y = x.slice(*args)
    up = _upstream(y.shape)
    ortho.sum(_mat(y) * _d(up, dev)).backward()

    xt = _t(arr, rg=True)
    (_narrow(xt, arr.shape, *args) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("label,arr,args", BACKWARD_CASES)
def test_only_the_sliced_region_receives_gradient(dev, label, arr, args):
    """Value parity alone cannot distinguish "wrote the right numbers in the
    right place" from a coincidence, so pin the zero regions separately."""
    x = _d(arr, dev)
    x.requires_grad_(True)
    y = x.slice(*args)
    up = _upstream(y.shape)
    ortho.sum(_mat(y) * _d(up, dev)).backward()

    g = _np(x.grad).reshape(arr.shape)
    dim, start, end = _norm(arr.shape, *args)
    sel = [slice(None)] * len(arr.shape)
    sel[dim] = slice(start, end)
    assert g[tuple(sel)] == pytest.approx(up, abs=1e-6)

    before = list(sel)
    before[dim] = slice(0, start)
    assert np.all(g[tuple(before)] == 0.0), "elements before start must be zero"
    after = list(sel)
    after[dim] = slice(end, arr.shape[dim])
    assert np.all(g[tuple(after)] == 0.0), "elements after end must be zero"


@pytest.mark.parametrize("dev", DEVICES)
def test_non_scalar_backward_seed_through_slice(dev):
    """s.backward() seeds ones() straight into SliceNode."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    x.slice(1, 1, 3).backward()

    xt = _t(X34, rg=True)
    xt.narrow(1, 1, 2).backward(torch.ones(3, 2))

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


def test_cpu_float64_slice_backward():
    x = ortho.from_numpy(np.ascontiguousarray(X253.astype(np.float64)))
    x.requires_grad_(True)
    up = ortho.from_numpy(np.ascontiguousarray(UP233.astype(np.float64)))
    ortho.sum(x.slice(1, 1, 4).contiguous() * up).backward()

    xt = _t(X253, rg=True, dtype=torch.float64)
    (xt.narrow(1, 1, 3) * _t(UP233, dtype=torch.float64)).sum().backward()

    assert x.grad.dtype == "float64"
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-12)


@requires_cuda
@pytest.mark.parametrize("dtype,tol,tdt", [("float32", 1e-6, torch.float32),
                                           ("float16", 1e-3, torch.float16),
                                           ("bfloat16", 1e-2, torch.bfloat16)])
def test_cuda_slice_backward_dtypes(dtype, tol, tdt):
    """For fp16/bf16 the float32 PyTorch gradient is the oracle and `tol` covers
    the dtype's own rounding (measured: 3.9e-4 fp16, 5.6e-3 bf16). The scatter
    itself is separately pinned as byte-exact below, so the tolerance is not
    hiding a placement error."""
    x = _d(X253, "cuda", dtype)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(UP233, "cuda", dtype)).backward()

    xt = _t(X253, rg=True)
    (xt.narrow(1, 1, 3) * _t(UP233)).sum().backward()

    assert x.grad.dtype == dtype
    assert x.grad.device == x.device
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=tol)

    # Byte-exactness: the region the scatter wrote must equal the upstream
    # rounded to the working dtype, with no tolerance at all.
    g = _np(x.grad).reshape(2, 5, 3)
    rounded = torch.tensor(UP233, dtype=tdt).double().numpy()
    assert g[:, 1:4, :] == pytest.approx(rounded, abs=0)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_half_slice_scatter_zeroes_are_exact(dtype):
    """The scatter is a byte copy into a zeroed buffer, so the untouched region
    must hold exact zeros in the narrow dtype, not merely small values."""
    x = _d(X253, "cuda", dtype)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(UP233, "cuda", dtype)).backward()
    g = _np(x.grad).reshape(2, 5, 3)
    assert np.all(g[:, 0, :] == 0.0)
    assert np.all(g[:, 4, :] == 0.0)
    assert np.count_nonzero(g[:, 1:4, :]) == UP233.size


# Multi-run scatter: what a general slice must prove beyond an integer index
@pytest.mark.parametrize("dev", DEVICES)
def test_inner_dimension_slice_scatters_into_every_prefix_block(dev):
    """parent [2, 5, 3], slice dim=1 [1:4].

    prefix = 2, suffix = 3, slice_len = 3: two runs of 9 elements landing 15
    elements apart. A single flat memcpy of 18 elements would put prefix block 1
    at offset 9 instead of offset 18, so it fails this test.
    """
    x = _d(X253, dev)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(UP233, dev)).backward()

    xt = _t(X253, rg=True)
    (xt.narrow(1, 1, 3) * _t(UP233)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 5, 3)
    for p in range(2):
        assert g[p, 1:4, :] == pytest.approx(UP233[p], abs=1e-6), \
            "prefix block %d landed in the wrong region" % p
        assert np.all(g[p, 0, :] == 0.0)
        assert np.all(g[p, 4, :] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_last_dimension_slice_scatters_into_every_prefix_block(dev):
    """parent [2, 3, 6], slice dim=2 [2:5]: prefix = 6, suffix = 1."""
    x = _d(X236, dev)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(2, 2, 5)) * _d(UP233, dev)).backward()

    xt = _t(X236, rg=True)
    (xt.narrow(2, 2, 3) * _t(UP233)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 3, 6)
    assert g[:, :, 2:5] == pytest.approx(UP233, abs=1e-6)
    assert np.all(g[:, :, 0:2] == 0.0)
    assert np.all(g[:, :, 5:6] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_prefix_blocks_are_not_interchangeable(dev):
    """Distinct per-block upstreams: a scatter that swapped or reused a prefix
    block would still match the element count and the zero pattern."""
    up = np.zeros((2, 3, 3))
    up[0] = 1.0
    up[1] = 7.0
    x = _d(X253, dev)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(up, dev)).backward()
    g = _np(x.grad).reshape(2, 5, 3)
    assert np.all(g[0, 1:4, :] == 1.0)
    assert np.all(g[1, 1:4, :] == 7.0)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("start,end", [(0, 2), (1, 4), (3, 5), (0, 5), (2, 3)])
def test_every_inner_slice_window_lands_where_it_should(dev, start, end):
    x = _d(X253, dev)
    x.requires_grad_(True)
    y = x.slice(1, start, end)
    up = _upstream(y.shape, 0.61)
    ortho.sum(_mat(y) * _d(up, dev)).backward()

    xt = _t(X253, rg=True)
    (xt.narrow(1, start, end - start) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 5, 3)
    assert g[:, start:end, :] == pytest.approx(up, abs=1e-6)
    assert np.all(g[:, :start, :] == 0.0)
    assert np.all(g[:, end:, :] == 0.0)


# empty slice
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("args,shape", [((1, 2, 2), (2, 0, 3)),
                                        ((0, 0, 0), (0, 5, 3)),
                                        ((2, 3, 3), (2, 5, 0))])
def test_empty_slice_forward_is_preserved(dev, args, shape):
    s = _d(X253, dev).slice(*args)
    assert s.shape == shape
    assert int(s.numel) == 0
    assert s.ndim == 3


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("args", [(1, 2, 2), (0, 0, 0), (2, 3, 3), (1, 5, 5)])
def test_empty_slice_backward_is_an_all_zero_parent_gradient(dev, args):
    """A zero-length slice is a legal forward, so its gradient is the correctly
    shaped all-zero parent gradient - not a crash on a null 0-byte copy."""
    x = _d(X253, dev)
    x.requires_grad_(True)
    x.slice(*args).backward()

    assert x.grad.shape == (2, 5, 3)
    assert x.grad.dtype == x.dtype
    assert x.grad.device == x.device
    assert np.all(_np(x.grad) == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_empty_slice_of_an_already_empty_parent(dev):
    """Parent with zero total elements: the gradient is the correctly shaped
    empty tensor, and nothing dereferences its (absent) storage."""
    x = _d(np.zeros((2, 0, 3)), dev)
    x.requires_grad_(True)
    s = x.slice(2, 1, 2)
    assert s.shape == (2, 0, 1)
    s.backward()
    assert x.grad.shape == (2, 0, 3)
    assert int(x.grad.numel) == 0


@pytest.mark.parametrize("dev", DEVICES)
def test_empty_slice_contributes_nothing_to_a_branch_sum(dev):
    x = _d(X253, dev)
    x.requires_grad_(True)
    x.slice(1, 2, 2).backward()
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(UP233, dev)).backward()

    xt = _t(X253, rg=True)
    (xt.narrow(1, 1, 3) * _t(UP233)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# non-contiguous parent
@pytest.mark.parametrize("dev", DEVICES)
def test_slice_a_transposed_parent_on_an_inner_dimension(dev):
    """SliceNode -> TransposeNode -> base, with the sliced dimension not the
    outermost one and the parent already strided."""
    base = _d(X345, dev)
    base.requires_grad_(True)
    parent = base.transpose(0, 1)               # [4, 3, 5], strided
    assert parent.is_contiguous is False
    sliced = parent.slice(1, 1, 3)              # inner dim of a strided view
    up = _upstream((4, 2, 5), 0.77)
    ortho.sum(_mat(sliced) * _d(up, dev)).backward()

    bt = _t(X345, rg=True)
    (bt.transpose(0, 1).narrow(1, 1, 2) * _t(up)).sum().backward()

    assert _np(base.grad) == pytest.approx(_ref(bt), abs=1e-6)
    g = _np(base.grad).reshape(3, 4, 5)
    assert np.all(g[0] == 0.0), "base row 0 is outside the sliced region"
    # sliced[i, j, k] == parent[i, j + 1, k] == base[j + 1, i, k], so the two
    # selected base rows receive the upstream with its first two axes swapped.
    assert g[1:3] == pytest.approx(np.transpose(up, (1, 0, 2)), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_a_transposed_parent_on_the_outer_dimension(dev):
    base = _d(X43, dev)
    base.requires_grad_(True)
    parent = base.transpose(0, 1)               # [3, 4]
    sliced = parent.slice(0, 1, 3)
    up = _upstream((2, 4), 1.31)
    ortho.sum(_mat(sliced) * _d(up, dev)).backward()

    bt = _t(X43, rg=True)
    (bt.transpose(0, 1).narrow(0, 1, 2) * _t(up)).sum().backward()

    assert _np(base.grad) == pytest.approx(_ref(bt), abs=1e-6)
    g = _np(base.grad).reshape(4, 3)
    assert np.all(g[:, 0] == 0.0)
    assert g[:, 1:3] == pytest.approx(up.T, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_of_a_reshaped_parent(dev):
    base = _d(X253, dev)
    base.requires_grad_(True)
    parent = base.reshape([2, 15])
    sliced = parent.slice(1, 3, 9)
    up = _upstream((2, 6), 0.42)
    ortho.sum(_mat(sliced) * _d(up, dev)).backward()

    bt = _t(X253, rg=True)
    (bt.reshape(2, 15).narrow(1, 3, 6) * _t(up)).sum().backward()

    assert _np(base.grad) == pytest.approx(_ref(bt), abs=1e-6)


# strided grad_output arriving at SliceNode
@pytest.mark.parametrize("dev", DEVICES)
def test_strided_grad_output_into_slice_node(dev):
    """TransposeNode hands SliceNode a non-contiguous grad_output, which
    slice_backward must materialise before doing any storage-order copying."""
    x = _d(X253, dev)
    x.requires_grad_(True)
    z = x.slice(1, 1, 4).transpose(1, 2).contiguous()
    up = _upstream((2, 3, 3), 1.61)
    ortho.sum(z * _d(up, dev)).backward()

    xt = _t(X253, rg=True)
    (xt.narrow(1, 1, 3).transpose(1, 2).contiguous() * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 5, 3)
    assert np.all(g[:, 0, :] == 0.0)
    assert np.all(g[:, 4, :] == 0.0)
    for p in range(2):
        assert g[p, 1:4, :] == pytest.approx(up[p].T, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_strided_grad_output_into_a_last_dimension_slice(dev):
    x = _d(X236, dev)
    x.requires_grad_(True)
    z = x.slice(2, 2, 5).transpose(0, 2).contiguous()
    up = _upstream((3, 3, 2), 0.93)
    ortho.sum(z * _d(up, dev)).backward()

    xt = _t(X236, rg=True)
    (xt.narrow(2, 2, 3).transpose(0, 2).contiguous() * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 3, 6)
    assert np.all(g[:, :, :2] == 0.0)
    assert np.all(g[:, :, 5:] == 0.0)


# branched / overlapping / disjoint slices
@pytest.mark.parametrize("dev", DEVICES)
def test_overlapping_leading_dimension_slices_accumulate(dev):
    up1 = _upstream((3, 5), 1.11)
    up2 = _upstream((3, 5), 2.02)
    x = _d(X45, dev)
    x.requires_grad_(True)
    a = x.slice(0, 0, 3)
    b = x.slice(0, 1, 4)
    ortho.sum(ortho.sum(_mat(a) * _d(up1, dev)) + ortho.sum(_mat(b) * _d(up2, dev))).backward()

    xt = _t(X45, rg=True)
    ((xt.narrow(0, 0, 3) * _t(up1)).sum()
     + (xt.narrow(0, 1, 3) * _t(up2)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(4, 5)
    assert g[0] == pytest.approx(up1[0], abs=1e-6)              # a only
    assert g[1] == pytest.approx(up1[1] + up2[0], abs=1e-6)     # overlap
    assert g[2] == pytest.approx(up1[2] + up2[1], abs=1e-6)     # overlap
    assert g[3] == pytest.approx(up2[2], abs=1e-6)              # b only


@pytest.mark.parametrize("dev", DEVICES)
def test_overlapping_inner_dimension_slices_accumulate(dev):
    """Overlap plus multi-run scatter: two independently scattered gradients
    must sum through central accumulation."""
    up1 = _upstream((2, 3, 3), 1.37)
    up2 = _upstream((2, 3, 3), 0.51)
    x = _d(X253, dev)
    x.requires_grad_(True)
    a = x.slice(1, 0, 3)
    b = x.slice(1, 2, 5)
    ortho.sum(ortho.sum(_mat(a) * _d(up1, dev)) + ortho.sum(_mat(b) * _d(up2, dev))).backward()

    xt = _t(X253, rg=True)
    ((xt.narrow(1, 0, 3) * _t(up1)).sum()
     + (xt.narrow(1, 2, 3) * _t(up2)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 5, 3)
    assert g[:, 0:2, :] == pytest.approx(up1[:, 0:2, :], abs=1e-6)
    assert g[:, 2, :] == pytest.approx(up1[:, 2, :] + up2[:, 0, :], abs=1e-6)
    assert g[:, 3:5, :] == pytest.approx(up2[:, 1:3, :], abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_disjoint_slices_land_separately(dev):
    up1 = _upstream((2, 2, 3), 1.72)
    up2 = _upstream((2, 2, 3), 0.29)
    x = _d(X253, dev)
    x.requires_grad_(True)
    a = x.slice(1, 0, 2)
    b = x.slice(1, 3, 5)
    ortho.sum(ortho.sum(_mat(a) * _d(up1, dev)) + ortho.sum(_mat(b) * _d(up2, dev))).backward()

    xt = _t(X253, rg=True)
    ((xt.narrow(1, 0, 2) * _t(up1)).sum()
     + (xt.narrow(1, 3, 2) * _t(up2)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 5, 3)
    assert g[:, 0:2, :] == pytest.approx(up1, abs=1e-6)
    assert np.all(g[:, 2, :] == 0.0), "the untouched gap must stay zero"
    assert g[:, 3:5, :] == pytest.approx(up2, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_the_same_slice_twice_accumulates(dev):
    up1 = _upstream((2, 3, 3), 1.44)
    up2 = _upstream((2, 3, 3), 0.66)
    x = _d(X253, dev)
    x.requires_grad_(True)
    a = x.slice(1, 1, 4)
    b = x.slice(1, 1, 4)
    ortho.sum(ortho.sum(_mat(a) * _d(up1, dev)) + ortho.sum(_mat(b) * _d(up2, dev))).backward()

    g = _np(x.grad).reshape(2, 5, 3)
    assert g[:, 1:4, :] == pytest.approx(up1 + up2, abs=1e-6)
    assert np.all(g[:, 0, :] == 0.0) and np.all(g[:, 4, :] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_slices_on_different_dimensions_accumulate(dev):
    up1 = _upstream((2, 3, 3), 0.83)
    up2 = _upstream((2, 5, 2), 1.92)
    x = _d(X253, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.sum(_mat(x.slice(1, 1, 4)) * _d(up1, dev))
           + ortho.sum(_mat(x.slice(2, 1, 3)) * _d(up2, dev))).backward()

    xt = _t(X253, rg=True)
    ((xt.narrow(1, 1, 3) * _t(up1)).sum()
     + (xt.narrow(2, 1, 2) * _t(up2)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_gradient_accumulates_across_backward_calls(dev):
    up1 = _upstream((2, 3, 3), 1.06)
    up2 = _upstream((2, 3, 3), 0.38)
    x = _d(X253, dev)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 0, 3)) * _d(up1, dev)).backward()
    first = _np(x.grad).copy()
    ortho.sum(_mat(x.slice(1, 2, 5)) * _d(up2, dev)).backward()
    second = _np(x.grad)

    xt = _t(X253, rg=True)
    (xt.narrow(1, 0, 3) * _t(up1)).sum().backward()
    assert first == pytest.approx(_ref(xt), abs=1e-6)
    (xt.narrow(1, 2, 3) * _t(up2)).sum().backward()
    assert second == pytest.approx(_ref(xt), abs=1e-6)
    assert not np.allclose(first, second)


def test_the_branch_upstreams_are_not_identical():
    """Oracle guard: the accumulation tests are only meaningful if the branch
    gradients actually differ."""
    assert not np.allclose(_upstream((2, 3, 3), 1.37), _upstream((2, 3, 3), 0.51))


# non-leaf parent
@pytest.mark.parametrize("dev", DEVICES)
def test_slice_of_a_computed_non_leaf_parent(dev):
    x = _d(X253, dev)
    x.requires_grad_(True)
    p = ortho.tanh(x)
    ortho.sum(_mat(p.slice(1, 1, 4)) * _d(UP233, dev)).backward()

    xt = _t(X253, rg=True)
    (torch.tanh(xt).narrow(1, 1, 3) * _t(UP233)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    # tanh' is nowhere zero, so the whole selected region is non-zero and the
    # rest is exactly zero: the gradient really travelled through the slice.
    g = _np(x.grad).reshape(2, 5, 3)
    assert np.all(g[:, 0, :] == 0.0) and np.all(g[:, 4, :] == 0.0)
    assert np.count_nonzero(g[:, 1:4, :]) == UP233.size


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_of_a_relu_parent_reaches_the_original_leaf(dev):
    x = _d(X345, dev)
    x.requires_grad_(True)
    up = _upstream((3, 2, 5), 1.19)
    ortho.sum(_mat(ortho.relu(x).slice(1, 1, 3)) * _d(up, dev)).backward()

    xt = _t(X345, rg=True)
    (torch.relu(xt).narrow(1, 1, 2) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# chained slices
@pytest.mark.parametrize("dev", DEVICES)
def test_chained_slices_on_the_same_dimension(dev):
    x = _d(X263, dev)
    x.requires_grad_(True)
    y = x.slice(1, 1, 5)
    z = y.slice(1, 1, 3)
    assert y._has_grad_fn is True and y.is_leaf is False
    assert z._has_grad_fn is True and z.is_leaf is False
    up = _upstream((2, 2, 3), 0.74)
    ortho.sum(_mat(z) * _d(up, dev)).backward()

    xt = _t(X263, rg=True)
    (xt.narrow(1, 1, 4).narrow(1, 1, 2) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 6, 3)
    assert g[:, 2:4, :] == pytest.approx(up, abs=1e-6)
    assert np.all(g[:, 0:2, :] == 0.0) and np.all(g[:, 4:6, :] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_chained_slices_on_different_dimensions(dev):
    x = _d(X345, dev)
    x.requires_grad_(True)
    z = x.slice(0, 1, 3).slice(2, 1, 4)
    up = _upstream((2, 4, 3), 1.55)
    ortho.sum(_mat(z) * _d(up, dev)).backward()

    xt = _t(X345, rg=True)
    (xt.narrow(0, 1, 2).narrow(2, 1, 3) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(3, 4, 5)
    assert np.all(g[0] == 0.0)
    assert g[1:3, :, 1:4] == pytest.approx(up, abs=1e-6)
    assert np.all(g[1:3, :, 0] == 0.0) and np.all(g[1:3, :, 4] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_chained_slices_with_a_negative_dim_stage(dev):
    x = _d(X345, dev)
    x.requires_grad_(True)
    z = x.slice(-1, 1, 4).slice(1, 1, 3)
    up = _upstream((3, 2, 3), 0.68)
    ortho.sum(_mat(z) * _d(up, dev)).backward()

    xt = _t(X345, rg=True)
    (xt.narrow(2, 1, 3).narrow(1, 1, 2) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_three_chained_slices(dev):
    x = _d(X345, dev)
    x.requires_grad_(True)
    z = x.slice(0, 0, 3).slice(1, 1, 4).slice(-1, -4, -1)
    assert z.shape == (3, 3, 3)
    up = _upstream((3, 3, 3), 1.83)
    ortho.sum(_mat(z) * _d(up, dev)).backward()

    xt = _t(X345, rg=True)
    (xt.narrow(0, 0, 3).narrow(1, 1, 3).narrow(2, 1, 3) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# SliceNode / IndexNode composition, both orders
@pytest.mark.parametrize("dev", DEVICES)
def test_index_of_a_slice(dev):
    x = _d(X345, dev)
    x.requires_grad_(True)
    y = x.slice(2, 1, 4)[1]
    up = _upstream((4, 3), 1.27)
    ortho.sum(_mat(y) * _d(up, dev)).backward()

    xt = _t(X345, rg=True)
    (xt.narrow(2, 1, 3)[1] * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(3, 4, 5)
    assert np.all(g[0] == 0.0) and np.all(g[2] == 0.0)
    assert g[1, :, 1:4] == pytest.approx(up, abs=1e-6)
    assert np.all(g[1, :, 0] == 0.0) and np.all(g[1, :, 4] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_of_an_index(dev):
    x = _d(X345, dev)
    x.requires_grad_(True)
    y = x[1].slice(1, 1, 4)
    up = _upstream((4, 3), 0.49)
    ortho.sum(_mat(y) * _d(up, dev)).backward()

    xt = _t(X345, rg=True)
    (xt[1].narrow(1, 1, 3) * _t(up)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(3, 4, 5)
    assert np.all(g[0] == 0.0) and np.all(g[2] == 0.0)
    assert g[1, :, 1:4] == pytest.approx(up, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_both_orders_agree_when_the_selected_region_is_the_same(dev):
    """x.slice(2, 1, 4)[1] and x[1].slice(1, 1, 4) select the same elements, so
    the same upstream must produce the same parent gradient."""
    up = _upstream((4, 3), 1.05)
    grads = []
    for build in (lambda t: t.slice(2, 1, 4)[1], lambda t: t[1].slice(1, 1, 4)):
        x = _d(X345, dev)
        x.requires_grad_(True)
        ortho.sum(_mat(build(x)) * _d(up, dev)).backward()
        grads.append(_np(x.grad))
    assert grads[0] == pytest.approx(grads[1], abs=0)


@pytest.mark.parametrize("dev", DEVICES)
def test_scalar_index_of_a_slice(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    x.slice(1, 1, 3)[2, 0].backward()

    xt = _t(X34, rg=True)
    xt.narrow(1, 1, 2)[2, 0].backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    expected = np.zeros((3, 4))
    expected[2, 1] = 1.0
    assert _np(x.grad).reshape(3, 4) == pytest.approx(expected, abs=0)


# the gradient is a plain first-order tensor
@pytest.mark.parametrize("dev", DEVICES)
def test_slice_gradient_carries_no_graph(dev):
    x = _d(X253, dev)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(UP233, dev)).backward()
    g = x.grad
    assert g.requires_grad is False
    assert g._has_grad_fn is False
    assert g.is_leaf is True
    assert g.is_contiguous is True
    assert g.shape == x.shape


@pytest.mark.parametrize("dev", DEVICES)
def test_slice_gradient_preserves_dtype_and_exact_device(dev):
    x = _d(X253, dev)
    x.requires_grad_(True)
    ortho.sum(_mat(x.slice(1, 1, 4)) * _d(UP233, dev)).backward()
    assert x.grad.dtype == x.dtype
    assert x.grad.device == x.device


# the forward error contract is unchanged
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("args", [
    (2, 0, 1),      # dim == ndim
    (5, 0, 1),      # dim far out of range
    (-3, 0, 1),     # negative dim below -ndim
    (0, -10, 2),    # start negative after normalisation
    (0, 0, 5),      # end beyond size
    (0, 3, 2),      # start > end
    (1, 0, -9),     # end negative after normalisation
    (1, 5, 5),      # start beyond size
])
def test_invalid_slice_arguments_still_raise(dev, args):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with pytest.raises(IndexError):
        x.slice(*args)


@pytest.mark.parametrize("dev", DEVICES)
def test_slicing_a_scalar_still_raises(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    scalar = x[1, 2]
    assert scalar.ndim == 0
    for dim in (0, -1, 1):
        with pytest.raises(IndexError):
            scalar.slice(dim, 0, 1)


@pytest.mark.parametrize("dev", DEVICES)
def test_a_failed_slice_leaves_no_gradient_behind(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with pytest.raises(IndexError):
        x.slice(0, 0, 99)
    assert x.grad is None


# The Python surface, and what it deliberately does not include
def test_slice_is_the_only_new_method():
    assert hasattr(ortho.Tensor, "slice")
    for absent in ("narrow", "__index_put__", "index_select", "select"):
        assert not hasattr(ortho.Tensor, absent)


def test_slice_takes_exactly_three_arguments():
    x = _d(X34)
    with pytest.raises(TypeError):
        x.slice(0, 0, 3, 2)          # no step
    with pytest.raises(TypeError):
        x.slice(0, 0)                # no omitted bound
    assert x.slice(dim=0, start=0, end=2).shape == (2, 4)


def test_python_slice_key_syntax_is_still_unsupported():
    """x.slice(dim, start, end) is the whole surface: slice objects, steps,
    omitted bounds and Ellipsis are out of scope."""
    x = _d(X34)
    for key in ("plain", "tuple", "step", "ellipsis"):
        with pytest.raises(Exception):
            {"plain": lambda t: t[1:3],
             "tuple": lambda t: t[:, 1:3],
             "step": lambda t: t[::2],
             "ellipsis": lambda t: t[..., 1:3]}[key](x)


def test_existing_integer_indexing_is_unchanged():
    x = _d(X34)
    x.requires_grad_(True)
    assert x[1].shape == (4,)
    assert x[1, 2].shape == ()
    assert x[[1, 2]].shape == ()                      # chained, not advanced
    assert torch.tensor(X34)[[1, 2]].shape == (2, 4)  # torch: gather two rows
    with pytest.raises(IndexError):
        x[9]
    with pytest.raises(ValueError):
        x[[]]


def test_setitem_semantics_are_untouched():
    x = _d(X34)
    x.requires_grad_(True)
    x[0, 1] = 7.0
    assert x.get_item([0, 1]) == 7.0
    x[2] = 3.0                       # partial index fills the sub-view
    assert [x.get_item([2, j]) for j in range(4)] == [3.0] * 4


# The CUDA device-context contract, pinned at source level
def _slice_ops_source():
    """The native scatter source, or None when running outside the repo."""
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "src", "backends", "cuda", "adapters.cpp")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _cuda_branch():
    src = _slice_ops_source()
    if src is None:
        pytest.skip("native source not available next to the test tree")
    start = src.index("Tensor cuda_slice_backward_adapter")
    import re
    return re.sub(r"//[^\n]*", "", src[start:src.index("\n} } }", start)])


def test_cuda_scatter_selects_the_parent_device():
    """The CUDA scatter selects the parent device before accessing its memory.
    Check DeviceGuard placement independently of available GPU ordinals."""
    branch = _cuda_branch()
    assert "DeviceGuard guard(input_device)" in branch
    assert branch.index("DeviceGuard guard(") < branch.index("context->memset_async")
    assert branch.index("DeviceGuard guard(") < branch.index("context->copy_d2d_2d_async")
    assert "cudaSetDevice" not in branch


def test_cuda_scatter_has_no_hardcoded_device():
    branch = _cuda_branch()
    assert "cudaSetDevice(0)" not in branch
    assert 'device_from_string("cuda")' not in branch
    assert '"cuda"' not in branch


def test_cuda_scatter_keeps_the_gradient_on_the_device():
    """No host round-trip for the gradient payload."""
    branch = _cuda_branch()
    assert "context->copy_d2d_2d_async" in branch
    assert "copy_h2d_snapshot" not in branch
    assert "copy_d2h_blocking" not in branch


def test_no_custom_cuda_kernel_was_added_for_slice():
    """The layout maps onto cudaMemset + cudaMemcpy2D, so slice backward
    needs no kernel launch site and no .cu file."""
    src = _slice_ops_source()
    if src is None:
        pytest.skip("native source not available next to the test tree")
    assert "__global__" not in src
    assert "<<<" not in src
    cuda_dir = os.path.join(os.path.dirname(__file__), "..", "..",
                            "src", "backends", "cuda")
    assert not os.path.isfile(os.path.join(cuda_dir, "slice_kernels.cu"))
