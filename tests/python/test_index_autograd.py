"""Integer-index views scatter gradients into the selected parent block.

Compare nonuniform values and untouched zero regions to detect misplaced writes."""
from tests.support.capabilities import orthon_cuda_available
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


X34 = _vals((3, 4), 1.7)
X43 = _vals((4, 3), 1.1)
X234 = _vals((2, 3, 4), 1.3)
X5 = _vals((5,), 0.9)
UP4 = _vals((4,), 2.1, 1.4)
UP4b = _vals((4,), 0.8, 1.2)
UP4c = _vals((4,), 1.9, 0.7)
UP43 = _vals((4, 3), 0.6, 1.1)


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


# Forward view contract - unchanged by autograd attachment
@pytest.mark.parametrize("dev", DEVICES)
def test_index_is_still_a_view_when_grad_is_tracked(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = x[1]
    assert y.shape == (4,)
    assert y.strides == (1,)
    assert y.storage_offset == 4
    assert y.tolist() == pytest.approx(X34[1].tolist(), rel=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_index_view_metadata_matches_untracked_index(dev):
    """Attaching a node must not change shape/strides/offset of the view."""
    plain = _d(X34, dev)[2]
    tracked = _d(X34, dev)
    tracked.requires_grad_(True)
    tracked = tracked[2]
    assert (tracked.shape, tracked.strides, tracked.storage_offset) == (
        plain.shape, plain.strides, plain.storage_offset)


def test_indexed_view_still_shares_storage_with_a_grad_tracking_parent():
    x = _d(X34)
    x.requires_grad_(True)
    row = x[1]
    row[0] = 99.0                       # __setitem__ on the view
    assert x.get_item([1, 0]) == 99.0   # ...lands in the parent's storage


@pytest.mark.parametrize("dev", DEVICES)
def test_scalar_index_view_is_zero_dimensional(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    s = x[1, 2]
    assert s.shape == ()
    assert s.ndim == 0
    assert s.storage_offset == 6


# Graph metadata
@pytest.mark.parametrize("dev", DEVICES)
def test_index_attaches_grad_fn_when_parent_requires_grad(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = x[1]
    assert y.requires_grad is True
    assert y._has_grad_fn is True
    assert y.is_leaf is False


@pytest.mark.parametrize("dev", DEVICES)
def test_index_attaches_nothing_when_parent_does_not_require_grad(dev):
    y = _d(X34, dev)[1]
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True


@pytest.mark.parametrize("dev", DEVICES)
def test_no_grad_index_returns_a_detached_view(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    with ortho.no_grad():
        y = x[1]
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True
    # ...but still the same view.
    assert (y.shape, y.strides, y.storage_offset) == ((4,), (1,), 4)
    assert y.tolist() == pytest.approx(X34[1].tolist(), rel=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_global_grad_disabled_index_returns_a_detached_view(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.set_grad_enabled(False)
    try:
        y = x[1]
    finally:
        ortho.set_grad_enabled(True)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.storage_offset == 4


# Basic parity: positive, negative, every index
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("i", [0, 1, 2, -1, -2, -3])
def test_row_index_gradient_matches_torch(dev, i):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(x[i] * _d(UP4, dev)).backward()

    xt = _t(X34, rg=True)
    (xt[i] * _t(UP4)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("i", [0, 1, 2, -1])
def test_only_the_selected_row_receives_gradient(dev, i):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(x[i] * _d(UP4, dev)).backward()

    g = _np(x.grad).reshape(3, 4)
    sel = i % 3
    assert g[sel] == pytest.approx(UP4, abs=1e-6)
    for r in range(3):
        if r != sel:
            assert np.all(g[r] == 0.0), "row %d should be untouched" % r


@pytest.mark.parametrize("dev", DEVICES)
def test_different_indices_land_in_different_rows(dev):
    """Guards against a scatter that always writes block 0."""
    rows = []
    for i in range(3):
        x = _d(X34, dev)
        x.requires_grad_(True)
        ortho.sum(x[i] * _d(UP4, dev)).backward()
        rows.append(np.nonzero(_np(x.grad).reshape(3, 4).any(axis=1))[0].tolist())
    assert rows == [[0], [1], [2]]


@pytest.mark.parametrize("dev", DEVICES)
def test_non_scalar_backward_seed_through_index(dev):
    """x[i].backward() seeds ones() straight into IndexNode."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    x[2].backward()

    xt = _t(X34, rg=True)
    xt[2].backward(torch.ones(4))

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# Scalar indexing
@pytest.mark.parametrize("dev", DEVICES)
def test_scalar_index_of_1d_backward(dev):
    x = _d(X5, dev)
    x.requires_grad_(True)
    x[2].backward()

    xt = _t(X5, rg=True)
    xt[2].backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    assert _np(x.grad).tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("form", ["tuple", "chained"])
def test_scalar_index_of_2d_backward(dev, form):
    x = _d(X34, dev)
    x.requires_grad_(True)
    (x[1, 2] if form == "tuple" else x[1][2]).backward()

    xt = _t(X34, rg=True)
    xt[1, 2].backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    expected = np.zeros((3, 4))
    expected[1, 2] = 1.0
    assert _np(x.grad).reshape(3, 4) == pytest.approx(expected, abs=0)


# Chained tuple / list indexing (existing binding semantics)
@pytest.mark.parametrize("dev", DEVICES)
@pytest.mark.parametrize("key", ["tuple", "paren_tuple", "list", "chained"])
def test_all_chained_index_forms_produce_the_same_gradient(dev, key):
    """The binding implements tuple/list keys as repeated cur = cur[i], so all
    four spellings must give the same graph. NOTE: x[[1, 2]] is chained
    indexing here (== x[1][2]), not PyTorch advanced indexing (which would
    gather rows 1 and 2). The torch reference is therefore written as the
    chained form the binding actually implements."""
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = {"tuple": lambda t: t[1, 2],
         "paren_tuple": lambda t: t[(1, 2)],
         "list": lambda t: t[[1, 2]],
         "chained": lambda t: t[1][2]}[key](x)
    y.backward()

    xt = _t(X34, rg=True)
    xt[1][2].backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_list_key_is_chained_not_advanced_indexing(dev):
    """Pin the documented divergence from PyTorch so a future change is loud."""
    x = _d(X34, dev)
    assert x[[1, 2]].shape == ()                        # orthon: x[1][2]
    assert torch.tensor(X34)[[1, 2]].shape == (2, 4)    # torch: gather two rows


@pytest.mark.parametrize("dev", DEVICES)
def test_intermediate_view_of_a_chained_index_has_its_own_node(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    mid = x[1]
    leaf = mid[2]
    assert mid._has_grad_fn is True and mid.is_leaf is False
    assert leaf._has_grad_fn is True and leaf.is_leaf is False
    assert mid.storage_offset == 4 and leaf.storage_offset == 6


@pytest.mark.parametrize("dev", DEVICES)
def test_chained_index_into_3d(dev):
    x = _d(X234, dev)
    x.requires_grad_(True)
    ortho.sum(x[1][2] * _d(UP4, dev)).backward()

    xt = _t(X234, rg=True)
    (xt[1][2] * _t(UP4)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 3, 4)
    assert g[1, 2] == pytest.approx(UP4, abs=1e-6)
    assert np.count_nonzero(g) == np.count_nonzero(UP4)


# Composition: non-contiguous parent, strided grad_output
@pytest.mark.parametrize("dev", DEVICES)
def test_index_a_non_contiguous_parent(dev):
    """IndexNode -> TransposeNode -> x.

    .contiguous() after the index is required on CUDA for the forward multiply
    (cuda_mul rejects strided operands - a pre-existing forward limitation that
    fires with no indexing involved), and is a no-op for the graph shape being
    tested here.
    """
    x = _d(X43, dev)
    x.requires_grad_(True)
    t = x.transpose(0, 1)
    assert t.is_contiguous is False
    ortho.sum(t[1].contiguous() * _d(UP4b, dev)).backward()

    xt = _t(X43, rg=True)
    (xt.transpose(0, 1)[1].contiguous() * _t(UP4b)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    # The gradient lands in column 1 of x, i.e. row 1 of the transposed view.
    g = _np(x.grad).reshape(4, 3)
    assert g[:, 1] == pytest.approx(UP4b, abs=1e-6)
    assert np.all(g[:, 0] == 0.0) and np.all(g[:, 2] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_strided_grad_output_into_index_node(dev):
    """The transpose upstream hands IndexNode a non-contiguous
    grad_output."""
    x = _d(X234, dev)
    x.requires_grad_(True)
    z = x[1].transpose(0, 1).contiguous()
    ortho.sum(z * _d(UP43, dev)).backward()

    xt = _t(X234, rg=True)
    (xt[1].transpose(0, 1).contiguous() * _t(UP43)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(2, 3, 4)
    assert np.all(g[0] == 0.0)
    assert g[1] == pytest.approx(UP43.T, abs=1e-6)


@pytest.mark.parametrize("dev", DEVICES)
def test_index_of_a_computed_non_leaf_parent(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    y = ortho.tanh(x)
    ortho.sum(y[1] * _d(UP4, dev)).backward()

    xt = _t(X34, rg=True)
    (torch.tanh(xt)[1] * _t(UP4)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)


# Branched / repeated indexing (exercises central gradient accumulation)
@pytest.mark.parametrize("dev", DEVICES)
def test_same_index_twice_accumulates(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.sum(x[0] * _d(UP4, dev)) + ortho.sum(x[0] * _d(UP4c, dev))).backward()

    xt = _t(X34, rg=True)
    ((xt[0] * _t(UP4)).sum() + (xt[0] * _t(UP4c)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(3, 4)
    assert g[0] == pytest.approx(UP4 + UP4c, abs=1e-6)
    assert np.all(g[1:] == 0.0)


@pytest.mark.parametrize("dev", DEVICES)
def test_two_different_indices_land_separately(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(ortho.sum(x[0] * _d(UP4, dev)) + ortho.sum(x[2] * _d(UP4c, dev))).backward()

    xt = _t(X34, rg=True)
    ((xt[0] * _t(UP4)).sum() + (xt[2] * _t(UP4c)).sum()).backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(3, 4)
    assert g[0] == pytest.approx(UP4, abs=1e-6)
    assert np.all(g[1] == 0.0)
    assert g[2] == pytest.approx(UP4c, abs=1e-6)


def test_the_two_upstreams_are_not_identical():
    """Oracle guard: the accumulation tests above are only meaningful if the
    two branch gradients actually differ."""
    assert not np.allclose(UP4, UP4c)


@pytest.mark.parametrize("dev", DEVICES)
def test_index_gradient_accumulates_across_backward_calls(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(x[0] * _d(UP4, dev)).backward()
    first = _np(x.grad).copy()
    ortho.sum(x[2] * _d(UP4c, dev)).backward()
    second = _np(x.grad)

    xt = _t(X34, rg=True)
    (xt[0] * _t(UP4)).sum().backward()
    assert first == pytest.approx(_ref(xt), abs=1e-6)
    (xt[2] * _t(UP4c)).sum().backward()
    assert second == pytest.approx(_ref(xt), abs=1e-6)
    assert not np.allclose(first, second)


# Gradient object properties
@pytest.mark.parametrize("dev", DEVICES)
def test_index_gradient_carries_no_graph(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(x[1] * _d(UP4, dev)).backward()
    g = x.grad
    assert g.requires_grad is False
    assert g._has_grad_fn is False
    assert g.is_leaf is True
    assert g.is_contiguous is True
    assert g.shape == x.shape


@pytest.mark.parametrize("dev", DEVICES)
def test_index_gradient_preserves_dtype_and_exact_device(dev):
    x = _d(X34, dev)
    x.requires_grad_(True)
    ortho.sum(x[1] * _d(UP4, dev)).backward()
    assert x.grad.dtype == x.dtype
    assert x.grad.device == x.device


# DTypes
def test_cpu_float64_index_gradient():
    x = ortho.from_numpy(np.ascontiguousarray(X34.astype(np.float64)))
    x.requires_grad_(True)
    up = ortho.from_numpy(np.ascontiguousarray(UP4.astype(np.float64)))
    ortho.sum(x[1] * up).backward()

    xt = _t(X34, rg=True, dtype=torch.float64)
    (xt[1] * _t(UP4, dtype=torch.float64)).sum().backward()

    assert x.grad.dtype == "float64"
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-12)


@requires_cuda
@pytest.mark.parametrize("dtype,tol", [("float32", 1e-6),
                                       ("float16", 1e-3),
                                       ("bfloat16", 1e-2)])
def test_cuda_index_gradient_dtypes(dtype, tol):
    x = _d(X34, "cuda", dtype)
    x.requires_grad_(True)
    ortho.sum(x[1] * _d(UP4, "cuda", dtype)).backward()

    xt = _t(X34, rg=True)
    (xt[1] * _t(UP4)).sum().backward()

    assert x.grad.dtype == dtype
    assert x.grad.device == x.device
    assert _np(x.grad) == pytest.approx(_ref(xt), abs=tol)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_half_scatter_preserves_bytes_outside_the_block(dtype):
    """The scatter is a typed copy, so the zero regions must be exact zeros in
    the narrow dtype, not merely small."""
    x = _d(X34, "cuda", dtype)
    x.requires_grad_(True)
    ortho.sum(x[1] * _d(UP4, "cuda", dtype)).backward()
    g = _np(x.grad).reshape(3, 4)
    assert np.all(g[0] == 0.0)
    assert np.all(g[2] == 0.0)
    assert np.count_nonzero(g[1]) == 4


# Existing forward behaviour must be unchanged
def test_out_of_range_still_raises():
    x = _d(X34)
    x.requires_grad_(True)
    with pytest.raises(IndexError):
        x[9]
    with pytest.raises(IndexError):
        x[-9]


def test_too_many_indices_still_raises():
    x = _d(X34)
    x.requires_grad_(True)
    with pytest.raises(IndexError):
        x[0, 0, 0]


def test_empty_index_sequence_still_raises():
    x = _d(X34)
    x.requires_grad_(True)
    with pytest.raises(ValueError):
        x[[]]


def test_indexing_a_zero_dim_tensor_still_raises():
    x = _d(X34)
    x.requires_grad_(True)
    with pytest.raises(RuntimeError):
        x[0][0][0]


def test_slice_syntax_is_still_unsupported():
    """Python slice keys are deliberately not part of the index API."""
    x = _d(X34)
    with pytest.raises(Exception):
        x[1:3]


def test_setitem_semantics_are_untouched():
    x = _d(X34)
    x.requires_grad_(True)
    x[0, 1] = 7.0
    assert x.get_item([0, 1]) == 7.0
    x[2] = 3.0                       # partial index fills the sub-view
    assert [x.get_item([2, j]) for j in range(4)] == [3.0] * 4


# Block-shape handling and the CUDA device-context contract
X324 = _vals((3, 2, 4), 0.7)
UP24 = _vals((2, 4), 1.5, 0.9)


@pytest.mark.parametrize("dev", DEVICES)
def test_non_square_block_shape_scatters_correctly(dev):
    """Parent [3, 2, 4] -> block [2, 4]. A shape-blind scatter that accepted
    [4, 2] would move the same bytes to the wrong elements, so pin the layout
    and not just the element count."""
    x = _d(X324, dev)
    x.requires_grad_(True)
    ortho.sum(x[1] * _d(UP24, dev)).backward()

    xt = _t(X324, rg=True)
    (xt[1] * _t(UP24)).sum().backward()

    assert _np(x.grad) == pytest.approx(_ref(xt), abs=1e-6)
    g = _np(x.grad).reshape(3, 2, 4)
    assert g[1] == pytest.approx(UP24, abs=1e-6)
    assert np.all(g[0] == 0.0) and np.all(g[2] == 0.0)


def _index_ops_source():
    """The native scatter source, or None when running outside the repo."""
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "src", "backends", "cuda", "adapters.cpp")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read()


def _cuda_branch():
    src = _index_ops_source()
    if src is None:
        pytest.skip("native source not available next to the test tree")
    start = src.index("Tensor cuda_index_backward_adapter")
    return src[start:src.index("Tensor cuda_slice_backward_adapter", start)]


def test_cuda_scatter_selects_the_parent_device():
    """CUDA scatter selects the parent Device before touching its memory.

    Pool reuse must not bypass the DeviceGuard that restores the caller's device."""
    branch = _cuda_branch()
    assert "DeviceGuard guard(input_device)" in branch
    # ...and it must come before the memory operations it protects.
    assert branch.index("DeviceGuard guard(") < branch.index("context->memset_async")
    assert branch.index("DeviceGuard guard(") < branch.index("context->copy_d2d_async")
    # The raw setter it replaced must not have come back.
    assert "cudaSetDevice" not in branch.replace(
        "raw cudaSetDevice because no", "")


def test_cuda_scatter_has_no_hardcoded_device():
    branch = _cuda_branch()
    assert "cudaSetDevice(0)" not in branch
    assert 'device_from_string("cuda")' not in branch
    assert '"cuda"' not in branch
