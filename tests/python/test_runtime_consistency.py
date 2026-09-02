"""CPU/CUDA runtime consistency.

The device an operation runs on must not change what it means. Four classes of
divergence are pinned here, each of which is silent rather than loud when it
goes wrong:

1. VIEWS. CPU walks arbitrary strides for ``add``/``mul``, every public unary,
   ``dropout``, real dtype ``cast``, ``fused_linear_relu`` and ``embedding``,
   so CUDA must consume the logically equivalent views. A raw CUDA kernel that
   receives a strided pointer and indexes storage LINEARLY returns storage-order
   values with no error at all, and can read past the end of a view's storage.

2. ``cast(x, x.dtype)`` means one thing on both devices: ``contiguous(x)``.
   Falling into cuda_cast's memcpy branch on CUDA instead returns a fresh
   DETACHED copy, so an operation that converts nothing behaves differently
   per device.

3. Python runtime helpers move tensors with a Device OBJECT, never the STRING
   ``"cuda"`` - the Device parser resolves that to cuda:0, so a module or
   collective living on cuda:1 would build its helper tensors on the wrong GPU.

4. ``set_item``, the full-index ``__setitem__`` and ``get_item`` route through
   the CUDA runtime. Dereferencing ``t.data()`` from host C++ on a CUDA tensor
   is device memory, and terminates the process with an access violation
   (0xC0000005).

The load-bearing design decision is that CUDA kernels stay contiguous-only.
Nothing here teaches a kernel about strides; the public op materialises only
its KERNEL OPERAND, under an explicit NoGradGuard, and attaches the public
autograd Node to the ORIGINAL user tensor. That is what keeps the public graph
shape, the saved-version watches and view backward routing pinned to what the
caller passed - and what would break, invisibly, if a future change saved the
temporary instead. The tests in TestMutationSelectivity exist specifically to
catch that mistake.

HONESTY NOTE: this machine has one GPU. Every genuinely multi-device assertion
skips explicitly rather than pretending to run; on a single GPU the exact-device
assertions read ``device('cuda:0')`` (not merely ``is_cuda``), which is a
regression tripwire, and the load-bearing evidence is the source-level audit at
the bottom of this file.
"""
import os
import re

import numpy as np
import pytest

import orthon as ortho
import orthon.nn as nn

CUDA = ortho.cuda_enabled()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

torch = pytest.importorskip("torch")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
requires_2_gpus = pytest.mark.skipif(
    N_GPUS < 2, reason="needs >=2 CUDA devices (this machine has %d)" % N_GPUS)

STALE = r"modified in-place"

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

BASE = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
POS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
MIX = np.array([[1.5, -2.0, 3.25], [0.5, -0.75, 4.0]], dtype=np.float32)


def _t(values, device="cpu", dtype=np.float32, grad=False):
    t = ortho.from_numpy(np.ascontiguousarray(np.asarray(values, dtype=dtype)))
    if device != "cpu":
        t = t.to(device)
    if grad:
        t.requires_grad_(True)
    return t


def _flat(t):
    """Flat host values (Tensor.tolist is flat by contract)."""
    if t.device.is_cuda():
        t = t.to("cpu")
    if t.dtype not in ("float32", "float64"):
        t = ortho.cast(t, "float32")
    return t.tolist()


def _read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def _strip_c_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _strip_py_comments_and_docstrings(src):
    src = re.sub(r'"""(?:.|\n)*?"""', "", src)
    src = re.sub(r"'''(?:.|\n)*?'''", "", src)
    return re.sub(r"#[^\n]*", "", src)


# ---------------------------------------------------------------------------
# CUDA binary ops accept logical views
# ---------------------------------------------------------------------------

class TestBinaryViews:
    """Every view/operand combination, checked against the CPU result.

    Each case is run on BOTH devices from the same source values, so the
    assertion is CPU/CUDA agreement on logical values, not just "CUDA does
    not raise". Without operand normalisation every CUDA case here raises
    "cuda_add/cuda_mul: input 'a' is not contiguous".
    """

    def _views(self, device):
        base = _t(BASE, device)
        other = _t(BASE * 10.0, device)
        return {
            # transpose views: strided in the last dim
            "transpose+transpose": (base.transpose(0, 1), other.transpose(0, 1)),
            "transpose+contig":    (base.transpose(0, 1), _t(BASE.T * 10.0, device)),
            "contig+transpose":    (_t(BASE.T * 10.0, device), base.transpose(0, 1)),
            # slice views: dim-1 slice is strided, dim-0 slice stays contiguous
            "slice1+slice1":       (base.slice(1, 1, 3), other.slice(1, 1, 3)),
            "slice0+slice0":       (base.slice(0, 0, 1), other.slice(0, 0, 1)),
            "slice1+contig":       (base.slice(1, 0, 2), _t(BASE[:, :2] * 10.0, device)),
            # indexed view (drops dim 0)
            "index+index":         (base[1], other[1]),
        }

    def _expected(self, name, op):
        cpu = self._views("cpu")[name]
        return _flat(op(cpu[0], cpu[1]))

    @pytest.mark.parametrize("name", ["transpose+transpose", "transpose+contig",
                                      "contig+transpose", "slice1+slice1",
                                      "slice0+slice0", "slice1+contig",
                                      "index+index"])
    @pytest.mark.parametrize("opname", ["add", "mul"])
    @requires_cuda
    def test_cuda_view_matches_cpu(self, name, opname):
        op = ortho.add if opname == "add" else ortho.mul
        a, b = self._views("cuda")[name]
        out = op(a, b)
        assert out.shape == a.shape or out.shape == b.shape
        assert _flat(out) == pytest.approx(self._expected(name, op), abs=1e-5)

    @requires_cuda
    def test_cuda_broadcast_with_a_strided_operand(self):
        # (3, 2) strided view + (2,) row vector -> broadcast over dim 0.
        base = _t(BASE, "cuda")
        row = _t([10.0, 100.0], "cuda")
        strided = base.transpose(0, 1)                     # (3, 2), strided
        got = _flat(ortho.mul(strided, row))
        want = (BASE.T * np.array([10.0, 100.0], dtype=np.float32)).reshape(-1)
        assert got == pytest.approx(list(want), abs=1e-4)
        # ...and the mirrored operand order.
        got2 = _flat(ortho.mul(row, strided))
        assert got2 == pytest.approx(list(want), abs=1e-4)

    @requires_cuda
    def test_cuda_view_result_is_a_fresh_contiguous_tensor(self):
        base = _t(BASE, "cuda")
        view = base.transpose(0, 1)
        out = ortho.add(view, view)
        assert out.is_contiguous
        assert out._impl_id != view._impl_id
        assert out._impl_id != base._impl_id

    @requires_cuda
    def test_view_normalisation_does_not_mutate_the_original(self):
        base = _t(BASE, "cuda")
        view = base.transpose(0, 1)
        before = base._storage_version
        ortho.mul(view, view)
        ortho.add(view, view)
        # Reading through a materialisation is not a mutation of anything.
        assert base._storage_version == before
        assert _flat(base) == list(BASE.reshape(-1))

    @requires_cuda
    @pytest.mark.parametrize("op", ["add_", "mul_"])
    def test_inplace_still_requires_a_contiguous_destination(self, op):
        """Deliberately NOT loosened: see the note above mul_ in binary_ops.cpp.

        An out-of-place op may materialise its operands because they are only
        read. An in-place op writes to its destination; materialising that
        would mutate a temporary and silently drop the caller's write.
        """
        base = _t(BASE, "cuda")
        view = base.transpose(0, 1)
        with pytest.raises(RuntimeError, match="contiguous"):
            if op == "add_":
                view.add_(_t(BASE.T, "cuda"), 1.0)
            else:
                view.mul_(2.0)
        # The rejected op left the storage untouched, version included.
        assert base._storage_version == 0
        assert _flat(base) == list(BASE.reshape(-1))


# ---------------------------------------------------------------------------
# CUDA unary ops accept logical views
# ---------------------------------------------------------------------------

UNARY = {
    "neg":        lambda t: ortho.neg(t),
    "relu":       lambda t: ortho.relu(t),
    "sigmoid":    lambda t: ortho.sigmoid(t),
    "tanh":       lambda t: ortho.tanh(t),
    "leaky_relu": lambda t: ortho.leaky_relu(t, 0.01),
    "gelu":       lambda t: ortho.gelu(t),
    "exp":        lambda t: ortho.exp(t),
    "sqrt":       lambda t: ortho.sqrt(t),
    "reciprocal": lambda t: ortho.reciprocal(t),
}

# sqrt/reciprocal need a strictly positive, non-degenerate source.
UNARY_SOURCE = {name: (POS if name in ("sqrt", "reciprocal") else MIX)
                for name in UNARY}


class TestUnaryViews:
    @pytest.mark.parametrize("name", sorted(UNARY))
    @requires_cuda
    def test_cuda_transpose_view_matches_cpu(self, name):
        """Every unary op returns logical-order values on a transposed CUDA
        view, matching the CPU result."""
        src = UNARY_SOURCE[name]
        fn = UNARY[name]
        want = _flat(fn(_t(src, "cpu").transpose(0, 1)))
        got = _flat(fn(_t(src, "cuda").transpose(0, 1)))
        assert got == pytest.approx(want, rel=1e-5, abs=1e-6)

    @pytest.mark.parametrize("name", sorted(UNARY))
    @requires_cuda
    def test_cuda_slice_view_matches_cpu(self, name):
        src = UNARY_SOURCE[name]
        fn = UNARY[name]
        want = _flat(fn(_t(src, "cpu").slice(1, 1, 3)))
        got = _flat(fn(_t(src, "cuda").slice(1, 1, 3)))
        assert got == pytest.approx(want, rel=1e-5, abs=1e-6)

    @pytest.mark.parametrize("name", ["sqrt", "reciprocal"])
    @requires_cuda
    def test_previously_silent_ops_are_no_longer_storage_order(self, name):
        """sqrt/reciprocal must produce LOGICAL-order results, never storage
        order.

        A transposed 2x3 view is not symmetric, so storage order and logical
        order differ in every interior position, making the two outcomes
        distinguishable.
        """
        fn = UNARY[name]
        got = _flat(fn(_t(POS, "cuda").transpose(0, 1)))
        logical = _flat(fn(_t(POS, "cpu").transpose(0, 1)))
        storage_order = _flat(fn(_t(POS, "cpu")))
        assert got == pytest.approx(logical, rel=1e-5)
        assert got != pytest.approx(storage_order, rel=1e-5)

    @pytest.mark.parametrize("name", sorted(UNARY))
    @requires_cuda
    def test_every_cuda_unary_rejects_an_unsupported_dtype_identically(self, name):
        """All nine unary wrappers reject an unsupported dtype loudly.

        Without a dtype check, cuda_sqrt's dispatch would fall through to the
        float32 kernel: a float64 CUDA tensor reinterpreted bit-for-bit with
        half its buffer written, returning [0.061, 0.125, 0.0, 0.0] for
        sqrt([1, 4, 9, 16]) - typed float64, no error.
        """
        x = ortho.from_numpy(np.asarray(POS, dtype=np.float64)).to("cuda")
        assert x.dtype == "float64"
        with pytest.raises(RuntimeError, match="float32, float16, or bfloat16"):
            UNARY[name](x)

    @pytest.mark.parametrize("name", sorted(UNARY))
    @requires_cuda
    def test_cuda_view_backward_returns_to_the_base(self, name):
        """Backward must route through the view node in LOGICAL coordinates.

        The saved input is the view itself, and the cuda_*_backward kernels
        index it linearly, so this also covers the backward-side normalisation.
        """
        src = UNARY_SOURCE[name]
        fn = UNARY[name]
        grads = {}
        for device in DEVICES:
            base = _t(src, device, grad=True)
            fn(base.transpose(0, 1)).sum().backward()
            assert base.grad.shape == base.shape
            assert base.grad.device == base.device
            grads[device] = _flat(base.grad)
        if "cuda" in grads:
            assert grads["cuda"] == pytest.approx(grads["cpu"], rel=1e-4, abs=1e-5)


# ---------------------------------------------------------------------------
# dropout on a CUDA view
# ---------------------------------------------------------------------------

class TestDropoutViews:
    @requires_cuda
    def test_dropout_accepts_a_cuda_view_with_no_kernel_change(self):
        got = _flat(ortho.dropout(_t(BASE, "cuda").transpose(0, 1), 0.5, 1234))
        want = _flat(ortho.dropout(_t(BASE, "cpu").transpose(0, 1), 0.5, 1234))
        assert got == pytest.approx(want, rel=1e-5)

    @requires_cuda
    def test_mask_is_keyed_to_logical_position_not_storage_position(self):
        """The mask has always been built over LOGICAL element index i.

        Which element gets dropped therefore depends only on (seed, i) - not on
        the operand's layout. Same seed, same drop pattern, view or not.
        """
        seed = 99
        contig_logical = _t(BASE.T, "cuda")             # same VALUES, contiguous
        view = _t(BASE, "cuda").transpose(0, 1)         # same values, strided
        assert _flat(ortho.dropout(view, 0.5, seed)) == pytest.approx(
            _flat(ortho.dropout(contig_logical, 0.5, seed)), rel=1e-5)

    @requires_cuda
    def test_seeded_reproducibility_and_stream_progression_are_unchanged(self):
        """A fixed module seed is deterministic, and repeated calls advance
        the mask. Accepting views must not perturb either."""
        ortho.manual_seed(4242)
        layer = nn.Dropout(0.5, seed=7)
        x = _t(BASE, "cuda").transpose(0, 1)
        first = _flat(layer(x))
        second = _flat(layer(x))
        layer2 = nn.Dropout(0.5, seed=7)
        assert _flat(layer2(x)) == pytest.approx(first, rel=1e-5)
        assert first != pytest.approx(second, rel=1e-5)

    @requires_cuda
    def test_dropout_view_backward_reaches_the_base(self):
        base = _t(BASE, "cuda", grad=True)
        ortho.dropout(base.transpose(0, 1), 0.5, 5).sum().backward()
        assert base.grad.shape == base.shape
        assert base.grad.device == base.device


# ---------------------------------------------------------------------------
# cast on views, and the unified same-dtype contract
# ---------------------------------------------------------------------------

CUDA_CAST_PAIRS = [("float32", "float16"), ("float32", "bfloat16"),
                   ("float16", "float32"), ("bfloat16", "float32")]


def _cuda_low(values, dtype):
    with ortho.no_grad():
        return ortho.cast(_t(values, "cuda"), dtype)


class TestCastViews:
    @pytest.mark.parametrize("pair", CUDA_CAST_PAIRS)
    @pytest.mark.parametrize("view", ["transpose", "slice"])
    @requires_cuda
    def test_real_cuda_cast_accepts_a_view(self, pair, view):
        src_dtype, dst_dtype = pair
        x = (_t(BASE, "cuda") if src_dtype == "float32"
             else _cuda_low(BASE, src_dtype))
        v = x.transpose(0, 1) if view == "transpose" else x.slice(1, 1, 3)
        z = ortho.cast(v, dst_dtype)
        assert z.dtype == dst_dtype
        assert z.shape == v.shape
        assert z.is_contiguous
        arr = BASE.T if view == "transpose" else BASE[:, 1:3]
        assert _flat(z) == pytest.approx(list(arr.reshape(-1)), abs=1e-2)

    @requires_cuda
    def test_cast_of_a_view_keeps_castnode_on_the_original(self):
        """Backward must reach the base through the view's own node, in
        logical coordinates - not through the internal contiguous copy."""
        base = _t(BASE, "cuda", grad=True)
        z = ortho.cast(base.transpose(0, 1), "float16")
        w = _cuda_low(np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]), "float16")
        (z * w).sum().backward()
        assert base.grad.dtype == "float32"
        assert base.grad.device == base.device
        # w is indexed transposed relative to base.
        assert _flat(base.grad) == pytest.approx([10.0, 30.0, 50.0, 20.0, 40.0, 60.0],
                                                 abs=1e-2)

    @requires_cuda
    def test_cast_of_a_slice_view_scatters_the_gradient(self):
        base = _t(BASE, "cuda", grad=True)
        z = ortho.cast(base.slice(1, 1, 3), "float16")
        w = _cuda_low(np.array([[1.0, 2.0], [4.0, 8.0]]), "float16")
        (z * w).sum().backward()
        assert _flat(base.grad) == pytest.approx([0.0, 1.0, 2.0, 0.0, 4.0, 8.0], abs=1e-2)


class TestSameDtypeCastContract:
    """One contract on both devices.

    ``cast(x, x.dtype)`` == ``contiguous(x)``. Not a copy API - callers wanting
    an independent physical copy use ``x.to(x.device)``, which is documented to
    copy and is differentiable.
    """

    @pytest.mark.parametrize("device", DEVICES)
    def test_contiguous_source_returns_the_same_object(self, device):
        x = _t(BASE, device, grad=True)
        y = ortho.cast(x, "float32")
        assert y._impl_id == x._impl_id
        assert y.requires_grad and y.is_leaf and not y._has_grad_fn

    @pytest.mark.parametrize("device", DEVICES)
    def test_contiguous_source_preserves_an_existing_grad_fn(self, device):
        x = _t(BASE, device, grad=True)
        mid = ortho.mul(x, _t(BASE, device))
        y = ortho.cast(mid, "float32")
        assert y._impl_id == mid._impl_id
        assert y._has_grad_fn and not y.is_leaf

    @pytest.mark.parametrize("device", DEVICES)
    def test_strided_source_is_contiguousnode_not_castnode(self, device):
        base = _t(BASE, device, grad=True)
        c = ortho.cast(base.transpose(0, 1), "float32")
        assert c._impl_id != base._impl_id
        assert c.is_contiguous
        assert c.requires_grad and c._has_grad_fn and not c.is_leaf
        assert _flat(c) == list(BASE.T.reshape(-1))
        c.sum().backward()
        assert _flat(base.grad) == [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_same_dtype_cast_is_not_a_copy_but_to_still_is(self, device):
        x = _t(BASE, device)
        assert ortho.cast(x, "float32")._impl_id == x._impl_id
        copy = x.to(x.device)
        assert copy._impl_id != x._impl_id
        copy.mul_(0.0)
        assert _flat(x) == list(BASE.reshape(-1))

    @requires_cuda
    @pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
    def test_same_dtype_contract_holds_for_low_precision_too(self, dtype):
        x = _cuda_low(BASE, dtype)
        assert ortho.cast(x, dtype)._impl_id == x._impl_id

    @pytest.mark.parametrize("device", DEVICES)
    def test_module_cast_to_the_current_dtype_keeps_parameter_identity(self, device):
        """The user-visible payoff of unifying this.

        ``Module.cast`` replaces each parameter with ``cast(p, dtype)``. On CUDA
        that used to hand back a fresh detached copy even when the dtype already
        matched, so a redundant ``.half()`` silently swapped every parameter
        object out from under any optimizer holding a reference.
        """
        model = nn.Linear(3, 2)
        if device != "cpu":
            model.to(device)
        before = [p._impl_id for p in model.parameters()]
        model.cast("float32")
        assert [p._impl_id for p in model.parameters()] == before


# ---------------------------------------------------------------------------
# exact CUDA device propagation
# ---------------------------------------------------------------------------

@requires_cuda
class TestExactDevicePropagationSingleGpu:
    """Single-GPU assertions read the EXACT device, not merely is_cuda().

    With one GPU these cannot distinguish a correct fix from a lucky cuda:0
    default, so they are regression tripwires for multi-GPU CI; the >=2 GPU
    class below is the real test and the source audit at the bottom is the
    load-bearing single-GPU evidence.
    """

    def _dev(self):
        return ortho.Device(ortho.DeviceType.CUDA, 0)

    def test_scalar_operand_carries_the_exact_device(self):
        """One check, instead of one per hand-written constant helper.

        A per-module helper that builds a small constant and hand-places it on
        the caller's device is where a bare "cuda" string resolves to cuda:0.
        The scalar contract carries the tensor's exact Device instead, so there
        is one place to check, and no future helper can reintroduce the bug at
        these call sites.
        """
        x = _t(BASE, "cuda")
        s = ortho._scalar_like(x, 0.5)
        assert str(s.device) == "device('cuda:0')"
        assert tuple(s.shape) == ()
        assert str((x * 0.5).device) == "device('cuda:0')"
        assert str((0.5 - x).device) == "device('cuda:0')"
        assert str((x / 2.0).device) == "device('cuda:0')"
        assert str((2.0 / x).device) == "device('cuda:0')"

    def test_embedding_indices_helper(self):
        from orthon.nn.embedding import _indices_tensor
        out = _indices_tensor([[0, 1], [1, 0]], self._dev())
        assert str(out.device) == "device('cuda:0')"

    def test_local_backend_payload_copy(self):
        from orthon.distributed.local_backend import _copy_payload_into_tensor
        dst = _t([0.0, 0.0, 0.0], "cuda")
        _copy_payload_into_tensor(((3,), "float32", [1.0, 2.0, 3.0]), dst)
        assert str(dst.device) == "device('cuda:0')"
        assert _flat(dst) == [1.0, 2.0, 3.0]

    def test_layer_norm_output_and_gradient(self):
        x = _t(BASE, "cuda", grad=True)
        y = ortho.layer_norm(x, (3,))
        assert str(y.device) == "device('cuda:0')"
        y.sum().backward()
        assert str(x.grad.device) == "device('cuda:0')"

    def test_attention_output_and_gradient(self):
        q = _t(BASE.reshape(1, 2, 3), "cuda", grad=True)
        k = _t(BASE.reshape(1, 2, 3), "cuda")
        v = _t(BASE.reshape(1, 2, 3), "cuda")
        out = ortho.scaled_dot_product_attention(q, k, v)
        assert str(out.device) == "device('cuda:0')"
        out.sum().backward()
        assert str(q.grad.device) == "device('cuda:0')"

    def test_embedding_module_output(self):
        emb = nn.Embedding(4, 2)
        emb.to("cuda")
        out = emb([[0, 1], [2, 3]])
        assert str(out.device) == "device('cuda:0')"


@requires_2_gpus
class TestExactDevicePropagationMultiGpu:
    """The real exact-device tests. They SKIP on this one-GPU machine."""

    # No helper-constant test here any more: it imported private helpers
    # (functional.math._scalar, nn.layernorm._constant_like_shape) that were
    # removed when those implementations moved to plain scalar arithmetic, so
    # it could never run - and the behaviour it wanted (internal constants on
    # the caller's exact device) is exactly what the layernorm/attention tests
    # below observe through the public API.

    def test_layernorm_on_cuda1(self):
        x = _t(BASE, "cuda:1", grad=True)
        y = ortho.layer_norm(x, (3,))
        assert str(y.device) == "device('cuda:1')"
        y.sum().backward()
        assert str(x.grad.device) == "device('cuda:1')"

    def test_functional_attention_on_cuda1(self):
        q = _t(BASE.reshape(1, 2, 3), "cuda:1", grad=True)
        k = _t(BASE.reshape(1, 2, 3), "cuda:1")
        v = _t(BASE.reshape(1, 2, 3), "cuda:1")
        out = ortho.scaled_dot_product_attention(q, k, v)
        assert str(out.device) == "device('cuda:1')"
        out.sum().backward()
        assert str(q.grad.device) == "device('cuda:1')"

    def test_embedding_indices_generated_for_a_cuda1_weight(self):
        emb = nn.Embedding(4, 2)
        emb.to("cuda:1")
        out = emb([[0, 1], [2, 3]])
        assert str(out.device) == "device('cuda:1')"

    def test_local_backend_payload_copy_into_cuda1(self):
        from orthon.distributed.local_backend import _copy_payload_into_tensor
        dst = _t([0.0, 0.0, 0.0], "cuda:1")
        _copy_payload_into_tensor(((3,), "float32", [1.0, 2.0, 3.0]), dst)
        assert str(dst.device) == "device('cuda:1')"
        assert _flat(dst) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# CUDA scalar mutation safety and version accounting
# ---------------------------------------------------------------------------

class TestScalarMutation:
    """The CUDA cases here fail as process kills, not exceptions, when they
    go wrong: 0xC0000005 for set_item, the full-index __setitem__ and
    get_item."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_set_item_writes_one_element(self, device):
        t = _t(np.zeros((2, 3)), device)
        t.set_item([1, 2], 7.5)
        assert _flat(t) == [0.0, 0.0, 0.0, 0.0, 0.0, 7.5]

    @pytest.mark.parametrize("device", DEVICES)
    def test_full_index_setitem_writes_one_element(self, device):
        t = _t(np.zeros((2, 3)), device)
        t[(0, 1)] = -3.25
        assert _flat(t) == [0.0, -3.25, 0.0, 0.0, 0.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_get_item_reads_one_element(self, device):
        t = _t(BASE, device)
        assert t.get_item([1, 2]) == pytest.approx(6.0)
        assert t.get_item([-1, -1]) == pytest.approx(6.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_scalar_access_respects_slice_view_strides(self, device):
        base = _t(np.zeros((2, 3)), device)
        view = base.slice(1, 1, 3)              # strided, storage_offset 1
        view.set_item([0, 1], 4.0)              # logical (0,1) -> base (0,2)
        view[(1, 0)] = 9.0                      # logical (1,0) -> base (1,1)
        assert _flat(base) == [0.0, 0.0, 4.0, 0.0, 9.0, 0.0]
        assert view.get_item([0, 1]) == pytest.approx(4.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_scalar_access_respects_transpose_view_strides(self, device):
        base = _t(np.zeros((2, 3)), device)
        view = base.transpose(0, 1)             # (3, 2)
        view.set_item([2, 0], 5.0)              # logical (2,0) -> base (0,2)
        view[(0, 1)] = 6.0                      # logical (0,1) -> base (1,0)
        assert _flat(base) == [0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
        assert view.get_item([2, 0]) == pytest.approx(5.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_successful_scalar_write_bumps_exactly_once(self, device):
        t = _t(np.zeros((2, 2)), device)
        assert t._storage_version == 0
        t.set_item([0, 0], 1.0)
        assert t._storage_version == 1
        t[(1, 1)] = 2.0
        assert t._storage_version == 2

    @pytest.mark.parametrize("device", DEVICES)
    def test_reads_never_bump(self, device):
        t = _t(BASE, device)
        t.get_item([0, 0])
        t.get_item([1, 1])
        assert t._storage_version == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_failed_scalar_write_does_not_bump(self, device):
        t = _t(np.zeros((2, 2)), device)
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            t.set_item([0, 5], 1.0)             # column index out of bounds
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            t[(9, 0)] = 1.0
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            t.set_item([0], 1.0)                # too few indices for a 2-D tensor
        assert t._storage_version == 0
        assert _flat(t) == [0.0, 0.0, 0.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_wrong_dtype_is_rejected_and_does_not_bump(self, device):
        t = ortho.from_numpy(np.zeros((2, 2), dtype=np.float64))
        if device != "cpu":
            t = t.to(device)
        with pytest.raises(RuntimeError, match="float32"):
            t.set_item([0, 0], 1.0)
        assert t._storage_version == 0

    @pytest.mark.parametrize("device", DEVICES)
    def test_scalar_write_through_a_view_bumps_the_base_storage(self, device):
        base = _t(np.zeros((2, 3)), device)
        view = base.slice(1, 1, 3)
        assert view._storage_version == base._storage_version == 0
        view.set_item([0, 0], 1.0)
        # Views share the Storage, so both aliases observe the one bump.
        assert base._storage_version == 1
        assert view._storage_version == 1


class TestPartialSetItem:
    @requires_cuda
    def test_partial_cuda_assignment_is_loudly_rejected(self):
        """Rejected explicitly, before any sub-view is built.

        The explicit check exists so the error names the actual limit of
        __setitem__ rather than surfacing "fill_: only CPU supported", which
        names an internal helper and reads like a missing dispatch. Supporting
        this would take a stride-aware CUDA fill.
        """
        t = _t(BASE, "cuda")
        with pytest.raises(RuntimeError,
                           match="partial CUDA assignment is not supported"):
            t[0] = 9.0
        assert t._storage_version == 0
        assert _flat(t) == list(BASE.reshape(-1))

    def test_cpu_partial_assignment_is_unchanged(self):
        t = _t(np.zeros((2, 3)), "cpu")
        t[0] = 9.0
        assert _flat(t) == [9.0, 9.0, 9.0, 0.0, 0.0, 0.0]
        assert t._storage_version == 1


# ---------------------------------------------------------------------------
# Versioning watches the ORIGINAL operand, not the materialised temporary
# ---------------------------------------------------------------------------

class TestMutationSelectivity:
    """The subtle mistake internal materialisation invites.

    If a CUDA op saved its internal contiguous COPY instead of the user's
    tensor, every test here would still produce correct forward values and
    correct gradients on an unmutated graph - and would silently stop detecting
    stale saved values. That is the whole point of this section.
    """

    @requires_cuda
    def test_mul_on_a_strided_operand_rejects_a_later_base_mutation(self):
        base = _t(BASE, "cuda", grad=True)
        view = base.transpose(0, 1)
        loss = ortho.mul(view, view).sum()
        base.mul_(3.0)                      # mutates the storage the view shares
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()

    @requires_cuda
    def test_mul_on_a_strided_operand_rejects_a_scalar_mutation(self):
        """Same rejection, driven by the CUDA scalar write."""
        base = _t(BASE, "cuda", grad=True)
        view = base.transpose(0, 1)
        loss = ortho.mul(view, view).sum()
        view.set_item([0, 0], 99.0)
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()

    @pytest.mark.parametrize("name", ["tanh", "sigmoid", "gelu", "exp",
                                      "relu", "leaky_relu", "sqrt", "reciprocal"])
    @requires_cuda
    def test_value_sensitive_activation_on_a_view_rejects_a_base_mutation(self, name):
        base = _t(POS, "cuda", grad=True)
        view = base.transpose(0, 1)
        loss = UNARY[name](view).sum()
        base.mul_(3.0)
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()

    @requires_cuda
    def test_neg_on_a_view_still_allows_a_base_mutation(self):
        """Neg's derivative is -1: value-INSENSITIVE, so no watch. Unchanged."""
        base = _t(BASE, "cuda", grad=True)
        loss = ortho.neg(base.transpose(0, 1)).sum()
        base.mul_(3.0)
        loss.backward()
        assert _flat(base.grad) == pytest.approx([-1.0] * 6)

    @requires_cuda
    def test_add_on_a_view_still_allows_a_base_mutation(self):
        """Add's derivative is 1: value-INSENSITIVE, so no watch. Unchanged."""
        base = _t(BASE, "cuda", grad=True)
        view = base.transpose(0, 1)
        loss = ortho.add(view, view).sum()
        base.mul_(3.0)
        loss.backward()
        assert _flat(base.grad) == pytest.approx([2.0] * 6)

    @requires_cuda
    def test_cast_of_a_view_still_allows_a_base_mutation(self):
        """CastNode is value-insensitive, and materialising the kernel
        operand must not accidentally introduce a watch."""
        base = _t(BASE, "cuda", grad=True)
        loss = ortho.cast(base.transpose(0, 1), "float16").sum()
        base.mul_(3.0)
        loss.backward()
        assert base.grad is not None

    @requires_cuda
    def test_fused_linear_relu_on_a_view_rejects_a_base_mutation(self):
        x = _t(np.array([[1.0, 2.0], [3.0, 4.0]]), "cuda", grad=True)
        w = _t(np.eye(2), "cuda", grad=True)
        b = _t(np.zeros(2), "cuda", grad=True)
        loss = ortho.fused_linear_relu(x.transpose(0, 1), w, b).sum()
        x.mul_(3.0)
        with pytest.raises(RuntimeError, match=STALE):
            loss.backward()


# ---------------------------------------------------------------------------
# public autograd graph shape survives internal normalisation
# ---------------------------------------------------------------------------

class TestPublicAutogradRegression:
    @pytest.mark.parametrize("device", DEVICES)
    def test_transpose_then_tanh_reaches_the_base(self, device):
        base = _t(MIX, device, grad=True)
        ortho.tanh(base.transpose(0, 1)).sum().backward()
        want = 1.0 - np.tanh(MIX) ** 2
        assert _flat(base.grad) == pytest.approx(list(want.reshape(-1)), abs=1e-5)

    @pytest.mark.parametrize("device", DEVICES)
    def test_slice_then_mul_scatters_correctly(self, device):
        base = _t(BASE, device, grad=True)
        other = _t(np.array([[10.0, 20.0], [30.0, 40.0]]), device)
        (base.slice(1, 1, 3) * other).sum().backward()
        assert _flat(base.grad) == pytest.approx([0.0, 10.0, 20.0, 0.0, 30.0, 40.0])

    @requires_cuda
    def test_transpose_then_cast_routes_through_the_transpose_node(self):
        base = _t(BASE, "cuda", grad=True)
        ortho.cast(base.transpose(0, 1), "float16").sum().backward()
        assert base.grad.shape == base.shape
        assert _flat(base.grad) == pytest.approx([1.0] * 6, abs=1e-2)

    @pytest.mark.parametrize("device", DEVICES)
    def test_deep_view_chain_matches_across_devices(self, device):
        base = _t(BASE, device, grad=True)
        y = ortho.relu(base.transpose(0, 1).slice(0, 1, 3))
        (y * y).sum().backward()
        grad_values = _flat(base.grad)
        assert base.grad.shape == base.shape
        assert grad_values == pytest.approx([0.0, 4.0, 6.0, 0.0, 10.0, 12.0], abs=1e-4)

    @requires_cuda
    def test_no_extra_graph_node_appears_for_a_contiguous_operand(self):
        """kernel_operand is identity for a contiguous tensor: same object in,
        same object out, so the common path allocates and attaches nothing."""
        x = _t(BASE, "cuda", grad=True)
        y = _t(BASE, "cuda")
        out = ortho.mul(x, y)
        assert out._has_grad_fn and not out.is_leaf
        assert out._impl_id != x._impl_id
        # A single MulNode: backward reaches the leaf in one hop with dL/dx = y.
        out.sum().backward()
        assert _flat(x.grad) == pytest.approx(list(BASE.reshape(-1)))

    @requires_cuda
    def test_grad_mode_off_still_suppresses_the_graph_for_views(self):
        x = _t(BASE, "cuda", grad=True)
        with ortho.no_grad():
            y = ortho.mul(x.transpose(0, 1), x.transpose(0, 1))
            z = ortho.tanh(x.transpose(0, 1))
            c = ortho.cast(x.transpose(0, 1), "float16")
        for t in (y, z, c):
            assert not t._has_grad_fn


# ---------------------------------------------------------------------------
# fused_linear_relu view parity
# ---------------------------------------------------------------------------

class TestFusedLinearReluViews:
    @requires_cuda
    def test_strided_x_matches_cpu(self):
        x_np = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        w_np = np.array([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
        b_np = np.array([0.1, -0.2], dtype=np.float32)
        want = _flat(ortho.fused_linear_relu(_t(x_np, "cpu").transpose(0, 1),
                                          _t(w_np, "cpu"), _t(b_np, "cpu")))
        got = _flat(ortho.fused_linear_relu(_t(x_np, "cuda").transpose(0, 1),
                                         _t(w_np, "cuda"), _t(b_np, "cuda")))
        assert got == pytest.approx(want, abs=1e-5)

    @requires_cuda
    def test_strided_weight_matches_cpu(self):
        x_np = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        w_np = np.array([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
        b_np = np.array([0.1, -0.2], dtype=np.float32)
        want = _flat(ortho.fused_linear_relu(_t(x_np, "cpu"),
                                          _t(w_np, "cpu").transpose(0, 1),
                                          _t(b_np, "cpu")))
        got = _flat(ortho.fused_linear_relu(_t(x_np, "cuda"),
                                         _t(w_np, "cuda").transpose(0, 1),
                                         _t(b_np, "cuda")))
        assert got == pytest.approx(want, abs=1e-5)

    @requires_cuda
    def test_strided_x_gradient_matches_cpu(self):
        x_np = np.array([[1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
        w_np = np.array([[1.0, 0.5], [-0.25, 2.0]], dtype=np.float32)
        b_np = np.array([0.1, -0.2], dtype=np.float32)
        grads = {}
        for device in DEVICES:
            x = _t(x_np, device, grad=True)
            w = _t(w_np, device, grad=True)
            b = _t(b_np, device, grad=True)
            ortho.fused_linear_relu(x.transpose(0, 1), w, b).sum().backward()
            grads[device] = (_flat(x.grad), _flat(w.grad), _flat(b.grad))
        if "cuda" in grads:
            for got, want in zip(grads["cuda"], grads["cpu"]):
                assert got == pytest.approx(want, abs=1e-5)


# ---------------------------------------------------------------------------
# Embedding: contiguity precondition is now device-symmetric
# ---------------------------------------------------------------------------

class TestEmbeddingContiguity:
    """The precondition is stated once, for both devices. Enforcing it only
    on CPU lets CUDA accept strided indices and read them in storage order,
    returning the wrong rows with no error; the weight carries the same
    linear-indexing assumption on both devices."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_strided_indices_are_rejected_identically(self, device):
        idx = ortho.from_numpy(np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int64))
        w = _t(np.arange(8, dtype=np.float32).reshape(4, 2), device)
        if device != "cpu":
            idx = idx.to(device)
        with pytest.raises(RuntimeError, match="indices must be contiguous"):
            ortho.embedding(idx.transpose(0, 1), w, None)

    @pytest.mark.parametrize("device", DEVICES)
    def test_strided_weight_is_rejected_identically(self, device):
        idx = ortho.from_numpy(np.array([0, 1], dtype=np.int64))
        w = _t(np.array([[1.0, 2.0], [3.0, 4.0]]), device)
        if device != "cpu":
            idx = idx.to(device)
        with pytest.raises(RuntimeError, match="weight must be contiguous"):
            ortho.embedding(idx, w.transpose(0, 1), None)

    @pytest.mark.parametrize("device", DEVICES)
    def test_contiguous_embedding_is_unaffected(self, device):
        idx = ortho.from_numpy(np.array([[0, 1], [1, 0]], dtype=np.int64))
        w = _t(np.array([[1.0, 2.0], [10.0, 20.0]]), device)
        if device != "cpu":
            idx = idx.to(device)
        assert _flat(ortho.embedding(idx, w, None)) == [1.0, 2.0, 10.0, 20.0,
                                                     10.0, 20.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# cross-GPU Tensor::to
# ---------------------------------------------------------------------------

@requires_2_gpus
class TestCrossGpuTransfer:
    """These SKIP on this one-GPU machine. The mechanism change is asserted at
    source level in TestSourceAudit instead."""

    def test_cuda0_to_cuda1_preserves_values_and_device(self):
        x = _t(BASE, "cuda:0")
        y = x.to(ortho.Device(ortho.DeviceType.CUDA, 1))
        assert str(y.device) == "device('cuda:1')"
        assert _flat(y) == list(BASE.reshape(-1))

    def test_cuda1_to_cuda0_preserves_values_and_device(self):
        x = _t(BASE, "cuda:1")
        y = x.to(ortho.Device(ortho.DeviceType.CUDA, 0))
        assert str(y.device) == "device('cuda:0')"
        assert _flat(y) == list(BASE.reshape(-1))

    def test_cross_gpu_transfer_makes_an_independent_copy(self):
        x = _t(BASE, "cuda:0")
        y = x.to(ortho.Device(ortho.DeviceType.CUDA, 1))
        y.mul_(0.0)
        assert _flat(x) == list(BASE.reshape(-1))

    def test_tracked_cross_gpu_backward_returns_to_the_original_gpu(self):
        x = _t(BASE, "cuda:0", grad=True)
        y = x.to(ortho.Device(ortho.DeviceType.CUDA, 1))
        (y * _t(BASE, "cuda:1")).sum().backward()
        assert str(x.grad.device) == "device('cuda:0')"
        assert _flat(x.grad) == pytest.approx(list(BASE.reshape(-1)))

    def test_a_strided_source_crosses_gpus_in_logical_order(self):
        x = _t(BASE, "cuda:0")
        y = x.transpose(0, 1).to(ortho.Device(ortho.DeviceType.CUDA, 1))
        assert _flat(y) == list(BASE.T.reshape(-1))


# ---------------------------------------------------------------------------
# Source-level audit (the load-bearing single-GPU evidence)
# ---------------------------------------------------------------------------

class TestSourceAudit:
    def test_kernel_operand_is_the_single_normalisation_helper(self):
        body = _strip_c_comments(_read("src/ops/cpu/shape_ops.cpp"))
        impl = body[body.index("Tensor kernel_operand(const Tensor& t)"):]
        impl = impl[:impl.index("Tensor transpose(")]
        assert "if (t.is_contiguous()) return t;" in impl
        assert "NoGradGuard guard;" in impl
        assert "return t.to(t.device());" in impl
        # It must not attach a public node.
        assert "ContiguousNode" not in impl
        assert "set_grad_fn" not in impl

    @pytest.mark.parametrize("rel,call,node_args", [
        ("src/ops/cpu/binary_ops.cpp", "cuda_add(kernel_operand(a), kernel_operand(b))",
         "AddNode>(a, b)"),
        ("src/ops/cpu/binary_ops.cpp", "cuda_mul(kernel_operand(a), kernel_operand(b))",
         "MulNode>(a, b)"),
    ])
    def test_binary_cuda_sites_normalise_operands_but_save_originals(
            self, rel, call, node_args):
        body = _strip_c_comments(_read(rel))
        assert call in body
        assert node_args in body

    @pytest.mark.parametrize("fn", ["cuda_neg", "cuda_relu", "cuda_sigmoid",
                                    "cuda_tanh", "cuda_leaky_relu", "cuda_gelu",
                                    "cuda_exp", "cuda_sqrt", "cuda_reciprocal"])
    def test_every_cuda_unary_forward_normalises_its_operand(self, fn):
        body = _strip_c_comments(_read("src/ops/cpu/unary_ops.cpp"))
        for m in re.finditer(r"(?<!\w)" + re.escape(fn) + r"\s*\(", body):
            tail = body[m.end():m.end() + 60]
            assert tail.startswith("kernel_operand("), (
                "%s is called without kernel_operand: ...%s" % (fn, tail[:40]))

    @pytest.mark.parametrize("fn", ["cuda_sigmoid_backward", "cuda_tanh_backward",
                                    "cuda_leaky_relu_backward", "cuda_gelu_backward",
                                    "cuda_exp_backward", "cuda_sqrt_backward",
                                    "cuda_reciprocal_backward"])
    def test_every_cuda_unary_backward_normalises_its_saved_input(self, fn):
        body = _strip_c_comments(_read("src/ops/cpu/unary_ops.cpp"))
        for m in re.finditer(r"(?<!\w)" + re.escape(fn) + r"\s*\(", body):
            tail = body[m.end():m.end() + 60]
            assert tail.startswith("kernel_operand("), fn

    def test_no_cuda_forward_wrapper_still_rejects_a_view(self):
        """The "CUDA path requires a contiguous tensor" guards are gone."""
        body = _strip_c_comments(_read("src/ops/cpu/unary_ops.cpp"))
        assert "CUDA path requires a contiguous tensor" not in body

    def test_cuda_kernels_were_not_taught_about_strides(self):
        """The design constraint: normalise at the op, not in the kernel.

        Each of these raw kernel wrappers must KEEP its contiguous-only
        precondition. Relaxing them instead of normalising at the public op
        would push layout handling into every optimised kernel.
        """
        for rel in ("src/ops/cuda/cast_kernels.cu",
                    "src/ops/cuda/fusion_kernels.cu",
                    "src/ops/cuda/binary_kernels.cu"):
            body = _strip_c_comments(_read(rel))
            assert "is_contiguous()" in body, (
                "%s must keep its contiguous-only precondition" % rel)

    def test_cast_resolves_same_dtype_before_device_dispatch(self):
        body = _strip_c_comments(_read("src/ops/cpu/cast_ops.cpp"))
        fn = body[body.index("Tensor cast(const Tensor& src, DType dst_dtype)"):]
        same = fn.index("if (src.dtype() == dst_dtype)")
        dispatch = fn.index("src.device().is_cuda()")
        assert same < dispatch, "same-dtype must short-circuit before dispatch"
        assert "return contiguous(src);" in fn[same:dispatch]
        # ...and the old CPU-only qualifier is gone.
        assert "src.dtype() == dst_dtype && src.device().is_cpu()" not in fn

    def test_cast_normalises_the_cuda_kernel_operand(self):
        body = _strip_c_comments(_read("src/ops/cpu/cast_ops.cpp"))
        assert "cuda_cast(kernel_operand(src), dst_dtype)" in body

    def test_fused_linear_relu_normalises_all_three_cuda_operands(self):
        body = _strip_c_comments(_read("src/ops/cpu/fusion_ops.cpp"))
        assert ("cuda_fused_linear_relu(\n            kernel_operand(x), "
                "kernel_operand(weight), kernel_operand(bias))") in body
        assert "attach_fused_linear_relu_grad_fn(out, x, weight, bias)" in body

    def test_scalar_accessors_never_dereference_device_memory_on_the_host(self):
        body = _strip_c_comments(_read("bindings/tensor_bindings.cpp"))
        # The three offending expressions are gone...
        assert "static_cast<float*>(t.data())[multi_index_offset" not in body
        assert "static_cast<const float*>(t.data())[multi_index_offset" not in body
        # ...replaced by helpers that branch on the device.
        for helper in ("write_scalar_float32", "read_scalar_float32"):
            impl = body[body.index("static " + ("void " if helper.startswith("write")
                                                else "float ") + helper):]
            impl = impl[:impl.index("\n}\n")]
            assert "t.device().is_cuda()" in impl
            assert "multi_index_offset(t, indices)" in impl
        cuda_helper = body[body.index("static void cuda_copy_scalar_float32"):]
        cuda_helper = cuda_helper[:cuda_helper.index("\n}\n")]
        assert "DeviceGuard guard(t.device())" in cuda_helper
        assert "cudaMemcpyHostToDevice" in cuda_helper
        assert "cudaMemcpyDeviceToHost" in cuda_helper

    def test_scalar_write_bumps_once_after_the_copy(self):
        body = _strip_c_comments(_read("bindings/tensor_bindings.cpp"))
        impl = body[body.index("static void write_scalar_float32"):]
        impl = impl[:impl.index("\n}\n")]
        assert impl.count("bump_version()") == 1
        assert impl.index("bump_version()") > impl.index("cuda_copy_scalar_float32")

    def test_partial_setitem_rejects_cuda_before_reaching_fill(self):
        body = _strip_c_comments(_read("bindings/tensor_bindings.cpp"))
        setitem = body[body.index('.def("__setitem__"'):]
        # End the slice at the next binding entry so the window is exactly the
        # __setitem__ lambda, not an arbitrary character count.
        setitem = setitem[:setitem.index("\n        .def_property_readonly(")]
        reject = setitem.index("partial CUDA assignment is not supported")
        fill = setitem.index("fill_(sub, value)")
        navigate = setitem.index("sub = sub[i]")
        assert reject < navigate < fill

    def test_cross_gpu_transfer_uses_an_explicit_peer_copy(self):
        body = _strip_c_comments(_read("src/core/tensor.cpp"))
        to_body = body[body.index("static Tensor to_physical"):]
        to_body = to_body[:to_body.index("Tensor Tensor::to(Device target)")]
        assert "cudaMemcpyPeer(" in to_body
        assert "cudaMemcpyDeviceToDevice" not in to_body
        peer = to_body[to_body.index("cudaMemcpyPeer("):]
        peer = peer[:peer.index(";")]
        # Both ordinals named explicitly, source and destination not swapped.
        assert "out.data(), target.index" in peer
        assert "src_c.data(), src_c.device().index" in peer
        # No permanent peer-access enabling was introduced as a side effect.
        assert "cudaDeviceEnablePeerAccess" not in body

    @pytest.mark.parametrize("rel,fn", [
        ("python/orthon/nn/embedding.py", "_indices_tensor"),
        ("python/orthon/distributed/local_backend.py",
         "_copy_payload_into_tensor"),
    ])
    def test_device_sensitive_helpers_do_not_collapse_to_cuda0(self, rel, fn):
        """The category "device-sensitive helper still collapses to cuda:0"
        must stay empty. Docstrings quoting the old form are stripped first."""
        body = _strip_py_comments_and_docstrings(_read(rel))
        start = body.index("def " + fn + "(")
        tail = body[start:]
        end = tail.index("\ndef ") if "\ndef " in tail else len(tail)
        impl = tail[:end]
        assert '.to("cuda")' not in impl and ".to('cuda')" not in impl, (
            "%s:%s still moves through the generic 'cuda' string" % (rel, fn))
        assert ".to(" in impl

    def test_scalar_constant_helpers_were_removed_not_merely_fixed(self):
        """The constant-placing helpers do not exist at all.

        functional/math.py:_scalar, nn/layernorm.py:_constant_like_shape,
        nn/batchnorm.py:_constant_like and the two _negate helpers each existed
        to build a small constant tensor and hand-place it on the caller's
        exact device - the exact spot where a bare "cuda" string resolves to
        cuda:0. Scalar arithmetic carries dtype and exact Device by contract,
        so the failure mode is unreachable at those call sites rather than
        merely corrected in each helper.
        """
        removed = [
            ("python/orthon/functional/math.py", "_scalar"),
            ("python/orthon/nn/layernorm.py", "_constant_like_shape"),
            ("python/orthon/nn/layernorm.py", "_negate"),
            ("python/orthon/nn/batchnorm.py", "_constant_like"),
            ("python/orthon/nn/batchnorm.py", "_negate"),
        ]
        for rel, fn in removed:
            body = _strip_py_comments_and_docstrings(_read(rel))
            assert ("def " + fn + "(") not in body, (
                "%s still defines %s" % (rel, fn))
            assert (fn + "(") not in body, "%s still calls %s" % (rel, fn)

    def test_intentional_generic_cuda_meanings_are_preserved(self):
        """Not everything spelled "cuda" is a bug. These stay."""
        device_cpp = _strip_c_comments(_read("src/device/device.cpp"))
        assert 'if (s == "cuda")' in device_cpp        # parser: "cuda" == cuda:0
        nccl = _strip_py_comments_and_docstrings(
            _read("python/orthon/distributed/nccl_backend.py"))
        assert 'if device == "cuda":' in nccl          # backend: rank-0 default

    def test_cuda_storage_allocates_on_its_exact_device(self):
        """Device memory never routes through the Allocator interface.

        Allocator serves host memory only. Storage calls get_allocator() inside
        an is_cpu() branch and sends every CUDA allocation straight to
        memory_pool_allocate(n_bytes, device) with the tensor's ACTUAL device,
        so a cuda:N tensor cannot end up allocated on cuda:0.
        """
        storage = _strip_c_comments(_read("src/core/storage.cpp"))
        ctor = storage[storage.index("Storage::Storage("):storage.index("Storage::~Storage")]
        assert "if (device.is_cpu()) {\n        alloc = get_allocator(device);" in ctor
        assert "memory_pool_allocate(n_bytes, device)" in ctor

        # There is no device-side Allocator implementation to fall into.
        alloc_cpp = _strip_c_comments(_read("src/memory/allocator.cpp"))
        assert "CPUAllocator cpu_allocator" in alloc_cpp
        assert "CUDAAllocator" not in alloc_cpp
        for rel in ("include/orthon/memory/cuda_allocator.h",
                    "src/memory/cuda_allocator.cpp"):
            assert not os.path.exists(os.path.join(REPO, rel)), rel

        # Nothing outside the allocator layer itself calls get_allocator.
        callers = []
        for root, _, files in os.walk(os.path.join(REPO, "src")):
            for name in files:
                if not name.endswith((".cpp", ".cu")):
                    continue
                rel = os.path.relpath(os.path.join(root, name), REPO).replace("\\", "/")
                if rel == "src/memory/allocator.cpp":
                    continue
                if "get_allocator(" in _strip_c_comments(_read(rel)):
                    callers.append(rel)
        assert callers == ["src/core/storage.cpp"], callers

    @requires_cuda
    def test_cuda_allocator_request_is_refused_rather_than_silently_served(self):
        """A CUDA Allocator request is a programming error, not a fallback."""
        alloc_cpp = _strip_c_comments(_read("src/memory/allocator.cpp"))
        assert "throw std::runtime_error(" in alloc_cpp
        # And real CUDA storage still lands on the requested device.
        t = ortho.zeros([4, 4]).to("cuda")
        assert t.device == ortho.Device("cuda:0")
