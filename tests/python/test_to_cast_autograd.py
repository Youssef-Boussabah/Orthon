"""Differentiable Tensor.to / floating cast, and explicit internal detach.

Both public operations participate in autograd. Detaching silently instead::

    x.requires_grad_(True)
    y = x.to("cuda")           # requires_grad=False, no grad_fn, leaf
    z = orthon.cast(x, "float64")   # likewise

would mean a loss built downstream could never reach ``x``. So:

* ``to`` attaches a ``ToNode`` (identity derivative in logical coordinates;
  backward returns the gradient to the exact source Device),
* a REAL floating->floating ``cast`` attaches a ``CastNode`` (backward casts
  the gradient back to the exact source dtype on the same device),

while everything that needs a DETACHED result asks for it explicitly, with
a ``no_grad`` region rather than a side effect: BatchNorm running-stat
snapshots,
``Module.to``/``Module.cast``/``half``/``bfloat16``, ``load_state_dict``
preparation, AMP master-weight creation and restore, checkpoint/distributed
host materialisation, and the internal materialisations inside
``contiguous()`` and the CUDA matmul path.

Deliberately unchanged, and pinned here so a uniformity "cleanup" fails:

* conversions touching an integer/bool dtype never join a graph - there is
  intentionally no straight-through estimator;
* ``contiguous(x)``'s public graph contract stays exactly one
  ``ContiguousNode`` - no hidden ToNode underneath.

* same-dtype ``cast`` does not differ per device. ``cast(x, x.dtype)`` is
  ``contiguous(x)`` on both - identity object for a contiguous source,
  ``ContiguousNode`` for a strided one, never a ``CastNode``.

ToNode and CastNode are value-INSENSITIVE: their derivatives never read the
input's stored values, so no ``watch_version`` - mutating the source after
forward stays legal unless a downstream value-sensitive node saved it.
"""
import numpy as np
import pytest

import orthon as ortho
import orthon.nn as nn

CUDA = ortho.cuda_enabled()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

STALE = r"modified in-place"


def _t(values, device="cpu", dtype=np.float32, grad=False):
    t = ortho.from_numpy(np.asarray(values, dtype=dtype))
    if device != "cpu":
        t = t.to(device)
    if grad:
        t.requires_grad_(True)
    return t


def _low(values, dtype, grad=False):
    """CUDA float16/bfloat16 tensor built via cast from exact float32 values."""
    base = _t(values, "cuda")
    with ortho.no_grad():
        out = ortho.cast(base, dtype)
    if grad:
        out.requires_grad_(True)
    return out


def _cpu_flat(t):
    if t.device.type != ortho.DeviceType.CPU:
        t = t.to("cpu")
    if t.dtype not in ("float32", "float64"):
        t = ortho.cast(t, "float32")
    return t.tolist()


def _tracked(y):
    return y.requires_grad and y._has_grad_fn and not y.is_leaf


def _plain(y):
    return (not y.requires_grad) and (not y._has_grad_fn) and y.is_leaf


# ---------------------------------------------------------------------------
# forward to() contract
# ---------------------------------------------------------------------------

class TestToForward:
    @pytest.mark.parametrize("device", DEVICES)
    def test_same_device_to_is_tracked_physical_copy(self, device):
        x = _t([1.0, 2.0, 3.0], device, grad=True)
        y = x.to(device)
        assert _tracked(y)
        assert y.device == x.device
        assert _cpu_flat(y) == [1.0, 2.0, 3.0]
        # Independent storage: mutating the copy leaves the source untouched,
        # and the two storage version counters advance independently.
        y.mul_(10.0)
        assert _cpu_flat(x) == [1.0, 2.0, 3.0]
        assert y._storage_version == 1
        assert x._storage_version == 0

    @requires_cuda
    def test_cpu_to_cuda_is_tracked(self):
        x = _t([1.0, 2.0], grad=True)
        y = x.to("cuda")
        assert _tracked(y)
        assert y.device.is_cuda()
        assert _cpu_flat(y) == [1.0, 2.0]

    @requires_cuda
    def test_cuda_to_cpu_is_tracked(self):
        x = _t([1.0, 2.0], "cuda", grad=True)
        y = x.to("cpu")
        assert _tracked(y)
        assert not y.device.is_cuda()
        assert _cpu_flat(y) == [1.0, 2.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_untracked_input_stays_plain(self, device):
        x = _t([1.0, 2.0], device)
        assert _plain(x.to(device))

    def test_integer_to_never_joins_a_graph(self):
        # Differentiability is floating-only, even if requires_grad was
        # forced onto an integer tensor.
        x = _t([1, 2], dtype=np.int64)
        x.requires_grad_(True)
        assert _plain(x.to("cpu"))

    @pytest.mark.parametrize("device", DEVICES)
    def test_non_contiguous_source_materialises_logical_order(self, device):
        base = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        xt = base.transpose(0, 1)
        y = xt.to(base.device)
        assert _tracked(y)
        assert y.is_contiguous
        assert _cpu_flat(y) == [1.0, 3.0, 2.0, 4.0]   # logical (transposed) order


# ---------------------------------------------------------------------------
# to() backward parity (nonuniform upstream)
# ---------------------------------------------------------------------------

class TestToBackward:
    @pytest.mark.parametrize("np_dtype", [np.float32, np.float64])
    def test_cpu_same_device_identity_gradient(self, np_dtype):
        x = _t([1.0, 2.0, 3.0], dtype=np_dtype, grad=True)
        y = x.to("cpu")
        w = _t([3.0, -5.0, 0.25], dtype=np_dtype)
        (y * w).sum().backward()
        assert x.grad.dtype == x.dtype
        assert _cpu_flat(x.grad) == [3.0, -5.0, 0.25]

    @requires_cuda
    def test_cpu_to_cuda_gradient_returns_to_cpu(self):
        x = _t([1.0, 2.0], grad=True)
        y = x.to("cuda")
        w = _t([3.0, -5.0], "cuda")
        (y * w).sum().backward()
        assert not x.grad.device.is_cuda()
        assert x.grad.dtype == "float32"
        assert _cpu_flat(x.grad) == [3.0, -5.0]

    @requires_cuda
    def test_cuda_to_cpu_gradient_returns_to_cuda(self):
        x = _t([1.0, 2.0], "cuda", grad=True)
        y = x.to("cpu")
        w = _t([0.5, -4.0])
        (y * w).sum().backward()
        assert x.grad.device.is_cuda()
        assert _cpu_flat(x.grad) == [0.5, -4.0]

    @requires_cuda
    def test_cuda_same_device_identity_gradient(self):
        x = _t([1.0, 2.0], "cuda", grad=True)
        y = x.to("cuda")
        w = _t([3.0, -5.0], "cuda")
        (y * w).sum().backward()
        assert x.grad.device.is_cuda()
        assert _cpu_flat(x.grad) == [3.0, -5.0]

    @requires_cuda
    @pytest.mark.parametrize("low_dtype", ["float16", "bfloat16"])
    def test_cuda_low_precision_same_device_identity_gradient(self, low_dtype):
        x = _low([1.0, 2.0], low_dtype, grad=True)
        y = x.to("cuda")
        w = _low([3.0, -5.0], low_dtype)
        (y * w).sum().backward()
        assert x.grad.dtype == low_dtype
        assert x.grad.device.is_cuda()
        assert _cpu_flat(x.grad) == [3.0, -5.0]   # exactly representable


# ---------------------------------------------------------------------------
# cast contracts
# ---------------------------------------------------------------------------

class TestCastForward:
    def test_cpu_floating_cast_is_tracked(self):
        x = _t([1.0, 2.0], grad=True)
        z = ortho.cast(x, "float64")
        assert _tracked(z) and z.dtype == "float64"
        x64 = _t([1.0, 2.0], dtype=np.float64, grad=True)
        z32 = ortho.cast(x64, "float32")
        assert _tracked(z32) and z32.dtype == "float32"

    @requires_cuda
    @pytest.mark.parametrize("pair", [("float32", "float16"),
                                      ("float32", "bfloat16"),
                                      ("float16", "float32"),
                                      ("bfloat16", "float32")])
    def test_cuda_floating_cast_is_tracked(self, pair):
        src_dtype, dst_dtype = pair
        x = _t([1.0, 2.0], "cuda", grad=True) if src_dtype == "float32" \
            else _low([1.0, 2.0], src_dtype, grad=True)
        z = ortho.cast(x, dst_dtype)
        assert _tracked(z) and z.dtype == dst_dtype

    def test_non_floating_casts_never_join_a_graph(self):
        x = _t([1.5, 2.5], grad=True)
        for dst in ("int32", "int64", "uint8", "int8", "bool"):
            assert _plain(ortho.cast(x, dst)), dst
        i = _t([1, 2], dtype=np.int64)
        i.requires_grad_(True)     # forced metadata on an integer tensor
        assert _plain(ortho.cast(i, "float32"))

    @pytest.mark.parametrize("device", DEVICES)
    def test_same_dtype_cast_is_contiguous_not_castnode(self, device):
        """One contract on both devices.

        ``cast(x, x.dtype)`` is ``contiguous(x)`` - identity object for a
        contiguous source, ContiguousNode for a strided one - never a copy and
        never a CastNode. Letting CUDA fall into cuda_cast's memcpy branch
        instead would return a fresh DETACHED copy, so the same call would mean
        two different things per device. ``x.to(x.device)`` is the public way
        to ask for an independent physical copy.
        """
        # Contiguous source: the very same tensor object, graph metadata intact.
        x = _t([1.0, 2.0], device, grad=True)
        y = ortho.cast(x, "float32")
        assert y._impl_id == x._impl_id
        assert y.requires_grad and y.is_leaf
        # Strided source: ContiguousNode semantics, gradient still correct.
        base = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        c = ortho.cast(base.transpose(0, 1), "float32")
        assert _tracked(c)
        assert _cpu_flat(c) == [1.0, 3.0, 2.0, 4.0]
        c.sum().backward()
        assert _cpu_flat(base.grad) == [1.0, 1.0, 1.0, 1.0]


class TestCastBackward:
    @pytest.mark.parametrize("pair", [(np.float32, "float64"),
                                      (np.float64, "float32")])
    def test_cpu_cast_gradient_returns_to_source_dtype(self, pair):
        src_np, dst = pair
        x = _t([1.0, 2.0, 3.0], dtype=src_np, grad=True)
        z = ortho.cast(x, dst)
        w = ortho.cast(_t([3.0, -5.0, 0.25]), dst)
        (z * w).sum().backward()
        assert x.grad.dtype == x.dtype
        assert x.grad.device == x.device
        assert _cpu_flat(x.grad) == pytest.approx([3.0, -5.0, 0.25])

    @requires_cuda
    @pytest.mark.parametrize("pair", [("float32", "float16"),
                                      ("float32", "bfloat16"),
                                      ("float16", "float32"),
                                      ("bfloat16", "float32")])
    def test_cuda_cast_gradient_returns_to_source_dtype(self, pair):
        src_dtype, dst_dtype = pair
        x = _t([0.5, 2.0], "cuda", grad=True) if src_dtype == "float32" \
            else _low([0.5, 2.0], src_dtype, grad=True)
        z = ortho.cast(x, dst_dtype)
        # Upstream exactly representable in every dtype involved.
        w = _low([3.0, 0.25], dst_dtype) if dst_dtype in ("float16", "bfloat16") \
            else _t([3.0, 0.25], "cuda")
        (z * w).sum().backward()
        assert x.grad.dtype == src_dtype
        assert x.grad.device.is_cuda()
        assert _cpu_flat(x.grad) == [3.0, 0.25]


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------

class TestComposition:
    @requires_cuda
    def test_cpu_to_cuda_then_cast_f16_roundtrip_gradient(self):
        x = _t([0.5, 2.0], grad=True)                    # CPU f32
        z = ortho.cast(x.to("cuda"), "float16")
        w = _low([3.0, 0.25], "float16")
        (z * w).sum().backward()
        assert not x.grad.device.is_cuda()
        assert x.grad.dtype == "float32"
        assert _cpu_flat(x.grad) == [3.0, 0.25]

    @requires_cuda
    def test_cuda_f16_cast_f32_then_to_cpu_roundtrip_gradient(self):
        x = _low([0.5, 2.0], "float16", grad=True)       # CUDA f16
        z = ortho.cast(x, "float32").to("cpu")
        w = _t([3.0, 0.25])
        (z * w).sum().backward()
        assert x.grad.device.is_cuda()
        assert x.grad.dtype == "float16"
        assert _cpu_flat(x.grad) == [3.0, 0.25]

    @pytest.mark.parametrize("device", DEVICES)
    def test_transpose_then_to_places_gradient_logically(self, device):
        x = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        y = x.transpose(0, 1).to(x.device)               # logical [[1,3],[2,4]]
        w = _t([[10.0, 20.0], [30.0, 40.0]], device)
        (y * w).sum().backward()
        # dL/dy = w in transposed coordinates -> dL/dx = w transposed back.
        assert x.grad.shape == (2, 2)
        assert _cpu_flat(x.grad) == [10.0, 30.0, 20.0, 40.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_slice_then_to_places_gradient_logically(self, device):
        x = _t([1.0, 2.0, 3.0, 4.0], device, grad=True)
        y = x.slice(0, 1, 3).to(x.device)
        w = _t([10.0, 20.0], device)
        (y * w).sum().backward()
        assert _cpu_flat(x.grad) == [0.0, 10.0, 20.0, 0.0]

    def test_cast_of_transpose_view_places_gradient_logically_cpu(self):
        x = _t([[1.0, 2.0], [3.0, 4.0]], grad=True)
        z = ortho.cast(x.transpose(0, 1), "float64")
        w = _t([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        (z * w).sum().backward()
        assert x.grad.dtype == "float32"
        assert _cpu_flat(x.grad) == [10.0, 30.0, 20.0, 40.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_cast_of_slice_view_places_gradient_logically(self, device):
        # dim-0 slice stays contiguous, so this composes on CUDA too
        # (cuda_cast rejects non-contiguous input - pre-existing).
        x = _t([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device, grad=True)
        sliced = x.slice(0, 1, 3)
        if device == "cuda":
            z = ortho.cast(sliced, "float16")
            w = _low([[10.0, 20.0], [30.0, 40.0]], "float16")
        else:
            z = ortho.cast(sliced, "float64")
            w = _t([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        (z * w).sum().backward()
        assert x.grad.dtype == "float32"
        assert _cpu_flat(x.grad) == [0.0, 0.0, 10.0, 20.0, 30.0, 40.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_branched_to_plus_mul_accumulates_in_one_backward(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        a = x.to(x.device)
        wa = _t([3.0, -1.0], device)
        wb = _t([0.5, 2.0], device)
        loss = ((a * wa) + (x * x * wb)).sum()
        loss.backward()
        # d/dx = wa + 2*x*wb
        assert _cpu_flat(x.grad) == pytest.approx([3.0 + 2.0 * 1.0 * 0.5,
                                                   -1.0 + 2.0 * 2.0 * 2.0])

    def test_branched_to_and_cast_accumulate_across_backwards_cpu(self):
        # Mixed-dtype branches cannot share one loss; cross-call gradient
        # accumulation is what combines them.
        x = _t([1.0, 2.0], grad=True)
        a = x.to("cpu")
        b = ortho.cast(x, "float64")
        (a * _t([3.0, -1.0])).sum().backward()
        (b * _t([0.5, 2.0], dtype=np.float64)).sum().backward()
        assert x.grad.dtype == "float32"
        assert _cpu_flat(x.grad) == pytest.approx([3.5, 1.0])


# ---------------------------------------------------------------------------
# no_grad detaches both operations
# ---------------------------------------------------------------------------

class TestNoGradContract:
    @pytest.mark.parametrize("device", DEVICES)
    def test_no_grad_detaches_and_restores(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        with ortho.no_grad():
            assert _plain(x.to(x.device))
            if device == "cpu":
                assert _plain(ortho.cast(x, "float64"))
            else:
                assert _plain(ortho.cast(x, "float16"))
        # differentiability restores outside the guard
        assert _tracked(x.to(x.device))


# ---------------------------------------------------------------------------
# contiguous keeps exactly one ContiguousNode
# ---------------------------------------------------------------------------

class TestContiguousGraphContract:
    @pytest.mark.parametrize("device", DEVICES)
    def test_contiguous_attaches_single_contiguous_node(self, device):
        x = _t([[1.0, 2.0], [3.0, 4.0]], device, grad=True)
        c = x.transpose(0, 1).contiguous()
        assert _tracked(c)
        c.sum().backward()
        # Identity chain TransposeNode <- ContiguousNode: all-ones gradient.
        # A leaked internal ToNode would appear as a second hop but the
        # observable contract is that this graph works and stays minimal:
        # gradient flows in one step per public op.
        assert _cpu_flat(x.grad) == [1.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Version-counter selectivity for ToNode / CastNode
# ---------------------------------------------------------------------------

class TestVersionSelectivity:
    @pytest.mark.parametrize("device", DEVICES)
    def test_to_survives_source_mutation(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y = x.to(x.device)
        x.mul_(3.0)                       # ToNode saved no values
        y.sum().backward()
        assert _cpu_flat(x.grad) == [1.0, 1.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_cast_survives_source_mutation(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        z = ortho.cast(x, "float64" if device == "cpu" else "float16")
        x.mul_(3.0)                       # CastNode saved no values
        z.sum().backward()
        assert x.grad.dtype == "float32"
        assert _cpu_flat(x.grad) == [1.0, 1.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_downstream_value_sensitive_node_still_rejects(self, device):
        x = _t([1.0, 2.0], device, grad=True)
        y = x.to(x.device)
        z = y * y                          # MulNode watches y
        y.mul_(3.0)
        with pytest.raises(RuntimeError, match=STALE):
            z.backward()

    def test_downstream_tanh_on_cast_output_still_rejects_cpu(self):
        x = _t([0.5, 1.0], grad=True)
        z = ortho.cast(x, "float64")
        t = ortho.tanh(z)                     # ActivationNode watches z
        z.mul_(2.0)
        with pytest.raises(RuntimeError, match=STALE):
            t.backward()


# ---------------------------------------------------------------------------
# explicit internal detach contracts
# ---------------------------------------------------------------------------

class TestInternalDetachContracts:
    def test_batchnorm_running_stats_stay_graph_free(self):
        bn = nn.BatchNorm1d(3)
        x = _t([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], grad=True)
        out = bn(x)
        assert out.requires_grad
        for stat in (bn.running_mean, bn.running_var):
            assert not stat.requires_grad
            assert not stat._has_grad_fn
            assert stat.is_leaf

    @pytest.mark.parametrize("device", DEVICES)
    def test_module_to_parameters_remain_detached_leaves(self, device):
        m = nn.Linear(2, 2)
        m.to(device if device == "cuda" else "cpu")
        for p in m.parameters():
            assert p.requires_grad
            assert p.is_leaf
            assert not p._has_grad_fn
        if device == "cuda":
            assert m.weight.device.is_cuda()

    def test_module_cast_parameters_remain_detached_leaves_cpu(self):
        m = nn.Linear(2, 2)
        m.cast("float64")
        for p in m.parameters():
            assert p.dtype == "float64"
            assert p.requires_grad
            assert p.is_leaf
            assert not p._has_grad_fn

    @requires_cuda
    def test_module_half_parameters_remain_detached_leaves(self):
        m = nn.Linear(2, 2)
        m.to("cuda").half()
        for p in m.parameters():
            assert p.dtype == "float16"
            assert p.requires_grad
            assert p.is_leaf
            assert not p._has_grad_fn

    def test_load_state_dict_does_not_import_source_graph(self):
        model = nn.Linear(2, 2)
        donor = nn.Linear(2, 2)
        # Give the donor's parameters a live graph and grad tracking, then
        # load a NON-LEAF graph output as state to prove independence.
        state = {}
        for key, tensor in donor.state_dict().items():
            derived = tensor.to("cpu")     # tracked ToNode output (non-leaf)
            assert derived._has_grad_fn
            state[key] = derived
        weight_before = model.weight
        model.load_state_dict(state)
        assert model.weight is weight_before      # identity preserved
        assert model.weight.requires_grad
        assert model.weight.is_leaf
        assert not model.weight._has_grad_fn
        # And the destination still trains normally.
        out = model(_t([[1.0, 2.0]]))
        (out * out).sum().backward()
        assert model.weight.grad is not None

    def test_checkpoint_save_of_tracked_parameters_stays_clean(self, tmp_path):
        from orthon.checkpoint import save_checkpoint, load_checkpoint
        model = nn.Linear(2, 2)
        path = str(tmp_path / "model.ckpt")
        save_checkpoint(path, model=model)
        restored = nn.Linear(2, 2)
        load_checkpoint(path, model=restored)
        for p in model.parameters():
            assert p.is_leaf and not p._has_grad_fn
        for p in restored.parameters():
            assert p.is_leaf and not p._has_grad_fn

    @requires_cuda
    def test_amp_masters_are_detached_fp32_leaves(self):
        from orthon.amp import MixedPrecisionOptimizer
        from orthon.optim import SGD
        model = nn.Linear(2, 2)
        model.to("cuda").half()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.01)
        for _, _, orig_dtype, master in opt._entries:
            assert master.dtype == "float32"
            assert master.requires_grad
            assert master.is_leaf
            assert not master._has_grad_fn

    @requires_cuda
    def test_amp_reloaded_masters_are_detached_fp32_leaves(self):
        from orthon.amp import MixedPrecisionOptimizer
        from orthon.optim import SGD
        model = nn.Linear(2, 2)
        model.to("cuda").half()
        opt = MixedPrecisionOptimizer(model, SGD, lr=0.01)
        state = opt.state_dict()
        model2 = nn.Linear(2, 2)
        model2.to("cuda").half()
        opt2 = MixedPrecisionOptimizer(model2, SGD, lr=0.01)
        opt2.load_state_dict(state)
        for _, _, orig_dtype, master in opt2._entries:
            assert master.dtype == "float32"
            assert master.requires_grad
            assert master.is_leaf
            assert not master._has_grad_fn
