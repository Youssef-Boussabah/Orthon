"""State loading retains parameter owners and copies destination-authoritative values.

Self-loads, aliases and dtype/device conversions must preserve optimizer bindings."""
from tests.support.capabilities import orthon_cuda_available
from tests.support.source import parse_python
import ast
import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Linear, Sequential
from orthon.optim import SGD

CUDA = orthon_cuda_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


def _fill(model, base=1.0):
    """Deterministic, distinctive, non-zero values.

    Zeros hide self-load corruption, and uniform values hide a
    transposed or partial copy.
    """
    for i, p in enumerate(model.parameters()):
        n = int(np.prod(p.shape)) if p.shape else 1
        vals = base + i * 10.0 + np.arange(n, dtype=np.float32) * 0.25
        flat = ortho.from_numpy(np.ascontiguousarray(vals.reshape(tuple(p.shape))))
        if p.device != ortho.from_numpy(np.zeros(1, dtype=np.float32)).device:
            flat = flat.to(p.device)
        p.mul_(0.0)
        p.add_(flat, alpha=1.0)
    return model


def _v(t):
    return np.array(t.tolist(), dtype=np.float64)


def _params(model):
    return list(model.parameters())


# Failure mode A: self-load
def test_self_load_preserves_values_and_identity():
    """A self-load must not zero every parameter."""
    m = _fill(Linear(3, 2))
    before_w, before_b = _v(m.weight).copy(), _v(m.bias).copy()
    w_obj, b_obj = m.weight, m.bias
    assert np.abs(before_w).max() > 0.0          # the test would be vacuous on zeros

    state = m.state_dict()
    assert state["weight"] is m.weight            # state_dict is live

    m.load_state_dict(state)

    assert _v(m.weight) == pytest.approx(before_w, abs=0)
    assert _v(m.bias) == pytest.approx(before_b, abs=0)
    assert m.weight is w_obj
    assert m.bias is b_obj
    assert m.weight.requires_grad is True


def test_nested_sequential_self_load_is_safe():
    s = _fill(Sequential(Linear(3, 2), Linear(2, 2)))
    before = [_v(p).copy() for p in _params(s)]
    objs = _params(s)

    s.load_state_dict(s.state_dict())

    for i, (b, p) in enumerate(zip(before, _params(s))):
        assert _v(p) == pytest.approx(b, abs=0), "param %d changed" % i
    assert all(a is b for a, b in zip(objs, _params(s)))


def test_repeated_self_load_is_idempotent():
    m = _fill(Linear(4, 3))
    before = [_v(p).copy() for p in _params(m)]
    for _ in range(3):
        m.load_state_dict(m.state_dict())
    for b, p in zip(before, _params(m)):
        assert _v(p) == pytest.approx(b, abs=0)


# Defects C/D: identity and independence
def test_destination_parameter_objects_are_never_replaced():
    src = _fill(Linear(3, 2), base=2.0)
    dst = _fill(Linear(3, 2), base=90.0)
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert all(a is b for a, b in zip(objs, _params(dst)))
    for a in _params(dst):
        assert all(a is not b for b in _params(src))


def test_source_and_destination_are_independent_after_load():
    src = _fill(Linear(3, 2), base=2.0)
    dst = _fill(Linear(3, 2), base=90.0)
    dst.load_state_dict(src.state_dict())
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=0)

    # Mutating the source must not move the destination...
    src.weight.mul_(0.0)
    assert np.abs(_v(dst.weight)).max() > 0.0
    # ...and mutating the destination must not move the source.
    src2 = _fill(Linear(3, 2), base=3.0)
    dst2 = _fill(Linear(3, 2), base=91.0)
    dst2.load_state_dict(src2.state_dict())
    keep = _v(src2.weight).copy()
    dst2.weight.mul_(0.0)
    assert _v(src2.weight) == pytest.approx(keep, abs=0)


def test_state_dict_entries_are_not_installed_into_the_destination():
    src = _fill(Linear(3, 2), base=2.0)
    dst = _fill(Linear(3, 2), base=90.0)
    state = src.state_dict()
    dst.load_state_dict(state)
    assert dst.weight is not state["weight"]
    assert dst.bias is not state["bias"]


@requires_cuda
def test_cuda_float32_independence():
    src = _fill(Linear(3, 2), base=2.0)
    src.to("cuda")
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=0)
    src.weight.mul_(0.0)
    assert np.abs(_v(dst.weight)).max() > 0.0


# Failure mode B: cross-device
@requires_cuda
def test_cpu_state_loads_into_a_cuda_destination():
    """Without device reconciliation: RuntimeError: add_: device mismatch."""
    src = _fill(Linear(3, 2), base=2.0)
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert str(dst.weight.device) == "device('cuda:0')"   # model stays put
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=1e-6)
    assert dst.weight.requires_grad is True


@requires_cuda
def test_cuda_state_loads_into_a_cpu_destination():
    src = _fill(Linear(3, 2), base=2.0)
    src.to("cuda")
    dst = _fill(Linear(3, 2), base=90.0)
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert str(dst.weight.device) == "device('cpu')"
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=1e-6)


@requires_cuda
def test_load_state_dict_does_not_move_the_destination_model():
    src = _fill(Linear(3, 2), base=2.0)
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    dst.load_state_dict(src.state_dict())
    for p in _params(dst):
        assert str(p.device) == "device('cuda:0')"


# Cross-dtype
def test_float32_state_into_float64_destination():
    """Without dtype reconciliation: RuntimeError: add_: dtype mismatch."""
    src = _fill(Linear(3, 2), base=2.0)
    dst = _fill(Linear(3, 2), base=90.0)
    dst.cast("float64")
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert dst.weight.dtype == "float64"          # destination dtype authoritative
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=1e-6)


def test_float64_state_into_float32_destination():
    src = _fill(Linear(3, 2), base=2.0)
    src.cast("float64")
    dst = _fill(Linear(3, 2), base=90.0)
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert dst.weight.dtype == "float32"
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=1e-6)


@requires_cuda
@pytest.mark.parametrize("meth,dtype,tol", [("half", "float16", 5e-2),
                                            ("bfloat16", "bfloat16", 5e-1)])
def test_float32_state_into_narrow_cuda_destination(meth, dtype, tol):
    src = _fill(Linear(3, 2), base=2.0)
    src.to("cuda")
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    getattr(dst, meth)()
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert dst.weight.dtype == dtype
    assert str(dst.weight.device) == "device('cuda:0')"
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=tol)


@requires_cuda
@pytest.mark.parametrize("meth,dtype,tol", [("half", "float16", 5e-2),
                                            ("bfloat16", "bfloat16", 5e-1)])
def test_narrow_cuda_state_into_float32_destination(meth, dtype, tol):
    src = _fill(Linear(3, 2), base=2.0)
    src.to("cuda")
    getattr(src, meth)()
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert dst.weight.dtype == "float32"
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=tol)


@requires_cuda
@pytest.mark.parametrize("meth,dtype", [("half", "float16"), ("bfloat16", "bfloat16")])
def test_narrow_precision_parameter_identity_is_preserved(meth, dtype):
    """The low-precision branch must copy, not install the source: a plain
    ``restored = src`` makes dst.weight is src.weight come out True."""
    src = _fill(Linear(3, 2), base=2.0)
    src.to("cuda")
    getattr(src, meth)()
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    getattr(dst, meth)()
    old_w, old_b = dst.weight, dst.bias
    state = src.state_dict()

    dst.load_state_dict(state)

    assert dst.weight is old_w
    assert dst.bias is old_b
    assert dst.weight is not src.weight
    assert dst.weight is not state["weight"]
    assert dst.weight.dtype == dtype


# Device + dtype together
@requires_cuda
@pytest.mark.parametrize("meth,dtype,tol", [("half", "float16", 5e-2),
                                            ("bfloat16", "bfloat16", 5e-1)])
def test_cpu_float32_state_into_narrow_cuda_destination(meth, dtype, tol):
    """Both attributes differ at once, so preparation ordering cannot be
    accidentally correct for only one of them."""
    src = _fill(Linear(3, 2), base=2.0)               # CPU float32
    dst = _fill(Linear(3, 2), base=90.0)
    dst.to("cuda")
    getattr(dst, meth)()                              # CUDA narrow
    objs = _params(dst)

    dst.load_state_dict(src.state_dict())

    assert str(dst.weight.device) == "device('cuda:0')"
    assert dst.weight.dtype == dtype
    assert all(a is b for a, b in zip(objs, _params(dst)))
    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=tol)


# Optimizer and hook safety
def test_optimizer_references_survive_a_load():
    model = _fill(Linear(3, 2))
    opt = SGD(model.parameters(), lr=0.1)
    originals = _params(model)

    src = _fill(Linear(3, 2), base=7.0)
    model.load_state_dict(src.state_dict())

    now = _params(model)
    assert all(a is b for a, b in zip(originals, now))
    opt_params = getattr(opt, "_params", None)
    if opt_params is not None:
        assert all(a is b for a, b in zip(list(opt_params), now))


def test_optimizer_step_after_load_updates_the_loaded_parameters():
    """The sharpest form: a loader that swapped the module's tensor while the
    optimizer kept the old one would leave the model unchanged here."""
    model = _fill(Linear(3, 2))
    opt = SGD(model.parameters(), lr=0.1)
    model.load_state_dict(_fill(Linear(3, 2), base=7.0).state_dict())

    loaded = _v(model.weight).copy()
    x = ortho.ones([2, 3])
    ortho.sum(model(x)).backward()
    opt.step()

    assert not np.allclose(_v(model.weight), loaded)


@requires_cuda
def test_optimizer_references_survive_a_narrow_precision_load():
    model = _fill(Linear(3, 2))
    model.to("cuda")
    model.half()
    opt = SGD(model.parameters(), lr=0.1)
    originals = _params(model)

    src = _fill(Linear(3, 2), base=7.0)
    src.to("cuda")
    src.half()
    model.load_state_dict(src.state_dict())

    now = _params(model)
    assert all(a is b for a, b in zip(originals, now))
    opt_params = getattr(opt, "_params", None)
    if opt_params is not None:
        assert all(a is b for a, b in zip(list(opt_params), now))


def test_a_parameter_hook_registered_before_loading_still_fires():
    """Hooks hang off the TensorImpl; replacing the parameter would drop them."""
    model = _fill(Linear(3, 2))
    fired = []
    model.weight._register_post_accumulate_grad_hook(lambda: fired.append(1))
    w_obj = model.weight

    model.load_state_dict(_fill(Linear(3, 2), base=7.0).state_dict())
    assert model.weight is w_obj

    ortho.sum(model(ortho.ones([2, 3]))).backward()
    assert fired == [1]


# Metadata and detached preparation
def test_parameter_autograd_metadata_is_unchanged_by_loading():
    model = _fill(Linear(3, 2))
    before = [(p.requires_grad, p.is_leaf, p._has_grad_fn) for p in _params(model)]
    model.load_state_dict(_fill(Linear(3, 2), base=7.0).state_dict())
    after = [(p.requires_grad, p.is_leaf, p._has_grad_fn) for p in _params(model)]
    assert after == before
    assert all(m == (True, True, False) for m in after)


def test_existing_gradients_are_not_cleared_by_loading():
    """Loading weights copies state; it does not rewrite autograd config."""
    model = _fill(Linear(3, 2))
    ortho.sum(model(ortho.ones([2, 3]))).backward()
    grad_before = _v(model.weight.grad).copy()

    model.load_state_dict(_fill(Linear(3, 2), base=7.0).state_dict())

    assert _v(model.weight.grad) == pytest.approx(grad_before, abs=0)


def test_a_trainable_source_does_not_import_its_graph():
    """State taken from a live trainable model, with a graph attached
    to nothing in particular, must not make the destination non-leaf."""
    src = _fill(Linear(3, 2), base=7.0)
    assert src.weight.requires_grad is True
    dst = _fill(Linear(3, 2))

    dst.load_state_dict(src.state_dict())

    assert dst.weight.requires_grad is True
    assert dst.weight.is_leaf is True
    assert dst.weight._has_grad_fn is False


# Strict / non-strict / atomicity
def test_strict_rejects_missing_and_unexpected_keys():
    model = Linear(3, 2)
    state = model.state_dict()
    partial = {k: v for k, v in state.items() if k != "bias"}
    with pytest.raises(KeyError):
        model.load_state_dict(partial)
    extra = dict(state)
    extra["nope"] = state["bias"]
    with pytest.raises(KeyError):
        model.load_state_dict(extra)


def test_non_strict_ignores_missing_and_unexpected_keys():
    src = _fill(Linear(3, 2), base=7.0)
    dst = _fill(Linear(3, 2), base=90.0)
    bias_before = _v(dst.bias).copy()

    partial = {"weight": src.state_dict()["weight"], "nope": src.state_dict()["bias"]}
    dst.load_state_dict(partial, strict=False)

    assert _v(dst.weight) == pytest.approx(_v(src.weight), abs=0)
    assert _v(dst.bias) == pytest.approx(bias_before, abs=0)   # untouched


def test_load_state_dict_returns_self():
    model = Linear(3, 2)
    assert model.load_state_dict(model.state_dict()) is model


def test_shape_mismatch_is_rejected():
    model = Linear(3, 2)
    bad = dict(model.state_dict())
    bad["weight"] = ortho.ones([5, 5])
    with pytest.raises(ValueError):
        model.load_state_dict(bad)


def test_non_tensor_entry_is_rejected():
    model = Linear(3, 2)
    bad = dict(model.state_dict())
    bad["bias"] = [1.0, 2.0]
    with pytest.raises(TypeError):
        model.load_state_dict(bad)


@pytest.mark.parametrize("bad_value,exc", [("shape", ValueError), ("type", TypeError)])
def test_a_malformed_later_entry_leaves_earlier_parameters_untouched(bad_value, exc):
    """Validation and preparation both complete before any commit."""
    model = _fill(Sequential(Linear(3, 2), Linear(2, 2)))
    before = [_v(p).copy() for p in _params(model)]

    good = _fill(Sequential(Linear(3, 2), Linear(2, 2)), base=70.0).state_dict()
    bad = dict(good)
    # "1.bias" is the LAST entry, so earlier parameters would already have been
    # committed by a loader that validated and mutated in one pass.
    bad["1.bias"] = ortho.ones([9]) if bad_value == "shape" else [1.0, 2.0]

    with pytest.raises(exc):
        model.load_state_dict(bad)

    for i, (b, p) in enumerate(zip(before, _params(model))):
        assert _v(p) == pytest.approx(b, abs=0), "param %d was mutated" % i


def test_preparation_loop_does_not_mutate():
    """Source-level companion to the test above: the prepare loop must contain
    no commit operation."""
    tree = parse_python("python/orthon/nn/module.py")
    module = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Module")
    method = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_prepare_load_state_dict")
    commit_index = next(i for i, n in enumerate(method.body)
                        if isinstance(n, ast.FunctionDef) and n.name == "commit")
    prepare = ast.unparse(ast.Module(body=method.body[:commit_index], type_ignores=[]))
    for forbidden in ("_copy_data_", "_parameters[", "_buffers[", ".mul_(", ".add_("):
        assert forbidden not in prepare, "prepare stage mutates: %s" % forbidden


def test_parameters_are_copied_into_not_replaced():
    """Source-level: the commit stage must not assign into _parameters."""
    tree = parse_python("python/orthon/nn/module.py")
    module = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Module")
    method = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_prepare_load_state_dict")
    commit = next(n for n in method.body if isinstance(n, ast.FunctionDef) and n.name == "commit")
    copy = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == "_copy_state_value")
    assert "_copy_state_value" in ast.unparse(commit)
    assert "_copy_data_" in ast.unparse(copy)
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store)
                   and isinstance(n.value, ast.Attribute) and n.value.attr == "_parameters"
                   for n in ast.walk(commit))


# Buffers are replaced, and stay independent of the source
def test_buffer_loading_still_replaces_and_stays_independent():
    from orthon.nn import BatchNorm1d
    src = BatchNorm1d(4)
    dst = BatchNorm1d(4)
    src.running_mean.mul_(0.0)
    src.running_mean.add_(ortho.full([4], 3.0), alpha=1.0)

    dst.load_state_dict(src.state_dict())
    assert _v(dst.running_mean) == pytest.approx(_v(src.running_mean), abs=0)
    assert dst.running_mean is not src.running_mean

    src.running_mean.mul_(0.0)
    assert _v(dst.running_mean) == pytest.approx(np.full(4, 3.0), abs=0)


def test_batchnorm_self_load_is_safe():
    from orthon.nn import BatchNorm1d
    m = BatchNorm1d(4)
    m.weight.mul_(0.0)
    m.weight.add_(ortho.full([4], 2.5), alpha=1.0)
    m.running_mean.mul_(0.0)
    m.running_mean.add_(ortho.full([4], 1.5), alpha=1.0)
    w_obj = m.weight

    m.load_state_dict(m.state_dict())

    assert _v(m.weight) == pytest.approx(np.full(4, 2.5), abs=0)
    assert _v(m.running_mean) == pytest.approx(np.full(4, 1.5), abs=0)
    assert m.weight is w_obj


# The internal copy primitive itself
def test_copy_data_rejects_mismatched_shape_dtype_and_device():
    a = ortho.ones([3, 2])
    with pytest.raises(RuntimeError):
        a._copy_data_(ortho.ones([2, 3]))
    with pytest.raises(RuntimeError):
        a._copy_data_(ortho.cast(ortho.ones([3, 2]), "float64"))
    if CUDA:
        with pytest.raises(RuntimeError):
            a._copy_data_(ortho.ones([3, 2]).to("cuda"))


def test_copy_data_rejects_a_non_contiguous_destination():
    a = ortho.transpose(ortho.ones([3, 2]), 0, 1)
    assert a.is_contiguous is False
    with pytest.raises(RuntimeError):
        a._copy_data_(ortho.ones([2, 3]))


def test_copy_data_preserves_identity_and_metadata():
    dst = ortho.from_numpy(np.zeros((2, 3), dtype=np.float32))
    dst.requires_grad_(True)
    impl_before = dst._impl_id
    src = ortho.from_numpy(np.arange(6, dtype=np.float32).reshape(2, 3))

    dst._copy_data_(src)

    assert dst._impl_id == impl_before          # same TensorImpl, not replaced
    assert _v(dst) == pytest.approx(np.arange(6), abs=0)
    assert dst.requires_grad is True
    assert dst.is_leaf is True
    assert dst._has_grad_fn is False
    assert dst is not src


def test_all_canonical_module_parameters_are_contiguous():
    """The copy primitive is deliberately narrow (contiguous destination only).
    That is only safe while every canonical parameter satisfies it."""
    from orthon import nn
    mods = [nn.Linear(4, 3), nn.Conv2d(2, 3, 3), nn.BatchNorm1d(4),
            nn.BatchNorm2d(4), nn.LayerNorm(4), nn.Embedding(6, 4),
            nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))]
    for m in mods:
        for key, t in m.state_dict().items():
            assert t.is_contiguous is True, "%s.%s is not contiguous" % (
                type(m).__name__, key)
