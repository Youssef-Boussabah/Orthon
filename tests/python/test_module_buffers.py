"""Tests for Module persistent/non-persistent buffers and state transformation.

Buffer behaviour is proven with small synthetic modules rather than BatchNorm,
so these tests stay valid independently of how BatchNorm is later migrated.
"""
import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Conv2d, Embedding, LayerNorm, Linear, Sequential
from orthon.nn.module import Module, Parameter

CUDA = ortho.cuda_enabled()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")


# Synthetic fixtures.

class Leaf(Module):
    """One parameter, one persistent buffer, one non-persistent buffer."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(ortho.ones([3]))
        self.register_buffer("running", ortho.zeros([3]))
        self.register_buffer("scratch", ortho.ones([3]), persistent=False)

    def forward(self, x):
        return x


class Nest(Module):
    """Wraps a Leaf and adds its own persistent buffer."""

    def __init__(self):
        super().__init__()
        self.leaf = Leaf()
        self.register_buffer("total", ortho.full([2], 4.0))

    def forward(self, x):
        return self.leaf(x)


class Bare(Module):
    """A module with no buffers at all."""

    def __init__(self):
        super().__init__()
        self.weight = Parameter(ortho.ones([2]))

    def forward(self, x):
        return x


def _int_buffer(values):
    return ortho.from_numpy(np.array(values, dtype=np.int64))


# 1-3. Registration and attribute access.

def test_register_persistent_buffer():
    m = Leaf()
    assert "running" in m._buffers
    assert isinstance(m.running, ortho.Tensor)


def test_register_non_persistent_buffer():
    m = Leaf()
    assert "scratch" in m._buffers
    assert "scratch" in m._non_persistent_buffers
    assert "running" not in m._non_persistent_buffers


def test_buffer_readable_as_attribute():
    m = Leaf()
    assert m.running.tolist() == [0.0, 0.0, 0.0]
    assert m.scratch.tolist() == [1.0, 1.0, 1.0]


def test_buffer_does_not_require_grad_by_default():
    m = Leaf()
    assert m.running.requires_grad is False
    assert m.scratch.requires_grad is False


def test_register_buffer_none_is_allowed():
    m = Bare()
    m.register_buffer("maybe", None)
    assert m.maybe is None
    assert "maybe" in m._buffers


# 4-7. Enumeration.

def test_named_buffers_local():
    m = Leaf()
    assert [n for n, _ in m.named_buffers()] == ["running", "scratch"]


def test_named_buffers_recursive():
    m = Nest()
    assert sorted(n for n, _ in m.named_buffers()) == [
        "leaf.running", "leaf.scratch", "total",
    ]


def test_named_buffers_non_recursive():
    m = Nest()
    assert [n for n, _ in m.named_buffers(recurse=False)] == ["total"]


def test_buffers_recursive():
    m = Nest()
    assert len(list(m.buffers())) == 3


def test_buffers_non_recursive():
    m = Nest()
    assert len(list(m.buffers(recurse=False))) == 1


def test_named_buffers_prefix():
    m = Leaf()
    names = [n for n, _ in m.named_buffers(prefix="mod.")]
    assert names == ["mod.running", "mod.scratch"]


def test_none_buffer_skipped_by_enumeration():
    m = Bare()
    m.register_buffer("maybe", None)
    assert [n for n, _ in m.named_buffers()] == []
    assert list(m.buffers()) == []


def test_buffers_excluded_from_parameters():
    m = Nest()
    param_ids = {id(p) for p in m.parameters()}
    buffer_ids = {id(b) for b in m.buffers()}
    assert param_ids.isdisjoint(buffer_ids)
    assert len(param_ids) == 1


def test_buffers_excluded_from_named_parameters():
    m = Nest()
    names = [n for n, _ in m.named_parameters()]
    assert names == ["leaf.weight"]
    assert not any("running" in n or "scratch" in n or "total" in n for n in names)


def test_optimizer_traversal_sees_only_parameters():
    from orthon.optim import SGD
    m = Nest()
    opt = SGD(m.parameters(), lr=0.1)
    assert len(opt._params) == 1


# 8-10. state_dict.

def test_persistent_buffer_in_state_dict():
    m = Leaf()
    assert "running" in m.state_dict()


def test_non_persistent_buffer_excluded_from_state_dict():
    m = Leaf()
    assert "scratch" not in m.state_dict()


def test_nested_state_dict_keys():
    m = Nest()
    assert sorted(m.state_dict().keys()) == [
        "leaf.running", "leaf.weight", "total",
    ]


def test_none_buffer_excluded_from_state_dict():
    m = Bare()
    m.register_buffer("maybe", None)
    assert sorted(m.state_dict().keys()) == ["weight"]


def test_state_dict_values_are_tensors():
    for v in Nest().state_dict().values():
        assert isinstance(v, ortho.Tensor)


def test_ordinary_attributes_excluded_from_state_dict():
    m = Bare()
    m.plain = ortho.ones([4])
    m.number = 7
    sd = m.state_dict()
    assert sorted(sd.keys()) == ["weight"]


# 11. load_state_dict.

def test_load_state_dict_restores_buffer():
    src, dst = Nest(), Nest()
    src.leaf.running.fill_(5.0)
    src.total.fill_(9.0)
    dst.load_state_dict(src.state_dict())
    assert dst.leaf.running.tolist() == pytest.approx([5.0, 5.0, 5.0])
    assert dst.total.tolist() == pytest.approx([9.0, 9.0])


def test_load_state_dict_keeps_buffer_registered():
    src, dst = Nest(), Nest()
    src.leaf.running.fill_(2.0)
    dst.load_state_dict(src.state_dict())
    assert "running" in dst.leaf._buffers
    assert dst.leaf.running.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_load_state_dict_does_not_expect_non_persistent_buffer():
    src, dst = Leaf(), Leaf()
    sd = src.state_dict()
    assert "scratch" not in sd
    dst.load_state_dict(sd, strict=True)   # must not raise


def test_load_state_dict_buffer_shape_mismatch_raises():
    m = Leaf()
    sd = m.state_dict()
    sd["running"] = ortho.zeros([5])
    with pytest.raises(ValueError):
        m.load_state_dict(sd)


def test_load_state_dict_rejects_non_tensor_entry():
    m = Leaf()
    sd = m.state_dict()
    sd["running"] = [0.0, 0.0, 0.0]
    with pytest.raises(TypeError):
        m.load_state_dict(sd)


def test_malformed_state_does_not_partially_apply():
    """A bad entry must abort the load before any value is written."""
    m = Nest()
    m.leaf.running.fill_(1.0)
    m.total.fill_(1.0)
    good = Nest()
    good.leaf.running.fill_(3.0)
    good.total.fill_(3.0)
    bad = dict(good.state_dict())
    bad["total"] = ortho.zeros([7])          # wrong shape, validated second
    with pytest.raises(ValueError):
        m.load_state_dict(bad)
    assert m.leaf.running.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert m.total.tolist() == pytest.approx([1.0, 1.0])


def test_load_state_dict_strict_rejects_unexpected_buffer_key():
    m = Leaf()
    sd = m.state_dict()
    sd["scratch"] = ortho.zeros([3])
    with pytest.raises(KeyError):
        m.load_state_dict(sd, strict=True)


# Buffer state-load independence.
#
# A loaded buffer must be an independent copy even when the source and
# destination already agree on device and dtype, otherwise two modules end up
# sharing one storage and mutating either corrupts the other.

def test_load_state_dict_buffer_is_not_source_object():
    """Primary regression: same device, same dtype must still copy."""
    src, dst = Leaf(), Leaf()
    src.running.fill_(1.0)
    dst.load_state_dict(src.state_dict())
    assert dst.running is not src.running
    assert dst.running._impl_id != src.running._impl_id


def test_load_state_dict_mutating_destination_does_not_touch_source():
    src, dst = Leaf(), Leaf()
    src.running.fill_(1.0)
    dst.load_state_dict(src.state_dict())
    dst.running.fill_(9.0)
    assert dst.running.tolist() == pytest.approx([9.0, 9.0, 9.0])
    assert src.running.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_load_state_dict_mutating_source_does_not_touch_destination():
    src, dst = Leaf(), Leaf()
    src.running.fill_(1.0)
    dst.load_state_dict(src.state_dict())
    src.running.fill_(4.0)
    assert src.running.tolist() == pytest.approx([4.0, 4.0, 4.0])
    assert dst.running.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_load_state_dict_nested_buffers_are_independent():
    src, dst = Nest(), Nest()
    src.leaf.running.fill_(2.0)
    src.total.fill_(3.0)
    dst.load_state_dict(src.state_dict())
    assert dst.leaf.running is not src.leaf.running
    assert dst.total is not src.total
    dst.leaf.running.fill_(7.0)
    dst.total.fill_(8.0)
    assert src.leaf.running.tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert src.total.tolist() == pytest.approx([3.0, 3.0])


def test_load_state_dict_integer_buffer_is_independent():
    class IntBuf(Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", _int_buffer([1, 2, 3]))

        def forward(self, x):
            return x

    src, dst = IntBuf(), IntBuf()
    dst.load_state_dict(src.state_dict())
    assert dst.counter is not src.counter
    assert dst.counter.dtype == "int64"
    assert dst.counter.tolist() == [1, 2, 3]


def test_load_state_dict_repeated_loads_stay_independent():
    """Loading the same state into two modules must not join them together."""
    src, a, b = Leaf(), Leaf(), Leaf()
    src.running.fill_(1.0)
    sd = src.state_dict()
    a.load_state_dict(sd)
    b.load_state_dict(sd)
    assert a.running is not b.running
    a.running.fill_(5.0)
    assert b.running.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert src.running.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_load_state_dict_preserves_destination_requires_grad_true():
    src, dst = Leaf(), Leaf()
    src.running.requires_grad_(False)
    dst.running.requires_grad_(True)
    dst.load_state_dict(src.state_dict())
    assert dst.running.requires_grad is True


def test_load_state_dict_preserves_destination_requires_grad_false():
    src, dst = Leaf(), Leaf()
    src.running.requires_grad_(True)
    dst.running.requires_grad_(False)
    dst.load_state_dict(src.state_dict())
    assert dst.running.requires_grad is False


def test_load_state_dict_reconciles_buffer_dtype_same_device():
    """Device already matches; dtype does not. Values convert, dtype is kept."""
    src, dst = Leaf(), Leaf()
    src.running.fill_(2.0)
    dst.cast("float16")
    assert dst.running.dtype == "float16"
    dst.load_state_dict(src.state_dict())
    assert dst.running.dtype == "float16"
    assert dst.running is not src.running
    assert dst.running.to("cpu").tolist() == pytest.approx([2.0, 2.0, 2.0], abs=1e-3)


def test_load_state_dict_keeps_persistence_bookkeeping():
    src, dst = Leaf(), Leaf()
    dst.load_state_dict(src.state_dict())
    assert "running" in dst._buffers
    assert "scratch" in dst._buffers
    assert dst._non_persistent_buffers == {"scratch"}
    assert [n for n, _ in dst.named_buffers()] == ["running", "scratch"]
    assert "running" in dst.state_dict()
    assert "scratch" not in dst.state_dict()


def test_load_state_dict_does_not_disturb_non_persistent_buffer():
    src, dst = Leaf(), Leaf()
    dst.scratch.fill_(3.0)
    dst.load_state_dict(src.state_dict())
    assert dst.scratch.tolist() == pytest.approx([3.0, 3.0, 3.0])


@requires_cuda
def test_load_state_dict_buffer_independent_on_same_cuda_device():
    """CUDA->CUDA same-device load must copy, not alias."""
    src = Leaf()
    src.running.fill_(1.0)          # fill_ is CPU-only, so fill before moving
    src.to("cuda")
    dst = Leaf().to("cuda")
    dst.load_state_dict(src.state_dict())
    assert dst.running is not src.running
    assert dst.running.device.is_cuda()
    dst.running.mul_(5.0)           # mul_ works on CUDA
    assert src.running.to("cpu").tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert dst.running.to("cpu").tolist() == pytest.approx([5.0, 5.0, 5.0])


@requires_cuda
def test_load_state_dict_cuda_buffer_preserves_destination_requires_grad():
    src = Leaf().to("cuda")
    dst = Leaf().to("cuda")
    src.running.requires_grad_(False)
    dst.running.requires_grad_(True)
    dst.load_state_dict(src.state_dict())
    assert dst.running.requires_grad is True


# 12-14. Device movement.

def test_to_cpu_moves_buffers():
    m = Nest().to("cpu")
    for _, b in m.named_buffers():
        assert b.device.is_cpu()


def test_to_keeps_buffers_registered():
    m = Nest()
    m.to("cpu")
    assert sorted(n for n, _ in m.named_buffers()) == [
        "leaf.running", "leaf.scratch", "total",
    ]


def test_to_preserves_buffer_values():
    m = Leaf()
    m.running.fill_(3.5)
    m.to("cpu")
    assert m.running.tolist() == pytest.approx([3.5, 3.5, 3.5])


def test_to_leaves_ordinary_attributes_untouched():
    m = Bare()
    m.number = 11
    m.text = "hello"
    m.to("cpu")
    assert m.number == 11
    assert m.text == "hello"


def test_to_returns_self():
    m = Nest()
    assert m.to("cpu") is m


@requires_cuda
def test_to_cuda_moves_buffers_recursively():
    m = Nest().to("cuda")
    for name, b in m.named_buffers():
        assert b.device.is_cuda(), name
    for p in m.parameters():
        assert p.device.is_cuda()


@requires_cuda
def test_to_cuda_moves_non_persistent_buffer():
    m = Leaf().to("cuda")
    assert m.scratch.device.is_cuda()


@requires_cuda
def test_to_cuda_preserves_buffer_values():
    m = Leaf()
    m.running.fill_(2.5)
    m.to("cuda")
    assert m.running.to("cpu").tolist() == pytest.approx([2.5, 2.5, 2.5])


@requires_cuda
def test_to_cuda_then_cpu_roundtrip():
    m = Leaf()
    m.running.fill_(1.25)
    m.to("cuda").to("cpu")
    assert m.running.device.is_cpu()
    assert m.running.tolist() == pytest.approx([1.25, 1.25, 1.25])


# 13. Dtype casting.

def test_cast_casts_buffers():
    m = Nest().cast("float16")
    for name, b in m.named_buffers():
        assert b.dtype == "float16", name


def test_cast_keeps_buffers_registered():
    m = Leaf().cast("float16")
    assert [n for n, _ in m.named_buffers()] == ["running", "scratch"]


def test_cast_casts_non_persistent_buffer():
    m = Leaf().cast("float16")
    assert m.scratch.dtype == "float16"


def test_cast_leaves_integer_buffers_alone():
    m = Bare()
    m.register_buffer("counter", _int_buffer([0, 1, 2]))
    m.cast("float16")
    assert m.counter.dtype == "int64"
    assert m.counter.tolist() == [0, 1, 2]


def test_cast_skips_none_buffer():
    m = Bare()
    m.register_buffer("maybe", None)
    m.cast("float16")
    assert m.maybe is None


def test_half_casts_buffers():
    m = Leaf().half()
    assert m.running.dtype == "float16"


def test_bfloat16_casts_buffers():
    m = Leaf().bfloat16()
    assert m.running.dtype == "bfloat16"


def test_cast_nested_buffers():
    m = Nest().cast("float64")
    assert m.leaf.running.dtype == "float64"
    assert m.total.dtype == "float64"


@requires_cuda
def test_cuda_then_cast_keeps_device_and_dtype():
    m = Nest().to("cuda").cast("float16")
    for name, b in m.named_buffers():
        assert b.device.is_cuda(), name
        assert b.dtype == "float16", name


@requires_cuda
def test_load_state_dict_reconciles_buffer_device_and_dtype():
    """A CUDA buffer loaded into a CPU module lands on CPU with its own dtype."""
    class OnlyBuffers(Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running", ortho.zeros([2]))
            self.register_buffer("counter", _int_buffer([0, 0]))

        def forward(self, x):
            return x

    src = OnlyBuffers()
    src.running.fill_(6.0)
    src.counter = _int_buffer([1, 2])
    src.to("cuda")

    dst = OnlyBuffers()          # stays on CPU
    dst.load_state_dict(src.state_dict())

    assert dst.running.device.is_cpu()
    assert dst.running.tolist() == pytest.approx([6.0, 6.0])
    assert dst.counter.device.is_cpu()
    assert dst.counter.dtype == "int64"
    assert dst.counter.tolist() == [1, 2]


# 15-17. requires_grad preservation.

def test_frozen_parameter_stays_frozen_after_to():
    m = Bare()
    m.weight.requires_grad_(False)
    m.to("cpu")
    assert m.weight.requires_grad is False


def test_frozen_parameter_stays_frozen_after_cast():
    m = Bare()
    m.weight.requires_grad_(False)
    m.cast("float16")
    assert m.weight.requires_grad is False


def test_trainable_parameter_stays_trainable_after_to():
    m = Bare()
    m.to("cpu")
    assert m.weight.requires_grad is True


def test_trainable_parameter_stays_trainable_after_cast():
    m = Bare()
    m.cast("float16")
    assert m.weight.requires_grad is True


def test_mixed_freeze_state_preserved_across_to():
    m = Sequential(Linear(4, 4), Linear(4, 2))
    params = list(m.parameters())
    params[0].requires_grad_(False)
    m.to("cpu")
    after = list(m.parameters())
    assert after[0].requires_grad is False
    assert all(p.requires_grad for p in after[1:])


def test_buffers_do_not_gain_requires_grad_from_to():
    m = Leaf().to("cpu")
    assert m.running.requires_grad is False


def test_buffers_do_not_gain_requires_grad_from_cast():
    m = Leaf().cast("float16")
    assert m.running.requires_grad is False


@requires_cuda
def test_frozen_parameter_stays_frozen_after_cuda_move():
    m = Bare()
    m.weight.requires_grad_(False)
    m.to("cuda")
    assert m.weight.requires_grad is False


# 18-21. Validation and collisions.

@pytest.mark.parametrize("bad", ["", "has.dot", "1abc", "with space", "a-b"])
def test_invalid_buffer_names_rejected(bad):
    m = Bare()
    with pytest.raises(ValueError):
        m.register_buffer(bad, ortho.zeros([1]))


def test_non_string_buffer_name_rejected():
    m = Bare()
    with pytest.raises(TypeError):
        m.register_buffer(42, ortho.zeros([1]))


def test_buffer_name_colliding_with_parameter_rejected():
    m = Bare()
    with pytest.raises(KeyError):
        m.register_buffer("weight", ortho.zeros([2]))


def test_buffer_name_colliding_with_child_module_rejected():
    m = Nest()
    with pytest.raises(KeyError):
        m.register_buffer("leaf", ortho.zeros([2]))


def test_buffer_name_colliding_with_plain_attribute_rejected():
    m = Bare()
    m.threshold = 0.5
    with pytest.raises(KeyError):
        m.register_buffer("threshold", ortho.zeros([1]))


def test_buffer_name_colliding_with_method_rejected():
    m = Bare()
    with pytest.raises(KeyError):
        m.register_buffer("parameters", ortho.zeros([1]))


def test_invalid_buffer_value_rejected():
    m = Bare()
    with pytest.raises(TypeError):
        m.register_buffer("bad", [1.0, 2.0])
    with pytest.raises(TypeError):
        m.register_buffer("bad2", np.zeros(3, dtype=np.float32))


def test_parameter_rejected_as_buffer_value():
    m = Bare()
    with pytest.raises(TypeError):
        m.register_buffer("bad", Parameter(ortho.ones([2])))


def test_module_rejected_as_buffer_value():
    m = Bare()
    with pytest.raises(TypeError):
        m.register_buffer("bad", Leaf())


def test_assigning_parameter_over_buffer_rejected():
    m = Leaf()
    with pytest.raises(KeyError):
        m.running = Parameter(ortho.ones([3]))


def test_assigning_module_over_buffer_rejected():
    m = Leaf()
    with pytest.raises(KeyError):
        m.running = Bare()


# 22. Reassignment semantics.

def test_buffer_reassignment_keeps_it_a_buffer():
    m = Leaf()
    m.running = ortho.full([3], 8.0)
    assert "running" in m._buffers
    assert "running" in m.state_dict()
    assert m.running.tolist() == pytest.approx([8.0, 8.0, 8.0])


def test_buffer_reassignment_keeps_persistence_flag():
    m = Leaf()
    m.scratch = ortho.full([3], 2.0)
    assert "scratch" in m._non_persistent_buffers
    assert "scratch" not in m.state_dict()


def test_buffer_reassignment_to_none_allowed():
    m = Leaf()
    m.running = None
    assert m.running is None
    assert "running" in m._buffers
    assert "running" not in m.state_dict()


def test_buffer_reassignment_rejects_bad_type():
    m = Leaf()
    with pytest.raises(TypeError):
        m.running = 5.0


def test_re_register_buffer_updates_value_and_persistence():
    m = Leaf()
    m.register_buffer("running", ortho.full([3], 6.0), persistent=False)
    assert m.running.tolist() == pytest.approx([6.0, 6.0, 6.0])
    assert "running" not in m.state_dict()


def test_plain_tensor_attribute_is_not_auto_registered():
    """Assigning a Tensor to an unregistered name must stay an ordinary attr."""
    m = Bare()
    m.notabuffer = ortho.ones([3])
    assert "notabuffer" not in m._buffers
    assert [n for n, _ in m.named_buffers()] == []
    assert "notabuffer" not in m.state_dict()


def test_parameter_and_module_registration_still_work():
    m = Bare()
    m.extra = Parameter(ortho.ones([2]))
    m.child = Leaf()
    assert "extra" in m._parameters
    assert "child" in m._modules
    assert sorted(m.state_dict().keys()) == [
        "child.running", "child.weight", "extra", "weight",
    ]


# Modules that register no buffers at all.

def test_bare_module_state_dict_unchanged():
    assert sorted(Bare().state_dict().keys()) == ["weight"]


def test_bare_module_has_no_buffers():
    m = Bare()
    assert list(m.buffers()) == []
    assert list(m.named_buffers()) == []


@pytest.mark.parametrize(
    "factory, keys, n_params",
    [
        (lambda: Linear(4, 2), ["bias", "weight"], 2),
        (lambda: Conv2d(2, 3, 2), ["bias", "weight"], 2),
        (lambda: LayerNorm(4), ["bias", "weight"], 2),
        (lambda: Embedding(5, 3), ["weight"], 1),
        (lambda: Sequential(Linear(4, 8), Linear(8, 2)),
         ["0.bias", "0.weight", "1.bias", "1.weight"], 4),
    ],
)
def test_existing_modules_state_dict_and_params_unchanged(factory, keys, n_params):
    m = factory()
    assert sorted(m.state_dict().keys()) == keys
    assert len(list(m.parameters())) == n_params
    assert list(m.buffers()) == []


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Linear(4, 2),
        lambda: Conv2d(2, 3, 2),
        lambda: LayerNorm(4),
        lambda: Embedding(5, 3),
        lambda: Sequential(Linear(4, 8), Linear(8, 2)),
    ],
)
def test_existing_modules_to_and_cast_still_work(factory):
    m = factory()
    before = sorted(m.state_dict().keys())
    m.to("cpu")
    assert sorted(m.state_dict().keys()) == before
    for p in m.parameters():
        assert p.device.is_cpu()
        assert p.requires_grad is True
    m.cast("float16")
    for p in m.parameters():
        assert p.dtype == "float16"
        assert p.requires_grad is True


def test_existing_module_load_state_dict_roundtrip():
    a, b = Linear(4, 2), Linear(4, 2)
    for i in range(4):
        for j in range(2):
            a.weight.set_item([i, j], float(i * 2 + j))
    b.load_state_dict(a.state_dict())
    assert b.weight.tolist() == pytest.approx(a.weight.tolist())


def test_sequential_forward_still_works_after_buffer_support():
    m = Sequential(Linear(4, 8), Linear(8, 2))
    out = m(ortho.ones([3, 4]))
    assert list(out.shape) == [3, 2]
