"""Stable-owner AMP contracts; CPU low precision uses manual gradients only."""
import numpy as np
import pytest

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD

from .workload import snapshot, assert_state, pair, write_master, gradient, observation, unchanged












@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
@pytest.mark.parametrize("scaled", [False, True], ids=["ordinary", "scaled"])
def test_missing_current_gradient_clears_master_and_preserves_binding_history(
    amp_storage_case, typed_tensor, kind, scaled,
):
    device, dtype = amp_storage_case
    model = pair(typed_tensor, dtype, device)
    opt = MixedPrecisionOptimizer(model, kind, lr=.125, weight_decay=.2, decay_bias=True)
    scaler = GradScaler(init_scale=4)
    gradient(model.a, 4. if scaled else 1.)
    if scaled:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    model.a.zero_grad()  # Deliberately do not call wrapper.zero_grad().
    before = snapshot(opt.state_dict())
    if scaled:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    after = opt.state_dict()
    assert after["masters"][0].grad is None
    assert_state(after["masters"], before["masters"])
    if kind is Adam:
        assert after["inner"]["t"] == before["inner"]["t"] + 1
        assert_state(after["inner"]["m"], before["inner"]["m"])
        assert_state(after["inner"]["v"], before["inner"]["v"])


def test_sync_preserves_owners_ties_gradients_and_shared_mutation_generation(amp_storage_case, typed_tensor):
    device, dtype = amp_storage_case
    model = pair(typed_tensor, dtype, device)
    model.tie = Parameter(model.a)
    opt = MixedPrecisionOptimizer(model, SGD, lr=.125)
    owner = model.a
    masters = opt.state_dict()["masters"]
    assert len(masters) == 2
    gradient(owner, 2.)
    grad = owner.grad
    impl = owner._impl_id
    for value in (.75, .5):
        write_master(masters[0], value)
        version = owner._storage_version
        opt.sync_to_model()
        assert model.a is model.tie is owner
        assert owner._impl_id == impl
        assert owner.dtype == dtype and owner.device == ortho.Device(device)
        assert owner.is_leaf and owner.requires_grad and owner.grad._impl_id == grad._impl_id
        assert owner.tolist() == [value] * 2
        assert owner._storage_version == version + (dtype != "float32")
    opt.step()
    assert model.a is model.tie is owner
    opt.zero_grad()
    assert owner.grad is None and masters[0].grad is None


def test_sync_invalidates_graphs_saved_from_fp32_alias_of_model_storage(typed_tensor):
    model = pair(typed_tensor, "float32")
    opt = MixedPrecisionOptimizer(model, SGD, lr=.1)
    # Native operation's saved-value protection also covers FP32 reconciliation.
    model.half()
    master = opt.state_dict()["masters"][0]
    master.fill_(1.0001)
    model.cast("float32")
    saved = (model.a * model.a).sum()
    opt.sync_to_model()
    with pytest.raises(RuntimeError, match="modif|version|mutat"):
        saved.backward()


@pytest.mark.parametrize("initial,final", [
    ("float32", "float16"), ("float32", "bfloat16"),
    ("float16", "bfloat16"), ("bfloat16", "float16"),
    ("float16", "float32"), ("bfloat16", "float32"),
    ("float16", "float16"), ("float32", "float32"),
])
@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
def test_post_binding_cast_preserves_owner_master_precision_and_initialized_history(typed_tensor, initial, final, kind):
    model = pair(typed_tensor, initial)
    model.tie = Parameter(model.a)
    owner = model.a
    opt = MixedPrecisionOptimizer(model, kind, lr=.125)
    gradient(owner)
    opt.step()
    old = opt.state_dict()
    old_master = old["masters"][0]
    old_master.fill_(1.0001)
    history = snapshot(old["inner"])
    model.cast(final).to("cpu")
    state = opt.state_dict()
    master = state["masters"][0]
    assert model.a is model.tie is owner
    assert state["orig_dtypes"] == [final] * 2
    assert master.dtype == "float32" and master.device == owner.device
    if final == "float32":
        assert master is owner
        assert master.tolist() == old_master.tolist()
    else:
        assert master is not owner and master.is_leaf and master.requires_grad
        if initial != "float32":
            assert master is old_master
            assert master.tolist() == pytest.approx([1.0001] * 2, abs=1e-7)
        else:
            assert master.tolist() == ortho.cast(owner, "float32").tolist()
    assert_state(state["inner"], history)
    opt.zero_grad()
    gradient(owner, 2.)
    opt.step()
    assert model.a is owner and master.grad.dtype == "float32"
    if kind is Adam:
        assert opt.state_dict()["inner"]["t"] == history["t"] + 1


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_post_binding_physical_device_moves_preserve_master_and_adam_history(cuda_device, typed_tensor, dtype):
    model = pair(typed_tensor, dtype)
    owner = model.a
    model.tie = Parameter(owner)
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    gradient(owner)
    opt.step()
    write_master(opt.state_dict()["masters"][0], 1.0001)
    for device in (cuda_device, cuda_device, "cpu"):
        before = snapshot(opt.state_dict())
        model.to(device)
        current = opt.state_dict()
        assert model.a is model.tie is owner
        assert current["masters"][0].device == owner.device == ortho.Device(device)
        assert current["masters"][0].tolist() == before["masters"][0].tolist()
        if dtype == "float32":
            assert current["masters"][0] is owner
        for name in ("m", "v"):
            for new, old in zip(current["inner"][name], before["inner"][name]):
                if new is not None:
                    assert new.tolist() == old.tolist() and new.device == owner.device
        assert current["inner"]["t"] == before["inner"]["t"]
        opt.zero_grad()
        gradient(owner)
        opt.step()


def test_explicit_registration_replacement_does_not_rebind_amp(typed_tensor):
    model = pair(typed_tensor)
    old = model.a
    opt = MixedPrecisionOptimizer(model, SGD, lr=.125)
    model.a = Parameter(typed_tensor([9., 10.], dtype="float16"))
    gradient(old)
    opt.step()
    assert old.tolist() == [.875, 1.875]
    assert model.a.tolist() == [9., 10.]


@pytest.mark.parametrize("dtype", ["float64", "int32", "int64", "int8", "uint8", "bool"])
def test_unsupported_model_precision_rejects_at_construction(typed_tensor, dtype):
    model = pair(typed_tensor, dtype)
    owners = list(model.parameters())
    with pytest.raises(ValueError, match="float32/float16/bfloat16"):
        MixedPrecisionOptimizer(model, SGD, lr=.1)
    assert all(p is q for p, q in zip(model.parameters(), owners))


@pytest.mark.parametrize("scaled", [False, True])
def test_later_gradient_conversion_failure_publishes_nothing(typed_tensor, monkeypatch, scaled):
    model = pair(typed_tensor)
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    for p in model.parameters():
        gradient(p)
    opt.step()
    for p in model.parameters():
        gradient(p, 8.)
    before = observation(model, opt)
    scaler = GradScaler(init_scale=8)
    scaler_before = scaler.state_dict()
    native_cast = ortho.cast
    count = 0
    def fail_later(value, dtype):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("injected later gradient conversion")
        return native_cast(value, dtype)
    monkeypatch.setattr(ortho, "cast", fail_later)
    with pytest.raises(RuntimeError, match="injected later"):
        scaler.unscale_(opt) if scaled else opt.step()
    unchanged(model, opt, before)
    assert scaler.state_dict() == scaler_before
    assert count == 2


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_low_precision_accepts_graph_bearing_fp32_grad_as_independent_leaf(typed_tensor, dtype):
    model = pair(typed_tensor, dtype)
    opt = MixedPrecisionOptimizer(model, SGD, lr=.125)
    x = typed_tensor([1., 2.]).requires_grad_(True)
    grad = x * 2
    model.a.set_grad(grad)
    opt.step()
    master_grad = opt.state_dict()["masters"][0].grad
    assert master_grad.dtype == "float32" and master_grad.is_leaf and not master_grad.requires_grad
    assert not master_grad._has_grad_fn and master_grad._impl_id != grad._impl_id
    assert master_grad.tolist() == grad.tolist() == [2., 4.]
    master_grad.fill_(9.)
    assert grad.tolist() == [2., 4.]


@pytest.mark.parametrize("scaled", [False, True])
def test_invalid_later_gradient_preflight_preserves_earlier_fp32_grad(typed_tensor, scaled):
    model = pair(typed_tensor, "float32")
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    gradient(model.a, 8.)
    model.b.set_grad(typed_tensor([1., 2.], dtype="int32"))
    before = observation(model, opt)
    scaler = GradScaler(init_scale=8)
    before_scaler = scaler.state_dict()
    with pytest.raises(ValueError, match="AMP gradients"):
        scaler.unscale_(opt) if scaled else opt.step()
    unchanged(model, opt, before)
    assert scaler.state_dict() == before_scaler


@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
def test_rebind_rejects_count_shape_and_late_conversion_without_history_publication(typed_tensor, monkeypatch, kind):
    a, b = typed_tensor([1., 2.]).requires_grad_(True), typed_tensor([3., 4.]).requires_grad_(True)
    opt = kind([a, b], lr=.125)
    gradient(a)
    gradient(b)
    opt.step()
    state = opt.state_dict()
    before = snapshot(state)
    for replacements in ([a], [a, typed_tensor([1.])]):
        with pytest.raises(ValueError):
            opt._rebind_parameters(replacements)
        assert_state(opt.state_dict(), before)
    if kind is Adam:
        native_cast = ortho.cast
        calls = 0
        def fail_late(value, dtype):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("late bridge conversion")
            return native_cast(value, dtype)
        monkeypatch.setattr(ortho, "cast", fail_late)
        new = [typed_tensor([7., 8.], dtype="float64").requires_grad_(True) for _ in range(2)]
        with pytest.raises(RuntimeError, match="late bridge"):
            opt._rebind_parameters(new)
        assert calls == 4
        assert_state(opt.state_dict(), before)
        assert all(x is y for x, y in zip(opt.state_dict()["m"], state["m"]))
        assert all(x is y for x, y in zip(opt.state_dict()["v"], state["v"]))
    opt.zero_grad()
    assert a.grad is b.grad is None
