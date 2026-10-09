"""Scaler validation and runtime cycle identity are independent of AMP storage."""
import pytest

from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import SGD
from .workload import pair, gradient, observation, unchanged


@pytest.mark.parametrize("field", ["init_scale", "growth_factor", "backoff_factor"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, False, "2"])
@pytest.mark.parametrize("enabled", [True, False])
def test_nonfinite_and_nonreal_settings_reject_even_when_disabled(field, value, enabled):
    with pytest.raises((TypeError, ValueError)):
        GradScaler(**{field: value}, enabled=enabled)


@pytest.mark.parametrize("field,value", [
    ("growth_interval", 2.5), ("growth_interval", 2.), ("growth_interval", True),
    ("enabled", 1), ("enabled", "yes"), ("enabled", None),
])
def test_constructor_rejects_noninteger_interval_and_nonbool_enabled(field, value):
    with pytest.raises((TypeError, ValueError)):
        GradScaler(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("version", True), ("scale", float("nan")), ("scale", True),
    ("growth_factor", float("inf")), ("backoff_factor", float("-inf")),
    ("growth_interval", 2.5), ("growth_interval", True),
    ("enabled", 1), ("unscaled", 0), ("found_inf", "false"),
    ("growth_tracker", -1), ("growth_tracker", 2),
    ("growth_tracker", 1.5), ("growth_tracker", True), ("enabled", "missing"),
])
def test_scaler_load_validates_complete_state_before_committing(field, value):
    scaler = GradScaler(init_scale=8, growth_interval=2)
    before = scaler.state_dict()
    state = GradScaler(init_scale=32, growth_interval=2).state_dict()
    if value == "missing":
        del state[field]
    else:
        state[field] = value
    with pytest.raises((RuntimeError, ValueError, TypeError, KeyError)):
        scaler.load_state_dict(state)
    assert scaler.state_dict() == before


@pytest.mark.parametrize("first", ["unscale", "step"])
@pytest.mark.parametrize("second", ["unscale", "step"])
def test_one_optimizer_per_enabled_cycle_rejects_second_before_any_mutation(typed_tensor, first, second):
    a, b = pair(typed_tensor), pair(typed_tensor)
    x, y = MixedPrecisionOptimizer(a, SGD, lr=.125), MixedPrecisionOptimizer(b, SGD, lr=.125)
    scaler = GradScaler(init_scale=8)
    gradient(a.a, 8.)
    gradient(b.a, 8.)
    (scaler.unscale_ if first == "unscale" else scaler.step)(x)
    before_x, before_y = observation(a, x), observation(b, y)
    before_scaler = scaler.state_dict()
    with pytest.raises(RuntimeError, match="one optimizer"):
        (scaler.unscale_ if second == "unscale" else scaler.step)(y)
    unchanged(a, x, before_x)
    unchanged(b, y, before_y)
    assert scaler.state_dict() == before_scaler
    scaler.update()
    scaler.step(y)
    assert y.state_dict()["masters"][0].tolist() == [.875, 1.875]


def test_same_optimizer_unscale_is_idempotent_and_restored_transient_requires_update(typed_tensor):
    model = pair(typed_tensor)
    opt = MixedPrecisionOptimizer(model, SGD, lr=.125)
    gradient(model.a, 8.)
    scaler = GradScaler(init_scale=8)
    scaler.unscale_(opt)
    grad = opt.state_dict()["masters"][0].grad
    scaler.unscale_(opt)
    assert opt.state_dict()["masters"][0].grad._impl_id == grad._impl_id and grad.tolist() == [1., 1.]
    restored = GradScaler()
    restored.load_state_dict(scaler.state_dict())
    before = observation(model, opt)
    with pytest.raises(RuntimeError, match="update"):
        restored.step(opt)
    unchanged(model, opt, before)
    restored.update()
    opt.zero_grad()
    gradient(model.a, 8.)
    restored.step(opt)
    assert opt.state_dict()["masters"][0].tolist() == [.875, 1.875]


def test_disabled_scaler_can_delegate_to_several_optimizers_without_bookkeeping(typed_tensor):
    scaler = GradScaler(init_scale=8, enabled=False)
    before = scaler.state_dict()
    for _ in range(2):
        model = pair(typed_tensor)
        opt = MixedPrecisionOptimizer(model, SGD, lr=.125)
        gradient(model.a)
        assert scaler.scale(model.a) is model.a
        scaler.unscale_(opt)
        assert opt.state_dict()["masters"][0].grad is None
        scaler.step(opt)
        scaler.update()
        assert scaler.state_dict() == before
        assert model.a.tolist() == [.875, 1.875]
