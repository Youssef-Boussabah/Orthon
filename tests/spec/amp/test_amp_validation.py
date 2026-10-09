import pytest

from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import Adam, SGD

from .workload import assert_state, snapshot


@pytest.mark.parametrize("settings", [
    {"init_scale": 0}, {"init_scale": -1}, {"growth_factor": 1}, {"growth_factor": .5},
    {"backoff_factor": 0}, {"backoff_factor": 1}, {"backoff_factor": -1},
    {"growth_interval": 0}, {"growth_interval": -1},
])
@pytest.mark.parametrize("enabled", [False, True])
def test_scaler_rejects_ordinary_invalid_factors_even_when_disabled(settings, enabled):
    with pytest.raises(ValueError):
        GradScaler(**settings, enabled=enabled)


@pytest.mark.parametrize("field,value,error", [
    ("version", 2, RuntimeError), ("version", None, RuntimeError),
    ("scale", 0, ValueError), ("scale", -1, ValueError),
    ("growth_factor", 1, ValueError), ("backoff_factor", 0, ValueError),
    ("backoff_factor", 1, ValueError), ("growth_interval", 0, ValueError),
    ("scale", "invalid", ValueError), ("growth_interval", None, TypeError),
])
def test_invalid_scaler_numeric_or_version_state_preserves_the_existing_scaler(field, value, error):
    scaler = GradScaler(init_scale=8, growth_interval=2)
    before = scaler.state_dict()
    incoming = GradScaler(init_scale=32, growth_factor=4).state_dict()
    incoming[field] = value
    with pytest.raises(error):
        scaler.load_state_dict(incoming)
    assert scaler.state_dict() == before


@pytest.mark.parametrize("field", ["scale", "growth_factor", "backoff_factor", "growth_interval"])
def test_missing_numeric_scaler_fields_reject_without_changing_existing_state(field):
    scaler = GradScaler(init_scale=8)
    before = scaler.state_dict()
    incoming = GradScaler(init_scale=32).state_dict()
    del incoming[field]
    with pytest.raises(KeyError):
        scaler.load_state_dict(incoming)
    assert scaler.state_dict() == before


@pytest.mark.parametrize("kind", [None, object(), SGD([], lr=.25)])
def test_optimizer_constructor_requires_a_callable_class_or_factory(amp_model, kind):
    before = snapshot(amp_model.state_dict())
    with pytest.raises(TypeError):
        MixedPrecisionOptimizer(amp_model, kind, lr=.25)
    assert_state(amp_model.state_dict(), before)
    assert all(p.grad is None and p.is_leaf for p in amp_model.parameters())


@pytest.mark.parametrize("fault", ["version", "dtype-count", "master-count", "dtype", "late-shape",
                                  "missing-dtypes", "missing-masters"])
def test_amp_outer_state_validation_preserves_model_and_optimizer_state(
    amp_model, model_gradient, fault,
):
    optimizer = MixedPrecisionOptimizer(amp_model, Adam, lr=.125)
    model_gradient(amp_model.weight, [[1., 2.], [3., 4.]])
    optimizer.step()
    optimizer.zero_grad()
    before_model, before_state = snapshot(amp_model.state_dict()), snapshot(optimizer.state_dict())
    state = snapshot(before_state)
    if fault == "version":
        state["version"] = 2
    elif fault == "dtype-count":
        state["orig_dtypes"].pop()
    elif fault == "master-count":
        state["masters"].pop()
    elif fault == "dtype":
        state["orig_dtypes"][-1] = "int32"
    elif fault == "late-shape":
        state["masters"][-1] = state["masters"][0]
    else:
        del state["orig_dtypes" if fault == "missing-dtypes" else "masters"]
    category = RuntimeError if fault == "version" else KeyError if fault.startswith("missing") else ValueError
    with pytest.raises(category):
        optimizer.load_state_dict(state)
    assert_state(amp_model.state_dict(), before_model)
    assert_state(optimizer.state_dict(), before_state)
