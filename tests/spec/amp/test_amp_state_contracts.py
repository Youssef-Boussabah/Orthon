import numpy as np
import pytest

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import Adam, SGD

from .workload import AffineModel, assert_state, snapshot


def test_optimizer_state_has_live_master_values_and_independent_configuration_scalars(
    amp_model, model_gradient,
):
    optimizer = MixedPrecisionOptimizer(amp_model, Adam, lr=.125)
    state = optimizer.state_dict()
    saved = snapshot(state)
    assert state["inner"]["m"] == [None] * 3 and state["inner"]["v"] == [None] * 3
    model_gradient(amp_model.weight, np.full((2, 2), 2.))
    optimizer.step()
    current = optimizer.state_dict()
    for name in ("m", "v"):
        moment = current["inner"][name][0]
        assert moment.dtype == "float32" and moment.device == current["masters"][0].device
        assert moment.shape == amp_model.weight.shape
        assert moment.requires_grad is False and moment.is_leaf
        assert current["inner"][name][1:] == [None, None]
    assert state["masters"][0].tolist() != saved["masters"][0].tolist()
    assert saved["masters"][0].tolist() == [1., 2., 3., 4.]
    optimizer.set_lr(.25)
    assert state["inner"]["lr"] == .125
    assert optimizer.state_dict()["inner"]["lr"] == .25


@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
def test_independent_amp_state_and_model_restore_continue_the_same_next_update(
    amp_storage_case, typed_tensor, model_gradient, kind,
):
    device, dtype = amp_storage_case
    create = lambda values: typed_tensor(values, dtype=dtype, device=device)
    source, restored = AffineModel(create), AffineModel(create)
    optimizer = MixedPrecisionOptimizer(source, kind, lr=.125)
    model_gradient(source.weight, [[1., 2.], [3., 4.]])
    optimizer.step()
    optimizer.zero_grad()
    optimizer.set_lr(.0625)
    model_state, amp_state = snapshot(source.state_dict()), snapshot(optimizer.state_dict())
    restored.load_state_dict(model_state)
    resumed = MixedPrecisionOptimizer(restored, kind, lr=1.)
    assert resumed.load_state_dict(amp_state) is None
    assert resumed.get_lr() == .0625
    assert_state(resumed.state_dict(), optimizer.state_dict())
    for model, opt in ((source, optimizer), (restored, resumed)):
        opt.zero_grad()
        model_gradient(model.weight, [[4., 3.], [2., 1.]])
        opt.step()
    assert_state(restored.state_dict(), source.state_dict())
    assert_state(resumed.state_dict(), optimizer.state_dict())
    before = snapshot(optimizer.state_dict())
    resumed.state_dict()["masters"][0].fill_(9)
    assert_state(optimizer.state_dict(), before)


@pytest.mark.parametrize("saved_dtype", ["float16", "float64"])
def test_converted_low_precision_master_state_restores_fp32_detached_leaves(
    saved_dtype, typed_tensor,
):
    model = AffineModel(lambda values: typed_tensor(values, dtype="float16"))
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    state = snapshot(optimizer.state_dict())
    for index, master in enumerate(state["masters"]):
        state["masters"][index] = master.cast(saved_dtype)
    assert optimizer.load_state_dict(state, map_location="cpu") is None
    for master, parameter in zip(optimizer.state_dict()["masters"], model.parameters()):
        assert master.dtype == "float32" and master.device == ortho.Device("cpu")
        assert master.is_leaf and master.requires_grad and master.grad is None
        assert master.shape == parameter.shape


@pytest.mark.parametrize("nonfinite", [False, True])
def test_scaler_state_restores_growth_progress_and_nonfinite_recovery(
    amp_model, model_gradient, nonfinite,
):
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    scaler = GradScaler(init_scale=8, growth_factor=4, backoff_factor=.25, growth_interval=2)
    model_gradient(amp_model.bias, [8., 8.])
    scaler.step(optimizer)
    scaler.update()
    state = scaler.state_dict()
    assert state == {"version": 1, "scale": 8., "growth_factor": 4., "backoff_factor": .25,
                     "growth_interval": 2, "enabled": True, "unscaled": False,
                     "found_inf": False, "growth_tracker": 1}
    restored = GradScaler(init_scale=1, enabled=False)
    assert restored.load_state_dict(state) is None
    assert restored.enabled is True and restored.scale_factor == 8 and restored.found_inf is False
    optimizer.zero_grad()
    model_gradient(amp_model.bias, [8., float("inf") if nonfinite else 8.])
    restored.step(optimizer)
    assert restored.scale_factor == (2 if nonfinite else 32)
    assert restored.found_inf is nonfinite
    assert state["scale"] == 8 and state["growth_tracker"] == 1
    restored.update()
    optimizer.zero_grad()
    model_gradient(amp_model.bias, [8., 8.])
    restored.step(optimizer)
    assert restored.scale_factor == (2 if nonfinite else 32)


def test_disabled_scaler_state_roundtrip_retains_the_disabled_training_policy(
    amp_model, model_gradient,
):
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    source = GradScaler(init_scale=32, growth_interval=1, enabled=False)
    restored = GradScaler(init_scale=1)
    assert restored.load_state_dict(source.state_dict()) is None
    model_gradient(amp_model.bias, [2., 2.])
    restored.step(optimizer)
    restored.update()
    assert restored.state_dict() == source.state_dict()
    assert restored.enabled is False and restored.scale_factor == 32


def test_defaulted_optional_scaler_progress_fields_start_a_clean_step():
    state = GradScaler(init_scale=8, growth_interval=2).state_dict()
    for key in ("unscaled", "found_inf", "growth_tracker"):
        del state[key]
    scaler = GradScaler(init_scale=1)
    assert scaler.load_state_dict(state) is None
    assert scaler.found_inf is False and scaler.scale_factor == 8
    assert scaler.state_dict()["unscaled"] is False
    assert scaler.state_dict()["growth_tracker"] == 0
