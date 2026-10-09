import numpy as np
import pytest

from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import Adam, SGD

from .workload import assert_state, snapshot


@pytest.mark.parametrize("explicit_unscale", [False, True])
def test_scaled_training_steps_once_without_double_unscale(
    training_model, amp_training_case, typed_tensor, assert_output, explicit_unscale,
):
    device, dtype = amp_training_case
    model = training_model
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    scaler = GradScaler(init_scale=8, growth_interval=1, enabled=dtype != "bfloat16")
    inputs = typed_tensor([[1., 2.]], dtype=dtype, device=device)
    result = model(inputs)
    loss = result.sum()
    scaled = scaler.scale(loss)
    assert scaled.requires_grad and not scaled.is_leaf
    scaled.backward()
    if explicit_unscale:
        assert scaler.unscale_(optimizer) is None
    assert scaler.step(optimizer) is None
    assert_output(model.weight, [[.75, 1.75], [2.5, 3.5]], dtype=dtype, device=device)
    assert_output(model.bias, [.25, -.75], dtype=dtype, device=device)
    for master, parameter in zip(optimizer.state_dict()["masters"], model.parameters()):
        assert master.dtype == "float32" and master.is_leaf and master.requires_grad
        assert parameter.dtype == dtype and parameter.is_leaf and parameter.requires_grad
    assert scaler.scale_factor == (8 if dtype == "bfloat16" else 16)
    assert scaler.found_inf is False
    assert result.grad is None and loss.grad is None and scaled.grad is None


@pytest.mark.parametrize("nonfinite", [float("inf"), float("-inf"), float("nan")],
                         ids=["positive-inf", "negative-inf", "nan"])
def test_nonfinite_step_preserves_model_masters_and_existing_adam_history(
    amp_model, amp_storage_case, model_gradient, nonfinite,
):
    optimizer = MixedPrecisionOptimizer(amp_model, Adam, lr=.125)
    scaler = GradScaler(init_scale=8, backoff_factor=.25)
    model_gradient(amp_model.weight, np.full((2, 2), 8.))
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    before_model = snapshot(amp_model.state_dict())
    state = optimizer.state_dict()
    before_masters, before_inner = snapshot(state["masters"]), snapshot(state["inner"])
    model_gradient(amp_model.weight, np.full((2, 2), 16.))
    model_gradient(amp_model.bias, [8., nonfinite])
    assert scaler.step(optimizer) is None
    assert scaler.found_inf is True and scaler.scale_factor == 2
    assert_state(amp_model.state_dict(), before_model)
    assert_state(optimizer.state_dict()["masters"], before_masters)
    assert_state(optimizer.state_dict()["inner"], before_inner)
    optimizer.zero_grad()
    assert scaler.found_inf is True
    assert scaler.update() is None
    assert scaler.found_inf is False and scaler.scale_factor == 2


def test_disabled_step_applies_unscaled_gradients_without_scale_adaptation(
    amp_model, amp_storage_case, model_gradient, assert_output,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    model_gradient(amp_model.weight, np.full((2, 2), 2.))
    scaler = GradScaler(init_scale=128, growth_interval=1, enabled=False)
    assert scaler.step(optimizer) is None
    assert_output(amp_model.weight, np.array([[1., 2.], [3., 4.]]) - .5, dtype=dtype, device=device)
    assert scaler.scale_factor == 128 and scaler.found_inf is False
    assert scaler.update() is None
    assert scaler.state_dict()["growth_tracker"] == 0


def test_cuda_sync_invalidates_a_saved_model_value_after_a_real_optimizer_step(
    cuda_device, typed_tensor,
):
    from orthon.nn import Module, Parameter
    model = Module()
    model.weight = Parameter(typed_tensor([1., 2.], dtype="float16", device=cuda_device))
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    root = (model.weight * model.weight).sum()
    owner = model.weight
    GradScaler(init_scale=4).scale((model.weight * 2).sum()).backward()
    scaler = GradScaler(init_scale=4)
    scaler.step(optimizer)
    assert model.weight is owner
    previous = owner.grad.tolist()
    with pytest.raises(RuntimeError, match="modified in-place"):
        root.backward()
    assert owner.grad.tolist() == previous
