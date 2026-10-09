import numpy as np
import pytest

from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import SGD


@pytest.mark.parametrize("scale", [.5, 1., 8.])
@pytest.mark.parametrize("enabled", [False, True])
def test_scaling_preserves_loss_values_metadata_and_its_gradient_path(
    tensor, tensor_case, assert_output, assert_grad, scale, enabled,
):
    device, dtype = tensor_case
    leaf = tensor([1., -2., 3.], requires_grad=True)
    loss = leaf * leaf + 1
    scaler = GradScaler(init_scale=scale, enabled=enabled)
    first, second = scaler.scale(loss), scaler.scale(loss)
    multiplier = scale if enabled else 1
    assert_output(first, np.array([2., 5., 10.]) * multiplier, dtype=dtype, device=device)
    assert_output(loss, [2., 5., 10.], dtype=dtype, device=device)
    if not enabled:
        assert first is loss and second is loss
    (first + second).sum().backward()
    assert_grad(leaf, np.array([1., -2., 3.]) * 4 * multiplier)
    assert loss.grad is None
    assert scaler.scale_factor == scale and scaler.found_inf is False


@pytest.mark.parametrize("nonfinite", [None, float("inf"), float("-inf"), float("nan")],
                         ids=["finite", "positive-inf", "negative-inf", "nan"])
def test_unscale_is_idempotent_and_checks_all_used_parameters(
    amp_model, amp_storage_case, model_gradient, assert_output, nonfinite,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    model_gradient(amp_model.weight, [[8., 16.], [24., 32.]])
    model_gradient(amp_model.bias, [8., 16. if nonfinite is None else nonfinite])
    scaler = GradScaler(init_scale=8)
    for _ in range(2):
        assert scaler.unscale_(optimizer) is None
        assert scaler.found_inf is (nonfinite is not None)
        master = optimizer.state_dict()["masters"][0]
        assert_output(master.grad, [[1., 2.], [3., 4.]], dtype="float32", device=device)
        assert optimizer.state_dict()["masters"][2].grad is None
    if dtype != "float32":
        assert_output(amp_model.weight.grad, [[8., 16.], [24., 32.]], dtype=dtype, device=device)


def test_disabled_unscale_neither_transfers_nor_checks_model_gradients(
    amp_model, amp_storage_case, model_gradient,
):
    _, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    model_gradient(amp_model.bias, [float("inf"), 8.])
    scaler = GradScaler(init_scale=8, enabled=False)
    assert scaler.unscale_(optimizer) is None
    assert scaler.found_inf is False and scaler.scale_factor == 8
    if dtype != "float32":
        assert all(master.grad is None for master in optimizer.state_dict()["masters"])
    assert amp_model.bias.grad.tolist()[0] == float("inf")


def test_update_makes_fresh_gradients_eligible_for_one_more_unscale(
    amp_model, amp_storage_case, model_gradient, assert_output,
):
    device, _ = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    scaler = GradScaler(init_scale=8)
    model_gradient(amp_model.bias, [8., 16.])
    scaler.unscale_(optimizer)
    assert scaler.update() is None
    optimizer.zero_grad()
    model_gradient(amp_model.bias, [24., 32.])
    scaler.unscale_(optimizer)
    assert_output(optimizer.state_dict()["masters"][1].grad, [3., 4.], dtype="float32", device=device)
    assert scaler.scale_factor == 8 and scaler.found_inf is False
