import numpy as np
import pytest

from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import SGD

from .workload import numbers


@pytest.mark.parametrize("strided", [False, True])
def test_unscale_transfers_logical_model_gradients_without_aliasing_low_precision_payloads(
    amp_model, amp_storage_case, model_gradient, assert_output, strided,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    raw = [[8., 16.], [24., 32.]]
    gradient = model_gradient(amp_model.weight, raw, strided=strided)
    logical = numbers(gradient).copy()
    scaler = GradScaler(init_scale=8)
    assert scaler.unscale_(optimizer) is None
    master = optimizer.state_dict()["masters"][0]
    assert_output(master.grad, logical / 8, dtype="float32", device=device)
    assert master.grad.requires_grad is False and master.grad.is_leaf
    assert_output(amp_model.weight.grad, logical / 8 if dtype == "float32" else logical,
                  dtype=dtype, device=device)
    master.grad.fill_(.5)
    if dtype != "float32":
        assert_output(amp_model.weight.grad, logical, dtype=dtype, device=device)
    assert amp_model.bias.grad is None and amp_model.unused.grad is None
    assert all(p.grad is None for p in optimizer.state_dict()["masters"][1:])


def test_zero_and_absent_model_gradients_have_different_optimizer_effects(
    amp_model, amp_storage_case, model_gradient, assert_output,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25, weight_decay=.5)
    model_gradient(amp_model.weight, np.zeros((2, 2)))
    assert optimizer.step() is None
    assert_output(amp_model.weight, np.array([[1., 2.], [3., 4.]]) * .875,
                  dtype=dtype, device=device)
    assert_output(amp_model.bias, [.5, -.5], dtype=dtype, device=device)
    assert_output(amp_model.unused, [6., 8.], dtype=dtype, device=device)
    masters = optimizer.state_dict()["masters"]
    assert_output(masters[0].grad, np.zeros((2, 2)), dtype="float32", device=device)
    assert masters[1].grad is None and masters[2].grad is None


def test_accumulated_model_backward_contributions_are_transferred_once(
    training_model, amp_training_case, typed_tensor, assert_output,
):
    device, dtype = amp_training_case
    model = training_model
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    inputs = typed_tensor([[1., 2.]], dtype=dtype, device=device)
    scaler = GradScaler(init_scale=4)
    for _ in range(2):
        scaler.scale(model(inputs).sum()).backward()
    masters = optimizer.state_dict()["masters"]
    if dtype != "float32":
        assert all(master.grad is None for master in masters)
    assert_output(model.weight.grad, [[8., 8.], [16., 16.]], dtype=dtype, device=device)
    scaler.unscale_(optimizer)
    assert_output(masters[0].grad, [[2., 2.], [4., 4.]], dtype="float32", device=device)
    assert_output(masters[1].grad, [2., 2.], dtype="float32", device=device)
    assert masters[2].grad is None
    scaler.step(optimizer)
    assert_output(model.weight, [[.5, 1.5], [2., 3.]], dtype=dtype, device=device)


def test_zero_grad_after_backward_and_explicit_transfer_clears_both_gradient_owners(
    training_model, amp_training_case, typed_tensor,
):
    device, dtype = amp_training_case
    model = training_model
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    scaler = GradScaler(init_scale=8, enabled=dtype != "bfloat16")
    scaler.scale(model(typed_tensor([[1., 2.]], dtype=dtype, device=device)).sum()).backward()
    assert model.weight.grad is not None
    scaler.unscale_(optimizer)
    for _ in range(2):
        assert optimizer.zero_grad() is None
        assert all(p.grad is None for p in model.parameters())
        assert all(p.grad is None for p in optimizer.state_dict()["masters"])
