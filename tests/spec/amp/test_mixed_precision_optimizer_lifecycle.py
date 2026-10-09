import numpy as np
import pytest

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import Adam, SGD

from .workload import AffineModel, assert_state


def test_manual_sync_uses_master_values_and_preserves_full_precision_passthrough(
    amp_model, amp_storage_case, assert_output,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    masters = optimizer.state_dict()["masters"]
    with ortho.no_grad():
        masters[0].fill_(.75)
    for _ in range(3):
        assert optimizer.sync_to_model() is None
        assert_output(amp_model.weight, np.full((2, 2), .75), dtype=dtype, device=device)
        assert_output(amp_model.bias, [.5, -.5], dtype=dtype, device=device)
        assert amp_model.weight.requires_grad and amp_model.weight.is_leaf
    if dtype == "float32":
        assert amp_model.weight is masters[0]


@pytest.mark.parametrize("phase", ["before-gradient", "after-gradient", "after-step", "after-skipped-step"])
def test_zero_grad_clears_both_owners_at_every_normal_loop_boundary(
    amp_model, model_gradient, phase,
):
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    scaler = GradScaler(init_scale=4)
    if phase != "before-gradient":
        value = float("inf") if phase == "after-skipped-step" else 8.
        model_gradient(amp_model.weight, np.full((2, 2), value))
    if phase in ("after-step", "after-skipped-step"):
        scaler.step(optimizer)
    for _ in range(2):
        assert optimizer.zero_grad() is None
        assert all(p.grad is None for p in amp_model.parameters())
        assert all(p.grad is None for p in optimizer.state_dict()["masters"])


@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
def test_fp32_training_matches_the_wrapped_optimizer_without_replacing_parameters(
    kind, operator_device, typed_tensor,
):
    create = lambda values: typed_tensor(values, device=operator_device)
    model, reference = AffineModel(create), AffineModel(create)
    optimizer = MixedPrecisionOptimizer(model, kind, lr=.125)
    plain = kind(reference.parameters(), lr=.125)
    inputs = create([[.5, 1.], [1., -.5]])
    owners = list(model.parameters())
    for rate in (.125, .0625):
        assert optimizer.set_lr(rate) is None
        plain.set_lr(rate)
        assert optimizer.get_lr() == plain.get_lr() == rate
        for net, opt in ((model, optimizer), (reference, plain)):
            opt.zero_grad()
            result = net(inputs)
            (result * result).mean().backward()
            assert opt.step() is None
        assert_state(model.state_dict(), reference.state_dict())
        assert_state(optimizer.state_dict()["inner"], plain.state_dict())
        assert all(current is saved for current, saved in zip(model.parameters(), owners))
        assert all(p.requires_grad and p.is_leaf for p in owners)


def test_lr_forwarding_changes_the_next_master_update_and_public_state(
    amp_model, amp_storage_case, model_gradient, assert_output,
):
    device, dtype = amp_storage_case
    optimizer = MixedPrecisionOptimizer(amp_model, SGD, lr=.25)
    assert optimizer.get_lr() == .25
    model_gradient(amp_model.weight, np.full((2, 2), 2.))
    optimizer.step()
    assert optimizer.set_lr(.125) is None
    assert optimizer.get_lr() == optimizer.state_dict()["inner"]["lr"] == .125
    optimizer.zero_grad()
    model_gradient(amp_model.weight, np.full((2, 2), 2.))
    optimizer.step()
    assert_output(amp_model.weight, np.array([[1., 2.], [3., 4.]]) - .75,
                  dtype=dtype, device=device)


def test_transforming_a_model_before_optimizer_construction_binds_current_leaves(typed_tensor):
    model = AffineModel(lambda v: typed_tensor(v))
    model.to("cpu").half()
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    assert optimizer.state_dict()["orig_dtypes"] == ["float16"] * 3
    assert all(p.is_leaf and p.requires_grad for p in model.parameters())
    assert all(p.dtype == "float32" and p.is_leaf and p.requires_grad
               for p in optimizer.state_dict()["masters"])


def test_fp32_tied_parameter_is_accumulated_and_optimized_once(typed_tensor, assert_output):
    model = AffineModel(lambda v: typed_tensor(v))
    model.tied = ortho.nn.Parameter(model.weight)
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    assert len(optimizer.state_dict()["masters"]) == 3
    (model.weight * 2 + model.tied * 3).sum().backward()
    assert_output(model.weight.grad, np.full((2, 2), 5.), dtype="float32", device="cpu")
    optimizer.step()
    assert model.weight is model.tied
    assert_output(model.weight, np.array([[1., 2.], [3., 4.]]) - 1.25, dtype="float32", device="cpu")


def test_cuda_low_precision_tie_survives_one_logical_master_update(
    cuda_device, typed_tensor, model_gradient, assert_output,
):
    model = AffineModel(lambda v: typed_tensor(v, dtype="float16", device=cuda_device))
    model.tied = ortho.nn.Parameter(model.weight)
    owner = model.weight
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=.25)
    assert len(optimizer.state_dict()["masters"]) == 3
    model_gradient(owner, np.full((2, 2), 5.))
    optimizer.step()
    assert model.weight is model.tied is owner
    assert_output(owner, np.array([[1., 2.], [3., 4.]]) - 1.25, dtype="float16", device=cuda_device)
