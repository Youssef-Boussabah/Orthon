# Parameter transformations preserve registered owners and clear stale gradients.
# Moves/changed casts still clear current grads; held gradient tensors survive.
import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import SGD


@pytest.mark.parametrize("destination", ["cpu", "cuda:0"])
def test_moves_replace_parameter_tensors_and_preserve_module_roles_and_values(
    request, module_factory, tensor_case, assert_output, assert_grad, destination,
):
    if destination != "cpu":
        request.getfixturevalue("cuda_device")
    model = module_factory().eval()
    model.branch.bias.requires_grad_(False)
    model.branch.running.requires_grad_(True)
    (model.weight * 3).sum().backward()
    (model.branch.running * 4).sum().backward()
    old_weight, child, scratch = model.weight, model.branch, model.scratch
    old_weight_grad = old_weight.grad
    old_buffers = dict(model.named_buffers())
    dtype = tensor_case[1]
    assert model.to(ortho.Device(destination)) is model
    assert model.weight is old_weight
    assert model.branch is child
    assert model.scratch is scratch
    assert model.training is model.branch.training is False
    assert model.weight.requires_grad is True
    assert model.branch.bias.requires_grad is False
    assert model.weight.is_leaf is model.branch.bias.is_leaf is True
    assert model.weight.grad is None
    assert old_weight_grad.tolist() == [3, 3, 3, 3]
    assert old_weight_grad.requires_grad is False
    assert_output(model.weight, [[1, 2], [3, 4]], dtype=dtype, device=destination)
    assert_output(model.branch.bias, [5, 6], dtype=dtype, device=destination)
    for name, moved in model.named_buffers():
        original = old_buffers[name]
        assert moved is not original
        assert_output(moved, original.tolist(), dtype=original.dtype, device=destination)
        assert moved.requires_grad is original.requires_grad
        assert moved.is_leaf is True
        assert moved.grad is None
    assert_grad(old_buffers["branch.running"], [4, 4])
    assert model.branch.optional is None
    assert "branch.cache" not in model.state_dict()


@pytest.mark.parametrize("destination_dtype", ["float32", "float64", "float16", "bfloat16"])
def test_casts_convert_floating_state_preserve_flags_and_leave_integer_buffers_alone(
    module_factory, tensor_case, assert_output, assert_grad, destination_dtype,
):
    model = module_factory()
    model.branch.bias.requires_grad_(False)
    model.branch.running.requires_grad_(True)
    (model.weight * 2).sum().backward()
    model.branch.running.sum().backward()
    old_weight, counter, scratch, child = model.weight, model.branch.counter, model.scratch, model.branch
    old_weight_grad = old_weight.grad
    device, old_dtype = tensor_case
    assert model.cast(destination_dtype) is model
    assert model.weight is old_weight
    assert model.branch is child
    assert model.scratch is scratch
    assert model.branch.counter is counter
    assert model.branch.counter.dtype == "int64"
    assert model.branch.optional is None
    assert model.weight.is_leaf is True
    assert model.weight.requires_grad is True
    assert model.branch.bias.requires_grad is False
    assert_output(model.weight, [[1, 2], [3, 4]], dtype=destination_dtype, device=device)
    assert_output(model.branch.bias, [5, 6], dtype=destination_dtype, device=device)
    assert_output(model.branch.running, [7, 8], dtype=destination_dtype, device=device)
    assert_output(model.branch.cache, [9, 10], dtype=destination_dtype, device=device)
    assert model.branch.running.requires_grad is True
    assert model.branch.running.is_leaf is True
    assert old_weight_grad.tolist() == [2, 2, 2, 2]
    assert old_weight_grad.requires_grad is False
    if destination_dtype == old_dtype:
        assert_grad(model.weight, [[2, 2], [2, 2]])
        assert_grad(model.branch.running, [1, 1])
        old_weight.mul_(2)
        assert_output(model.weight, [[2, 4], [6, 8]], dtype=destination_dtype, device=device)
    else:
        assert model.weight.grad is None
        assert model.branch.running.grad is None
        old_weight_grad.mul_(2)
        assert_output(model.weight, [[1, 2], [3, 4]], dtype=destination_dtype, device=device)


@pytest.mark.parametrize("method,dtype", [("half", "float16"), ("bfloat16", "bfloat16")])
def test_precision_convenience_methods_follow_the_recursive_cast_contract(module_factory, tensor_case, method, dtype):
    model = module_factory()
    assert getattr(model, method)() is model
    assert {p.dtype for p in model.parameters()} == {dtype}
    assert model.branch.running.dtype == model.branch.cache.dtype == dtype
    assert model.branch.counter.dtype == "int64"
    assert all(p.device == ortho.Device(tensor_case[0]) for p in model.parameters())


def test_model_cast_keeps_integer_and_boolean_buffer_dtypes(storage_dtype, typed_tensor):
    model = Module()
    source = typed_tensor([0, 1], dtype=storage_dtype)
    model.register_buffer("value", source)
    model.cast("float64")
    floating = storage_dtype in ("float16", "bfloat16", "float32", "float64")
    assert model.value.dtype == ("float64" if floating else storage_dtype)
    if not floating:
        assert model.value is source
    assert model.value.tolist() == [0, 1]


@pytest.mark.parametrize("operation", ["to", "cast"])
def test_invalid_movement_requests_fail_before_changing_registered_state(module_factory, operation):
    model = module_factory()
    before = model.state_dict()
    with pytest.raises(ValueError):
        getattr(model, operation)("not-a-device-or-dtype")
    assert all(model.state_dict()[name] is value for name, value in before.items())


def test_cpu_only_build_rejects_module_cuda_movement(typed_tensor):
    if ortho.cuda_enabled():
        pytest.skip("contract applies to CPU-only builds")
    model = Module()
    source = typed_tensor([2, 3])
    model.weight = Parameter(source)
    with pytest.raises(RuntimeError):
        model.to("cuda:0")
    assert model.weight is source
    assert model.weight.device == ortho.Device("cpu")


@pytest.mark.parametrize("operation", ["to", "changed_cast", "same_cast"])
def test_existing_optimizer_references_follow_tensor_ownership_after_transformation(
    tensor, tensor_case, assert_tensor, operation,
):
    model = Module()
    original = tensor([2, 3])
    model.weight = Parameter(original)
    optimizer = SGD(model.parameters(), lr=.25)
    (model.weight * 2).sum().backward()
    if operation == "to":
        model.to(tensor_case[0])
    elif operation == "same_cast":
        model.cast(tensor_case[1])
    else:
        target = "float64" if tensor_case[1] != "float64" else "float32"
        model.cast(target)
    optimizer.step()
    assert model.weight is original
    assert_tensor(original, [1.5, 2.5] if operation == "same_cast" else [2, 3])
    assert_tensor(model.weight, [1.5, 2.5] if operation == "same_cast" else [2, 3])
