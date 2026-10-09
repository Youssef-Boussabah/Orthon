import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


def test_adam_state_is_lazy_per_parameter_and_preserves_metadata(
    adam_parameter, assign_gradient, assert_tensor,
):
    used, zero, never = adam_parameter(3.), adam_parameter([[4., -2.]]), adam_parameter([5.])
    opt = Adam([used, zero, never], betas=(0.5, 0.75), eps=1.)
    assert opt.state_dict()["t"] == 0
    assert opt.state_dict()["m"] == opt.state_dict()["v"] == [None, None, None]
    assign_gradient(used, 2.)
    assign_gradient(zero, [[0., 0.]])
    opt.step()
    state = opt.state_dict()
    for key, expected in (("m", 1.), ("v", 1.)):
        assert_tensor(state[key][0], expected)
        assert_tensor(state[key][1], [[0., 0.]])
        assert state[key][2] is None
        for p, value in zip((used, zero), state[key][:2]):
            assert value.shape == p.shape
            assert value.dtype == "float32"
            assert value.device == p.device
            assert value.requires_grad is False
            assert value.is_leaf is True
            assert value.grad is None


def test_equal_parameters_have_separate_moments_from_each_other_and_gradients(
    adam_parameter, assign_gradient, assert_tensor,
):
    a, b = adam_parameter([3.]), adam_parameter([3.])
    for p in (a, b):
        assign_gradient(p, [2.])
    opt = Adam([a, b], lr=0.25, betas=(0.5, 0.75), eps=1.)
    opt.step()
    state = opt.state_dict()
    state["m"][0].mul_(3.)
    assert_tensor(state["m"][1], [1.])
    assert_tensor(state["v"][0], [1.])
    assert_tensor(state["v"][1], [1.])
    for p in (a, b):
        assert_tensor(p, [3. - 1 / 6])
        assert_tensor(p.grad, [2.])


def test_separate_adam_constructions_own_independent_state(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    left, right = (Adam([p], lr=0., betas=(0.5, 0.75), eps=1.) for _ in range(2))
    assign_gradient(p, [2.])
    left.step()
    assign_gradient(p, [4.])
    right.step()
    assert_tensor(left.state_dict()["m"][0], [1.])
    assert_tensor(right.state_dict()["m"][0], [2.])
    right.state_dict()["m"][0].mul_(2.)
    assert_tensor(left.state_dict()["m"][0], [1.])
    assert_tensor(p, [3.])


def test_state_exports_new_mapping_and_scalar_values(parameter, optimizer):
    opt = optimizer([parameter([3.])])
    first, second = opt.state_dict(), opt.state_dict()
    assert first is not second
    assert list(first) == list(second)
    first["lr"] = 9.
    first["weight_decay"] = 9.
    assert opt.get_lr() == 0.25
    assert opt.state_dict()["weight_decay"] == 0.
    opt.set_lr(0.5)
    assert second["lr"] == 0.25


def test_adam_exported_moment_values_are_live_but_counter_is_a_value(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    opt = Adam([p], betas=(0.5, 0.75), eps=1.)
    assign_gradient(p, [2.])
    opt.step()
    earlier = opt.state_dict()
    assign_gradient(p, [4.])
    opt.step()
    assert earlier["t"] == 1
    assert opt.state_dict()["t"] == 2
    assert_tensor(earlier["m"][0], [2.5])
    assert_tensor(earlier["v"][0], [4.75])


def test_zero_grad_does_not_reset_adam_counter_or_history(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    assign_gradient(p, [2.])
    opt = Adam([p], betas=(0.5, 0.75), eps=1.)
    opt.step()
    opt.zero_grad()
    state = opt.state_dict()
    assert state["t"] == 1
    assert_tensor(state["m"][0], [1.])
    assert_tensor(state["v"][0], [1.])
    assert p.grad is None


def test_loading_copied_state_restores_resume_trajectory_and_configuration(
    parameter, optimizer, optimizer_case, assign_gradient, assert_tensor, copy_optimizer_state,
):
    kind, _, _ = optimizer_case
    p, unused = parameter([[3., -2.]]), parameter([5.])
    options = {"betas": (0.5, 0.75), "eps": 1.} if kind is Adam else {}
    opt = optimizer([p, unused], lr=0.125, weight_decay=0.5, decay_bias=True, **options)
    assign_gradient(p, [[2., -4.]])
    opt.step()
    # Public state contains live tensors; copy before branching the training path.
    checkpoint = copy_optimizer_state(opt.state_dict())
    q, other_unused = parameter([p.tolist()]), parameter([5.])
    restored = optimizer([q, other_unused], lr=0.5)
    assert restored.load_state_dict(checkpoint) is None
    exported = restored.state_dict()
    for key in ("lr", "weight_decay", "decay_bias"):
        assert exported[key] == checkpoint[key]
    if kind is Adam:
        for key in ("betas", "eps", "t"):
            assert exported[key] == checkpoint[key]
        assert exported["m"][1] is exported["v"][1] is None
    for gradient in ([[4., 2.]], [[-1., 3.]]):
        assign_gradient(p, gradient)
        assign_gradient(q, gradient)
        opt.step()
        restored.step()
        assert_tensor(q, [p.tolist()])
        assert_tensor(other_unused, [5.])
        if kind is Adam:
            for key in ("m", "v"):
                assert_tensor(restored.state_dict()[key][0], [opt.state_dict()[key][0].tolist()])
            assert restored.state_dict()["t"] == opt.state_dict()["t"]


def test_loading_configuration_does_not_load_parameter_values_or_gradients(
    parameter, optimizer, assign_gradient, assert_tensor,
):
    p, q = parameter([3.]), parameter([7.])
    opt = optimizer([p], lr=0.125)
    assign_gradient(q, [1.])
    restored = optimizer([q], lr=0.5)
    restored.load_state_dict(opt.state_dict())
    assert_tensor(q, [7.])
    assert_tensor(q.grad, [1.])
    restored.step()
    assert_tensor(q, [6.875])
    assert_tensor(p, [3.])


def test_loading_uninitialized_adam_state_resets_history_without_clearing_gradients(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    opt = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    initial = opt.state_dict()
    # Capture the uninitialized state independently of its mutable moment sequences.
    initial["m"], initial["v"] = [None], [None]
    assign_gradient(p, [2.])
    opt.step()
    opt.load_state_dict(initial)
    assert opt.state_dict()["t"] == 0
    assert opt.state_dict()["m"] == opt.state_dict()["v"] == [None]
    assert_tensor(p.grad, [2.])
    opt.step()
    assert opt.state_dict()["t"] == 1
    assert_tensor(p, [3. - 1 / 3])


def test_repeated_adam_self_load_preserves_continuation(
    adam_parameter, assign_gradient, assert_tensor,
):
    p = adam_parameter([3.])
    opt = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    assign_gradient(p, [2.])
    opt.step()
    for _ in range(2):
        assert opt.load_state_dict(opt.state_dict()) is None
    opt.step()
    assert_tensor(p, [3. - 1 / 3])
    assert opt.state_dict()["t"] == 2


def test_legacy_state_without_decay_fields_restores_no_decay_defaults(
    parameter, optimizer, assign_gradient, assert_tensor,
):
    p = parameter([[3.]])
    source = optimizer([p], lr=0.125)
    state = source.state_dict()
    del state["weight_decay"], state["decay_bias"]
    restored = optimizer([p], weight_decay=0.5, decay_bias=True)
    restored.load_state_dict(state)
    assert restored.state_dict()["weight_decay"] == 0.
    assert restored.state_dict()["decay_bias"] is False
    assign_gradient(p, [[1.]])
    restored.step()
    assert_tensor(p, [[2.875]])


@pytest.mark.parametrize("key", ["lr", "betas", "eps", "t", "m", "v"])
def test_adam_missing_required_state_entry_is_rejected_without_parameter_write(
    adam_parameter, assign_gradient, assert_tensor, key,
):
    p = adam_parameter([3.])
    assign_gradient(p, [2.])
    opt = Adam([p])
    state = opt.state_dict()
    del state[key]
    with pytest.raises(KeyError):
        opt.load_state_dict(state)
    assert_tensor(p, [3.])
    assert_tensor(p.grad, [2.])


def test_sgd_state_requires_learning_rate(tensor, assert_tensor):
    p = tensor([3.], requires_grad=True)
    opt = SGD([p], lr=0.25)
    with pytest.raises(KeyError):
        opt.load_state_dict({"weight_decay": 0.5})
    assert_tensor(p, [3.])


@pytest.mark.parametrize("key", ["m", "v"])
@pytest.mark.parametrize("count", [0, 2])
def test_adam_state_parameter_count_mismatch_is_rejected(
    adam_parameter, assert_tensor, key, count,
):
    p = adam_parameter([3.])
    opt = Adam([p])
    state = opt.state_dict()
    state[key] = [None] * count
    with pytest.raises(ValueError):
        opt.load_state_dict(state)
    assert_tensor(p, [3.])


@pytest.mark.parametrize("key", ["m", "v"])
def test_adam_state_parameter_shape_mismatch_is_rejected(
    adam_parameter, assert_tensor, key,
):
    p = adam_parameter([3.])
    opt = Adam([p])
    state = opt.state_dict()
    state[key] = [adam_parameter([[0.]])]
    with pytest.raises(ValueError):
        opt.load_state_dict(state)
    assert_tensor(p, [3.])


def test_sgd_configuration_load_does_not_require_equal_parameter_count(tensor, assert_tensor):
    a, b = tensor([3.], requires_grad=True), tensor([4.], requires_grad=True)
    state = SGD([], lr=0.125).state_dict()
    opt = SGD([a, b], lr=0.5)
    assert opt.load_state_dict(state, map_location="cpu") is None
    a.set_grad(tensor([1.]))
    b.set_grad(tensor([2.]))
    opt.step()
    assert_tensor(a, [2.875])
    assert_tensor(b, [3.75])


@pytest.mark.parametrize("map_location", [None, "cpu", ortho.Device("cpu")])
def test_adam_cpu_state_load_supports_explicit_and_default_placement(
    typed_tensor, assert_tensor, copy_optimizer_state, map_location,
):
    p = typed_tensor([3.]).requires_grad_(True)
    p.set_grad(typed_tensor([2.]))
    source = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    source.step()
    q = typed_tensor([3. - 1 / 6]).requires_grad_(True)
    restored = Adam([q])
    restored.load_state_dict(copy_optimizer_state(source.state_dict()), map_location=map_location)
    for key in ("m", "v"):
        value = restored.state_dict()[key][0]
        assert value.device == q.device
        assert value.dtype == "float32"
        assert_tensor(value, [1.])
    q.set_grad(typed_tensor([2.]))
    restored.step()
    assert_tensor(q, [3. - 1 / 3])


@pytest.mark.parametrize("target", ["cpu", "cuda:0"])
@pytest.mark.parametrize("explicit", [False, True])
def test_adam_state_cross_device_load_can_continue_updates(
    cuda_device, typed_tensor, assert_tensor, copy_optimizer_state, target, explicit,
):
    source_device = "cuda:0" if target == "cpu" else "cpu"
    p = typed_tensor([3.], device=source_device).requires_grad_(True)
    p.set_grad(typed_tensor([2.], device=source_device))
    source = Adam([p], lr=0.25, betas=(0.5, 0.75), eps=1.)
    source.step()
    q = typed_tensor([3. - 1 / 6], device=target).requires_grad_(True)
    restored = Adam([q])
    state = copy_optimizer_state(source.state_dict())
    restored.load_state_dict(state, map_location=ortho.Device(target) if explicit else None)
    for key in ("m", "v"):
        assert restored.state_dict()[key][0].device == q.device
    q.set_grad(typed_tensor([2.], device=target))
    restored.step()
    assert_tensor(q, [3. - 1 / 3])


def test_parameter_replacement_does_not_retarget_existing_optimizer(
    parameter, optimizer, assign_gradient, assert_tensor,
):
    original, replacement = parameter([3.]), parameter([7.])
    model = Module()
    model.weight = Parameter(original)
    opt = optimizer(model.parameters())
    assign_gradient(original, [1.])
    opt.step()
    model.weight = Parameter(replacement)
    assign_gradient(replacement, [1.])
    opt.step()
    assert model.weight is replacement
    assert_tensor(original, [2.5])
    assert_tensor(model.weight, [7.])
    opt.zero_grad()
    assert original.grad is None
    assert_tensor(replacement.grad, [1.])


def test_same_device_module_move_does_not_move_optimizer_ownership(
    parameter, optimizer, assert_tensor,
):
    p = parameter([3.])
    model = Module()
    model.weight = Parameter(p)
    opt = optimizer(model.parameters())
    p.sum().backward()
    opt.step()
    model.to(p.device)
    current = model.weight
    # Movement keeps the bound Python owner and clears its gradient.
    assert current is p
    opt.step()
    assert_tensor(p, [2.75])
    assert_tensor(current, [2.75])
    assert current.grad is None


@pytest.mark.parametrize("kind", [SGD, Adam])
def test_constructor_and_no_gradient_step_accept_storage_without_promising_compute(
    typed_tensor, storage_dtype, kind, assert_output,
):
    p = typed_tensor([1], dtype=storage_dtype)
    opt = kind([p], lr=0.125)
    assert opt.step() is None
    assert_output(p, [1], dtype=storage_dtype, device="cpu")
    assert p.grad is None


@pytest.mark.parametrize("kind", [SGD, Adam])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int32", "int64", "uint8", "int8", "bool"])
def test_storage_only_dtypes_cannot_be_updated_by_bare_optimizer(
    typed_tensor, operator_device, kind, dtype, assert_output,
):
    p = typed_tensor([1], dtype=dtype, device=operator_device)
    p.set_grad(typed_tensor([1], dtype=dtype, device=operator_device))
    opt = kind([p], lr=0.125)
    with pytest.raises(RuntimeError):
        opt.step()
    assert_output(p, [1], dtype=dtype, device=operator_device)


def test_cuda_float64_sgd_compute_is_unsupported(cuda_device, typed_tensor, assert_output):
    p = typed_tensor([2.], dtype="float64", device=cuda_device)
    p.set_grad(typed_tensor([1.], dtype="float64", device=cuda_device))
    opt = SGD([p], lr=0.125)
    with pytest.raises(RuntimeError):
        opt.step()
    assert_output(p, [2.], dtype="float64", device=cuda_device)
