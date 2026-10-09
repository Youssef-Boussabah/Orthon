import pytest

from orthon.nn import Module, Parameter
from orthon.optim import SGD


def test_state_dict_orders_parameters_then_persistent_buffers_by_hierarchical_name(module_factory, tensor):
    model = module_factory()
    model.register_buffer("root_stat", tensor([0]))
    expected = ["weight", "root_stat", "branch.bias", "branch.running", "branch.counter"]
    first, second = model.state_dict(), model.state_dict()
    assert first is not second
    assert list(first) == list(second) == expected
    assert all(first[name] is second[name] for name in expected)
    assert "branch.cache" not in first
    assert "branch.optional" not in first
    assert "scratch" not in first
    assert "training" not in first


def test_state_mapping_values_are_live_but_editing_mapping_membership_does_not_register_state(
    module_factory, tensor, assert_tensor,
):
    model = module_factory()
    state = model.state_dict()
    assert state["weight"] is model.weight
    state["weight"].mul_(2)
    assert_tensor(model.weight, [[2, 4], [6, 8]])
    model.branch.running.mul_(3)
    assert_tensor(state["branch.running"], [21, 24])
    state["weight"] = tensor([[0, 0], [0, 0]])
    del state["branch.running"]
    assert_tensor(model.weight, [[2, 4], [6, 8]])
    assert "branch.running" in model.state_dict()


def test_state_load_preserves_parameters_replaces_buffers_and_keeps_existing_gradients(
    module_factory, tensor, assert_tensor, assert_grad,
):
    model = module_factory().eval()
    model.branch.bias.requires_grad_(False)
    model.branch.running.requires_grad_(True)
    (model.weight * 2).sum().backward()
    model.branch.running.sum().backward()
    weight, bias, running, cache = model.weight, model.branch.bias, model.branch.running, model.branch.cache
    source_weight = tensor([[10, 20], [30, 40]], requires_grad=True)
    source_bias, source_running = tensor([50, 60]), tensor([70, 80])
    state = dict(model.state_dict(), weight=source_weight)
    state["branch.bias"], state["branch.running"] = source_bias, source_running
    assert model.load_state_dict(state) is model
    assert model.weight is weight
    assert model.branch.bias is bias
    assert model.branch.running is not running
    assert model.branch.running is not source_running
    assert model.branch.cache is cache
    assert model.training is model.branch.training is False
    assert model.weight.requires_grad is True
    assert model.branch.bias.requires_grad is False
    assert model.branch.running.requires_grad is True
    assert model.weight.is_leaf is model.branch.running.is_leaf is True
    assert model.branch.running.grad is None
    assert_grad(weight, [[2, 2], [2, 2]])
    assert_tensor(running.grad, [1, 1])
    assert_tensor(model.weight, [[10, 20], [30, 40]])
    source_weight.mul_(2)
    source_running.mul_(2)
    assert_tensor(model.weight, [[10, 20], [30, 40]])
    assert_tensor(model.branch.running, [70, 80])
    assert source_weight.grad is None


def test_self_load_and_swapped_live_sources_copy_values_before_writing(tensor, assert_tensor):
    model = Module()
    model.a, model.b = Parameter(tensor([2, 3])), Parameter(tensor([5, 7]))
    a, b = model.a, model.b
    for _ in range(2):
        assert model.load_state_dict(model.state_dict()) is model
    assert_tensor(a, [2, 3])
    assert_tensor(b, [5, 7])
    model.load_state_dict({"a": b, "b": a})
    assert model.a is a
    assert model.b is b
    assert_tensor(a, [5, 7])
    assert_tensor(b, [2, 3])


def test_consistent_shared_state_load_keeps_the_shared_parameter(tensor, assert_tensor):
    model = Module()
    original = tensor([2, 3])
    shared = Parameter(original)
    model.a, model.b = shared, shared
    incoming = tensor([5, 7])
    model.load_state_dict({"a": incoming, "b": incoming})
    assert model.a is model.b is original
    assert list(model.parameters()) == [original]
    incoming.mul_(2)
    assert_tensor(original, [5, 7])


@pytest.mark.parametrize("key", ["weight", "branch.running"])
def test_strict_load_requires_every_persistent_key(module_factory, key):
    model = module_factory()
    state = model.state_dict()
    del state[key]
    with pytest.raises(KeyError):
        model.load_state_dict(state)


@pytest.mark.parametrize("key", ["unknown", "branch.cache", "branch.optional"])
def test_strict_load_rejects_non_state_keys_even_for_registered_temporary_slots(module_factory, tensor, key):
    model = module_factory()
    state = model.state_dict()
    state[key] = tensor([1])
    with pytest.raises(KeyError):
        model.load_state_dict(state)


def test_non_strict_load_ignores_missing_and_unexpected_entries(module_factory, tensor, assert_tensor):
    model = module_factory()
    weight, running, cache = model.weight, model.branch.running, model.branch.cache
    assert model.load_state_dict({"branch.bias": tensor([2, 3]), "ignored": object()}, strict=False) is model
    assert_tensor(model.branch.bias, [2, 3])
    assert model.weight is weight
    assert model.branch.running is running
    assert model.branch.cache is cache
    assert_tensor(weight, [[1, 2], [3, 4]])
    assert_tensor(running, [7, 8])


@pytest.mark.parametrize("invalid", ["shape", "non_tensor"])
def test_invalid_later_state_entry_leaves_all_previous_values_and_ownership_intact(
    module_factory, tensor, invalid,
):
    model = module_factory()
    before = model.state_dict()
    values = {name: entry.tolist() for name, entry in before.items()}
    state = dict(before, weight=tensor([[99, 99], [99, 99]]))
    state["branch.counter"] = tensor([1, 2]) if invalid == "shape" else [1]
    with pytest.raises(ValueError if invalid == "shape" else TypeError):
        model.load_state_dict(state)
    after = model.state_dict()
    assert {name: entry.tolist() for name, entry in after.items()} == values
    assert all(after[name] is entry for name, entry in before.items())


def test_state_load_reconciles_strided_source_dtype_and_device_to_the_destination(
    module_factory, typed_tensor, tensor_case, assert_output,
):
    model = module_factory()
    original = model.weight
    source = typed_tensor([[4, 5], [6, 7]], dtype="float64").requires_grad_(True)
    incoming = source.transpose(0, 1)
    model.load_state_dict({"weight": incoming}, strict=False)
    assert model.weight is original
    assert model.weight.is_leaf is True
    assert_output(model.weight, [[4, 6], [5, 7]], dtype=tensor_case[1], device=tensor_case[0])
    model.weight.sum().backward()
    assert source.grad is None


def test_state_import_accepts_storage_dtypes_and_preserves_destination_buffer_dtype(
    storage_dtype, typed_tensor, assert_output,
):
    model = Module()
    model.weight = Parameter(typed_tensor([0, 0]))
    model.register_buffer("mask", typed_tensor([0, 1], dtype=storage_dtype))
    original = model.weight
    model.load_state_dict({"weight": typed_tensor([1, 0], dtype=storage_dtype),
                           "mask": typed_tensor([1, 0], dtype="float64")})
    assert model.weight is original
    assert_output(model.weight, [1, 0], dtype="float32", device="cpu")
    assert_output(model.mask, [1, 0], dtype=storage_dtype, device="cpu")


def test_cuda_source_state_loads_into_existing_cpu_parameters_without_importing_its_graph(
    cuda_device, typed_tensor, assert_output,
):
    model = Module()
    model.weight = Parameter(typed_tensor([2, 3], dtype="float64"))
    original = model.weight
    source = typed_tensor([5, 7], device=cuda_device).requires_grad_(True)
    model.load_state_dict({"weight": source})
    assert model.weight is original
    assert model.weight.is_leaf is True
    assert_output(model.weight, [5, 7], dtype="float64", device="cpu")
    model.weight.sum().backward()
    assert source.grad is None


def test_optimizer_created_before_load_updates_the_loaded_parameter(tensor, assert_tensor):
    model = Module()
    model.weight = Parameter(tensor([2, 3]))
    original = model.weight
    optimizer = SGD(model.parameters(), lr=.25)
    (model.weight * 3).sum().backward()
    model.load_state_dict({"weight": tensor([10, 20])})
    optimizer.step()
    assert model.weight is original
    assert_tensor(model.weight, [9.25, 19.25])


def test_loading_state_invalidates_a_backward_that_saved_parameter_values(tensor, assert_grad):
    model = Module()
    model.weight = Parameter(tensor([2, 3]))
    (model.weight * 2).sum().backward()
    output = (model.weight * model.weight).sum()
    model.load_state_dict({"weight": tensor([5, 7])})
    with pytest.raises(RuntimeError):
        output.backward()
    assert_grad(model.weight, [2, 2])
