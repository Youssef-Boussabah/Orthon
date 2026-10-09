import pytest

from orthon.nn import Module, Parameter


def test_bare_tensors_and_python_containers_are_not_automatically_registered(tensor):
    model = Module()
    model.tensor = tensor([1, 2], requires_grad=True)
    model.layers = [Module()]
    model.marks = {"weight": Parameter(tensor([3]))}
    assert list(model.parameters()) == []
    assert list(model.buffers()) == []
    assert model.state_dict() == {}


def test_replacing_a_parameter_changes_discovery_without_mutating_the_old_tensor(tensor, assert_tensor):
    model = Module()
    first, second, replacement = tensor([1]), tensor([2]), tensor([3])
    model.first = Parameter(first)
    model.second = Parameter(second)
    model.first = Parameter(replacement)
    assert list(model.named_parameters()) == [("first", replacement), ("second", second)]
    assert list(model.parameters()) == [replacement, second]
    assert model.first is replacement
    assert_tensor(first, [1])


@pytest.mark.parametrize("replacement", ["none", "plain_tensor", "number"])
def test_assigning_unmarked_values_unregisters_a_parameter(tensor, replacement):
    model = Module()
    source = tensor([2, 3])
    model.weight = Parameter(source)
    value = {"none": None, "plain_tensor": tensor([5, 6]), "number": 7}[replacement]
    model.weight = value
    assert model.weight is value
    assert list(model.parameters()) == []
    assert model.state_dict() == {}
    assert source.requires_grad is True


def test_parameter_and_child_can_replace_each_other_in_a_registered_slot(tensor):
    model = Module()
    old, current = tensor([1]), tensor([2])
    model.slot = Parameter(old)
    child = Module()
    child.weight = Parameter(current)
    model.slot = child
    assert model.slot is child
    assert list(model.named_parameters()) == [("slot.weight", current)]
    model.slot = Parameter(old)
    assert model.slot is old
    assert list(model.named_parameters()) == [("slot", old)]


def test_buffer_slots_keep_their_role_and_persistence_across_reassignment(tensor):
    model = Module()
    model.register_buffer("cache", None, persistent=False)
    assert model.cache is None
    assert list(model.named_buffers()) == []
    model.cache = tensor([2, 3])
    assert dict(model.named_buffers()) == {"cache": model.cache}
    assert list(model.parameters()) == []
    assert model.state_dict() == {}
    model.register_buffer("cache", tensor([5, 6]), persistent=True)
    assert model.state_dict()["cache"] is model.cache
    model.cache = None
    assert list(model.buffers()) == []
    assert model.state_dict() == {}
    model.cache = tensor([7, 8])
    assert model.state_dict()["cache"] is model.cache


def test_shared_buffers_are_deduplicated_only_by_unnamed_discovery(tensor):
    model = Module()
    shared = tensor([2, 3])
    model.register_buffer("persistent", shared)
    model.register_buffer("temporary", shared, persistent=False)
    assert list(model.buffers()) == [shared]
    assert list(model.named_buffers()) == [("persistent", shared), ("temporary", shared)]
    assert model.state_dict() == {"persistent": shared}


@pytest.mark.parametrize("name", ["", "a.b", "with space", "0value"])
def test_buffer_names_must_be_valid_single_attribute_names(tensor, name):
    with pytest.raises(ValueError):
        Module().register_buffer(name, tensor([1]))


def test_buffer_names_must_be_strings(tensor):
    with pytest.raises(TypeError):
        Module().register_buffer(1, tensor([1]))


@pytest.mark.parametrize("value", ["number", "parameter", "module"])
def test_buffer_registration_requires_a_tensor_or_none(tensor, value):
    incoming = {"number": 1, "parameter": Parameter(tensor([1])), "module": Module()}[value]
    model = Module()
    with pytest.raises(TypeError):
        model.register_buffer("value", incoming)
    assert list(model.buffers()) == []
    assert model.state_dict() == {}


@pytest.mark.parametrize("collision", ["parameter", "child", "attribute", "method"])
def test_buffer_registration_rejects_existing_names_without_changing_state(tensor, collision):
    model = Module()
    if collision == "parameter":
        model.slot = Parameter(tensor([1]))
    elif collision == "child":
        model.slot = Module()
    elif collision == "attribute":
        model.slot = 1
    name = "train" if collision == "method" else "slot"
    before = model.state_dict()
    with pytest.raises(KeyError):
        model.register_buffer(name, tensor([2]))
    assert model.state_dict() == before
    assert list(model.buffers()) == []


@pytest.mark.parametrize("replacement", ["parameter", "child", "number"])
def test_registered_buffer_rejects_non_tensor_roles(tensor, replacement):
    model = Module()
    source = tensor([2, 3])
    model.register_buffer("slot", source)
    value = {"parameter": Parameter(tensor([1])), "child": Module(), "number": 1}[replacement]
    with pytest.raises(TypeError if replacement == "number" else KeyError):
        model.slot = value
    assert model.slot is source
    assert list(model.parameters()) == []
