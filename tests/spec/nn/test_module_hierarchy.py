import pytest

from orthon.nn import Module, Parameter, Sequential


def test_recursive_discovery_is_local_first_then_child_registration_order(tensor):
    root = Module()
    root.first = Parameter(tensor([1]))
    root.branch = Module()
    root.branch.weight = Parameter(tensor([2]))
    root.last = Parameter(tensor([3]))
    root.other = Module()
    root.other.bias = Parameter(tensor([4]))
    expected = [("first", root.first), ("last", root.last),
                ("branch.weight", root.branch.weight), ("other.bias", root.other.bias)]
    assert list(root.named_parameters()) == expected
    assert list(root.named_parameters()) == expected
    assert list(root.parameters()) == [value for _, value in expected]
    assert list(root.named_parameters(prefix="net.")) == [("net." + name, p) for name, p in expected]


def test_buffer_recursion_and_names_are_separate_from_parameter_discovery(module_factory):
    model = module_factory()
    assert [name for name, _ in model.named_parameters()] == ["weight", "branch.bias"]
    assert [name for name, _ in model.named_buffers()] == ["branch.running", "branch.cache", "branch.counter"]
    assert list(model.named_buffers(recurse=False)) == []
    assert list(model.buffers(recurse=False)) == []
    assert [name for name, _ in model.named_buffers(prefix="net.")] == [
        "net.branch.running", "net.branch.cache", "net.branch.counter",
    ]


@pytest.mark.parametrize("replacement", ["child", "none"])
def test_child_replacement_or_removal_updates_all_recursive_discovery(tensor, replacement):
    root = Module()
    old = Module()
    old.weight = Parameter(tensor([1]))
    old.register_buffer("stat", tensor([2]))
    root.branch = old
    if replacement == "child":
        new = Module()
        new.bias = Parameter(tensor([3]))
        root.branch = new
        assert list(root.named_parameters()) == [("branch.bias", new.bias)]
        assert list(root.state_dict()) == ["branch.bias"]
    else:
        root.branch = None
        assert list(root.parameters()) == []
        assert root.state_dict() == {}
    assert list(root.buffers()) == []
    assert old.weight.requires_grad is True


def test_shared_child_has_all_paths_but_one_parameter_in_unnamed_discovery(tensor):
    child = Module()
    child.weight = Parameter(tensor([2, 3]))
    root = Module()
    root.left = child
    root.right = child
    assert root.left is root.right
    assert list(root.parameters()) == [child.weight]
    assert list(root.named_parameters()) == [("left.weight", child.weight), ("right.weight", child.weight)]
    assert list(root.state_dict()) == ["left.weight", "right.weight"]


def test_sequential_owns_children_and_invokes_them_in_order(tensor, assert_tensor):
    class Offset(Module):
        def __init__(self, value):
            super().__init__()
            self.offset = Parameter(tensor(value))

        def forward(self, x):
            return x + self.offset

    class Scale(Module):
        def __init__(self, value):
            super().__init__()
            self.scale = Parameter(tensor(value))

        def forward(self, x):
            return x * self.scale

    model = Sequential(Offset(2), Scale(3))
    assert [name for name, _ in model.named_parameters()] == ["0.offset", "1.scale"]
    assert_tensor(model(tensor([1, 4])), [9, 18])


def test_call_forwards_positional_and_keyword_arguments_to_forward():
    class Echo(Module):
        def forward(self, *args, **kwargs):
            return args, kwargs

    model = Echo()
    first, second = object(), object()
    for result in (model(first, second, marker=first), model.forward(first, second, marker=first)):
        assert result == ((first, second), {"marker": first})


def test_nested_call_routes_values_and_gradients_to_registered_parameters(tensor, assert_tensor, assert_grad):
    class Weighted(Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(tensor([2, 3]))

        def forward(self, x):
            return x * self.weight

    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.child = Weighted()

        def forward(self, x, *, factor):
            return self.child(x) * factor

    model = Parent()
    output = model(tensor([1, 4]), factor=2)
    assert_tensor(output, [4, 24])
    output.sum().backward()
    assert_grad(model.child.weight, [2, 8])
    assert output.grad is None


def test_call_preserves_forward_exceptions_and_base_forward_is_abstract():
    failure = ValueError("test forward failure")

    class Broken(Module):
        def forward(self):
            raise failure

    with pytest.raises(ValueError) as error:
        Broken()()
    assert error.value is failure
    with pytest.raises(NotImplementedError):
        Module()()
