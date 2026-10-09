"""Module registration, shared owner transformations and logical state copies."""
import numpy as np
import pytest
import subprocess
import sys

import orthon as ortho
from orthon.nn import Module, Parameter, Sequential
from orthon.optim import SGD


def observable(model):
    return (list(model.named_parameters()), list(model.named_buffers()), model.state_dict())


@pytest.mark.parametrize('initial', ['ordinary', 'none', 'parameter', 'module'])
@pytest.mark.parametrize('replacement', ['parameter', 'module', 'ordinary', 'tensor'])
def test_registration_replacement_has_one_visible_meaning(tensor, initial, replacement):
    model = Module()
    values = {'ordinary': 7, 'none': None, 'parameter': Parameter(tensor([2])), 'module': Module()}
    model.slot = values[initial]
    expected = tensor([3]) if replacement in ('parameter', 'tensor') else Module() if replacement == 'module' else 9
    model.slot = Parameter(expected) if replacement == 'parameter' else expected
    assert model.slot is expected
    assert sum('slot' in registry for registry in (model._parameters, model._modules, model._buffers)) == int(replacement in ('parameter', 'module'))
    assert list(model.parameters()) == ([expected] if replacement == 'parameter' else [])
    assert model.state_dict() == ({'slot': expected} if replacement == 'parameter' else {})
    assert list(model.named_buffers()) == []
    assert 'slot' not in model._non_persistent_buffers
    if replacement == 'module':
        expected.weight = Parameter(tensor([4]))
        assert list(model.named_parameters()) == [('slot.weight', expected.weight)]


@pytest.mark.parametrize('persistent', [True, False])
@pytest.mark.parametrize('initial_none', [True, False])
@pytest.mark.parametrize('invalid', ['parameter', 'module', 'number'])
def test_buffer_assignment_rejection_is_atomic(tensor, persistent, initial_none, invalid):
    model = Module()
    original = None if initial_none else tensor([2])
    model.register_buffer('slot', original, persistent)
    before = observable(model)
    value = {'parameter': Parameter(tensor([3])), 'module': Module(), 'number': 5}[invalid]
    with pytest.raises(TypeError if invalid == 'number' else KeyError):
        model.slot = value
    assert model.slot is original
    assert observable(model) == before
    assert ('slot' in model._non_persistent_buffers) is (not persistent)


@pytest.mark.parametrize('initial', ['parameter', 'module', 'ordinary'])
def test_register_buffer_collision_preserves_every_member(tensor, initial):
    model = Module()
    model.slot = {'parameter': Parameter(tensor([2])), 'module': Module(), 'ordinary': 5}[initial]
    original, before = model.slot, observable(model)
    with pytest.raises(KeyError):
        model.register_buffer('slot', tensor([3]), False)
    assert model.slot is original
    assert observable(model) == before
    assert 'slot' not in model._non_persistent_buffers


@pytest.mark.parametrize('name', ['training', 'forward', '_parameters', 'a.b', ''])
@pytest.mark.parametrize('kind', ['parameter', 'module'])
def test_registered_name_rejection_does_not_damage_module(tensor, name, kind):
    model = Module()
    model.weight = Parameter(tensor([2]))
    before = observable(model)
    with pytest.raises(ValueError if name in ('a.b', '') else KeyError):
        setattr(model, name, Parameter(tensor([3])) if kind == 'parameter' else Module())
    assert observable(model) == before
    assert model.training is True


@pytest.mark.parametrize('persistent', [True, False])
def test_none_slot_replacement_and_persistence_survive_transform(tensor, persistent):
    model = Module()
    model.register_buffer('slot', None, persistent)
    model.to('cpu').half().bfloat16()
    assert model.slot is None
    model.slot = tensor([2])
    assert list(model.named_buffers()) == [('slot', model.slot)]
    assert ('slot' in model.state_dict()) is persistent
    model.slot = None
    assert model._buffers['slot'] is None
    assert observable(model) == ([], [], {})
    model.slot = tensor([3])
    model.register_buffer('slot', model.slot, not persistent)
    assert ('slot' in model.state_dict()) is (not persistent)


@pytest.mark.parametrize('method,target', [('to', None), ('cast', 'float32'), ('cast', 'float64'), ('half', None), ('bfloat16', None)])
def test_tied_transforms_keep_owner_names_values_and_leaf_flags(tensor, tensor_case, method, target):
    # CUDA float64 cast is a deliberately unsupported lower-stack conversion.
    if tensor_case[0] != 'cpu' and target == 'float64':
        with pytest.raises(RuntimeError):
            ModuleWithParameter(tensor([2, 3])).cast(target)
        return
    p = tensor([2, 3])
    wrapper = Parameter(p)
    model = Module()
    model.a = wrapper
    model.b = wrapper
    child = ModuleWithParameter(p)
    model.left = child
    model.right = child
    model.other = ModuleWithParameter(p)
    buf = tensor([4, 5])
    model.register_buffer('stat', buf)
    child.register_buffer('cache', buf, False)
    child.register_buffer('optional', None)
    before = list(model.state_dict())
    if method == 'to':
        model.to('cpu')
    elif target is None:
        getattr(model, method)()
    else:
        model.cast(target)
    assert model.a is model.b is child.weight is model.other.weight is p
    assert wrapper._data is p
    assert list(model.parameters()) == [p]
    assert len(list(model.named_parameters())) == 5
    assert list(model.state_dict()) == before
    assert p.tolist() == [2, 3]
    assert p.is_leaf and p.requires_grad
    assert p.grad is None
    assert model.stat is child.cache
    assert model.stat is not buf
    assert model.stat.tolist() == [4, 5]
    assert child.optional is None
    assert model.left is model.right


class ModuleWithParameter(Module):
    def __init__(self, value):
        super().__init__()
        self.weight = Parameter(value)


@pytest.mark.parametrize('operation', ['same_cast', 'changed_cast', 'to'])
def test_transform_gradient_policy_and_references(tensor, tensor_case, operation):
    model = ModuleWithParameter(tensor([2, 3]))
    p = model.weight
    optimizer = SGD(model.parameters(), lr=.25)
    (p * 5).sum().backward()
    previous_grad = p.grad
    if operation == 'same_cast':
        model.cast(p.dtype)
        assert p.grad.tolist() == [5, 5]
        optimizer.step()
        assert p.tolist() == [.75, 1.75]
    else:
        model.to('cpu') if operation == 'to' else model.cast('float32' if p.dtype == 'float64' else 'float16')
        assert p.grad is None
        assert previous_grad.tolist() == [5, 5]
        assert previous_grad.requires_grad is False
    assert model.weight is p
    assert p.is_leaf and p.requires_grad


def test_tied_transformed_parameter_backward_and_sgd_run_once(tensor, assert_grad, assert_tensor):
    p = tensor([2, 3])
    model = ModuleWithParameter(p)
    model.alias = Parameter(p)
    model.to('cpu').cast('float64')
    (model.weight * 2 + model.alias * 3).sum().backward()
    assert_grad(p, [5, 5])
    SGD(model.parameters(), lr=.25).step()
    assert_tensor(p, [.75, 1.75])
    model.zero_grad()
    assert p.grad is None


def test_unique_transform_calls_and_failure_before_publication(tensor):
    p = tensor([2])
    model = ModuleWithParameter(p)
    model.alias = Parameter(p)
    model.left = ModuleWithParameter(p)
    model.right = model.left
    buf = tensor([3])
    model.register_buffer('first', buf)
    model.left.register_buffer('second', buf)
    calls = []
    def failing(value):
        calls.append(value)
        if value is buf:
            raise RuntimeError('injected later conversion failure')
        return value.to('cpu')
    before = observable(model)
    with pytest.raises(RuntimeError, match='later conversion'):
        model._apply_tensor_transform(failing)
    assert calls == [p, buf]
    assert observable(model) == before
    assert p.tolist() == [2]
    calls.clear()
    def transform(value):
        calls.append(value)
        return value.to('cpu')
    model._apply_tensor_transform(transform)
    assert calls == [p, buf]
    assert model.first is model.left.second


def test_distinct_tensor_owners_sharing_values_transform_separately(tensor):
    first = tensor([2, 3])
    with ortho.no_grad():
        second = first.reshape([2])
    model = ModuleWithParameter(first)
    model.other = Parameter(second)
    model.to('cpu')
    assert model.weight is first and model.other is second
    assert list(model.parameters()) == [first, second]
    first.mul_(2)
    assert second.tolist() == [2, 3]


def test_parameter_buffer_alias_uses_the_stable_parameter_owner(tensor):
    p = tensor([2, 3])
    model = ModuleWithParameter(p)
    model.register_buffer('stat', p)
    model.cast('float16').to('cpu')
    assert model.weight is model.stat is p
    assert p.dtype == 'float16'


def test_same_dtype_transform_detaches_registered_nonleaf(tensor):
    source = tensor([2, 3], requires_grad=True)
    p = source * 2
    model = ModuleWithParameter(p)
    assert not p.is_leaf
    model.cast(p.dtype)
    assert model.weight is p
    assert p.is_leaf and p.requires_grad
    p.sum().backward()
    assert p.grad.tolist() == [1, 1]
    assert source.grad is None


@pytest.mark.parametrize('layout', ['transpose', 'offset', 'scalar', 'empty'])
@pytest.mark.parametrize('source_view', [False, True])
def test_logical_state_load_preserves_parameter_layout_and_buffer_independence(typed_tensor, operator_device, layout, source_view):
    device = operator_device
    base = typed_tensor(np.arange(12).reshape(3, 4), device=device)
    if layout == 'transpose':
        dest = base.transpose(0, 1)
    elif layout == 'offset':
        dest = base.slice(1, 1, 3)
    elif layout == 'scalar':
        dest = base[1][2]
    else:
        dest = base.slice(1, 4, 4)
    model = ModuleWithParameter(dest)
    model.register_buffer('stat', dest)
    shape, strides, offset = dest.shape, dest.strides, dest.storage_offset
    # A source view with the exact destination logical shape, plus an offset.
    source = typed_tensor(np.arange(int(dest.numel)).reshape(shape) + 30, device=device)
    if source_view and source.ndim:
        if layout == 'offset':
            padded = np.full((shape[0], shape[1] + 2), -1, dtype=np.float32)
            padded[:, 1:-1] = np.arange(int(dest.numel)).reshape(shape) + 30
            source = typed_tensor(padded, device=device).slice(1, 1, shape[1] + 1)
            assert source.storage_offset == 1
        elif source.ndim == 2:
            source = source.transpose(0, 1).to(device).transpose(0, 1)
    values = source.tolist()
    old_buffer = model.stat
    model.load_state_dict({'weight': source, 'stat': source})
    assert model.weight is dest
    assert (dest.shape, dest.strides, dest.storage_offset) == (shape, strides, offset)
    assert dest.tolist() == values
    assert model.stat is not old_buffer and model.stat is not source
    assert model.stat.tolist() == values
    assert source.tolist() == values
    assert dest.requires_grad and dest.is_leaf
    assert source.grad is None
    if layout == 'offset':
        assert base.tolist() == [0, 30, 31, 3, 4, 32, 33, 7, 8, 34, 35, 11]


@pytest.mark.parametrize('reverse', [False, True])
def test_conflicting_tied_state_rejects_before_any_write(tensor, reverse):
    model = ModuleWithParameter(tensor([2, 3]))
    model.alias = Parameter(model.weight)
    model.earlier = Parameter(tensor([4, 5]))
    model.register_buffer('stat', tensor([6, 7]))
    before = {k: v.tolist() for k, v in model.state_dict().items()}
    state = {'weight': tensor([8, 9]), 'alias': tensor([10, 11]), 'earlier': tensor([12, 13]), 'stat': tensor([14, 15])}
    if reverse:
        state = dict(reversed(list(state.items())))
    with pytest.raises(ValueError, match='conflicting tied parameter'):
        model.load_state_dict(state)
    assert {k: v.tolist() for k, v in model.state_dict().items()} == before


def test_consistent_tied_state_applies_once_and_preserves_existing_gradient(tensor):
    p = tensor([2, 3])
    model = ModuleWithParameter(p)
    model.alias = Parameter(p)
    p.sum().backward()
    version = p._storage_version
    model.load_state_dict({'alias': tensor([4, 5]), 'weight': tensor([4, 5])})
    assert model.weight is model.alias is p
    assert p.tolist() == [4, 5]
    assert p.grad.tolist() == [1, 1]
    assert p._storage_version == version + 1


def test_strided_state_swaps_snapshot_sources_and_keep_saved_value_checks(tensor):
    base = tensor([[1, 2], [3, 4]])
    with ortho.no_grad():
        first, second = base.transpose(0, 1), base.reshape([2, 2])
    model = ModuleWithParameter(first)
    model.other = Parameter(second)
    graph = (first * first).sum()
    model.load_state_dict({'weight': second, 'other': first})
    # Both independent owners share one Buffer: commits retain traversal order.
    assert first.tolist() == [1, 2, 3, 4]
    assert second.tolist() == [1, 3, 2, 4]
    with pytest.raises(RuntimeError):
        graph.backward()


def test_private_replacement_validates_paths_and_preserves_parameter_ties(tensor):
    # Assignment replaces the selected slot and preserves other ties;
    # tied-wide private replacement is unsupported.
    from orthon.optim import SGD
    p = tensor([2])
    model = ModuleWithParameter(p)
    model.child = ModuleWithParameter(p)
    model.other = model.child
    optimizer = SGD(model.parameters(), lr=.1)
    assert not hasattr(model, '_replace_parameter')
    p.sum().backward()
    replacement = tensor([4], requires_grad=True)
    model.child.weight = Parameter(replacement)
    assert model.child.weight is model.other.weight is replacement
    assert model.weight is p
    assert list(model.parameters()) == [p, replacement]
    # Replace the remaining slot explicitly too. The old captured owner is now
    # absent from discovery, and both owners have gradients: a dynamic retarget
    # would update the replacement instead of the external old reference.
    model.weight = Parameter(replacement)
    assert list(model.parameters()) == [replacement]
    replacement.sum().backward()
    optimizer.step()
    assert p.tolist() == pytest.approx([1.9])
    assert replacement.tolist() == [4]
    before = observable(model)
    with pytest.raises((KeyError, ValueError)):
        model.register_buffer('weight', tensor([3]))
    assert observable(model) == before


def test_trivial_cycle_diagnostics_terminate_without_losing_shared_paths(tensor):
    model = ModuleWithParameter(tensor([2]))
    model.self_ref = model
    assert list(model.parameters()) == [model.weight]
    assert list(model.state_dict()) == ['weight']
    model.eval().train().to('cpu')
    assert model.training


def test_sequential_uses_validated_child_registration_and_preserves_numbered_names(tensor):
    child = ModuleWithParameter(tensor([2]))
    seq = Sequential(child, child)
    assert getattr(seq, '0') is getattr(seq, '1') is child
    assert list(seq.state_dict()) == ['0.weight', '1.weight']
    assert list(seq.parameters()) == [child.weight]
    with pytest.raises(TypeError):
        Sequential(child, object())


def test_owner_rebind_rejects_graph_mode_shape_graph_and_flags_without_mutation(tensor):
    p = tensor([2, 3], requires_grad=True)
    before = p.tolist()
    with pytest.raises(RuntimeError, match='no_grad'):
        p._rebind_model_value_(p)
    with ortho.no_grad():
        for value, message in [(tensor([2], requires_grad=True), 'shape'), (tensor([2, 3]), 'requires_grad')]:
            with pytest.raises(RuntimeError, match=message):
                p._rebind_model_value_(value)
    graph = p * 2
    with ortho.no_grad(), pytest.raises(RuntimeError, match='graph-free leaf'):
        p._rebind_model_value_(graph)
    assert p.tolist() == before


def test_physical_cuda_roundtrip_and_exact_ordinal_preserve_tied_owners(cuda_device, typed_tensor):
    p = typed_tensor([2, 3])
    model = ModuleWithParameter(p)
    model.alias = Parameter(p)
    buf = typed_tensor([4, 5])
    model.register_buffer('stat', buf)
    model.child = Module()
    model.child.register_buffer('scratch', buf, False)
    assert model.to(ortho.Device(cuda_device)) is model
    assert model.weight is model.alias is p
    assert p.device == ortho.Device('cuda:0')
    assert model.stat.device == p.device
    assert model.stat is model.child.scratch
    (model.weight * 2 + model.alias * 3).sum().backward()
    assert p.grad.tolist() == [5, 5]
    model.to('cpu')
    assert model.weight is model.alias is p
    assert p.device == ortho.Device('cpu')
    assert p.grad is None
    assert p.tolist() == [2, 3]
    assert model.stat is model.child.scratch
    # The existing CUDA invalid-ordinal path leaves a pending runtime error.
    # Isolate rejection so the next test cannot consume that old error.
    probe = subprocess.run([sys.executable, '-c', '''
import orthon as o
m = o.nn.Module()
p = o.ones([2])
m.weight = o.nn.Parameter(p)
m.register_buffer('stat', o.ones([2]))
try:
    m.to('cuda:9999')
except RuntimeError:
    pass
else:
    raise AssertionError('unavailable ordinal selected ambient device')
assert m.weight is p
assert p.device == m.stat.device == o.Device('cpu')
assert p.tolist() == [1, 1]
'''], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
def test_physical_low_precision_transformed_tie_backward(cuda_device, typed_tensor, dtype):
    p = typed_tensor([2, 3])
    model = ModuleWithParameter(p)
    model.alias = Parameter(p)
    model.to(cuda_device).cast(dtype)
    assert model.weight is model.alias is p
    assert p.dtype == dtype and p.is_leaf and p.requires_grad
    (model.weight * 2 + model.alias * 3).sum().backward()
    assert p.grad.dtype == dtype and p.grad.device == p.device
    assert p.grad.tolist() == [5, 5]
    model.to('cpu')
    assert model.weight is p and p.grad is None


def test_invalid_persistence_truth_value_rejects_before_registration(tensor):
    class Invalid:
        def __bool__(self):
            raise TypeError('invalid persistence')
    model = Module()
    original = tensor([2])
    model.register_buffer('slot', original, False)
    with pytest.raises(TypeError, match='invalid persistence'):
        model.register_buffer('slot', tensor([3]), Invalid())
    assert model.slot is original
    assert model.state_dict() == {}
    assert model._non_persistent_buffers == {'slot'}
