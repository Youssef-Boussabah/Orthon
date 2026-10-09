"""Optimizer Python Tensor-owner bindings, independent of Module registries."""
import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


@pytest.mark.parametrize('kind', [SGD, Adam])
@pytest.mark.parametrize('container', [list, tuple, iter], ids=['list', 'tuple', 'generator'])
@pytest.mark.parametrize('positions', [(0, 1), (0, 3), (1, 2), (2, 3)])
def test_duplicate_owners_reject_before_update(kind, container, positions, typed_tensor):
    parameters = [typed_tensor([2.]).requires_grad_(True) for _ in range(4)]
    parameters[positions[1]] = parameters[positions[0]]
    for p in parameters:
        p.set_grad(typed_tensor([3.]))
    with pytest.raises(ValueError, match='duplicate'):
        kind(container(parameters), lr=.1)
    assert all(p.tolist() == [2.] and p.grad.tolist() == [3.] for p in parameters)


@pytest.mark.parametrize('kind', [SGD, Adam])
@pytest.mark.parametrize('bad', ['object', 'dict', 'wrapper', 'number', 'none'])
@pytest.mark.parametrize('position', [0, 1])
def test_non_tensor_entries_reject_at_capture(kind, bad, position, typed_tensor):
    p = typed_tensor([2.])
    invalid = {'object': object(), 'dict': {'params': [p]}, 'wrapper': Parameter(p),
               'number': 3., 'none': None}[bad]
    supplied = [p]
    supplied.insert(position, invalid)
    with pytest.raises(TypeError, match='Tensor'):
        kind(supplied, lr=.1)
    assert p.tolist() == [2.] and p.grad is None


def test_parameter_iterable_is_materialized_exactly_once(optimizer, parameter, assign_gradient):
    p = parameter([2.])

    class Once:
        calls = 0

        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            yield p

    supplied = Once()
    opt = optimizer(supplied)
    assign_gradient(p, [1.])
    opt.step()
    opt.zero_grad()
    assert supplied.calls == 1 and p.grad is None


def test_distinct_owners_sharing_storage_remain_distinct_bindings(
    optimizer, parameter, optimizer_case, assign_gradient, assert_tensor,
):
    kind, _, _ = optimizer_case
    a = parameter([3.])
    b = a.reshape([1])
    assert b is not a
    assign_gradient(a, [1.])
    assign_gradient(b, [2.])
    opt = optimizer([a, b], lr=.1)
    opt.step()
    assert_tensor(a, [2.7 if kind is SGD else 2.8])
    assert_tensor(b, [2.7 if kind is SGD else 2.8])
    if kind is Adam:
        state = opt.state_dict()
        assert len(state['m']) == len(state['v']) == 2
        assert state['m'][0] is not state['m'][1]
        assert_tensor(state['m'][0], [.1])
        assert_tensor(state['m'][1], [.2])


@pytest.mark.parametrize('kind', [SGD, Adam])
@pytest.mark.parametrize('transform', ['same-cpu', 'float64', 'float32'])
def test_existing_optimizer_survives_cpu_module_transform(kind, transform, typed_tensor):
    dtype = 'float64' if transform == 'float32' else 'float32'
    model = Module()
    model.weight = Parameter(typed_tensor([2., -3.], dtype=dtype))
    owner = model.weight
    opt = kind(model.parameters(), lr=.1234567890123, **(
        {'betas': (.712345678901, .823456789012), 'eps': .0123456789012} if kind is Adam else {}))
    owner.set_grad(typed_tensor([.5, -1.], dtype=dtype))
    opt.step()
    old = opt.state_dict()
    if transform == 'same-cpu':
        model.to('cpu')
    else:
        model.cast(transform)
    assert model.weight is owner
    before = np.array(owner.tolist(), dtype=np.float64)
    gradient = np.array([-.25, .75])
    owner.set_grad(typed_tensor(gradient, dtype=owner.dtype))
    if kind is Adam:
        b1, b2 = old['betas']
        # Conversion rounds history to the destination representation first.
        m0 = np.array(old['m'][0].tolist(), dtype=owner.dtype).astype(np.float64)
        v0 = np.array(old['v'][0].tolist(), dtype=owner.dtype).astype(np.float64)
        m = b1 * m0 + (1-b1) * gradient
        v = b2 * v0 + (1-b2) * gradient**2
        expected = before - old['lr'] * (m / (1-b1**2)) / (np.sqrt(v / (1-b2**2)) + old['eps'])
    else:
        expected = before - old['lr'] * gradient
    opt.step()
    tolerance = 2e-14 if owner.dtype == 'float64' else 2e-6
    assert owner.tolist() == pytest.approx(expected.tolist(), rel=tolerance, abs=tolerance)
    if kind is Adam:
        state = opt.state_dict()
        for key, expected_moment in [('m', m), ('v', v)]:
            moment = state[key][0]
            assert moment.dtype == owner.dtype and moment.device == owner.device
            assert moment.tolist() == pytest.approx(expected_moment.tolist(), rel=tolerance, abs=tolerance)
            assert moment.is_leaf and not moment.requires_grad and moment.grad is None
            assert (moment is old[key][0]) == (transform == 'same-cpu')
    opt.zero_grad()
    assert owner.grad is None


@pytest.mark.parametrize('kind', [SGD, Adam])
@pytest.mark.parametrize('origin,target', [('cpu', 'cuda:0'), ('cuda:0', 'cpu'), ('cuda:0', 'cuda:0')])
def test_existing_optimizer_survives_physical_cuda_module_move(
    cuda_device, kind, origin, target, typed_tensor, assert_tensor,
):
    model = Module()
    model.weight = Parameter(typed_tensor([2.], device=origin))
    owner = model.weight
    opt = kind(model.parameters(), lr=.25, **({'betas': (.5, .75), 'eps': 1.} if kind is Adam else {}))
    owner.set_grad(typed_tensor([2.], device=origin))
    opt.step()
    old = opt.state_dict()
    model.to(target)
    assert model.weight is owner and owner.grad is None
    owner.set_grad(typed_tensor([2.], device=target))
    opt.step()
    assert_tensor(owner, [1. if kind is SGD else 2. - 1/3])
    if kind is Adam:
        state = opt.state_dict()
        for key, expected in [('m', [1.5]), ('v', [1.75])]:
            moment = state[key][0]
            assert moment.device == ortho.Device(target) and moment.dtype == 'float32'
            assert_tensor(moment, expected)
            assert moment.is_leaf and not moment.requires_grad
            assert (moment is old[key][0]) == (origin == target)


def test_missing_gradient_defers_initialized_state_migration(typed_tensor):
    model = Module()
    model.weight = Parameter(typed_tensor([2.]))
    opt = Adam(model.parameters())
    model.weight.sum().backward()
    opt.step()
    old = opt.state_dict()
    model.cast('float64')
    opt.step()
    state = opt.state_dict()
    assert state['t'] == 2
    assert state['m'][0] is old['m'][0] and state['v'][0] is old['v'][0]
    assert state['m'][0].dtype == state['v'][0].dtype == 'float32'
    model.weight.sum().backward()
    opt.step()
    assert opt.state_dict()['m'][0].dtype == opt.state_dict()['v'][0].dtype == 'float64'


def test_bare_cuda_float64_adam_is_not_a_new_compute_profile(cuda_device, typed_tensor):
    p = typed_tensor([2.], dtype='float64', device=cuda_device)
    p.set_grad(typed_tensor([1.], dtype='float64', device=cuda_device))
    opt = Adam([p])
    with pytest.raises(RuntimeError):
        opt.step()
    assert p.tolist() == [2.]
    assert opt.state_dict()['m'] == opt.state_dict()['v'] == [None]


@pytest.mark.parametrize('kind', [SGD, Adam])
@pytest.mark.parametrize('container', [list, tuple, iter], ids=['list', 'tuple', 'generator'])
def test_two_identical_direct_parameters_reject(kind, container, typed_tensor):
    p = typed_tensor([2.]).requires_grad_(True)
    p.set_grad(typed_tensor([1.]))
    with pytest.raises(ValueError, match='duplicate'):
        kind(container([p, p]), lr=.1)
    assert p.tolist() == [2.] and p.grad.tolist() == [1.]


def test_binding_retains_a_strong_reference_to_supplied_owner(parameter, optimizer, assign_gradient):
    import gc
    import weakref

    p = parameter([2.])
    assign_gradient(p, [1.])
    retained = weakref.ref(p)
    supplied = [p]
    opt = optimizer(supplied)
    supplied.clear()
    del p
    gc.collect()
    assert retained() is not None
    opt.step()
    assert retained().tolist() == pytest.approx([1.75], abs=1e-6)
    opt.zero_grad()
    assert retained().grad is None
    del opt
    gc.collect()
    assert retained() is None


def test_tied_parameter_discovered_through_many_paths_updates_once(
    parameter, optimizer, optimizer_case, assert_tensor, assign_gradient,
):
    kind, _, _ = optimizer_case
    model, child = Module(), Module()
    p = parameter([2.])
    shared = Parameter(p)
    model.direct = shared
    child.left = shared
    child.right = shared
    model.child = child
    model.alias_child = child
    assert list(model.parameters()) == [p]
    opt = optimizer(model.parameters())
    assign_gradient(p, [2.])
    opt.step()
    assert_tensor(p, [1.5 if kind is SGD else 1.75])
    if kind is Adam:
        assert len(opt.state_dict()['m']) == len(opt.state_dict()['v']) == 1
