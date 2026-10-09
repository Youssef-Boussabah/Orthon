"""Independent moment snapshots and locally transactional optimizer loading."""
import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


def _snapshot(opt, parameters):
    state = opt.state_dict()
    return state, {key: [None if x is None else x.tolist() for x in state[key]]
                   for key in ('m', 'v')}, [(p.tolist(), p.grad.tolist()) for p in parameters]


def _assert_unchanged(opt, parameters, snapshot):
    old, values, model = snapshot
    now = opt.state_dict()
    for key in ('lr', 'betas', 'eps', 'weight_decay', 'decay_bias', 't'):
        assert now[key] == old[key]
    for key in ('m', 'v'):
        for x, y, expected in zip(now[key], old[key], values[key]):
            assert x is y
            assert (None if x is None else x.tolist()) == expected
    assert [(p.tolist(), p.grad.tolist()) for p in parameters] == model
    # Prove subsequent step still targets exactly the held owners.
    opt.zero_grad()
    assert all(p.grad is None for p in parameters)


@pytest.mark.parametrize('malformed', [
    'missing-lr', 'missing-betas', 'missing-eps', 'missing-t', 'missing-m', 'missing-v',
    'short-betas', 'long-betas', 'm-count', 'v-count', 'm-shape', 'v-shape',
    'm-type', 'v-type', 'm-none', 'v-none', 'invalid-location', 'wrong-location',
    'location-type',
])
def test_failed_adam_load_preserves_complete_state(
    adam_parameter, assign_gradient, malformed,
):
    parameters = [adam_parameter([2.]), adam_parameter([3.])]
    opt = Adam(parameters, lr=.125, betas=(.5, .75), eps=.25, weight_decay=.125, decay_bias=True)
    for p in parameters:
        assign_gradient(p, [2.])
    opt.step()
    snapshot = _snapshot(opt, parameters)
    incoming = Adam(parameters, lr=.5, betas=(.2, .3), eps=.5).state_dict()
    incoming.update(t=19, m=[p.to(p.device) for p in parameters], v=[p.to(p.device) for p in parameters])
    location = None
    if malformed.startswith('missing-'):
        del incoming[malformed[8:]]
    elif malformed.endswith('betas'):
        incoming['betas'] = (.2,) if malformed.startswith('short') else (.2, .3, .4)
    elif malformed.endswith('count'):
        incoming[malformed[0]].pop()
    elif malformed.endswith('shape'):
        incoming[malformed[0]][1] = adam_parameter([[1.]])
    elif malformed.endswith('type') and malformed != 'location-type':
        incoming[malformed[0]][1] = [1.]
    elif malformed.endswith('none'):
        incoming[malformed[0]][1] = None
    else:
        location = {'invalid-location': 'moon', 'wrong-location': 'cuda:1', 'location-type': {}}[malformed]
    with pytest.raises((KeyError, TypeError, ValueError, RuntimeError)):
        opt.load_state_dict(incoming, map_location=location)
    _assert_unchanged(opt, parameters, snapshot)


@pytest.mark.parametrize('kind', [SGD, Adam])
def test_configuration_preparation_failure_is_atomic(kind, typed_tensor):
    p = typed_tensor([2.]).requires_grad_(True)
    p.set_grad(typed_tensor([1.]))
    opt = kind([p], lr=.125, weight_decay=.125, decay_bias=True)
    opt.step()
    before = opt.state_dict()

    class BadMapping(dict):
        def get(self, key, default=None):
            if key == 'decay_bias':
                raise ValueError('late configuration read failure')
            return super().get(key, default)

    incoming = BadMapping(before, lr=.5, weight_decay=.5)
    with pytest.raises(ValueError):
        opt.load_state_dict(incoming)
    after = opt.state_dict()
    assert after['lr'] == before['lr'] and after['weight_decay'] == before['weight_decay']
    assert after['decay_bias'] == before['decay_bias']
    if kind is Adam:
        assert after['t'] == before['t']
        assert after['m'][0] is before['m'][0] and after['v'][0] is before['v'][0]


def test_late_moment_conversion_failure_is_atomic(typed_tensor, monkeypatch):
    parameters = [typed_tensor([2.]).requires_grad_(True), typed_tensor([3.]).requires_grad_(True)]
    opt = Adam(parameters, lr=.125)
    for p in parameters:
        p.set_grad(typed_tensor([1.]))
    opt.step()
    before = _snapshot(opt, parameters)
    incoming = dict(opt.state_dict(), lr=.5, t=11)
    incoming['m'] = [typed_tensor([4.], dtype='float64') for _ in parameters]
    incoming['v'] = [typed_tensor([5.], dtype='float64') for _ in parameters]
    real_cast = ortho.cast
    calls = []

    def fail_late(tensor, dtype):
        assert not ortho.is_grad_enabled()
        calls.append(tensor)
        if len(calls) == 4:
            raise RuntimeError('injected last moment conversion failure')
        return real_cast(tensor, dtype)

    monkeypatch.setattr(ortho, 'cast', fail_late)
    with pytest.raises(RuntimeError, match='injected'):
        opt.load_state_dict(incoming)
    assert len(calls) == 4 and ortho.is_grad_enabled()
    _assert_unchanged(opt, parameters, before)


def test_transform_moment_pair_is_published_only_after_both_conversions(typed_tensor, monkeypatch):
    model = Module()
    model.weight = Parameter(typed_tensor([2.]))
    opt = Adam(model.parameters())
    model.weight.sum().backward()
    opt.step()
    old = opt.state_dict()
    values = [old[key][0].tolist() for key in ('m', 'v')]
    model.cast('float64')
    model.weight.sum().backward()
    before = model.weight.tolist()
    real_cast = ortho.cast
    calls = []

    def fail_second(tensor, dtype):
        assert not ortho.is_grad_enabled()
        calls.append(tensor)
        if len(calls) == 2:
            raise RuntimeError('injected second conversion failure')
        return real_cast(tensor, dtype)

    monkeypatch.setattr(ortho, 'cast', fail_second)
    with pytest.raises(RuntimeError, match='injected'):
        opt.step()
    state = opt.state_dict()
    assert model.weight.tolist() == before and model.weight.grad.tolist() == [1.]
    for key, expected in zip(('m', 'v'), values):
        assert state[key][0] is old[key][0]
        assert state[key][0].dtype == 'float32' and state[key][0].tolist() == expected
    assert ortho.is_grad_enabled()


def test_direct_load_owns_independent_graph_free_moment_snapshots(
    adam_parameter, assign_gradient, assert_tensor,
):
    p, q = adam_parameter([2.]), adam_parameter([2.])
    source, target = Adam([p], lr=0.), Adam([q], lr=0.)
    assign_gradient(p, [2.])
    source.step()
    incoming = source.state_dict()
    before = {key: incoming[key][0].tolist() for key in ('m', 'v')}
    target.load_state_dict(incoming)
    loaded = target.state_dict()
    for key in ('m', 'v'):
        assert loaded[key][0] is not incoming[key][0]
        assert loaded[key][0].is_leaf and not loaded[key][0].requires_grad
        assert loaded[key][0].grad is None
    assign_gradient(q, [4.])
    target.step()
    for key in ('m', 'v'):
        assert incoming[key][0].tolist() == before[key]
    current = {key: target.state_dict()[key][0].tolist() for key in ('m', 'v')}
    for key in ('m', 'v'):
        incoming[key][0].mul_(7.)
        assert target.state_dict()[key][0].tolist() == current[key]
    assert_tensor(p, [2.])
    assert_tensor(q, [2.])


def test_load_breaks_reused_moment_and_cross_binding_aliases(adam_parameter):
    parameters = [adam_parameter([2.]), adam_parameter([3.])]
    shared = adam_parameter([1.])
    state = Adam(parameters).state_dict()
    state.update(m=[shared, shared], v=[shared, shared])
    opt = Adam(parameters)
    opt.load_state_dict(state)
    exported = opt.state_dict()
    moments = exported['m'] + exported['v']
    assert len({id(x) for x in moments}) == 4
    assert all(x is not shared and x.is_leaf and not x.requires_grad for x in moments)
    moments[0].mul_(3.)
    assert all(x.tolist() == [1.] for x in moments[1:])
    assert shared.tolist() == [1.]
    with ortho.no_grad():
        shared.mul_(5.)
    assert all(x.tolist() == [1.] for x in moments[1:])


def test_self_load_snapshots_then_retains_live_moment_owners(adam_parameter, assign_gradient):
    p = adam_parameter([2.])
    opt = Adam([p], lr=0., betas=(.5, .75))
    assign_gradient(p, [2.])
    opt.step()
    old = opt.state_dict()
    opt.load_state_dict(old)
    loaded = opt.state_dict()
    for key in ('m', 'v'):
        assert loaded[key][0] is not old[key][0]
    assert loaded is not opt.state_dict()
    assign_gradient(p, [4.])
    opt.step()
    assert loaded['m'][0] is opt.state_dict()['m'][0]
    assert loaded['v'][0] is opt.state_dict()['v'][0]
    assert loaded['m'][0].tolist() == [2.5] and loaded['v'][0].tolist() == [4.75]
    assert old['m'][0].tolist() == old['v'][0].tolist() == [1.]


def test_exported_sequence_edits_do_not_replace_owned_state(adam_parameter, assign_gradient):
    p = adam_parameter([2.])
    opt = Adam([p])
    assign_gradient(p, [1.])
    opt.step()
    before = opt.state_dict()
    m, v = before['m'][0], before['v'][0]
    before['m'].clear()
    before['v'][0] = None
    assert opt.state_dict()['m'][0] is m and opt.state_dict()['v'][0] is v
