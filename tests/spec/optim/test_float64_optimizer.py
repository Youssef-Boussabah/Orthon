"""Double-precision optimizer references, continuation and checkpoint integration."""
import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD


@pytest.mark.parametrize('lr', [.123456789012345, 0.])
@pytest.mark.parametrize('weight_decay', [0., .234567890123456])
@pytest.mark.parametrize('betas,eps', [((.712345678901234, .823456789012345), .012345678901234),
                                      ((.234567890123456, .912345678901234), .123456789012345)])
def test_float64_adam_first_and_multistep_match_double_reference(
    typed_tensor, lr, weight_decay, betas, eps,
):
    initial = np.array([[2.123456789012345, -3.234567890123456, .123456789012345]])
    p = typed_tensor(initial, dtype='float64').requires_grad_(True)
    unused = typed_tensor([5.], dtype='float64')
    opt = Adam([p, unused], lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    assert opt.state_dict()['m'] == opt.state_dict()['v'] == [None, None]
    m, v, expected = np.zeros_like(initial), np.zeros_like(initial), initial.copy()
    b1, b2 = betas
    live = None
    gradients = [np.array([[.123456789012345, -.345678901234567, 0.]]),
                 np.array([[-.456789012345678, .567890123456789, .67890123456789]]),
                 np.zeros_like(initial), np.array([[.789012345678901, -.890123456789012, -.901234567890123]])]
    for t, g in enumerate(gradients, 1):
        p.set_grad(typed_tensor(g, dtype='float64'))
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g*g
        expected = (1-lr*weight_decay)*expected - lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
        assert opt.step() is None
        state = opt.state_dict()
        assert state['t'] == t
        assert p.tolist() == pytest.approx(expected.ravel().tolist(), rel=2e-14, abs=2e-15)
        for key, value in [('m', m), ('v', v)]:
            moment = state[key][0]
            assert moment.dtype == 'float64' and moment.device == p.device
            assert moment.tolist() == pytest.approx(value.ravel().tolist(), rel=2e-14, abs=2e-15)
            assert moment.is_leaf and not moment.requires_grad and moment.grad is None
            assert state[key][1] is None
            if live is not None:
                assert moment is live[key][0]
        assert state['m'][0] is not state['v'][0]
        assert p.grad.tolist() == g.ravel().tolist()
        live = state


@pytest.mark.parametrize('decay_bias', [False, True])
def test_float64_sgd_decimal_coefficients_and_decay_are_double_precision(typed_tensor, decay_bias):
    lr, wd = .123456789012345, .234567890123456
    initial = np.array([[2.123456789012345, -3.234567890123456]])
    g = np.array([[.345678901234567, -.456789012345678]])
    p = typed_tensor(initial, dtype='float64').requires_grad_(True)
    opt = SGD([p], lr=lr, weight_decay=wd, decay_bias=decay_bias)
    expected = initial.copy()
    for _ in range(4):
        p.set_grad(typed_tensor(g, dtype='float64'))
        expected = (1-lr*wd)*expected - lr*g
        opt.step()
        assert p.tolist() == pytest.approx(expected.ravel().tolist(), rel=2e-15, abs=2e-15)


@pytest.mark.parametrize('placement', [None, 'cpu', ortho.Device('cpu')])
def test_float64_adam_direct_resume_matches_uninterrupted_updates(typed_tensor, placement):
    p = typed_tensor([2., -3.], dtype='float64').requires_grad_(True)
    source = Adam([p], lr=.123456789012345, betas=(.712345678901, .823456789012), eps=.012345678901)
    p.set_grad(typed_tensor([.345678901234, -.456789012345], dtype='float64'))
    source.step()
    q = p.to('cpu').requires_grad_(True)
    target = Adam([q])
    target.load_state_dict(source.state_dict(), map_location=placement)
    for gradient in ([0., 0.], [-.567890123456, .678901234567], [.789012345678, -.890123456789]):
        for owner in (p, q):
            owner.set_grad(typed_tensor(gradient, dtype='float64'))
        source.step()
        target.step()
        assert p.tolist() == q.tolist()
        for key in ('m', 'v'):
            a, b = source.state_dict()[key][0], target.state_dict()[key][0]
            assert a is not b and a.dtype == b.dtype == 'float64'
            assert a.tolist() == b.tolist()


@pytest.mark.parametrize('tied', [False, True])
def test_float64_adam_checkpoint_resume_preserves_owners_and_history(typed_tensor, tmp_path, tied):
    def create():
        model = Module()
        marked = Parameter(typed_tensor([2., -3.], dtype='float64'))
        model.weight = marked
        model.child = Module()
        if tied:
            model.child.alias = marked
        model.unused = Parameter(typed_tensor([5.], dtype='float64'))
        opt = Adam(model.parameters(), lr=.123456789012345, betas=(.712345678901, .823456789012),
                   eps=.012345678901, weight_decay=.234567890123, decay_bias=True)
        return model, opt

    def advance(model, opt):
        opt.zero_grad()
        value = model.weight + model.child.alias if tied else model.weight
        loss = (value*value).sum()
        loss.backward()
        opt.step()
        return loss.item()

    source, opt = create()
    for _ in range(3):
        advance(source, opt)
    path = tmp_path / 'float64.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=opt)
    expected = [advance(source, opt) for _ in range(3)]
    resumed, restored = create()
    held = list(resumed.parameters())
    ortho.load_checkpoint(path, model=resumed, optimizer=restored, map_location='cpu')
    assert all(p is q for p, q in zip(resumed.parameters(), held))
    if tied:
        assert resumed.weight is resumed.child.alias
    assert [advance(resumed, restored) for _ in range(3)] == expected
    assert resumed.weight.tolist() == source.weight.tolist()
    for key in ('m', 'v'):
        assert len(restored.state_dict()[key]) == 2
        assert restored.state_dict()[key][1] is None
        assert restored.state_dict()[key][0].dtype == 'float64'
        assert restored.state_dict()[key][0].tolist() == opt.state_dict()[key][0].tolist()
