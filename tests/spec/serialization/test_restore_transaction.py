"""Whole-checkpoint preparation, stable owner placement, and defensive rollback."""
import pytest
import numpy as np

import orthon as ortho
from orthon.nn import Module, Parameter
from orthon.optim import Adam, SGD
from orthon.amp import MixedPrecisionOptimizer
from .workload import RegressionModel, StateSource, advance, make_optimizer, training_state


def independent(value):
    if isinstance(value, ortho.Tensor):
        with ortho.no_grad():
            return value.to(value.device)
    if isinstance(value, dict):
        return {key: independent(item) for key, item in value.items()}
    if isinstance(value, list):
        return [independent(item) for item in value]
    if isinstance(value, tuple):
        return tuple(independent(item) for item in value)
    return value


def values(value):
    if isinstance(value, ortho.Tensor):
        with ortho.no_grad():
            return (tuple(value.shape), value.dtype, str(value.device),
                    ortho._tensor_to_bytes(value.to('cpu')))
    if isinstance(value, dict):
        return {key: values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(values(item) for item in value)
    return value


def capture(model, optimizer):
    owners = tuple(model.parameters())
    tensors = dict(model.state_dict())
    cache = model.cache
    state = values(tensors)
    flags = [(p._impl_id, p.requires_grad, p.is_leaf, p.dtype, str(p.device),
              None if p.grad is None else (p.grad._impl_id, values(p.grad))) for p in owners]
    modes = model.training, model.head.training
    history = values(optimizer.state_dict())

    def unchanged():
        assert all(p is q for p, q in zip(owners, model.parameters()))
        assert all(model.state_dict()[name] is tensor for name, tensor in tensors.items())
        assert model.cache is cache
        assert values(model.state_dict()) == state
        assert [(p._impl_id, p.requires_grad, p.is_leaf, p.dtype, str(p.device),
                 None if p.grad is None else (p.grad._impl_id, values(p.grad))) for p in owners] == flags
        assert (model.training, model.head.training) == modes
        assert values(optimizer.state_dict()) == history
    return unchanged


def destination(device='cpu', kind='Adam'):
    model = RegressionModel(device=device).eval()
    optimizer = make_optimizer(model, kind, lr=.5)
    advance(model, optimizer, 1)
    return model, optimizer


@pytest.mark.parametrize('damage', ['count', 'one-sided', 'shape', 'type', 'missing-config'])
@pytest.mark.parametrize('explicit', [False, True])
def test_combined_adam_semantic_failure_preserves_whole_destination(tmp_path, damage, explicit):
    source = RegressionModel()
    optimizer = make_optimizer(source, 'Adam')
    advance(source, optimizer, 2)
    state = independent(optimizer.state_dict())
    if damage == 'count':
        state['m'].pop()
    elif damage == 'one-sided':
        state['v'][0] = None
    elif damage == 'shape':
        state['m'][0] = ortho.ones([7])
    elif damage == 'type':
        state['v'][0] = 7
    else:
        del state['eps']
    path = tmp_path / 'adam.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=StateSource(state))
    model, recipient = destination()
    unchanged = capture(model, recipient)
    with pytest.raises((ValueError, TypeError, KeyError)):
        ortho.load_checkpoint(path, model=model, optimizer=recipient,
                              map_location='cpu' if explicit else None)
    unchanged()


def test_wrong_destination_optimizer_parameter_count_cannot_apply_valid_model(tmp_path):
    source = RegressionModel()
    opt = make_optimizer(source, 'Adam')
    advance(source, opt, 2)
    path = tmp_path / 'count.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=opt)
    model, _ = destination()
    recipient = Adam([model.weight], lr=.75)
    model.weight.set_grad(ortho.ones(list(model.weight.shape)))
    recipient.step()
    unchanged = capture(model, recipient)
    with pytest.raises(ValueError, match='count mismatch'):
        ortho.load_checkpoint(path, model=model, optimizer=recipient)
    unchanged()


def test_combined_sgd_missing_required_config_cannot_apply_model(tmp_path):
    source = RegressionModel()
    path = tmp_path / 'sgd.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=StateSource({'weight_decay': 1}))
    model, opt = destination(kind='SGD')
    unchanged = capture(model, opt)
    with pytest.raises(KeyError):
        ortho.load_checkpoint(path, model=model, optimizer=opt)
    unchanged()


@pytest.mark.parametrize('damage', ['count', 'shape', 'dtype', 'inner-count', 'inner-pair'])
def test_combined_amp_preparation_failure_preserves_model_masters_and_inner_state(tmp_path, damage):
    source = RegressionModel().half()
    optimizer = MixedPrecisionOptimizer(source, Adam, lr=.125)
    source.weight.set_grad(ortho.cast(ortho.ones(list(source.weight.shape)), 'float16'))
    optimizer.step()
    state = independent(optimizer.state_dict())
    if damage == 'count':
        state['masters'].pop()
    elif damage == 'shape':
        state['masters'][0] = ortho.ones([7])
    elif damage == 'dtype':
        state['orig_dtypes'][0] = 'float64'
    elif damage == 'inner-count':
        state['inner']['m'].pop()
    else:
        state['inner']['v'][0] = None
    path = tmp_path / 'amp.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=StateSource(state))
    model = RegressionModel().half().eval()
    recipient = MixedPrecisionOptimizer(model, Adam, lr=.5)
    model.weight.set_grad(ortho.cast(ortho.ones(list(model.weight.shape)), 'float16'))
    recipient.step()
    unchanged = capture(model, recipient)
    with pytest.raises(ValueError):
        ortho.load_checkpoint(path, model=model, optimizer=recipient, map_location='cpu')
    unchanged()


@pytest.mark.parametrize('damage', ['model-shape', 'optimizer-count', 'amp-inner'])
@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
def test_failed_explicit_device_remap_never_moves_or_clears_destination(
    cuda_device, tmp_path, damage, direction,
):
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    source = RegressionModel()
    source_opt = make_optimizer(source, 'Adam')
    advance(source, source_opt, 1)
    model_state, opt_state = independent(source.state_dict()), independent(source_opt.state_dict())
    if damage == 'model-shape':
        model_state['head.bias'] = ortho.ones([7])
    elif damage == 'optimizer-count':
        opt_state['m'].pop()
    else:
        source_opt = MixedPrecisionOptimizer(source, Adam, lr=.125)
        opt_state = independent(source_opt.state_dict())
        opt_state['inner']['m'].pop()
    path = tmp_path / 'move.ckpt'
    ortho.save_checkpoint(path, model=StateSource(model_state), optimizer=StateSource(opt_state))
    model, opt = destination(original)
    if damage == 'amp-inner':
        opt = MixedPrecisionOptimizer(model, Adam, lr=.5)
        model.weight.set_grad(ortho.ones(list(model.weight.shape)).to(original))
        opt.step()
    unchanged = capture(model, opt)
    with pytest.raises(ValueError):
        ortho.load_checkpoint(path, model=model, optimizer=opt, map_location=target)
    unchanged()


@pytest.mark.parametrize('kind', ['SGD', 'Adam'])
@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu', 'same-cuda'])
def test_combined_explicit_remap_matches_model_first_reference_and_next_steps(
    cuda_device, tmp_path, kind, direction,
):
    original, target = {'cpu-to-cuda': ('cpu', cuda_device),
                        'cuda-to-cpu': (cuda_device, 'cpu'),
                        'same-cuda': (cuda_device, cuda_device)}[direction]
    source = RegressionModel()
    opt = make_optimizer(source, kind)
    advance(source, opt, 2)
    path = tmp_path / 'resume.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=opt, metadata={'step': 2})
    reference = RegressionModel()
    ortho.load_checkpoint(path, model=reference, map_location=target)
    reference_opt = make_optimizer(reference, kind, lr=1)
    ortho.load_checkpoint(path, optimizer=reference_opt)
    model, recipient = destination(original, kind)
    held = tuple(model.parameters())
    assert ortho.load_checkpoint(path, model=model, optimizer=recipient, map_location=target) == {'step': 2}
    assert all(p is q for p, q in zip(held, model.parameters()))
    assert all(p.device == ortho.Device(target) for p in model.parameters())
    assert model.cache.device == ortho.Device(target)
    assert all(p.grad is None for p in model.parameters()) if original != target else all(
        p.grad is not None for p in model.parameters() if p is not model.spare)
    assert advance(model, recipient, 3) == pytest.approx(advance(reference, reference_opt, 3), rel=1e-6)
    assert training_state(model, recipient) == training_state(reference, reference_opt)


@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16'])
@pytest.mark.parametrize('target_direction', ['cpu-to-cuda', 'cuda-to-cpu'])
def test_amp_combined_explicit_placement_matches_reference(
    cuda_device, tmp_path, dtype, target_direction,
):
    original, target = ('cpu', cuda_device) if target_direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    source = RegressionModel()
    source.cast(dtype)
    optimizer = MixedPrecisionOptimizer(source, Adam, lr=.125)
    source.weight.set_grad(ortho.cast(ortho.ones(list(source.weight.shape)), dtype))
    optimizer.step()
    path = tmp_path / 'amp-move.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    reference = RegressionModel().cast(dtype)
    ortho.load_checkpoint(path, model=reference, map_location=target)
    reference_opt = MixedPrecisionOptimizer(reference, Adam, lr=1)
    ortho.load_checkpoint(path, optimizer=reference_opt)
    model = RegressionModel().cast(dtype).to(original)
    recipient = MixedPrecisionOptimizer(model, Adam, lr=1)
    owners = tuple(model.parameters())
    ortho.load_checkpoint(path, model=model, optimizer=recipient, map_location=target)
    assert all(p is q for p, q in zip(owners, model.parameters()))
    assert values(recipient.state_dict()) == values(reference_opt.state_dict())
    for current, opt in ((model, recipient), (reference, reference_opt)):
        current.weight.set_grad(ortho.cast(ortho.ones(list(current.weight.shape)), dtype).to(target))
        opt.step()
    assert values(model.state_dict()) == values(reference.state_dict())
    assert values(recipient.state_dict()) == values(reference_opt.state_dict())


@pytest.mark.parametrize('kind', [SGD, Adam])
def test_optimizer_only_placement_conflict_rejects_without_moving_parameters(cuda_device, tmp_path, kind):
    source = RegressionModel()
    optimizer = kind(source.parameters(), lr=.125)
    advance(source, optimizer, 1)
    path = tmp_path / 'optimizer.ckpt'
    ortho.save_checkpoint(path, optimizer=optimizer)
    model = RegressionModel().eval()
    recipient = kind(model.parameters(), lr=.5)
    model.weight.set_grad(ortho.ones(list(model.weight.shape)))
    unchanged = capture(model, recipient)
    with pytest.raises(ValueError, match='conflicts'):
        ortho.load_checkpoint(path, optimizer=recipient, map_location=cuda_device)
    unchanged()


class GenericOwner:
    def __init__(self, state, rollback_fails=False):
        self.state = state
        self.rollback_fails = rollback_fails

    def state_dict(self):
        return self.state

    def load_state_dict(self, state, map_location=None):
        self.state = state
        if state.get('fail'):
            raise ValueError('generic late commit failure')
        if self.rollback_fails:
            raise ValueError('generic rollback failure')


def test_generic_optimizer_encoded_null_is_present_state(tmp_path):
    class NullOwner:
        def __init__(self, state):
            self.state = state

        def state_dict(self):
            return self.state

        def load_state_dict(self, state, map_location=None):
            self.state = state

    path = tmp_path / 'null-state.ckpt'
    ortho.save_checkpoint(path, optimizer=NullOwner(None), metadata={'step': 1})
    assert ortho.load_checkpoint(path) == {'step': 1}
    recipient = NullOwner({'previous': True})
    assert ortho.load_checkpoint(path, optimizer=recipient) == {'step': 1}
    assert recipient.state is None


@pytest.mark.parametrize('builtin_model', [False, True])
def test_generic_late_failure_rolls_back_prior_and_failing_recipients(tmp_path, builtin_model):
    source = RegressionModel()
    bad = GenericOwner({'lr': .125, 'fail': True})
    path = tmp_path / 'generic.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=bad)
    if builtin_model:
        model, unused = destination()
        unchanged = capture(model, unused)
    else:
        model = GenericOwner(independent(RegressionModel().state_dict()))
        before = values(model.state)
        unchanged = lambda: values(model.state) == before
    recipient = GenericOwner({'lr': 1, 'fail': False})
    with pytest.raises(ValueError, match='generic late commit failure'):
        ortho.load_checkpoint(path, model=model, optimizer=recipient)
    if builtin_model:
        unchanged()
    else:
        assert unchanged()
    assert recipient.state == {'lr': 1, 'fail': False}


def test_generic_rollback_failure_reports_transaction_error_and_chains_original(tmp_path):
    path = tmp_path / 'rollback.ckpt'
    ortho.save_checkpoint(path, optimizer=GenericOwner({'fail': True}))
    recipient = GenericOwner({'fail': False}, rollback_fails=True)
    with pytest.raises(RuntimeError, match='checkpoint: transaction rollback failed') as caught:
        ortho.load_checkpoint(path, optimizer=recipient)
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == 'generic late commit failure'


@pytest.mark.parametrize('kind', [SGD, Adam])
def test_file_and_direct_load_call_the_same_preparer(monkeypatch, tmp_path, kind):
    source = RegressionModel()
    optimizer = kind(source.parameters(), lr=.125)
    advance(source, optimizer, 1)
    model = RegressionModel()
    recipient = kind(model.parameters(), lr=1)
    calls = []
    original_model = Module._prepare_load_state_dict
    original_optimizer = kind._prepare_load_state_dict

    def model_prepare(self, *args, **kwargs):
        calls.append('model')
        return original_model(self, *args, **kwargs)

    def opt_prepare(self, *args, **kwargs):
        calls.append('optimizer')
        return original_optimizer(self, *args, **kwargs)

    monkeypatch.setattr(Module, '_prepare_load_state_dict', model_prepare)
    monkeypatch.setattr(kind, '_prepare_load_state_dict', opt_prepare)
    model.load_state_dict(source.state_dict())
    recipient.load_state_dict(optimizer.state_dict())
    path = tmp_path / 'same-path.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    ortho.load_checkpoint(path, model=model, optimizer=recipient)
    assert calls == ['model', 'optimizer', 'model', 'optimizer']


def test_strided_model_and_tied_owners_restore_through_file_transaction(operator_device, tmp_path):
    source = Module()
    source.a = Parameter(ortho.ones([2, 3]).to(operator_device).transpose(0, 1))
    source.b = Parameter(source.a)
    path = tmp_path / 'strided.ckpt'
    ortho.save_state_dict(source, path)
    model = Module()
    original = ortho.zeros([2, 3]).to(operator_device).transpose(0, 1)
    model.a = Parameter(original)
    model.b = Parameter(original)
    optimizer = SGD(model.parameters(), lr=.25)
    ortho.load_checkpoint(path, model=model)
    assert model.a is model.b is original
    assert len(list(model.parameters())) == 1
    assert not original.is_contiguous
    assert original.tolist() == [1] * 6
    # Optimizer updates require contiguous owners; the established stable-owner
    # transform materializes this layout while retaining the existing binding.
    model.to(operator_device)
    assert model.a is model.b is original
    original.set_grad(ortho.ones([3, 2]).to(operator_device))
    optimizer.step()
    assert original.tolist() == [.75] * 6


def test_module_synchronous_partial_commit_restores_parameters_buffers_and_gradients(monkeypatch, tmp_path):
    source = RegressionModel()
    path = tmp_path / 'commit.ckpt'
    ortho.save_checkpoint(path, model=source)
    model, optimizer = destination()
    unchanged = capture(model, optimizer)
    original = Module._copy_state_value
    calls = 0

    def fail_late(dst, src):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('injected synchronous model copy failure')
        return original(dst, src)

    monkeypatch.setattr(Module, '_copy_state_value', staticmethod(fail_late))
    with pytest.raises(RuntimeError, match='injected synchronous model copy failure'):
        ortho.load_checkpoint(path, model=model)
    unchanged()


def test_amp_synchronous_commit_failure_rolls_back_wrapper_and_already_loaded_model(monkeypatch, tmp_path):
    source = RegressionModel().half()
    source_opt = MixedPrecisionOptimizer(source, Adam, lr=.125)
    path = tmp_path / 'amp-commit.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=source_opt)
    model = RegressionModel().half().eval()
    recipient = MixedPrecisionOptimizer(model, Adam, lr=.5)
    model.weight.set_grad(ortho.cast(ortho.ones(list(model.weight.shape)), 'float16'))
    recipient.step()
    unchanged = capture(model, recipient)
    original = ortho._amp_copy_master_to_model_
    calls = 0

    def fail_late(parameter, master):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected synchronous AMP copy failure')
        return original(parameter, master)

    monkeypatch.setattr(ortho, '_amp_copy_master_to_model_', fail_late)
    with pytest.raises(RuntimeError, match='injected synchronous AMP copy failure'):
        ortho.load_checkpoint(path, model=model, optimizer=recipient)
    unchanged()


def test_future_placement_is_only_granted_to_the_models_bound_owners(cuda_device, tmp_path):
    source = RegressionModel()
    optimizer = make_optimizer(source, 'Adam')
    advance(source, optimizer, 1)
    path = tmp_path / 'external.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    model = RegressionModel()
    foreign = ortho.ones(list(model.weight.shape)).requires_grad_(True)
    recipient = Adam([foreign, model.spare, model.head.bias], lr=.5)
    unchanged = capture(model, recipient)
    with pytest.raises(ValueError, match='conflicts'):
        ortho.load_checkpoint(path, model=model, optimizer=recipient, map_location=cuda_device)
    unchanged()


@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
def test_late_failure_after_explicit_move_restores_original_representation(cuda_device, tmp_path, direction):
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    source = RegressionModel()
    path = tmp_path / 'late-move.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=GenericOwner({'fail': True}))
    model, opt = destination(original)
    unchanged = capture(model, opt)
    events = []
    model.weight._register_post_accumulate_grad_hook(lambda: events.append('weight'))
    with pytest.raises(ValueError, match='generic late commit failure'):
        ortho.load_checkpoint(path, model=model, optimizer=GenericOwner({'fail': False}),
                              map_location=target)
    unchanged()
    model.zero_grad()
    (model.weight * model.weight).sum().backward()
    assert events == ['weight']


def _r3_model(device='cpu', layout='transpose', shift=0.):
    model = Module()
    raw = (np.arange(12, dtype=np.float32).reshape(3, 4) + shift) / 8
    base = ortho.from_numpy(raw).to(device)
    if layout == 'transpose':
        view = base.transpose(0, 1)
    elif layout == 'offset':
        view = base.slice(0, 1, 3).slice(1, 1, 3)
    elif layout == 'scalar':
        view = base[1][1]
    else:
        view = base.transpose(0, 1).slice(0, 4, 4)
    model.a = Parameter(view)
    model.b = Parameter(view)
    model.spare = Parameter(ortho.from_numpy(np.array([.5, 1.], np.float32)).to(device))
    model.register_buffer('stats', ortho.from_numpy(np.array([shift + .25], np.float32)).to(device))
    cache = ortho.from_numpy(np.array([99.], np.float32)).to(device)
    model.register_buffer('cache', cache, persistent=False)
    model.register_buffer('cache_tie', cache, persistent=False)
    model.register_buffer('owner_alias', view, persistent=False)
    return model, base


def _r3_tensor_state(tensor):
    with ortho.no_grad():
        payload = ortho._tensor_to_bytes(tensor.to('cpu'))
    return (tensor.shape, tensor.strides, tensor.storage_offset, tensor.dtype,
            str(tensor.device), tensor.requires_grad, tensor.is_leaf,
            tensor._has_grad_fn, payload)


def _r3_capture(model, optimizer, base):
    parameters = tuple(model.parameters())
    slots = dict(model._buffers)
    tensors = [_r3_tensor_state(p) for p in parameters]
    grads = [None if p.grad is None else _r3_tensor_state(p.grad) for p in parameters]
    prior_base = _r3_tensor_state(base)
    prior_optimizer = values(optimizer.state_dict())

    def unchanged():
        assert tuple(model.parameters()) == parameters
        assert model.a is model.b is parameters[0] and model.owner_alias is model.a
        assert all(model._buffers[name] is value for name, value in slots.items())
        assert [_r3_tensor_state(p) for p in parameters] == tensors
        assert [None if p.grad is None else _r3_tensor_state(p.grad) for p in parameters] == grads
        assert _r3_tensor_state(base) == prior_base
        assert values(optimizer.state_dict()) == prior_optimizer
        assert all(b.parameter is p for b, p in zip(optimizer._bindings, parameters))
    return unchanged


def _r3_optimizer(model, kind, state=None):
    optimizer = kind(model.parameters(), lr=.03125)
    if state is not None:
        optimizer.load_state_dict(state)
    else:
        # Nonzero moments are legal for strided owners; stepping them first
        # requires the established contiguous Module transform.
        if kind is Adam:
            state = optimizer.state_dict()
            state['t'] = 2
            state['m'] = [ortho.full(list(p.shape), .125).to(p.device) for p in model.parameters()]
            state['v'] = [ortho.full(list(p.shape), .25).to(p.device) for p in model.parameters()]
            optimizer.load_state_dict(state)
    return optimizer


@pytest.mark.parametrize('layout', ['transpose', 'offset'])
@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
@pytest.mark.parametrize('kind', [None, SGD, Adam], ids=['model-only', 'SGD', 'Adam'])
def test_r3_strided_checkpoint_move_preserves_owners_and_optimizer(
    cuda_device, tmp_path, layout, direction, kind,
):
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    source, _ = _r3_model(layout=layout, shift=8.)
    model, _ = _r3_model(original, layout)
    owner = model.a
    owner.set_grad(ortho.ones(list(owner.shape)).to(original))
    optimizer = None if kind is None else _r3_optimizer(model, kind)
    saved_optimizer = None if kind is None else _r3_optimizer(source, kind)
    path = tmp_path / 'strided-move.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=saved_optimizer)
    expected = source.a.tolist()
    events = []
    owner._register_post_accumulate_grad_hook(lambda: events.append('old-hook'))
    ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location=target)
    assert model.a is model.b is owner and model.owner_alias is owner
    assert owner.device == ortho.Device(target) and owner.dtype == 'float32'
    assert owner.shape == source.a.shape and owner.is_contiguous
    assert owner.tolist() == expected and owner.requires_grad and owner.is_leaf
    assert owner.grad is None and not owner._has_grad_fn
    assert model.stats.tolist() == source.stats.tolist()
    assert model.cache is model.cache_tie and model.cache.tolist() == [99.]
    assert all(b.device == ortho.Device(target) for b in model._buffers.values())
    if optimizer is not None:
        assert all(b.parameter is p for b, p in zip(optimizer._bindings, model.parameters()))
        if kind is Adam:
            assert optimizer.state_dict()['t'] == 2
            for moment in optimizer.state_dict()['m'] + optimizer.state_dict()['v']:
                assert moment.device == ortho.Device(target)
                assert any(v != 0 for v in moment.tolist())
        optimizer.zero_grad()
        (owner * owner).sum().backward()
        optimizer.step()
        assert owner.tolist() != expected
        # A true move follows Module.to: prior gradient/hooks belong to the
        # original representation, rather than being copied to the new leaf.
        assert events == []


@pytest.mark.parametrize('layout', ['transpose', 'offset', 'scalar', 'empty'])
@pytest.mark.parametrize('kind', [SGD, Adam])
def test_r3_same_device_strided_load_preserves_layout_gradient_and_hooks(tmp_path, layout, kind):
    source, _ = _r3_model(layout=layout, shift=8.)
    model, base = _r3_model(layout=layout)
    owner = model.a
    gradient = ortho.ones(list(owner.shape))
    owner.set_grad(gradient)
    layout_before = owner.strides, owner.storage_offset
    optimizer = _r3_optimizer(model, kind)
    events = []
    owner._register_post_accumulate_grad_hook(lambda: events.append('hook'))
    path = tmp_path / 'same-device.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=_r3_optimizer(source, kind))
    ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location='cpu')
    assert model.a is model.b is owner and model.owner_alias is owner
    assert (owner.strides, owner.storage_offset) == layout_before
    assert owner.tolist() == source.a.tolist() and owner.grad.tolist() == gradient.tolist()
    assert owner._storage_version == base._storage_version
    optimizer.zero_grad()
    (owner * owner).backward()
    assert events == ['hook']
    # Stepping follows the existing optimizer contiguous-owner requirement.
    model.to('cpu')
    model.a.set_grad(ortho.ones(list(owner.shape)))
    optimizer.step()


@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
@pytest.mark.parametrize('phase', ['one-model-move', 'before-optimizer', 'inside-optimizer'])
def test_r3_strided_late_failure_restores_original_layout_aliases_and_hooks(
    monkeypatch, cuda_device, tmp_path, direction, phase,
):
    from orthon._state import _PreparedLoad
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    source, _ = _r3_model(layout='offset', shift=8.)
    model, base = _r3_model(original, 'offset')
    optimizer = _r3_optimizer(model, Adam)
    model.a.set_grad(ortho.ones(list(model.a.shape)).to(original))
    old_gradient = model.a.grad
    unchanged = _r3_capture(model, optimizer, base)
    events = []
    model.a._register_post_accumulate_grad_hook(lambda: events.append('hook'))
    path = tmp_path / 'late-strided.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=_r3_optimizer(source, Adam))
    if phase == 'one-model-move':
        rebind = ortho.Tensor._rebind_model_value_
        failed = False

        def fail_once(owner, value):
            nonlocal failed
            rebind(owner, value)
            if owner is model.a and owner.device == ortho.Device(target) and not failed:
                failed = True
                raise ValueError('R3 injected late failure')
        monkeypatch.setattr(ortho.Tensor, '_rebind_model_value_', fail_once)
    else:
        prepare = Adam._prepare_load_state_dict

        def prepare_failure(self, *args, **kwargs):
            prepared = prepare(self, *args, **kwargs)

            def fail():
                assert model.a.device == model.spare.device == ortho.Device(target)
                if phase == 'inside-optimizer':
                    prepared.commit()
                raise ValueError('R3 injected late failure')
            return _PreparedLoad(fail, prepared.rollback)
        monkeypatch.setattr(Adam, '_prepare_load_state_dict', prepare_failure)
    with pytest.raises(ValueError, match='R3 injected late failure'):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location=target)
    unchanged()
    old_gradient.mul_(2)
    assert model.a.grad.tolist() == old_gradient.tolist()
    assert model.a._storage_version == base._storage_version
    optimizer.zero_grad()
    (model.a * model.a).sum().backward()
    assert events == ['hook']
    np.testing.assert_allclose(model.a.grad.to('cpu').tolist(), 2 * np.array(model.a.tolist()))
    # The old view still shares its original Buffer after rollback.
    prior = np.array(model.a.tolist())
    base.mul_(2)
    np.testing.assert_equal(model.a.tolist(), 2 * prior)
    model.to(original)
    model.a.set_grad(ortho.ones(list(model.a.shape)).to(original))
    optimizer.step()


@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
def test_r3_strided_prepare_failure_never_publishes_model(cuda_device, tmp_path, direction):
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    model, base = _r3_model(original, 'transpose')
    optimizer = _r3_optimizer(model, Adam)
    unchanged = _r3_capture(model, optimizer, base)
    state = independent(optimizer.state_dict())
    state['m'].pop()
    path = tmp_path / 'prepare.ckpt'
    ortho.save_checkpoint(path, model=model, optimizer=StateSource(state))
    with pytest.raises(ValueError, match='parameter count mismatch'):
        ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location=target)
    unchanged()


def test_r3_private_representation_reference_keeps_layout_and_gradient_owner():
    model, base = _r3_model(layout='offset')
    owner = model.a
    owner.set_grad(ortho.ones(list(owner.shape)))
    with pytest.raises(RuntimeError, match='model value reference requires no_grad'):
        owner._model_value_reference()
    with ortho.no_grad():
        reference = owner._model_value_reference()
        replacement = owner.to('cpu')
        replacement.requires_grad_(True)
        owner._rebind_model_value_(replacement)
        assert reference is not owner
        assert reference.strides == (4, 1) and reference.storage_offset == 5
        assert reference.grad is not None
        owner._rebind_model_value_(reference)
    assert owner.grad.tolist() == [1.] * 4
    base.mul_(2)
    assert owner.tolist() == reference.tolist()
    assert not hasattr(ortho.Tensor, 'detach')


@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
@pytest.mark.parametrize('dtype', ['float16', 'bfloat16'])
@pytest.mark.parametrize('fail', [False, True], ids=['success', 'late-rollback'])
def test_r3_strided_amp_masters_follow_true_move_and_rollback(
    monkeypatch, cuda_device, tmp_path, direction, dtype, fail,
):
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')

    def pair(device, shift):
        model = Module()
        model.weight = Parameter(ortho.full([2, 2], .5 + shift).cast(dtype).to(device))
        optimizer = MixedPrecisionOptimizer(model, Adam, lr=.03125)
        model.weight.set_grad(ortho.full([2, 2], .25).cast(dtype).to(device))
        optimizer.step()
        return model, optimizer
    source, source_optimizer = pair('cpu', .25)
    model, optimizer = pair(original, 0.)
    model._apply_tensor_transform(lambda t: t.transpose(0, 1))
    owner = model.weight
    owner.set_grad(ortho.ones([2, 2]).cast(dtype).to(original))
    before = _r3_tensor_state(owner), _r3_tensor_state(owner.grad)
    masters, inner = optimizer._states, optimizer._inner
    master_values = [_r3_tensor_state(s.master) for s in masters]
    inner_values = values(inner.state_dict())
    # Public AMP state inspection reconciles the current representation and
    # requires contiguous low-precision owners. Future-device load prepares a
    # contiguous representation before publishing either participant.
    with pytest.raises(RuntimeError, match='AMP synchronization requires contiguous'):
        optimizer.state_dict()
    path = tmp_path / 'amp-strided.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=source_optimizer)
    if fail:
        copy = ortho._amp_copy_master_to_model_
        failed = False

        def fail_once(parameter, master):
            nonlocal failed
            copy(parameter, master)
            if not failed:
                failed = True
                raise ValueError('R3 injected AMP failure')
        monkeypatch.setattr(ortho, '_amp_copy_master_to_model_', fail_once)
        with pytest.raises(ValueError, match='R3 injected AMP failure'):
            ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location=target)
        assert (_r3_tensor_state(owner), _r3_tensor_state(owner.grad)) == before
        assert optimizer._states is masters and optimizer._inner is inner
        assert [_r3_tensor_state(s.master) for s in masters] == master_values
        assert values(inner.state_dict()) == inner_values
        assert model.weight is owner and not owner.is_contiguous
    else:
        ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location=target)
        assert model.weight is owner and owner.is_contiguous and owner.grad is None
        assert owner.dtype == dtype and owner.device == ortho.Device(target)
        assert owner.tolist() == source.weight.tolist()
        state = optimizer.state_dict()
        assert state['masters'][0].device == owner.device
        assert state['masters'][0].dtype == 'float32'
        assert state['inner']['t'] == source_optimizer.state_dict()['inner']['t']
        assert any(v != 0 for v in state['inner']['m'][0].tolist())
        owner.set_grad(ortho.ones([2, 2]).cast(dtype).to(target))
        optimizer.step()


@pytest.mark.parametrize('direction', ['cpu-to-cuda', 'cuda-to-cpu'])
def test_r3_foreign_numpy_strided_checkpoint_resume_integrity(cuda_device, tmp_path, direction):
    original, target = ('cpu', cuda_device) if direction == 'cpu-to-cuda' else (cuda_device, 'cpu')
    native = (np.arange(12, dtype=np.float32).reshape(3, 4) + 1) / 8
    foreign = native.astype(native.dtype.newbyteorder('S'))
    source, _ = _r3_model(original, 'offset')
    imported = ortho.from_numpy(foreign).to(original).slice(0, 1, 3).slice(1, 1, 3)
    source.a = Parameter(imported)
    source.b = Parameter(imported)
    source.owner_alias = imported
    expected = native[1:3, 1:3].copy()
    foreign.fill(0)
    assert imported.tolist() == expected.ravel().tolist()
    saved_optimizer = _r3_optimizer(source, Adam)
    path = tmp_path / 'endian-strided.ckpt'
    ortho.save_checkpoint(path, model=source, optimizer=saved_optimizer)
    model, _ = _r3_model(original, 'offset', shift=16.)
    optimizer = _r3_optimizer(model, Adam)
    owner = model.a
    ortho.load_checkpoint(path, model=model, optimizer=optimizer, map_location=target)
    assert model.a is model.b is model.owner_alias is owner
    assert owner.device == ortho.Device(target) and owner.dtype == 'float32'
    assert owner.tolist() == expected.ravel().tolist()
    optimizer.zero_grad()
    (owner * owner).sum().backward()
    gradient = 2 * expected.astype(np.float64)
    np.testing.assert_allclose(owner.grad.to('cpu').tolist(), gradient.ravel(), atol=1e-6)
    moment = .9 * .125 + .1 * gradient
    variance = .999 * .25 + .001 * gradient ** 2
    reference = expected - .03125 * (moment / (1 - .9 ** 3)) / (
        np.sqrt(variance / (1 - .999 ** 3)) + 1e-8)
    optimizer.step()
    np.testing.assert_allclose(owner.to('cpu').tolist(), reference.ravel(), atol=2e-6, rtol=2e-5)
    logits = ortho.from_numpy(np.full((2,), 1e8, np.float32)).to(target)
    np.testing.assert_allclose(ortho.softmax(logits, 0).to('cpu').tolist(), [.5, .5], atol=2e-6)
