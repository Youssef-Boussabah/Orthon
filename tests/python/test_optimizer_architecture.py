"""Optimizer diagnostics for one parameter/state owner and narrow dependencies."""
from tests.support.source import REPO_ROOT
import ast
from pathlib import Path

import orthon as ortho
from orthon.optim import Adam, SGD

REPO = REPO_ROOT
OPTIM = REPO / 'python/orthon/optim'


def test_optimizers_own_one_ordered_binding_sequence_with_per_owner_state():
    p = ortho.ones([1]).requires_grad_(True)
    for kind in (SGD, Adam):
        opt = kind([p], lr=.1)
        assert not {'_params', '_m', '_v'} & vars(opt).keys()
        sequences = [value for value in vars(opt).values() if isinstance(value, (list, tuple))]
        assert len(sequences) == 1
        binding = sequences[0][0]
        tensor_fields = [value for value in vars(binding).values() if isinstance(value, ortho.Tensor)]
        assert tensor_fields == [p]
        assert binding.state is None
        p.sum().backward()
        opt.step()
        if kind is Adam:
            pair = [value for value in vars(binding.state).values() if isinstance(value, ortho.Tensor)]
            assert len(pair) == 2 and pair[0] is not pair[1]
            exported = opt.state_dict()
            assert exported['m'][0] in pair and exported['v'][0] in pair
        else:
            assert binding.state is None
        opt.zero_grad()


def test_optimizers_have_no_parallel_parameter_or_moment_containers():
    forbidden = {'_params', '_m', '_v', '_restore_state_tensors'}
    for path in OPTIM.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, (path, node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in forbidden, (path, node.value)
            elif isinstance(node, ast.FunctionDef):
                assert node.name not in forbidden


def test_optimizer_dependencies_stop_at_semantic_tensor_api():
    forbidden = {'_parameters', '_modules', '_buffers', '_inner', '_entries', '_C',
                 '_impl_id', '_storage_version', 'impl', 'data', 'Buffer', 'TensorImpl'}
    for path in OPTIM.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [x.name for x in node.names]
                assert not any(part in name.split('.') for name in names for part in
                               ('amp', 'checkpoint', 'nn', 'distributed', 'backends', 'dispatch', '_C'))
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, (path, node.attr)
            if isinstance(node, ast.Name):
                assert node.id not in forbidden


def test_parameter_capture_has_one_identity_validation_route():
    # Both public constructors and the AMP-only bridge must share capture.
    from orthon.optim import adam, sgd
    assert adam._capture_bindings is sgd._capture_bindings
    for module in (adam, sgd):
        tree = ast.parse(Path(module.__file__).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('SGD', 'Adam'))
        for name in ('__init__', '_rebind_parameters'):
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
            assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == '_capture_bindings' for n in ast.walk(method))


def test_amp_rebinding_bridge_has_only_the_named_production_consumers():
    allowed = {'python/orthon/optim/adam.py', 'python/orthon/optim/sgd.py', 'python/orthon/amp/optimizer.py'}
    found = set()
    for path in (REPO / 'python/orthon').rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            name = node.attr if isinstance(node, ast.Attribute) else (
                node.name if isinstance(node, ast.FunctionDef) else (
                    node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None))
            if name == '_rebind_parameters':
                found.add(path.relative_to(REPO).as_posix())
            if isinstance(node, ast.Attribute) and node.attr == '_params':
                raise AssertionError((path, 'retired optimizer parameter list'))
    assert found == allowed


def test_amp_bridge_rejects_invalid_replacement_without_losing_bindings():
    p = ortho.ones([1]).requires_grad_(True)
    q = ortho.ones([1]).requires_grad_(True)
    for kind in (SGD, Adam):
        opt = kind([p], lr=.1)
        p.sum().backward()
        opt.step()
        old = opt.state_dict()
        for supplied, error in (([q, q], ValueError), ([q, object()], TypeError)):
            try:
                opt._rebind_parameters(supplied)
            except error:
                pass
            else:
                raise AssertionError('invalid bridge input accepted')
            if kind is Adam:
                assert opt.state_dict()['m'][0] is old['m'][0]
                assert opt.state_dict()['v'][0] is old['v'][0]
            opt.zero_grad()
            assert p.grad is None
        opt._rebind_parameters([q])
        state = opt.state_dict()
        assert state['lr'] == old['lr']
        if kind is Adam:
            assert state['t'] == old['t']
            assert state['m'][0].tolist() == old['m'][0].tolist()
            assert state['v'][0].tolist() == old['v'][0].tolist()
            assert state['m'][0] is not old['m'][0]
        q.sum().backward()
        before = q.tolist()
        opt.step()
        assert q.tolist() != before and p.grad is None
        opt.zero_grad()
