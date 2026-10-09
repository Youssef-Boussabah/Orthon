"""AMP ownership fences expressed through AST dependencies, not formatting."""
from tests.support.source import REPO_ROOT
import ast
from pathlib import Path

REPO = REPO_ROOT
AMP = REPO / 'python/orthon/amp'


def tree(name):
    return ast.parse((AMP / name).read_text())


def attributes(node):
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def test_amp_holds_tensor_owners_without_module_registries_or_retired_entries():
    code = tree('optimizer.py')
    assert not attributes(code) & {'_entries', '_parameters', '_modules', '_bindings',
                                   '_replace_parameter', '_copy_data_', '_impl_id'}
    methods = {n.name: n for n in ast.walk(code) if isinstance(n, ast.FunctionDef)}
    assert '_iter_params_with_module' not in methods
    capture = [n for n in ast.walk(methods['__init__']) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'parameters']
    assert len(capture) == 1


def test_scaler_only_calls_named_wrapper_protocols():
    code = tree('grad_scaler.py')
    assert not attributes(code) & {'_entries', '_states', '_inner', '_bindings', 'masters',
                                   '_parameters', '_modules', '_all_reduce_master_grads'}
    calls = {n.func.attr for n in ast.walk(code) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'mp_optimizer'}
    assert calls == {'step', '_unscale_and_check', '_step_prepared', '_amp_collective_device'}
    assert not any(isinstance(n, (ast.For, ast.AsyncFor)) for n in ast.walk(code))


def test_one_canonical_amp_collection_and_no_upward_or_raw_backend_dependencies():
    code = tree('optimizer.py')
    for node in ast.walk(code):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
                    assert target.attr not in {'_masters', '_parameters', '_names', '_dtypes', '_entries'}
    for path in AMP.glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [n.name for n in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                assert all(not name or not any(part in name for part in ('checkpoint', 'backends', '._C')) for name in names)
            if isinstance(node, ast.Attribute):
                assert not node.attr.startswith(('cpu_', 'cuda_'))


def test_amp_copy_uses_dispatch_and_semantic_version_boundary():
    semantic = (REPO / 'src/ops/tensor_access.cpp').read_text()
    assert 'resolve<CopyKernel>' in semantic and 'OperatorId::AmpCopy' in semantic
    assert 'bump_version()' in semantic
    for path in (REPO / 'src/backends').rglob('*'):
        if path.suffix in {'.cpp', '.cu'}:
            assert 'bump_version' not in path.read_text()
