"""Permanent fences for the five final test layers and their support owners."""
import ast
from itertools import combinations
import sys

import orthon
import pytest

from tests.support import examples
from tests.support.capabilities import build_profile, orthon_cuda_available, verify_profile
from tests.support.layers import ARCHITECTURE_MODULES, PRIMARY_LAYERS, primary_layer_for_path
from tests.support.source import REPO_ROOT, parse_python, read_text


def collected_sources():
    return sorted((REPO_ROOT / 'tests/python').glob('test_*.py')) + sorted(
        (REPO_ROOT / 'tests/spec').rglob('test_*.py'))


def test_every_collected_source_and_item_has_one_reviewed_primary_layer(request):
    paths = collected_sources()
    assert ARCHITECTURE_MODULES <= {p.name for p in paths}
    for path in paths:
        expected = ('spec' if 'spec' in path.relative_to(REPO_ROOT).parts else
                    'architecture' if path.name in ARCHITECTURE_MODULES else 'regression')
        assert primary_layer_for_path(path) == expected
    groups = {layer: set() for layer in PRIMARY_LAYERS}
    for item in request.session.items:
        primary = [m.name for m in item.iter_markers() if m.name in PRIMARY_LAYERS]
        assert primary == [primary_layer_for_path(item.path)], item.nodeid
        groups[primary[0]].add(item.nodeid)
    assert set.union(*groups.values()) == {item.nodeid for item in request.session.items}
    for a, b in combinations(groups.values(), 2):
        assert not a & b


@pytest.mark.parametrize('path', ['tests/smoke/package_smoke.py', 'tests/support/source.py',
                                 'tests/other/test_unknown.py'])
def test_unknown_or_non_collected_locations_reject_primary_classification(path):
    with pytest.raises(ValueError, match='unclassified pytest source'):
        primary_layer_for_path(path)


def test_collected_modules_do_not_import_other_test_modules():
    for path in collected_sources():
        for node in ast.walk(parse_python(path)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                if isinstance(node, ast.ImportFrom):
                    names.append(node.module or '')
                assert not any(part.startswith('test_') for name in names
                               for part in name.split('.')), (path, names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == 'import_module' and node.args and isinstance(node.args[0], ast.Constant):
                    assert not any(part.startswith('test_') for part in str(node.args[0].value).split('.')), path


def test_conftests_never_guess_build_paths_and_example_paths_are_scoped():
    assert not (REPO_ROOT / 'tests/python/conftest.py').exists()
    for path in list((REPO_ROOT / 'tests').rglob('conftest.py')) + collected_sources():
        for node in ast.walk(parse_python(path)):
            # No conftest/test module may mutate sys.path, regardless of spelling
            # of the guessed directory. The one scoped owner is support/examples.
            def is_sys_path(value):
                return (isinstance(value, ast.Attribute) and value.attr == 'path' and
                        isinstance(value.value, ast.Name) and value.value.id == 'sys')
            if is_sys_path(node):
                assert not isinstance(node.ctx, (ast.Store, ast.Del)), path
            if isinstance(node, ast.Subscript) and is_sys_path(node.value):
                assert not isinstance(node.ctx, (ast.Store, ast.Del)), path
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and is_sys_path(node.func.value):
                assert node.func.attr not in {'insert', 'append', 'extend', 'remove', 'pop', 'clear'}, path
    support = parse_python('tests/support/examples.py')
    assert any(isinstance(n, ast.Try) and n.finalbody for n in ast.walk(support))


def test_shared_support_is_not_collected_or_a_runtime_dependency():
    support = list((REPO_ROOT / 'tests/support').glob('*.py'))
    assert support and not any(p.name.startswith('test_') for p in support)
    for path in support:
        assert not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                       n.name.startswith('test_') for n in ast.walk(parse_python(path)))
    for path in (REPO_ROOT / 'python/orthon').rglob('*.py'):
        for node in ast.walk(parse_python(path)):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith('tests'), path
            if isinstance(node, ast.Import):
                assert not any(a.name.startswith('tests') for a in node.names), path
    assert 'wheel.packages = ["python/orthon"]' in read_text('pyproject.toml')


def test_cuda_probes_and_architecture_source_helpers_have_one_owner():
    for path in collected_sources():
        functions = [n for n in ast.walk(parse_python(path)) if isinstance(n, ast.FunctionDef)]
        assert not any(n.name == '_cuda_available' for n in functions), path
        if path.name in ARCHITECTURE_MODULES:
            assert not any(n.name in ('source', '_source', '_read', 'function', '_strip_c_comments')
                           for n in functions), path
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
                           n.func.id == 'Path' and n.args and isinstance(n.args[0], ast.Name) and
                           n.args[0].id == '__file__' for n in ast.walk(parse_python(path))), path
    functions = [n for n in ast.walk(parse_python('tests/support/capabilities.py'))
                 if isinstance(n, ast.FunctionDef) and n.name == 'orthon_cuda_available']
    assert len(functions) == 1


def test_strict_primary_markers_and_import_mode_are_configured(request):
    assert request.config.getini('testpaths') == ['tests/python', 'tests/spec']
    assert request.config.getoption('strict_markers')
    assert request.config.getoption('importmode') == 'importlib'
    registered = {entry.split(':', 1)[0] for entry in request.config.getini('markers')}
    assert set(PRIMARY_LAYERS) <= registered


def test_cpu_ci_executes_all_layers_with_profile_and_preserves_matrix():
    workflow = read_text('.github/workflows/ci.yml')
    python_job = workflow.split('\n  python:', 1)[1].split('\n  package:', 1)[0]
    assert 'python-version: ["3.10", "3.13"]' in python_job
    for layer in PRIMARY_LAYERS:
        step = next(s for s in python_job.split('      - name:')
                    if 'python -m pytest -q -m ' + layer in s)
        assert 'ORTHON_TEST_PROFILE: cpu' in step
        assert 'continue-on-error' not in step
    assert 'os: [ubuntu-latest, windows-latest]' in workflow
    assert '\n  package:' in workflow
    assert 'cpp-cpu-release' in workflow
    assert 'cpp-cuda-release' not in workflow
    assert 'tests/smoke/package_smoke.py' in workflow


@pytest.mark.parametrize('cuda,nccl,expected', [(False, False, 'cpu'), (True, False, 'cuda'),
                                             (True, True, 'cuda-nccl')])
def test_profile_mapping_and_optional_requested_profile(cuda, nccl, expected):
    assert build_profile(cuda, nccl) == expected
    assert verify_profile(expected, cuda, nccl) == expected
    assert verify_profile(None, cuda, nccl) == expected


def test_invalid_or_mismatched_profile_rejects_actionably():
    for requested in ('gpu', '', 'CPU'):
        with pytest.raises(ValueError, match='ORTHON_TEST_PROFILE must be'):
            verify_profile(requested, False, False)
    with pytest.raises(ValueError, match='prepare the matching installed package'):
        verify_profile('cuda', False, False)
    with pytest.raises(ValueError, match='NCCL support requires CUDA'):
        build_profile(False, True)


def test_cuda_helper_probes_the_requested_orthon_device_and_runtime_errors(monkeypatch):
    visited = []

    class Probe:
        def to(self, device):
            visited.append(device)

    monkeypatch.setattr(orthon, 'zeros', lambda shape: Probe())
    assert orthon_cuda_available('cuda:7')
    assert visited == ['cuda:7']

    def unavailable(shape):
        raise RuntimeError('device unavailable')

    monkeypatch.setattr(orthon, 'zeros', unavailable)
    assert not orthon_cuda_available()


@pytest.mark.parametrize('raises', [False, True])
def test_example_loader_restores_the_complete_path_on_success_and_failure(tmp_path, monkeypatch, raises):
    directory = tmp_path / 'examples'
    directory.mkdir()
    text = 'import sys\nvisible = sys.path[0]\n'
    if raises:
        text += 'raise RuntimeError("example failed")\n'
    (directory / 'probe.py').write_text(text)
    monkeypatch.setattr(examples, 'REPO_ROOT', tmp_path)
    before = sys.path[:]
    if raises:
        with pytest.raises(RuntimeError, match='example failed'):
            examples.load_example('probe.py')
    else:
        assert examples.load_example('probe.py').visible == str(directory)
    assert sys.path == before


def test_loaded_provenance_records_package_extension_flags_and_python(request):
    header = '\n'.join(request.config._orthon_test_provenance)
    assert str(orthon.__file__) in header and str(orthon._C.__file__) in header
    assert 'cuda_enabled=' in header and 'nccl_enabled=' in header
    assert 'expected=' in header and 'Python:' in header
