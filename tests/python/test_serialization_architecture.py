"""Ownership fences for the private v1 serialization and state transaction layer."""
from tests.support.source import REPO_ROOT
import ast
from pathlib import Path

import orthon as ortho
import orthon.checkpoint as checkpoint

ROOT = REPO_ROOT
PACKAGE = ROOT / 'python/orthon'


def sources():
    return {path: ast.parse(path.read_text()) for path in (PACKAGE / '_serialization').glob('*.py')}


def test_public_checkpoint_facade_preserves_identity_and_has_no_codec():
    tree = ast.parse((PACKAGE / 'checkpoint.py').read_text())
    for name in ('save_checkpoint', 'load_checkpoint', 'save_state_dict', 'load_state_dict'):
        assert getattr(ortho, name) is getattr(checkpoint, name)
    assert (checkpoint.FORMAT_ID, checkpoint.FORMAT_VERSION) == ('ml-runtime-checkpoint', 1)
    assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
    forbidden = {'zipfile', 'json', 'hashlib', 'tempfile'}
    assert not any(isinstance(node, ast.Import) and any(alias.name in forbidden for alias in node.names)
                   for node in ast.walk(tree))
    assert not (PACKAGE / 'serialization').exists()


def test_serialization_uses_owner_protocols_without_state_representation_access():
    forbidden = {'_parameters', '_buffers', '_modules', '_non_persistent_buffers',
                 '_bindings', '_states', '_inner', '_AdamState',
                 '_copy_data_', '_rebind_model_value_'}
    for path, tree in sources().items():
        attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert not attributes & forbidden, path
        assert not any(isinstance(node, ast.ImportFrom) and node.module and
                       any(part in node.module for part in ('runtime', 'distributed', 'backends'))
                       for node in ast.walk(tree)), path


def test_byte_binding_consumption_is_confined_to_tensor_snapshot_codec():
    consumers = {}
    for path, tree in sources().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ('_tensor_to_bytes', '_tensor_from_bytes'):
                consumers.setdefault(node.attr, set()).add(path.name)
    assert consumers == {'_tensor_to_bytes': {'tensor.py'}, '_tensor_from_bytes': {'tensor.py'}}
    tensor_tree = sources()[PACKAGE / '_serialization/tensor.py']
    assert any(isinstance(node, ast.ClassDef) and node.name == 'TensorSnapshotCodec'
               for node in ast.walk(tensor_tree))
    assert callable(ortho._tensor_to_bytes) and callable(ortho._tensor_from_bytes)


def test_archive_io_has_one_owner_and_never_executes_or_extracts():
    for path, tree in sources().items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(alias.name in ('pickle', 'marshal', 'importlib') for alias in node.names)
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                    node.func.id if isinstance(node.func, ast.Name) else '')
                assert name not in {'eval', 'exec', 'extract', 'extractall'}, path
                if name == 'loads':
                    assert path.name == 'archive.py', path
            if isinstance(node, ast.Attribute) and node.attr in {'ZipFile', 'mkstemp', 'fsync', 'replace'}:
                assert path.name == 'archive.py', path
    tree = sources()[PACKAGE / '_serialization/archive.py']
    assert any(isinstance(node, ast.ClassDef) and node.name == 'ArchiveCodec' for node in ast.walk(tree))


def test_service_decodes_before_preparing_and_committing_recipients():
    tree = sources()[PACKAGE / '_serialization/service.py']
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == 'load_checkpoint')
    calls = [(node.lineno, node.func.attr) for node in ast.walk(method)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    line = lambda name: min(n for n, attr in calls if attr == name)
    assert line('decode') < line('prepare') < line('commit')
    assert not any(attr == 'load_state_dict' for _, attr in calls)
    assert not any(isinstance(node, ast.Attribute) and node.attr in {'_tensor_to_bytes', '_tensor_from_bytes'}
                   for node in ast.walk(tree))


def test_all_builtin_local_loads_use_the_same_private_preparation_protocol():
    for filename in ('nn/module.py', 'optim/sgd.py', 'optim/adam.py', 'amp/optimizer.py'):
        tree = ast.parse((PACKAGE / filename).read_text())
        local = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                     and node.name == 'load_state_dict')
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == '_prepare_load_state_dict' for node in ast.walk(local)), filename
        assert any(isinstance(node, ast.FunctionDef) and node.name == '_prepare_load_state_dict'
                   for node in ast.walk(tree)), filename
