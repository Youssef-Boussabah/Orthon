"""Module ownership boundaries, with behavioral coverage in spec/nn."""
from tests.support.source import REPO_ROOT
import ast
from pathlib import Path
import re

REPO = REPO_ROOT
MODULE_PATH = REPO / 'python/orthon/nn/module.py'


def module_methods():
    tree = ast.parse(MODULE_PATH.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Module')
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def calls(node):
    return {n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_nn_has_no_raw_core_backend_allocator_or_dispatcher_dependency():
    forbidden = re.compile(r'\b(?:Storage|TensorImpl|Buffer|allocator_for|get_allocator|cudaMemcpy|cudaMalloc|cudaFree)\b')
    for path in (REPO / 'python/orthon/nn').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                name = node.module if isinstance(node, ast.ImportFrom) else ' '.join(n.name for n in node.names)
                assert not re.search(r'(?:backends|dispatch|memory_pool|_C|core)', name or '')
            if isinstance(node, ast.Attribute):
                assert node.attr not in ('_impl_id', 'impl', '_storage_version', 'data')
            if isinstance(node, ast.Name):
                assert not forbidden.fullmatch(node.id)


def test_registration_dictionary_mutations_have_one_protocol():
    methods = module_methods()
    registries = {'_parameters', '_buffers', '_modules', '_non_persistent_buffers'}
    for name, node in methods.items():
        if name in ('__init__', '_set_member'):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript) and isinstance(child.ctx, (ast.Store, ast.Del)):
                assert not isinstance(child.value, ast.Attribute) or child.value.attr not in registries
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr in ('pop', 'clear', 'update', 'add', 'discard'):
                    assert not isinstance(child.func.value, ast.Attribute) or child.func.value.attr not in registries
    assert '_prepare_load_state_dict' in calls(methods['load_state_dict'])
    assert '_replace_parameter' not in methods
    for name in ('__setattr__', 'register_buffer', '_register_module', '_prepare_load_state_dict'):
        assert '_set_member' in calls(methods[name]) or '_register_module' in calls(methods[name])
    seq = ast.parse((REPO / 'python/orthon/nn/sequential.py').read_text())
    assert '_register_module' in calls(seq)
    assert not any(isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) for n in ast.walk(seq))


def test_discovery_and_state_methods_share_explicit_state_traversal():
    methods = module_methods()
    assert '_prepare_load_state_dict' in calls(methods['load_state_dict'])
    for name in ('named_parameters', 'named_buffers', 'state_dict', '_prepare_load_state_dict'):
        assert '_state_entries' in calls(methods[name])
    assert '_walk_modules' in calls(methods['_state_entries'])
    assert 'named_parameters' in calls(methods['parameters'])
    assert 'named_buffers' in calls(methods['buffers'])


def test_transforms_share_the_no_grad_preparation_and_owner_protocol():
    methods = module_methods()
    for name in ('to', 'cast'):
        assert '_apply_tensor_transform' in calls(methods[name])
        assert not any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(methods[name]))
    for name in ('half', 'bfloat16'):
        assert 'cast' in calls(methods[name])
    transform = methods['_apply_tensor_transform']
    assert 'no_grad' in calls(transform)
    assert '_walk_modules' in calls(transform)
    assert '_rebind_model_value_' in calls(transform)
    assert '_set_member' in calls(transform)


def test_parameter_wraps_one_tensor_without_a_second_model_payload():
    tree = ast.parse(MODULE_PATH.read_text())
    parameter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Parameter')
    stores = [n.attr for n in ast.walk(parameter) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)]
    assert stores == ['_data']
    assert calls(parameter) == {'requires_grad_'}


def test_owner_rebind_is_private_validated_and_does_not_execute_a_kernel():
    text = (REPO / 'bindings/tensor_bindings.cpp').read_text()
    body = text.split('.def("_rebind_model_value_"', 1)[1].split('}, py::arg("value"))', 1)[0]
    for invariant in ('is_grad_enabled', 'shape()', 'is_leaf()', 'has_grad_fn()', 'requires_grad()', 'self = value;'):
        assert invariant in body
    assert not re.search(r'cuda\w*\(|\.data\(|new TensorImpl|make_shared|memcpy', body)
    assert not any(isinstance(n, ast.Attribute) and n.attr == '_rebind_model_value_' for p in (REPO / 'python/orthon').rglob('*.py') if p != MODULE_PATH for n in ast.walk(ast.parse(p.read_text())))
