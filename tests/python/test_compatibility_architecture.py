"""Compatibility APIs retain one current owner and explicit consumer boundaries."""
from tests.support.source import read_cpp_source as source
from tests.support.source import REPO_ROOT, iter_source_files
import ast
import re
from functools import lru_cache
import orthon
from tests.support.compatibility import (
    NATIVE_CONSUMERS, PYTHON_CONSUMERS, PRIVATE_BINDINGS, SIGNATURE_CONSUMERS,
)
from tests.support.source import function

REPO = REPO_ROOT
NATIVE_ROOTS = ("include/orthon", "src", "bindings", "tests/cpp")
NATIVE_SUFFIXES = {".h", ".hpp", ".cuh", ".cpp", ".cu"}
COMPATIBILITY_TEST = "tests/cpp/test_compatibility_surface.cpp"


def native_files():
    return list(iter_source_files(*NATIVE_ROOTS, suffixes=NATIVE_SUFFIXES))


def native_argument_counts(code, name):
    """Distinguish retained overload signatures from canonical same-name calls."""
    pattern = (r'\b' + re.escape(name) + r'(?:\s+\w+)?\s*\(|'
               r'\b(?:make_shared|make_unique)\s*<\s*' + re.escape(name) + r'\s*>\s*\(')
    for match in re.finditer(pattern, code):
        start = match.end()
        depth, commas, position = 1, 0, start
        while position < len(code) and depth:
            char = code[position]
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == ',' and depth == 1:
                commas += 1
            position += 1
        if depth == 0:
            yield 0 if not code[start:position - 1].strip() else commas + 1


@lru_cache(maxsize=None)
def python_references(text):
    """Non-definition consumers, including imports and reflective names."""
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
    return names & PYTHON_CONSUMERS.keys()


def test_canonical_includes_do_not_depend_on_compatibility_headers():
    for path in native_files():
        rel = path.relative_to(REPO).as_posix()
        if rel == COMPATIBILITY_TEST or rel.startswith(
            ("include/orthon/ops/cpu/", "include/orthon/ops/cuda/")
        ):
            continue
        assert not re.search(r'#\s*include\s*[<"]orthon/(?:ops/(?:cpu|cuda)/|dispatch/adapters\.h)',
                             source(path)), rel


def test_native_compatibility_has_only_named_consumers():
    for path in native_files():
        rel, text = path.relative_to(REPO).as_posix(), source(path)
        # Error strings are not dependencies.
        code = re.sub(r'"(?:\\.|[^"\\])*"', '', text)
        # Identity's global sentinel variables prove namespace isolation;
        # declaring a shadow variable is not a compatibility API invocation.
        code = re.sub(r'(?m)^\s*int\s+\w+\s*=\s*0\s*;\s*$', '', code)
        for symbol, allowed in NATIVE_CONSUMERS.items():
            if re.search(r"\b" + symbol + r"\b", code):
                assert rel in allowed, (symbol, rel, sorted(allowed))
        for signature, allowed in SIGNATURE_CONSUMERS.items():
            name, arity = signature.split(':')
            if int(arity) in native_argument_counts(code, name):
                assert rel in allowed, (signature, rel, allowed)
        if re.search(r'#\s*include\s*[<"]orthon/core/storage\.h', text):
            assert rel in {"src/core/storage.cpp", COMPATIBILITY_TEST}, rel


def test_retired_dispatch_aliases_have_no_source_or_build_route():
    assert not (REPO / "include/orthon/dispatch/adapters.h").exists()
    for path in native_files() + [REPO / "CMakeLists.txt"]:
        text = source(path)
        assert "orthon/dispatch/adapters.h" not in text, path
        assert not re.search(r"\badapters::|namespace\s+adapters\b", text), path
    for root in ("src/ops/cpu", "src/ops/cuda"):
        assert not list((REPO / root).glob("*"))


def test_python_bridges_have_only_named_production_consumers():
    for path in (REPO / "python/orthon").rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        for name in python_references(path.read_text()):
            assert rel in PYTHON_CONSUMERS[name], (name, rel, sorted(PYTHON_CONSUMERS[name]))


def test_private_binding_inventory_requires_explicit_review():
    names = set()
    for path in (REPO / "bindings").glob("*.cpp"):
        names.update(re.findall(
            r'\.(?:def|def_property_readonly|def_property)\(\s*"(_[^"]+)"', source(path)
        ))
    assert {name for name in names if not name.startswith("__")} == PRIVATE_BINDINGS


def test_consumer_fence_catches_reflection_and_imports():
    references = python_references("""
import orthon as o
from orthon import _tensor_to_bytes as bytes_for_tensor
p._copy_data_(q)
getattr(p, '_impl_id')
object.__getattribute__(m, '_parameters')
""")
    assert references == {"_tensor_to_bytes", "_copy_data_", "_impl_id", "_parameters"}
    assert all("python/orthon/new_consumer.py" not in PYTHON_CONSUMERS[n] for n in references)


def test_retained_lookup_adapters_terminate_directly_in_runtime():
    allocator = source(REPO / "src/memory/allocator.cpp")
    assert "default_runtime().allocator(device)" in function(allocator, "allocator_for")
    assert "default_runtime().allocator(device).get()" in function(allocator, "get_allocator")
    assert "default_runtime().scratch(device)" in function(
        source(REPO / "src/memory/scratch.cpp"), "scratch_arena_for"
    )
    assert "default_runtime().memory_pool()" in function(
        source(REPO / "src/memory/pool.cpp"), "default_memory_pool"
    )
    for name in ("buffer", "storage"):
        text = source(REPO / f"src/core/{name}.cpp")
        assert "default_runtime().allocator(device)->allocate(" in text
    pool = source(REPO / "src/core/memory_pool.cpp")
    for name in ("memory_pool_set_enabled", "memory_pool_stats", "memory_pool_clear_cache"):
        assert "default_runtime().memory_pool()" in function(pool, name)
    for name in ("memory_pool_scratch_allocate", "memory_pool_reset_scratch", "memory_pool_scratch_stats"):
        assert "default_runtime().scratch(device)" in function(pool, name)
    wrapper = function(source(REPO / "src/ops/shape_ops.cpp"), "kernel_operand")
    assert re.fullmatch(r"\s*return\s+execution_operand\(t\);\s*", wrapper)


def test_native_forwarding_headers_have_one_canonical_destination():
    for kind, count in (("cpu", 17), ("cuda", 12)):
        headers = list((REPO / "include/orthon/ops" / kind).glob("*.h"))
        assert len(headers) == count
        for path in headers:
            target = f"orthon/ops/{path.name}" if kind == "cpu" else f"orthon/backends/cuda/{path.name}"
            # Whitespace does not affect the ownership contract.
            directives = re.findall(r'#\s*(pragma\s+once|include\s*[<"][^>"]+[>"])', source(path))
            assert len(directives) == 2 and directives[0] == "pragma once", path
            assert re.fullmatch(r"include\s*<" + re.escape(target) + r">", directives[1]), path
            assert not re.sub(r"#.*", "", source(path)).strip(), path


def test_public_backward_has_only_the_engine_route():
    tensor = source(REPO / "src/core/tensor.cpp")
    body = function(tensor, "Tensor::backward")
    assert re.fullmatch(r"\s*default_autograd_engine\(\)\.backward\(\*this, retain_graph\);\s*", body)
    assert not re.search(r"build_topo|privatise_leaf_grad|pending_grads|cuda_\w+\(", tensor)


def test_python_bridge_consumers_match_current_owners():
    actual = {name: set() for name in PYTHON_CONSUMERS}
    for path in (REPO / 'python/orthon').rglob('*.py'):
        rel = path.relative_to(REPO).as_posix()
        for name in python_references(path.read_text()):
            actual[name].add(rel)
    assert actual == PYTHON_CONSUMERS


def test_current_tools_use_the_private_diagnostic_facade():
    diagnostic_names = {n for n in PRIVATE_BINDINGS
                        if n.startswith(('_matmul_', '_memory_pool_', '_conv2d_'))}
    for root in ('benchmarks', 'examples'):
        for path in (REPO / root).rglob('*.py'):
            tree = ast.parse(path.read_text())
            attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                       for a in n.names}
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                       and isinstance(n.value, str)}
            assert not diagnostic_names & (attributes | imports | strings), path


def test_deleted_python_surfaces_have_no_source_binding_or_binary_route():
    from orthon.nn import Module
    assert not hasattr(orthon, '_scalar_like')
    assert not hasattr(orthon._C, '_scalar_like')
    assert not hasattr(Module, '_replace_parameter')
    assert not hasattr(Module(), '_replace_parameter')
    for path in (REPO / 'python/orthon').rglob('*.py'):
        assert not re.search(r'\b_scalar_like\b|\b_replace_parameter\b', path.read_text()), path
    for path in (REPO / 'bindings').glob('*.cpp'):
        assert '_scalar_like' not in path.read_text(), path
    for path in (REPO / 'tests').rglob('*.py'):
        tree = ast.parse(path.read_text())
        assert not any(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == '_replace_parameter'
            or isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in ('_scalar_like', '_replace_parameter')
            for n in ast.walk(tree)
        ), path


def test_retained_compatibility_does_not_restore_parallel_owners():
    storage = source(REPO / 'include/orthon/core/storage.h')
    assert re.findall(r'Allocation\s+\w+_;', storage) == ['Allocation allocation_;']
    assert not re.search(r'DType|AutogradMeta|registry|generation', storage)
    for name in ('adam.py', 'sgd.py'):
        tree = ast.parse((REPO / 'python/orthon/optim' / name).read_text())
        assert not {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        } & {'_params', '_m', '_v'}
    tree = ast.parse((REPO / 'python/orthon/amp/optimizer.py').read_text())
    assert not {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    } & {'_entries', '_masters', '_names'}
    binding = source(REPO / 'bindings/nccl_bindings.cpp')
    assert not re.search(r'ncclComm_t|cudaStream_t|ncclComm(?:Init|Destroy|Abort)|cudaStream(?:Create|Destroy)', binding)
    ddp = ast.parse((REPO / 'python/orthon/distributed/ddp.py').read_text())
    assert not {
        n.attr for n in ast.walk(ddp) if isinstance(n, ast.Attribute)
    } & {'buckets', 'outstanding', 'launch_count'}
    assert 'host_scatter' not in source(REPO / 'src/backends/cuda/reduce_kernels.cu')
