"""Layer ownership checks; native tests exercise execution and lifetime contracts."""
from tests.support.source import read_cpp_source as source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

import pytest

REPO = REPO_ROOT




@pytest.mark.parametrize('layer', ['cpu', 'cuda'])
def test_backend_sources_are_graph_free_and_do_not_depend_upward_or_across(layer):
    files = list((REPO / 'src/backends' / layer).glob('*')) + list((REPO / 'include/orthon/backends' / layer).glob('*.h'))
    assert files
    for path in files:
        text = source(path)
        assert not re.search(r'#include.*(?:autograd|pybind|Python\.h|orthon/ops/)', text), path
        assert not re.search(r'\b(?:set_grad_fn|requires_grad_|set_requires_grad|set_is_leaf|watch_version|NoGradGuard)\b|\b\w+Node\b', text), path
        other = 'cuda' if layer == 'cpu' else 'cpu'
        assert f'backends/{other}/' not in text and f'backend::{other}::' not in text, path
        if layer == 'cpu':
            assert not re.search(r'\bcuda\w*\s*\(|\bcublas\w*\s*\(', text), path


def test_semantic_sources_do_not_execute_backend_math_or_call_raw_backends():
    files = list((REPO / 'src/ops').glob('*.cpp'))
    assert len(files) >= 17
    for path in files:
        text = source(path)
        assert not re.search(r'#include.*(?:backends/|cuda_runtime|cublas|immintrin)', text), path
        assert not re.search(r'\b(?:cuda_\w+|cudaMemcpy\w*|cpu_\w+)\s*\(|backend::(?:cpu|cuda)::', text), path
        assert not re.search(r'static_cast<[^>]+\*>\([^)]*\.data\(\)', text), path


def test_autograd_formulas_have_no_raw_execution_dependencies():
    files = list((REPO / 'include/orthon/autograd').glob('*.h')) + list((REPO / 'src/autograd').glob('*.cpp'))
    for path in files:
        text = source(path)
        assert not re.search(r'#include.*(?:backends/|ops/cpu|ops/cuda|cuda_runtime|cublas)', text), path
        assert not re.search(r'\b(?:cuda_\w+|cpu_\w+)\s*\(|backend::(?:cpu|cuda)::', text), path
        assert '.data()' not in text, path


def test_bootstrap_uses_backend_owned_checked_callables_and_no_graph_state():
    text = source(REPO / 'src/dispatch/builtin_registration.cpp')
    assert '#include <orthon/backends/cpu/kernels.h>' in text
    assert '#include <orthon/backends/cuda/kernels.h>' in text
    assert 'using namespace backend::cpu;' in text and 'using namespace backend::cuda;' in text
    assert 'using namespace adapters;' not in text
    for name in ['Relu', 'DropoutMask', 'Conv2d', 'MaxPool2d', 'CrossEntropy', 'NllLoss', 'Embedding', 'Fill', 'AddInplace', 'Finite', 'AmpUnscale', 'IndexBackward', 'Conv2dBackward', 'MaxBackward']:
        assert f'OperatorId::{name}' in text
    assert '->freeze()' in text
    assert not re.search(r'Node|set_grad_fn|requires_grad', text)


def test_compatibility_headers_forward_without_duplicate_implementations():
    for kind in ['cpu', 'cuda']:
        for path in (REPO / 'include/orthon/ops' / kind).glob('*.h'):
            target = f'orthon/ops/{path.name}' if kind == 'cpu' else f'orthon/backends/cuda/{path.name}'
            assert source(path).strip() == f'#pragma once\n\n#include <{target}>'
    assert not list((REPO / 'src/ops/cpu').glob('*.cpp'))
    assert not list((REPO / 'src/ops/cuda').glob('*.cu'))


def test_cuda_broadcast_and_execution_materialization_have_neutral_ownership():
    raw = source(REPO / 'src/backends/cuda/binary_kernels.cu')
    assert '#include <orthon/core/broadcast.h>' in raw
    assert '#include <orthon/ops/' not in raw
    helper = source(REPO / 'src/core/execution_copy.cpp')
    assert 'if (t.is_contiguous()) return t;' in helper
    assert 'resolve<TransferKernel>' in helper and 'kernel.call(t, t.device())' in helper
    assert 'grad' not in helper.lower()
    adapter = source(REPO / 'src/backends/cuda/adapters.cpp')
    assert 'execution_operand(x)' in adapter
    assert 'context->peer_copy_blocking(out.data(),' in adapter
    assert '*default_runtime().cuda(src_c.device().index), src_c.data()' in adapter


def test_mutation_policy_stays_at_semantic_boundaries():
    for path in (REPO / 'src/backends').rglob('*'):
        if path.suffix in ('.cpp', '.cu'):
            assert 'bump_version' not in source(path), path
    text = source(REPO / 'src/ops/binary_ops.cpp')
    assert text.count('bump_version()') == 2
    assert 'bump_version()' in source(REPO / 'src/ops/fill_ops.cpp')
    binding = source(REPO / 'bindings/tensor_bindings.cpp')
    assert 'orthon::scalar_write_value' in binding and 'orthon::copy_range_values' in binding
    assert not re.search(r'\bcuda_\w+\s*\(|\bcudaMemcpy\w*\s*\(', binding)
