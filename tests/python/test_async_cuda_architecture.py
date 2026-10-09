"""CUDA runtime source fences complement injected and physical native tests."""
from tests.support.source import read_cpp_source as source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

REPO = REPO_ROOT




def production():
    for root in ('src', 'include', 'bindings'):
        for path in (REPO / root).rglob('*'):
            if path.suffix in {'.cpp', '.cu', '.h', '.cuh'}:
                yield path


def test_all_production_launches_use_explicit_context_stream():
    count = 0
    for path in (REPO / 'src/backends/cuda').glob('*.cu'):
        text = source(path)
        launches = re.findall(r'<<<(.*?)>>>', text, re.S)
        for launch in launches:
            # Split top-level commas, ignoring function arguments and templates.
            depth = 0
            args = []
            start = 0
            for index, char in enumerate(launch):
                if char in '(<[':
                    depth += 1
                elif char in ')>]':
                    depth -= 1
                elif char == ',' and depth == 0:
                    args.append(launch[start:index].strip())
                    start = index + 1
            args.append(launch[start:].strip())
            assert len(args) == 4 and args[-1] == 'stream', (path, launch)
        if launches:
            assert 'context->compute_stream().native_handle()' in text, path
            assert 'default_runtime().cuda(' in text, path
        count += len(launches)
    assert count > 100


def test_native_resource_and_transfer_primitives_have_runtime_owner():
    primitive = re.compile(r'\b(?:cuda(?:Memcpy\w*|Memset\w*|Event\w+|Stream(?:Create\w*|Destroy|Synchronize|WaitEvent))|cublasSetStream)\s*\(')
    for path in production():
        if primitive.search(source(path)):
            assert path.relative_to(REPO).as_posix() in {
                'src/runtime/cuda_context.cpp', 'bindings/nccl_bindings.cpp',
            }, path


def test_device_wide_waits_are_only_runtime_compatibility_or_nccl():
    for path in production():
        if re.search(r'\bcudaDeviceSynchronize\s*\(', source(path)):
            assert path.relative_to(REPO).as_posix() in {
                'src/runtime/cuda_context.cpp', 'bindings/nccl_bindings.cpp',
            }, path
    conv = source(REPO / 'src/backends/cuda/conv2d_cuda.cu')
    matmul = source(REPO / 'src/backends/cuda/matmul_kernels.cu')
    assert 'synchronize_compute()' not in conv
    assert 'CudaTimingEvent' not in matmul and '->synchronize_compute()' not in matmul


def test_cuda_metadata_does_not_enter_buffer_allocation_or_plans():
    for name in ('core/buffer.h', 'core/tensor_impl.h', 'memory/allocation.h',
                 'planning/matmul_planner.h', 'reduction/reduction.h'):
        text = source(REPO / 'include/orthon' / name)
        assert not re.search(r'\b(?:CudaStream|CudaTimingEvent|CompletionToken|cudaStream_t|cudaEvent_t)\b', text), name
    for name in ('memory/pool.h', 'memory/scratch.h', 'runtime/completion.h', 'runtime/cuda_stream.h'):
        text = source(REPO / 'include/orthon' / name)
        assert 'cuda_runtime' not in text and 'cublas_v2' not in text
        assert 'cudaEvent_t' not in text and 'cudaStream_t' not in text


def test_shared_state_has_no_context_cycle_or_global_registry():
    for name in ('src/runtime/cuda_stream.cpp', 'src/runtime/completion.cpp',
                 'include/orthon/runtime/cuda_stream.h', 'include/orthon/runtime/completion.h',
                 'src/memory/pool.cpp', 'src/memory/scratch.cpp'):
        text = source(REPO / name)
        assert not re.search(r'shared_ptr\s*<\s*(?:CudaContext|Runtime)\s*>|\[\s*this\s*\]|thread_local', text), name
        assert not re.search(r'static\s+std::(?:map|unordered_map|vector)', text), name
    stream = source(REPO / 'src/runtime/cuda_context.cpp')
    assert 'cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)' in stream
    assert 'bind_default_stream' not in stream and 'default_stream' not in stream


def test_pool_and_scratch_retire_and_order_before_reuse_or_free():
    for name in ('src/memory/pool.cpp', 'src/memory/scratch.cpp'):
        text = source(REPO / name)
        assert 'record_completion_noexcept()' in text
        assert '.enqueue_wait_on(queue)' in text
        assert '.wait_host_noexcept()' in text
    backends = '\n'.join(source(p) for p in (REPO / 'src/backends').rglob('*') if p.suffix in {'.cpp', '.cu'})
    assert not re.search(r'\b(?:legacy_allocate|legacy_reset)\s*\(', backends)


def test_no_public_python_stream_api_or_future_tensor_results():
    python = '\n'.join(source(p) for p in (REPO / 'python/orthon').rglob('*.py'))
    bindings = '\n'.join(source(p) for p in (REPO / 'bindings').glob('*.cpp') if p.name != 'nccl_bindings.cpp')
    assert not re.search(r'\b(?:CudaStream|CudaTimingEvent|CompletionToken|current_stream|record_stream)\b', python + bindings)
    assert 'default_runtime().synchronize_current_cuda()' in bindings
    cmake = (REPO / 'CMakeLists.txt').read_text()
    assert 'orthon_add_cpp_test(test_async_cuda tests/cpp/test_async_cuda.cu runtime)' in cmake
    assert 'if(_ORTHON_CUDA_ENABLED)' in cmake
