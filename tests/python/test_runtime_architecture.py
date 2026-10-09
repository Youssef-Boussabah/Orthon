"""Runtime ownership diagnostics, backed by native lifecycle tests."""
from tests.support.source import read_cpp_source as source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

REPO = REPO_ROOT




def production():
    for directory in ("include", "src", "bindings"):
        for path in (REPO / directory).rglob("*"):
            if path.suffix in (".h", ".cpp", ".cu", ".cuh"):
                yield path.relative_to(REPO).as_posix()


def test_allocator_scratch_and_pool_facades_forward_to_runtime():
    for path, call in (
        ("src/memory/allocator.cpp", "default_runtime().allocator(device)"),
        ("src/memory/scratch.cpp", "default_runtime().scratch(device)"),
        ("src/memory/pool.cpp", "default_runtime().memory_pool()"),
    ):
        text = source(path)
        assert call in text
        assert not re.search(r"static\s+std::(?:map|unordered_map|mutex)", text)
    owners = [path for path in production()
              if re.search(r"std::(?:map|unordered_map)<[^;]*shared_ptr<CudaContext>", source(path))]
    assert owners == ["include/orthon/runtime/runtime.h"]


def test_buffer_and_allocation_keep_their_independent_lifetimes():
    buffer = source("include/orthon/core/buffer.h")
    allocation = source("include/orthon/memory/allocation.h")
    assert "Allocation allocation_" in buffer
    assert not re.search(r"\b(?:Runtime|CpuContext|CudaContext)\b", buffer + allocation)
    assert "std::shared_ptr<Allocator>" in allocation
    assert "std::shared_ptr<ScratchArena>" in source("include/orthon/memory/scratch.h")


def test_cublas_creation_destruction_and_capabilities_have_one_owner():
    for primitive in ("cublasCreate", "cublasDestroy", "cudaGetDeviceProperties"):
        owners = [path for path in production()
                  if re.search(r"\b" + primitive + r"\s*\(", source(path))]
        assert owners == ["src/runtime/cuda_context.cpp"]
    matmul = source("src/ops/cuda/matmul_kernels.cu")
    assert "thread_local" not in matmul
    assert "unordered_map" not in matmul
    assert "acquire_cublas()" in matmul
    assert "->capabilities()" in source("src/planning/matmul_planner.cpp")
    assert "->capabilities()" not in matmul
    assert "cublasSetStream" in source("src/runtime/cuda_context.cpp")


def test_default_stream_and_synchronization_remain_internal_and_scoped():
    runtime = source("src/runtime/runtime.cpp")
    module = source("bindings/module.cpp")
    assert "cudaGetDevice(&ordinal)" in runtime
    assert "cuda(ordinal)->synchronize()" in runtime
    assert "std::atexit(&Runtime::shutdown_default_cuda)" in runtime
    assert "runtime.cuda_.clear()" in runtime
    assert "runtime.pool_->clear_cache()" in runtime
    assert "default_runtime().synchronize_current_cuda()" in module
    assert "cudaDeviceSynchronize" not in module
    public_python = "\n".join(path.read_text() for path in (REPO / "python/orthon").rglob("*.py"))
    assert not re.search(r"\b(?:Runtime|CpuContext|CudaContext|CudaStream|CudaEvent|CompletionToken)\b", public_python)
    headers = "\n".join(source(path) for path in production() if path.startswith("include/"))
    assert "class CudaStream" in source("include/orthon/runtime/cuda_stream.h")
    assert "cuda_runtime" not in headers and "cublas_v2" not in headers
