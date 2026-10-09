"""Allocator source ownership boundaries that runtime values cannot establish."""
from tests.support.source import read_cpp_source as _source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

REPO = REPO_ROOT


def test_raw_cuda_allocation_is_confined_to_the_allocator_backend():
    offenders = []
    for directory in ("src", "bindings", "include"):
        for path in (REPO / directory).rglob("*"):
            if path.suffix not in (".cpp", ".cu", ".h", ".cuh"):
                continue
            if re.search(r"\bcuda(?:Malloc|Free)\s*\(", _source(path)):
                if path.relative_to(REPO).as_posix() != "src/memory/cuda_allocator.cpp":
                    offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == []

def test_core_temporaries_use_scoped_ownership_and_not_legacy_pointer_adapters():
    for directory in ("src/ops", "src/backends", "bindings"):
        for path in (REPO / directory).rglob("*"):
            if path.suffix not in (".cpp", ".cu"):
                continue
            # module.cpp exposes the explicitly retained diagnostic adapters.
            if path.name == "module.cpp":
                continue
            assert not re.search(r"\bmemory_pool_(?:allocate|deallocate|scratch_allocate|reset_scratch)\s*\(", _source(path)), path
    reductions = _source(REPO / "src/ops/cuda/reduce_kernels.cu")
    assert "default_runtime().scratch(t.device())->acquire(" in reductions
    assert "scratch.allocate" in reductions
