"""Primary pytest layer ownership; package smoke and native are separate."""
from pathlib import Path, PurePosixPath
from .source import REPO_ROOT

PRIMARY_LAYERS = ("regression", "spec", "architecture")
ARCHITECTURE_MODULES = frozenset({
    "test_allocator_architecture.py",
    "test_amp_architecture.py",
    "test_async_cuda_architecture.py",
    "test_autograd_engine_architecture.py",
    "test_backend_architecture.py",
    "test_build_architecture.py",
    "test_benchmark_architecture.py",
    "test_dispatcher_architecture.py",
    "test_distributed_runtime_architecture.py",
    "test_documentation_architecture.py",
    "test_compatibility_architecture.py",
    "test_matmul_planner_architecture.py",
    "test_nn_architecture.py",
    "test_optimizer_architecture.py",
    "test_package_identity.py",
    "test_reduction_architecture.py",
    "test_repository_architecture.py",
    "test_runtime_architecture.py",
    "test_serialization_architecture.py",
    "test_tensor_core_architecture.py",
    "test_test_architecture.py",
})

def primary_layer_for_path(path):
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(REPO_ROOT)
    path = PurePosixPath(path.as_posix())
    if not path.name.startswith("test_") or path.suffix != ".py" or ".." in path.parts:
        raise ValueError("unclassified pytest source: %s" % path)
    if path.parts[:2] == ("tests", "spec"):
        return "spec"
    if path.parent == PurePosixPath("tests/python"):
        return "architecture" if path.name in ARCHITECTURE_MODULES else "regression"
    raise ValueError("unclassified pytest source: %s" % path)
