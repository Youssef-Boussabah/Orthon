"""Narrow ownership checks for dispatcher boundaries; execution/lifetimes are tested natively."""
from tests.support.source import read_cpp_source as source, function
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

import pytest

REPO = REPO_ROOT






@pytest.mark.parametrize("path,name", [
    ("src/ops/cpu/binary_ops.cpp", "dispatch::binary_semantic"),
    ("src/ops/cpu/unary_ops.cpp", "neg"),
    ("src/ops/cpu/unary_ops.cpp", "reciprocal_t"),
    ("src/ops/cpu/cast_ops.cpp", "cast"),
    ("src/ops/cpu/matmul_ops.cpp", "matmul"),
    ("src/ops/cpu/reduce_ops.cpp", "reduce_impl"),
    ("src/ops/cpu/reduce_ops.cpp", "reduce_max"),
    ("src/ops/cpu/reduce_ops.cpp", "argmax"),
    ("src/core/tensor.cpp", "Tensor::to"),
])
def test_semantic_frontend_resolves_once_and_has_no_backend_execution_route(path, name):
    body = function(source(path), name)
    assert body.count(".resolve<") == 1
    assert "kernel.call(" in body
    assert not re.search(r"\bcuda_\w+\s*\(|\bcudaMemcpy\w*\s*\(", body)
    assert not re.search(r"\.is_(?:cpu|cuda)\s*\(|DeviceType::(?:CPU|CUDA)", body)
    if "set_grad_fn" in body:
        assert body.index("kernel.call(") < body.index("set_grad_fn")


def test_public_binary_facades_and_compositions_share_canonical_schemas():
    text = source("src/ops/cpu/binary_ops.cpp")
    for name in ("add", "mul"):
        body = function(text, name)
        assert "dispatch::binary_semantic(" in body
        assert not re.search(r"\bcuda_\w+\s*\(|\.is_(?:cpu|cuda)\(", body)
    for name, primitive in (("sub", "return add(a, neg(b));"),
                            ("div", "return mul(a, reciprocal_t(b));")):
        body = function(text, name)
        assert ".schema(" in body
        assert primitive in body
        assert "set_grad_fn" not in body


def test_every_registered_adapter_is_discoverable_and_graph_free():
    headers = [source(f"include/orthon/backends/{kind}/kernels.h") for kind in ("cpu", "cuda")]
    names = [name for header in headers for name in re.findall(
        r"(?m)^(?:Tensor|ReductionResult|std::vector<Tensor>|bool|void|float|std::pair<Tensor, bool>)\s+(\w+)\(", header)]
    assert names
    bootstrap = source("src/dispatch/builtin_registration.cpp")
    texts = [source(path.relative_to(REPO)) for path in (REPO / "src/backends").rglob("*.cpp")]
    forbidden = r"set_grad_fn|requires_grad_|requires_grad\(|set_is_leaf|\bbackward\(|\b\w+Node\b"
    for name in names:
        assert re.search(r"\b" + name + r"\b", bootstrap), name
        definitions = [text for text in texts if re.search(r"(?m)^(?:Tensor|ReductionResult|std::vector<Tensor>|bool|void|float|std::pair<Tensor, bool>)\s+" + re.escape(name) + r"\(", text)]
        assert len(definitions) == 1, name
        body = function(definitions[0], name)
        assert not re.search(forbidden, body), name
    # Registered CUDA binary execution consumes the shape already validated
    # by the semantic wrapper. Only compatibility entry points validate again.
    raw = source("src/ops/cuda/binary_kernels.cu")
    for name in ("cuda_add", "cuda_mul"):
        checked = raw.replace(f"Tensor {name}(const Tensor& a, const Tensor& b)",
                              f"Tensor checked_{name}(const Tensor& a, const Tensor& b)")
        execution = function(checked, name)
        assert "cuda_broadcast_shape" not in execution
        assert "require_cuda_fp_contiguous" not in execution
        assert "assert(" in execution
        assert not re.search(forbidden, execution)
    # The reused CPU math must also be pure, not just its registered wrapper.
    for path, helper in (("binary", "broadcast_apply_typed"),
                         ("unary", "elementwise_typed"),
                         ("matmul", "matmul_values"),
                         ("reduce", "cpu_reduce_all"),
                         ("reduce", "cpu_reduce_dim"),
                         ("reduce", "cpu_max_all"),
                         ("reduce", "cpu_max_dim")):
        assert not re.search(forbidden, function(source(f"src/backends/cpu/{path}.cpp"), helper))


def test_registry_bootstrap_is_explicit_frozen_and_uses_checked_signatures():
    bootstrap = source("src/dispatch/builtin_registration.cpp")
    body = function(bootstrap, "build_builtin_registry")
    for before, after in (("register_schemas", "register_cpu_kernels"),
                          ("register_cpu_kernels", "register_cuda_kernels"),
                          ("register_cuda_kernels", "->freeze()")):
        assert body.index(before) < body.index(after)
    assert "#ifdef WITH_CUDA" in body
    assert "expect_kernel(id, c)" in bootstrap
    registry = source("src/dispatch/kernel_registry.cpp")
    assert "signature_of(value.kernel) != s.signature" in registry
    assert "ambiguous overlapping kernels" in registry
    assert "expected built-in kernel missing" in registry
    headers = source("include/orthon/dispatch/operator.h") + source("include/orthon/dispatch/dispatcher.h")
    assert "std::variant<" in headers
    assert "std::get_if<K>" in headers
    assert "reinterpret_cast" not in headers + registry
    assert "void*" not in headers + registry
    assert "KernelRegistry& operator=(const KernelRegistry&) = delete" in headers


def test_dispatcher_queries_existing_context_without_hardware_cache_or_graph_state():
    paths = list((REPO / "src/dispatch").glob("*.cpp")) + list((REPO / "include/orthon/dispatch").glob("*.h"))
    text = "\n".join(source(path.relative_to(REPO)) for path in paths
                     if path.name != "adapters.h")
    assert "runtime.cuda(request.source.index)->capabilities().cublas_bfloat16" in text
    assert not re.search(r"cudaGetDeviceProperties|cudaDeviceGetAttribute|cublasGetVersion", text)
    assert not re.search(r"set_grad_fn|requires_grad|watch_version|\b\w+Node\b", text)
    request = source("include/orthon/dispatch/operator.h")
    body = request[request.index("struct DispatchRequest"):request.index("struct SpatialArgs")]
    assert not re.search(r"Tensor\s+\w|shared_ptr|Autograd|Buffer", body)
    assert not re.search(r"\bRuntime\s+\w+\s*[;{]", text)


def test_transfer_dispatch_names_both_backends_and_preserves_ordinal_copy_policy():
    bootstrap = source("src/dispatch/builtin_registration.cpp")
    for name in ("cpu.to.cpu", "cpu.to.cuda", "cuda.to.cpu", "cuda.to.cuda"):
        assert name in bootstrap
    semantic = function(source("src/core/tensor.cpp"), "Tensor::to")
    text = source("src/backends/cuda/adapters.cpp")
    assert "transfer_request(*this, target)" in semantic
    physical = function(text, "cuda_to_cuda")
    assert "self.device() == target" in physical
    assert "return cuda_make_contiguous(self)" in physical
    assert "context->peer_copy_blocking(out.data()," in physical
    assert "*default_runtime().cuda(src_c.device().index), src_c.data()" in physical
    assert "cudaDeviceEnablePeerAccess" not in text
    assert "make_contiguous_copy(self)" in function(source("src/backends/cpu/copy.cpp"), "cpu_to")


def test_saved_nodes_retain_original_operands_above_backend_materialization():
    binary = source("src/ops/cpu/binary_ops.cpp")
    semantic = function(binary, "dispatch::binary_semantic")
    assert "AddNode>(a, b)" in semantic and "MulNode>(a, b)" in semantic
    for name in ("cuda_add_adapter", "cuda_mul_adapter"):
        body = function(source("src/backends/cuda/adapters.cpp"), name)
        assert "execution_operand(a), execution_operand(b)" in body
    matmul = source("src/ops/cpu/matmul_ops.cpp")
    assert "MatMulNode>(a, b)" in function(matmul, "matmul")
    assert "execution_operand(a)" in function(source("src/backends/cuda/adapters.cpp"), "cuda_matmul_adapter")
    cast = source("src/ops/cpu/cast_ops.cpp")
    assert "CastNode>(src)" in function(cast, "cast")
    reduce = source("src/ops/cpu/reduce_ops.cpp")
    assert "MaxNode>(" in function(reduce, "reduce_max")
    assert "attach_reduce_grad(t, result.values" in function(reduce, "reduce_impl")
