"""Planner policy ownership; native tests prove metadata and execution behavior."""
from tests.support.source import read_cpp_source as source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re
from tests.support.source import function

REPO = REPO_ROOT




def test_plan_records_are_neutral_immutable_metadata():
    text = source("include/orthon/planning/matmul_plan.h")
    assert all(name in text for name in ("MatmulProblem", "MatmulPlan", "MatmulAlgorithm", "MatmulPreference"))
    assert not re.search(r"Tensor|Buffer|shared_ptr|cuda_runtime|cublas_v2|backends/|autograd|pybind", text)
    assert "const MatmulProblem problem" in text
    assert "const MatmulAlgorithm algorithm" in text


def test_planner_is_the_only_policy_owner_and_reads_exact_runtime_ordinal():
    text = source("src/planning/matmul_planner.cpp")
    assert "runtime.cuda(p.device.index)->capabilities().cublas_bfloat16" in text
    assert not re.search(r"cudaGet\w+|cublasGet\w+|acquire_cublas|cudaEvent\w+|cudaMalloc|cudaMemcpy|synchronize|backends/|getenv|unordered_map", text)
    assert "static const MatmulPlanner planner" in text
    for path in (REPO / "src/backends").rglob("*"):
        if path.suffix not in (".cpp", ".cu"):
            continue
        body = source(path.relative_to(REPO))
        assert not re.search(r"kCublasWorkThreshold|use_library_matmul|use_library_bfloat16_matmul|cublas_bfloat16_supported|default_matmul_planner|cublas_bfloat16", body), path


def test_semantic_dispatch_then_one_plan_then_execution_then_graph_attachment():
    body = function(source("src/ops/matmul_ops.cpp"), "matmul")
    assert body.count(".resolve<MatmulKernel>") == 1
    assert body.count(".plan(") == 1
    assert body.index(".resolve<") < body.index(".plan(") < body.index("kernel.call(a, b, plan)") < body.index("set_grad_fn")
    assert "MatMulNode>(a, b)" in body
    contract = source("include/orthon/dispatch/operator.h")
    assert "Tensor (*)(const Tensor&, const Tensor&, const MatmulPlan&)" in contract
    assert "MatmulCustom" not in contract and "MatmulCublas" not in contract


def test_backends_obey_plan_and_autograd_uses_semantic_transposes():
    executor = function(source("src/backends/cuda/matmul_kernels.cu"), "cuda_matmul_execute")
    assert "switch (plan.algorithm)" in executor
    assert ".plan(" not in executor and "capabilities()" not in executor
    adapter = function(source("src/backends/cuda/adapters.cpp"), "cuda_matmul_adapter")
    assert "execution_operand(a)" in adapter and "execution_operand(b)" in adapter
    assert "cuda_matmul_execute(a_kernel, b_kernel, plan)" in adapter
    node = source("include/orthon/autograd/matmul_node.h")
    assert "watch_version(a" in node and "watch_version(b" in node
    assert ".is_cuda()" not in node and ".to(" not in node


def test_diagnostics_share_planner_and_timing_keeps_events_in_execution():
    text = source("src/diagnostics/matmul.cpp")
    assert text.count("default_matmul_planner().plan(") == 1
    assert "MatmulPreference::ForceCublas" in function(text, "cuda_matmul_tiled_cublas")
    assert "MatmulPreference::ForceCublas" in function(text, "cuda_batched_matmul_tiled_cublas")
    assert not re.search(r"work_items\s*[<>](?![<>])|capabilities\(|cudaEvent|acquire_cublas|\b16\s*\}?\s*<<", text)
    raw = source("src/backends/cuda/matmul_kernels.cu")
    assert "CudaTimingEvent" not in raw and "chrono" not in raw and "acquire_cublas()" in raw
    assert "context->compute_stream()" in raw and "cudaEventRecord" not in raw
