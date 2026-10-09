"""Ownership fences complement behavioral and native plan/lifetime checks."""
from tests.support.source import REPO_ROOT
from pathlib import Path
import re
from tests.support.source import function, read_cpp_source as source

REPO = REPO_ROOT


def test_reduction_metadata_has_one_immutable_resource_free_owner():
    header = source("include/orthon/reduction/reduction_plan.h")
    assert "class ReductionPlan" in header and "const ReductionKind kind" in header
    assert "friend ReductionPlan normalize_reduction" in header
    assert not re.search(r"\b(?:Tensor|TensorImpl|Buffer|Allocation|ScratchLease|Node|cudaStream_t)\b", header)
    for path in list((REPO / "include/orthon/reduction").glob("*.h")) + list((REPO / "src/reduction").glob("*.cpp")):
        assert not re.search(r"#\s*include\s*[<\"](?:cuda|orthon/backends/)", source(path.relative_to(REPO)))
    dispatch = source("include/orthon/dispatch/operator.h")
    assert not re.search(r"\b(?:ReductionArgs|MaxScatterArgs)\b", dispatch)
    assert "const ReductionPlan&" in dispatch


def test_backends_consume_plan_without_deriving_shape_or_geometry():
    paths = ["src/backends/cpu/reduce.cpp", "src/backends/cpu/max_backward.cpp", "src/backends/cuda/reduce_kernels.cu"]
    for path in paths:
        text = source(path)
        assert not re.search(r"\b(?:outer|inner|out_numel)\s*\*=|output_shape\s*\.(?:erase|insert|push_back)", text)
        assert not re.search(r"\b(?:reduction_shape|resolve_dim|reduction_factor)\s*\(", text)
        assert "plan.output_numel" in text
        assert not re.search(r"set_grad_fn|requires_grad_|\b\w+Node\b", text)
    adapters = source("src/backends/cuda/adapters.cpp")
    for name in ["cuda_sum_adapter", "cuda_mean_adapter", "cuda_max_adapter"]:
        body = function(adapters, name)
        assert "execution_operand(t)" in body and "plan" in body
        assert not re.search(r"\.shape\(|outer|inner|value_or", body)


def test_nodes_save_plan_and_original_identity_without_duplicate_geometry():
    for name in ["reduce", "max"]:
        text = source(f"include/orthon/autograd/{name}_node.h")
        assert "const ReductionPlan plan_" in text and "inputs = {input}" in text
        assert not re.search(r"\b(?:dim_|keepdim_|is_mean_|factor_|input_shape_|outer_|reduce_|inner_)\b", text)
    maximum = source("include/orthon/autograd/max_node.h")
    assert 'watch_version(argmax_indices_, "argmax_indices")' in maximum
    assert "watch_version(input" not in maximum
    assert "max_backward(input, grad, argmax_indices_, plan_)" in maximum


def test_max_backward_stays_on_device_for_global_and_dimension():
    body = function(source("src/backends/cuda/adapters.cpp"), "cuda_max_backward_adapter")
    assert "context->memset_async" in body and "cuda_max_scatter_grad" in body
    assert "DeviceGuard guard(plan.device)" in body
    assert not re.search(r"cuda_to_host|host_to_cuda|host_full_max_scatter|make_zeroed_host_tensor|if\s*\([^)]*\.dim", body)
    assert not (REPO / "src/core/host_scatter.cpp").exists()
    assert not (REPO / "include/orthon/core/host_scatter.h").exists()


def test_workspace_is_one_scoped_backend_lease_without_legacy_reset():
    text = source("src/backends/cuda/reduce_kernels.cu")
    assert text.count("->acquire(") == 1
    assert "ScratchLease scratch" in text and "checked_allocation_add" in text
    assert text.count("ReductionWorkspace workspace(t, num_blocks,") == 2
    assert not re.search(r"legacy_allocate|legacy_reset|memory_pool_\w+|cudaMalloc|thread_local", text)
    neutral = source("include/orthon/reduction/reduction_plan.h") + source("src/reduction/reduction.cpp")
    assert "ScratchLease" not in neutral


def test_all_extrema_comparisons_share_the_same_selector():
    selector = source("include/orthon/reduction/extrema.h")
    assert "candidate_index < current_index" in selector
    for path in ["src/backends/cpu/reduce.cpp", "src/backends/cuda/reduce_kernels.cu"]:
        text = source(path)
        assert "#include <orthon/reduction/extrema.h>" in text
        assert "extrema_candidate_wins(" in text
        assert not re.search(r"\b(?:v|input\[i\]|partial_vals\[i\])\s*>\s*(?:best_val|local.val)|\.val\s*>\s*smem", text)


def test_semantic_reductions_normalize_once_then_resolve_one_route():
    text = source("src/ops/reduce_ops.cpp")
    for name in ["reduce_impl", "reduce_max", "argmax"]:
        body = function(text, name)
        assert body.count("normalize_reduction(") == 1
        assert body.count(".resolve<ReductionKernel>") == 1
        assert "kernel.call(t, plan)" in body
        assert not re.search(r"cpu_\w+\(|cuda_\w+\(|\b(?:outer|reduce|inner)_sz\b", body)
