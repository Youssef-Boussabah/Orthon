"""Narrow engine ownership diagnostics, backed by native/public contracts."""
from tests.support.source import read_cpp_source as source
from tests.support.source import REPO_ROOT
from pathlib import Path
import re

REPO = REPO_ROOT




def test_tensor_backward_only_forwards_to_engine():
    from tests.support.source import function
    body = function(source("src/core/tensor.cpp"), "Tensor::backward")
    assert "default_autograd_engine().backward(*this, retain_graph)" in body
    assert not re.search(r"\b(?:for|while|unordered_map|unordered_set|Node|NoGradGuard)\b", body)
    assert "build_topo" not in source("src/core/tensor.cpp")


def test_task_owns_discovery_dependencies_scheduling_and_commit():
    text = source("src/autograd/engine.cpp")
    assert re.search(r"struct\s+GraphTask\b", text)
    for responsibility in ("discover", "dependencies", "ready", "gradient", "preflight",
                           "evaluate", "commit_leaves", "release_history"):
        assert responsibility in text
    assert "std::unordered_map<TensorImpl*, size_t>" in text
    assert "validate_saved_versions()" in text
    assert not re.search(r"static\s+std::(?:vector|map|unordered_map|deque)", text)


def test_attachment_seals_edges_centrally_and_discovery_uses_them():
    text = source("src/core/autograd_meta.cpp")
    assert "fn->seal_input_edges()" in text
    assert "GraphEdge::capture(input)" in source("src/autograd/node.cpp")
    from tests.support.source import function
    discover = function(source("src/autograd/engine.cpp"), "discover")
    assert "input_edges()" in discover
    assert "grad_fn(" not in discover
    assert "std::vector<GraphEdge> input_edges_" in source("include/orthon/autograd/node.h")


def test_engine_and_backends_keep_their_dependency_boundaries():
    for directory in ("src/autograd", "include/orthon/autograd"):
        for path in (REPO / directory).glob("*"):
            text = source(path.relative_to(REPO))
            assert not re.search(r"#include\s*[<\"](?:orthon/backends/|cuda|cublas)", text)
    for path in (REPO / "src/backends").rglob("*"):
        if path.is_file():
            text = source(path.relative_to(REPO))
            assert not re.search(r"AutogradEngine|GraphTask|set_grad_fn|post_accumulate", text)


def test_backward_guard_covers_formulas_commits_hooks_and_release():
    from tests.support.source import function
    body = function(source("src/autograd/engine.cpp"), "AutogradEngine::backward")
    assert body.count("NoGradGuard") == 1
    assert body.index("NoGradGuard") < body.index("task.discover")
    calls = ["task.preflight", "task.evaluate", "task.commit_leaves", "task.release_history"]
    assert [body.index(call) for call in calls] == sorted(body.index(call) for call in calls)
    assert "set_grad_enabled(" not in body


def test_existing_formulas_remain_local_adjoints():
    for path in (REPO / "include/orthon/autograd").glob("*_node.h"):
        text = source(path.relative_to(REPO))
        assert not re.search(r"AutogradEngine|GraphTask|seal_input_edges|input_edges\(", text)
        # A formula may directly call another local adjoint (fused ReLU).
        assert "backward(const Tensor& grad_output)" in text
