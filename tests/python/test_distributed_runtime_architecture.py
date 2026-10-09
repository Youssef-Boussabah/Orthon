from tests.support.source import REPO_ROOT
import gc
from pathlib import Path
import weakref

import pytest
import orthon as ortho
import orthon.distributed as dist
from orthon.nn import Module, Parameter
from orthon.amp import MixedPrecisionOptimizer
from orthon.optim import SGD

ROOT = REPO_ROOT


def single_group():
    dist.register_backend("runtime_single", lambda **kw: dist._SingleRankBackend())
    return dist.init_process_group("runtime_single")


def teardown_function():
    dist.destroy_process_group()


def test_captured_group_rejects_old_ddp_after_default_group_reinitialization():
    from orthon.distributed import _state
    backend = single_group()
    assert _state.get_active() is not backend
    model = Module()
    model.weight = Parameter(ortho.ones([1]))
    old = dist.DDP(model)
    dist.destroy_process_group()
    new_backend = single_group()
    new = dist.DDP(model, sync_grads=False)
    for call in (old.all_reduce_gradients, old.wait_for_gradient_sync, backend.barrier):
        with pytest.raises(RuntimeError, match="shut down"):
            call()
    new.all_reduce_gradients()
    assert new_backend is not backend


def test_callbacks_do_not_keep_ddp_reducer_or_module_alive():
    single_group()
    model = Module()
    model.weight = Parameter(ortho.ones([1]))
    ddp = dist.DDP(model)
    reference, reducer = weakref.ref(ddp), weakref.ref(ddp._reducer)
    parameter = model.weight
    module = weakref.ref(model)
    del ddp, model
    gc.collect()
    assert reference() is None and reducer() is None and module() is None
    (parameter * 2).sum().backward()
    assert parameter.grad.tolist() == [2.]


def test_finalizer_deduplicates_outputs_replaces_callback_and_runs_after_hooks():
    a = ortho.ones([1]).requires_grad_(True)
    b = ortho.ones([1]).requires_grad_(True)
    order = []
    a._register_post_accumulate_grad_hook(lambda: order.append("a"))
    b._register_post_accumulate_grad_hook(lambda: order.append("b"))
    x, y = a * 2, b * 3
    x._register_backward_finalizer(37, lambda: order.append("replaced"))
    callback = lambda: order.append((a.grad.tolist(), b.grad.tolist()))
    x._register_backward_finalizer(37, callback)
    y._register_backward_finalizer(37, callback)
    loss = (x + y).sum()
    loss.backward(retain_graph=True)
    assert set(order[:2]) == {"a", "b"}
    assert order[2:] == [([2.], [3.])]
    order.clear()
    loss.backward()
    assert order[2:] == [([4.], [6.])]


def test_failed_backward_does_not_call_finalizer():
    a = ortho.ones([1]).requires_grad_(True)
    root = (a * a).sum()
    calls = []
    root._register_backward_finalizer(38, lambda: calls.append(1))
    with ortho.no_grad():
        a.fill_(2)
    with pytest.raises(RuntimeError):
        root.backward()
    assert not calls and a.grad is None


def test_amp_rejects_direct_auto_sync_wrapper_before_creating_masters():
    single_group()
    model = Module()
    model.weight = Parameter(ortho.ones([1]))
    with pytest.raises(ValueError, match="sync_grads=False"):
        MixedPrecisionOptimizer(dist.DDP(model), SGD, lr=.1)


def test_distributed_services_keep_vendor_ownership_out_of_bindings_and_headers():
    binding = (ROOT / "bindings/nccl_bindings.cpp").read_text()
    assert "nccl.h" not in binding and "cuda_runtime" not in binding
    for name in ("ncclCommInitRank", "ncclCommDestroy", "ncclBroadcast", "ncclAllReduce",
                 "cudaStreamCreate", "cudaStreamDestroy", "cudaStreamSynchronize", "cudaMemcpy", "cudaDeviceSynchronize"):
        assert name not in binding
    headers = list((ROOT / "include/orthon/distributed").glob("*.h"))
    assert headers
    for path in headers:
        source = path.read_text()
        assert "nccl.h" not in source and "cuda_runtime" not in source and "pybind11" not in source
    for path in (ROOT / "src").rglob("*.cpp"):
        if path.name != "system_nccl_transport.cpp":
            assert "nccl.h" not in path.read_text()
    ddp = (ROOT / "python/orthon/distributed/ddp.py").read_text()
    assert "mul_(" not in ddp and "async_op=" not in ddp and "_copy_flat_range" not in ddp
    reducer = (ROOT / "python/orthon/distributed/_reducer.py").read_text()
    assert "state_dict" not in reducer and "orthon.amp" not in reducer and "orthon.optim" not in reducer
    amp = (ROOT / "python/orthon/amp/optimizer.py").read_text()
    assert "_reduce_master_gradients" in amp and "ddist.all_reduce" not in amp
