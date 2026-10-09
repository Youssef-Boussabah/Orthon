"""Direct AMP restoration prepares independent state before publication."""
import pytest

import orthon as ortho
from orthon.amp import MixedPrecisionOptimizer
from orthon.optim import Adam, SGD

from .workload import pair, gradient, observation, unchanged, write_master
from .workload import assert_state, snapshot


@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
def test_same_device_direct_load_snapshots_masters_and_source_history(amp_storage_case, typed_tensor, kind):
    device, dtype = amp_storage_case
    source, target = pair(typed_tensor, dtype, device), pair(typed_tensor, dtype, device)
    opt = MixedPrecisionOptimizer(source, kind, lr=.125)
    restored = MixedPrecisionOptimizer(target, kind, lr=.5)
    for p in source.parameters():
        gradient(p)
    opt.step()
    opt.zero_grad()
    incoming = opt.state_dict()
    restored.load_state_dict(incoming, map_location=ortho.Device(device))
    before_source = snapshot(opt.state_dict())
    current = restored.state_dict()
    for old, new in zip(incoming["masters"], current["masters"]):
        assert new is not old
        assert new.is_leaf and new.requires_grad and new.grad is None
        assert not new._has_grad_fn
    for p in target.parameters():
        gradient(p)
    restored.step()
    assert_state(opt.state_dict(), before_source)
    before_target = snapshot(restored.state_dict())
    write_master(incoming["masters"][0], 9.)
    assert_state(restored.state_dict(), before_target)


def test_reused_graph_bearing_input_creates_independent_fp32_leaves(typed_tensor):
    model = pair(typed_tensor)
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    x = typed_tensor([2., 3.]).requires_grad_(True)
    incoming = x * 2
    assert not incoming.is_leaf and incoming._has_grad_fn
    state = opt.state_dict()
    state["masters"] = [incoming, incoming]
    opt.load_state_dict(state)
    a, b = opt.state_dict()["masters"]
    assert a is not b and a is not incoming and b is not incoming
    for master in (a, b):
        assert master.dtype == "float32" and master.is_leaf and master.requires_grad
        assert not master._has_grad_fn and master.grad is None
        assert master.tolist() == [4., 6.]
    a.fill_(7.)
    assert b.tolist() == incoming.tolist() == [4., 6.]
    incoming.sum().backward()
    assert x.grad.tolist() == [2., 2.]


@pytest.mark.parametrize("fault", ["missing-inner", "inner-count", "late-moment-shape",
    "late-moment-type", "late-conversion", "outer-dtype", "outer-count", "outer-shape",
    "map-conflict", "map-type"])
def test_failed_amp_load_leaves_all_owners_values_history_gradients_and_configuration(
    typed_tensor, monkeypatch, fault,
):
    model = pair(typed_tensor)
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    for p in model.parameters():
        gradient(p)
    opt.step()
    before = observation(model, opt)
    state = snapshot(opt.state_dict())
    state["inner"]["lr"] = .75
    state["masters"][0].fill_(6.)
    location = None
    if fault == "missing-inner":
        del state["inner"]
    elif fault == "inner-count":
        state["inner"]["v"].pop()
    elif fault == "late-moment-shape":
        state["inner"]["v"][-1] = typed_tensor([1.])
    elif fault == "late-moment-type":
        state["inner"]["v"][-1] = object()
    elif fault == "late-conversion":
        state["inner"]["v"][-1] = typed_tensor([1., 1.], dtype="float64")
        def fail(*args):
            raise RuntimeError("injected late inner conversion")
        monkeypatch.setattr(ortho, "cast", fail)
    elif fault == "outer-dtype":
        state["orig_dtypes"][-1] = "bfloat16"
    elif fault == "outer-count":
        state["masters"].pop()
    elif fault == "outer-shape":
        state["masters"][-1] = typed_tensor([1.])
    elif fault == "map-conflict":
        location = "cuda:0"  # Device metadata; no unavailable allocation occurs.
    else:
        location = 3
    with pytest.raises((KeyError, ValueError, TypeError, RuntimeError)):
        opt.load_state_dict(state, map_location=location)
    unchanged(model, opt, before)


def test_self_load_preserves_values_and_detaches_new_master_snapshots(amp_storage_case, typed_tensor):
    device, dtype = amp_storage_case
    model = pair(typed_tensor, dtype, device)
    opt = MixedPrecisionOptimizer(model, Adam, lr=.125)
    for p in model.parameters():
        gradient(p)
    opt.step()
    old = opt.state_dict()
    before = snapshot(old)
    owners = list(model.parameters())
    opt.load_state_dict(old)
    current = opt.state_dict()
    assert_state(current, before)
    assert all(a is b for a, b in zip(model.parameters(), owners))
    for old_master, new_master in zip(old["masters"], current["masters"]):
        if dtype == "float32":
            assert new_master is old_master
        else:
            assert new_master is not old_master and new_master.grad is None
