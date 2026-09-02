import pytest

import orthon as ortho
import orthon.distributed as ddist
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.distributed import DistributedDataParallel as DDP
from orthon.nn import Linear
from orthon.optim import SGD


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _flatten(values):
    out = []
    for value in values:
        if isinstance(value, list):
            out.extend(_flatten(value))
        else:
            out.append(float(value))
    return out


def _flatten_mapping(mapping):
    out = []
    for key in sorted(mapping):
        out.extend(_flatten(mapping[key]))
    return out


def _amp_ddp_finite_worker(rank):
    model = Linear(2, 1)
    for parameter in model.parameters():
        parameter.fill_(0.25)
    model = model.to("cuda")
    model.half()
    ddp_model = DDP(model, sync_grads=False)
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=0.01)
    scaler = GradScaler(init_scale=8.0, growth_interval=1)

    x = ortho.cast(ortho.full([1, 2], float(rank + 1)).to("cuda"), "float16")
    optimizer.zero_grad()
    loss = ortho.sum(ddp_model(x))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    return {
        "params": {
            name: _flatten(ortho.cast(param.to("cpu"), "float32").tolist())
            for name, param in model.named_parameters()
        },
        "scale": scaler.scale_factor,
        "found_inf": scaler.found_inf,
    }


def _amp_ddp_nonfinite_worker(rank):
    model = Linear(2, 1)
    for parameter in model.parameters():
        parameter.fill_(0.25)
    model = model.to("cuda")
    model.half()
    ddp_model = DDP(model, sync_grads=False)
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=0.01)
    scaler = GradScaler(init_scale=8.0, backoff_factor=0.25)

    x = ortho.cast(ortho.full([1, 2], 1.0).to("cuda"), "float16")
    optimizer.zero_grad()
    loss = ortho.sum(ddp_model(x))
    scaler.scale(loss).backward()
    if rank == 1:
        first_param = next(model.parameters())
        first_param.set_grad(
            ortho.cast(ortho.full(list(first_param.shape), float("inf")).to("cuda"), "float16")
        )
    before = {
        name: _flatten(ortho.cast(param.to("cpu"), "float32").tolist())
        for name, param in model.named_parameters()
    }
    scaler.step(optimizer)
    scaler.update()
    after = {
        name: _flatten(ortho.cast(param.to("cpu"), "float32").tolist())
        for name, param in model.named_parameters()
    }

    return {
        "before": before,
        "after": after,
        "scale": scaler.scale_factor,
        "found_inf": scaler.found_inf,
    }


def _amp_ddp_scaler_state_worker(rank):
    model = Linear(2, 1)
    for parameter in model.parameters():
        parameter.fill_(0.25)
    model = model.to("cuda")
    model.half()
    ddp_model = DDP(model, sync_grads=False)
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=0.01)
    scaler = GradScaler(init_scale=8.0, growth_interval=2)

    x = ortho.cast(ortho.full([1, 2], float(rank + 1)).to("cuda"), "float16")
    optimizer.zero_grad()
    scaler.scale(ortho.sum(ddp_model(x))).backward()
    scaler.step(optimizer)
    scaler.update()

    state = scaler.state_dict()
    restored = GradScaler(init_scale=1.0)
    restored.load_state_dict(state)

    optimizer.zero_grad()
    scaler.scale(ortho.sum(ddp_model(x))).backward()
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()
    restored.scale(ortho.sum(ddp_model(x))).backward()
    if rank == 1:
        first_param = next(model.parameters())
        first_param.set_grad(
            ortho.cast(ortho.full(list(first_param.shape), float("inf")).to("cuda"), "float16")
        )
    restored.step(optimizer)
    restored.update()

    return {
        "saved_scale": state["scale"],
        "saved_growth_tracker": state["growth_tracker"],
        "continuous_scale": scaler.scale_factor,
        "restored_scale_after_nonfinite": restored.scale_factor,
    }


@requires_cuda
def test_amp_ddp_finite_master_gradients_sync_and_grow_scale():
    results = ddist.spawn(_amp_ddp_finite_worker, nprocs=2, backend="local")

    assert _flatten_mapping(results[0]["params"]) == pytest.approx(
        _flatten_mapping(results[1]["params"])
    )
    assert [result["scale"] for result in results] == [16.0, 16.0]
    assert [result["found_inf"] for result in results] == [False, False]


@requires_cuda
def test_amp_ddp_global_nonfinite_skips_all_ranks_and_backs_off_scale():
    results = ddist.spawn(_amp_ddp_nonfinite_worker, nprocs=2, backend="local")

    for result in results:
        assert _flatten_mapping(result["after"]) == pytest.approx(
            _flatten_mapping(result["before"])
        )
        assert result["scale"] == pytest.approx(2.0)
        assert result["found_inf"] is False


@requires_cuda
def test_amp_ddp_restored_scaler_state_remains_consistent_across_ranks():
    results = ddist.spawn(_amp_ddp_scaler_state_worker, nprocs=2, backend="local")

    assert [result["saved_scale"] for result in results] == [8.0, 8.0]
    assert [result["saved_growth_tracker"] for result in results] == [1, 1]
    assert [result["continuous_scale"] for result in results] == [16.0, 16.0]
    assert [result["restored_scale_after_nonfinite"] for result in results] == [4.0, 4.0]
