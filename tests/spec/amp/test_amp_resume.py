import pytest

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.optim import Adam, SGD

from .workload import AffineModel, assert_state, snapshot, train


@pytest.mark.parametrize("kind", [SGD, Adam], ids=["sgd", "adam"])
@pytest.mark.parametrize("transport", ["independent-memory", "checkpoint" ])
def test_amp_model_optimizer_and_caller_managed_scaler_resume_match_continuation(
    amp_training_case, typed_tensor, tmp_path, kind, transport,
):
    device, dtype = amp_training_case
    create = lambda values: typed_tensor(values, dtype=dtype, device=device)
    model = AffineModel(create)
    optimizer = MixedPrecisionOptimizer(model, kind, lr=.03125)
    scaler = GradScaler(init_scale=8, growth_factor=2, growth_interval=2, enabled=dtype != "bfloat16")
    inputs = create([[.5, 1.], [1., -.5]])
    for _ in range(2):
        train(model, optimizer, scaler, inputs)
    optimizer.set_lr(.015625)
    path = tmp_path / "amp.ckpt"
    if transport == "checkpoint":
        assert ortho.save_checkpoint(path, model=model, optimizer=optimizer,
                                     metadata={"scaler": scaler.state_dict(), "step": 2}) is None
    else:
        model_state = snapshot(model.state_dict())
        amp_state = snapshot(optimizer.state_dict())
        scaler_state = dict(scaler.state_dict())
    restored_model = AffineModel(create)
    restored_optimizer = MixedPrecisionOptimizer(restored_model, kind, lr=1.)
    restored_scaler = GradScaler(init_scale=1, enabled=True)
    if transport == "checkpoint":
        metadata = ortho.load_checkpoint(path, model=restored_model, optimizer=restored_optimizer)
        assert metadata["step"] == 2
        assert restored_scaler.scale_factor == 1
        restored_scaler.load_state_dict(metadata["scaler"])
    else:
        restored_model.load_state_dict(model_state)
        restored_optimizer.load_state_dict(amp_state)
        restored_scaler.load_state_dict(scaler_state)
    assert restored_optimizer.get_lr() == optimizer.get_lr() == .015625
    assert_state(restored_model.state_dict(), model.state_dict())
    assert_state(restored_optimizer.state_dict(), optimizer.state_dict())
    assert restored_scaler.state_dict() == scaler.state_dict()
    for _ in range(3):
        expected = train(model, optimizer, scaler, inputs)
        actual = train(restored_model, restored_optimizer, restored_scaler, inputs)
        assert actual == pytest.approx(expected, rel=4e-2 if dtype == "bfloat16" else 5e-3,
                                       abs=2e-6)
        assert_state(restored_model.state_dict(), model.state_dict())
        assert_state(restored_optimizer.state_dict(), optimizer.state_dict())
        assert restored_scaler.state_dict() == scaler.state_dict()


def test_file_restore_produces_independent_low_precision_master_and_moment_owners(
    tmp_path, typed_tensor, model_gradient,
):
    create = lambda values: typed_tensor(values, dtype="float16")
    source = AffineModel(create)
    optimizer = MixedPrecisionOptimizer(source, Adam, lr=.125)
    model_gradient(source.weight, [[1., 2.], [3., 4.]])
    optimizer.step()
    optimizer.zero_grad()
    path = tmp_path / "amp-storage.ckpt"
    ortho.save_checkpoint(path, model=source, optimizer=optimizer)
    branches = []
    for _ in range(2):
        model = AffineModel(create)
        opt = MixedPrecisionOptimizer(model, Adam, lr=1.)
        ortho.load_checkpoint(path, model=model, optimizer=opt)
        branches.append((model, opt))
    unchanged = [snapshot(opt.state_dict()) for _, opt in branches]
    branches[0][1].state_dict()["masters"][0].fill_(9)
    branches[0][1].state_dict()["inner"]["m"][0].fill_(7)
    assert_state(branches[1][1].state_dict(), unchanged[1])
    assert_state(optimizer.state_dict(), unchanged[1])
