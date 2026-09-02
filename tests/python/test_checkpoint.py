"""Checkpoint BEHAVIOUR: model/optimizer round-trips, resume, devices.

The container format itself (raw byte codec, manifest, determinism, corruption
handling, path safety) is covered by test_checkpoint_format.py.

Round-trips are asserted on EXACT bytes rather than approximate values: the
codec moves raw storage, so there is no reason for a saved value to come back
merely close to what was written.
"""
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import orthon as ortho
from orthon.amp import GradScaler, MixedPrecisionOptimizer
from orthon.nn import BatchNorm1d, Linear, ReLU, Sequential
from orthon.optim import Adam, SGD


CUDA = ortho.cuda_enabled()


def _checkpoint_path(name):
    return Path(f".checkpoint_{name}_{uuid4().hex}.ckpt")


def _raw(tensor):
    """Exact payload bytes of a tensor, wherever it lives."""
    with ortho.no_grad():
        return ortho._tensor_to_bytes(tensor.to("cpu"))


def _assert_model_states_bit_identical(left, right):
    left_state, right_state = left.state_dict(), right.state_dict()
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert _raw(right_state[key]) == _raw(left_state[key]), key


def _make_model():
    return Sequential(Linear(3, 4), ReLU(), Linear(4, 2))


def _x(device="cpu"):
    tensor = ortho.full([4, 3], 0.25)
    return tensor.to(device) if device != "cpu" else tensor


def _loss(model, x):
    out = model(x)
    return ortho.sum(out * out)


def _train_steps(model, optimizer, steps, device="cpu"):
    losses = []
    x = _x(device)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _loss(model, x)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def _train_amp_steps(model, optimizer, scaler, steps, device="cuda"):
    x = ortho.cast(_x(device), "float16")
    for _ in range(steps):
        optimizer.zero_grad()
        loss = _loss(model, x)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def _state_values(model):
    return {
        name: ortho.cast(tensor.to("cpu"), "float32").tolist()
        for name, tensor in model.state_dict().items()
    }


def _assert_model_states_close(left, right):
    left_state = _state_values(left)
    right_state = _state_values(right)
    assert left_state.keys() == right_state.keys()
    for key in left_state:
        assert right_state[key] == pytest.approx(left_state[key], rel=1e-6, abs=1e-6)


def test_checkpoint_model_parameter_roundtrip():
    model = _make_model()
    for param in model.parameters():
        param.fill_(0.125)

    path = _checkpoint_path("model")
    try:
        ortho.save_checkpoint(path, model=model)

        restored = _make_model()
        ortho.load_checkpoint(path, model=restored)
    finally:
        path.unlink(missing_ok=True)

    _assert_model_states_close(model, restored)
    # The raw-byte codec makes "close" an understatement.
    _assert_model_states_bit_identical(model, restored)


def test_checkpoint_preserves_parameter_identity():
    """File loading copies INTO parameters, it never replaces them."""
    model = _make_model()
    path = _checkpoint_path("identity")
    try:
        ortho.save_checkpoint(path, model=model)
        destination = _make_model()
        before = {name: tensor._impl_id
                  for name, tensor in destination.state_dict().items()}
        held = list(destination.parameters())
        ortho.load_checkpoint(path, model=destination)
    finally:
        path.unlink(missing_ok=True)

    after = {name: tensor._impl_id
             for name, tensor in destination.state_dict().items()}
    assert after == before
    assert all(p is q for p, q in zip(held, destination.parameters()))
    assert all(p.requires_grad and p.is_leaf for p in destination.parameters())


def test_checkpoint_batchnorm_buffers_roundtrip():
    """Persistent buffers are model state and must survive exactly."""
    norm = BatchNorm1d(4)
    model = Sequential(Linear(3, 4), norm)
    model.train()
    x = ortho.from_numpy(
        np.arange(24, dtype=np.float32).reshape(8, 3) * 0.1 - 1.0)
    model(x)                              # moves the running statistics
    running_mean = _raw(norm.running_mean)
    running_var = _raw(norm.running_var)
    assert running_mean != _raw(ortho.zeros([4]))    # the test would be vacuous
    assert running_var != _raw(ortho.ones([4]))

    path = _checkpoint_path("bn")
    try:
        ortho.save_checkpoint(path, model=model)
        restored_norm = BatchNorm1d(4)
        restored = Sequential(Linear(3, 4), restored_norm)
        ortho.load_checkpoint(path, model=restored)
    finally:
        path.unlink(missing_ok=True)

    assert _raw(restored_norm.running_mean) == running_mean
    assert _raw(restored_norm.running_var) == running_var
    _assert_model_states_bit_identical(model, restored)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_state_dict_file_preserves_dtype_exactly(dtype):
    """dtype survives save_state_dict exactly.

    Storing ``tensor.tolist()`` with no dtype and rebuilding every tensor as
    float32 ``zeros(shape)`` brings a float64 parameter back with the float32
    rounding baked in - 1/3 restored as 0.3333333432674408.
    """
    model = Linear(3, 2)
    if dtype != "float32":
        model.cast(dtype)
    with ortho.no_grad():
        exact = ortho.from_numpy(np.full((3, 2), 1.0 / 3.0, dtype=np.float64))
        model.weight._copy_data_(exact if dtype == "float64"
                                 else ortho.cast(exact, "float32"))
    before = _raw(model.weight)

    path = _checkpoint_path("dtype_" + dtype)
    try:
        ortho.save_state_dict(model, path)
        restored = Linear(3, 2)
        if dtype != "float32":
            restored.cast(dtype)
        ortho.load_state_dict(restored, path)
    finally:
        path.unlink(missing_ok=True)

    assert restored.weight.dtype == dtype
    assert _raw(restored.weight) == before
    if dtype == "float64":
        assert restored.weight.tolist()[0] == 1.0 / 3.0


def test_checkpoint_metadata_roundtrip():
    metadata = {"epoch": 3, "step": 17, "loss": 0.25, "tag": "smoke"}
    path = _checkpoint_path("metadata")
    try:
        ortho.save_checkpoint(path, metadata=metadata)
        loaded = ortho.load_checkpoint(path)
    finally:
        path.unlink(missing_ok=True)

    assert loaded == metadata


def test_checkpoint_sgd_state_roundtrip():
    model = _make_model()
    optimizer = SGD(model.parameters(), lr=0.05, weight_decay=0.1, decay_bias=True)

    path = _checkpoint_path("sgd")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer)

        restored_model = _make_model()
        restored_optimizer = SGD(restored_model.parameters(), lr=1.0)
        metadata = ortho.load_checkpoint(
            path,
            model=restored_model,
            optimizer=restored_optimizer,
        )
    finally:
        path.unlink(missing_ok=True)

    assert metadata == {}
    assert restored_optimizer.get_lr() == pytest.approx(0.05)
    assert restored_optimizer._weight_decay == pytest.approx(0.1)
    assert restored_optimizer._decay_bias is True


def test_checkpoint_adam_state_roundtrip():
    model = _make_model()
    optimizer = Adam(model.parameters(), lr=0.01)
    _train_steps(model, optimizer, 2)

    path = _checkpoint_path("adam")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer)

        restored_model = _make_model()
        restored_optimizer = Adam(restored_model.parameters(), lr=1.0)
        ortho.load_checkpoint(path, model=restored_model, optimizer=restored_optimizer)
    finally:
        path.unlink(missing_ok=True)

    assert restored_optimizer.get_lr() == pytest.approx(0.01)
    assert restored_optimizer._t == optimizer._t
    assert restored_optimizer._eps == optimizer._eps
    assert restored_optimizer._beta1 == optimizer._beta1
    assert restored_optimizer._beta2 == optimizer._beta2
    assert restored_optimizer._weight_decay == optimizer._weight_decay
    assert restored_optimizer._decay_bias == optimizer._decay_bias
    for got, expected in zip(restored_optimizer._m, optimizer._m):
        assert got is not None
        assert _raw(got) == _raw(expected)         # exact, not approx
    for got, expected in zip(restored_optimizer._v, optimizer._v):
        assert got is not None
        assert _raw(got) == _raw(expected)


def test_checkpoint_adam_betas_stay_a_tuple():
    """The safe codec preserves the tuple/list distinction."""
    model = _make_model()
    optimizer = Adam(model.parameters(), lr=0.01, betas=(0.85, 0.995))
    path = _checkpoint_path("betas")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer)
        restored_model = _make_model()
        restored = Adam(restored_model.parameters(), lr=1.0)
        ortho.load_checkpoint(path, model=restored_model, optimizer=restored)
    finally:
        path.unlink(missing_ok=True)
    assert restored.state_dict()["betas"] == (0.85, 0.995)
    assert isinstance(restored.state_dict()["betas"], tuple)


def test_checkpoint_adam_uninitialised_moments_round_trip():
    """Adam lazily creates moments, so its state legitimately contains None."""
    model = _make_model()
    optimizer = Adam(model.parameters(), lr=0.01)
    assert optimizer.state_dict()["m"] == [None] * len(list(model.parameters()))
    path = _checkpoint_path("lazy")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer)
        restored_model = _make_model()
        restored = Adam(restored_model.parameters(), lr=1.0)
        ortho.load_checkpoint(path, model=restored_model, optimizer=restored)
    finally:
        path.unlink(missing_ok=True)
    assert restored._m == [None] * len(list(restored_model.parameters()))
    assert restored._t == 0


def test_checkpoint_resume_training_matches_continuous_adam_path():
    model = _make_model()
    optimizer = Adam(model.parameters(), lr=0.01)
    _train_steps(model, optimizer, 2)

    path = _checkpoint_path("resume")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer, metadata={"step": 2})

        _train_steps(model, optimizer, 3)

        resumed_model = _make_model()
        resumed_optimizer = Adam(resumed_model.parameters(), lr=0.01)
        metadata = ortho.load_checkpoint(
            path,
            model=resumed_model,
            optimizer=resumed_optimizer,
        )
        _train_steps(resumed_model, resumed_optimizer, 3)
    finally:
        path.unlink(missing_ok=True)

    assert metadata == {"step": 2}
    _assert_model_states_close(model, resumed_model)


def test_checkpoint_resume_is_bit_exact_and_matches_the_loss_trajectory():
    """Resume equivalence, checked on values AND on losses.

    A checkpoint that merely restores nearby values is not a checkpoint. Both
    runs are seeded identically, so continuing from the file must
    reproduce the continuous run's parameters, moments, step counter and
    per-step losses exactly.
    """
    ortho.manual_seed(1234)
    model = _make_model()
    optimizer = Adam(model.parameters(), lr=0.01)
    _train_steps(model, optimizer, 4)

    path = _checkpoint_path("bitresume")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer,
                           metadata={"step": 4})
        continuous_losses = _train_steps(model, optimizer, 5)

        ortho.manual_seed(1234)
        resumed_model = _make_model()
        resumed_optimizer = Adam(resumed_model.parameters(), lr=1.0)
        metadata = ortho.load_checkpoint(path, model=resumed_model,
                                      optimizer=resumed_optimizer)
        resumed_losses = _train_steps(resumed_model, resumed_optimizer, 5)
    finally:
        path.unlink(missing_ok=True)

    assert metadata == {"step": 4}
    assert resumed_optimizer._t == optimizer._t
    assert resumed_losses == continuous_losses
    _assert_model_states_bit_identical(model, resumed_model)
    for got, expected in zip(resumed_optimizer._m, optimizer._m):
        assert _raw(got) == _raw(expected)
    for got, expected in zip(resumed_optimizer._v, optimizer._v):
        assert _raw(got) == _raw(expected)


def test_checkpoint_sgd_resume_matches_continuous_path():
    ortho.manual_seed(99)
    model = _make_model()
    optimizer = SGD(model.parameters(), lr=0.02, weight_decay=0.01)
    _train_steps(model, optimizer, 3)

    path = _checkpoint_path("sgdresume")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer)
        continuous = _train_steps(model, optimizer, 4)

        ortho.manual_seed(99)
        resumed_model = _make_model()
        resumed_optimizer = SGD(resumed_model.parameters(), lr=1.0)
        ortho.load_checkpoint(path, model=resumed_model, optimizer=resumed_optimizer)
        resumed = _train_steps(resumed_model, resumed_optimizer, 4)
    finally:
        path.unlink(missing_ok=True)

    assert resumed == continuous
    _assert_model_states_bit_identical(model, resumed_model)


# ---- map_location ----

def test_load_state_dict_default_preserves_the_models_devices():
    """The destination is authoritative: loading never moves a model."""
    model = _make_model()
    path = _checkpoint_path("nomove")
    try:
        ortho.save_state_dict(model, path)
        destination = _make_model()
        ortho.load_state_dict(destination, path)
    finally:
        path.unlink(missing_ok=True)
    assert all(p.device.is_cpu() for p in destination.parameters())
    _assert_model_states_bit_identical(model, destination)


def test_map_location_cpu_string_is_accepted():
    model = _make_model()
    path = _checkpoint_path("maploc_cpu")
    try:
        ortho.save_state_dict(model, path)
        destination = _make_model()
        ortho.load_state_dict(destination, path, map_location="cpu")
    finally:
        path.unlink(missing_ok=True)
    assert all(p.device.is_cpu() for p in destination.parameters())


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_cpu_file_loads_into_a_cuda_model():
    model = _make_model()
    for param in model.parameters():
        param.fill_(0.375)
    path = _checkpoint_path("cpu_to_cuda")
    try:
        ortho.save_state_dict(model, path)          # saved from a CPU model
        destination = _make_model().to("cuda")
        ortho.load_state_dict(destination, path)
    finally:
        path.unlink(missing_ok=True)
    assert all(p.device.is_cuda() for p in destination.parameters())
    _assert_model_states_bit_identical(model, destination)


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_cuda_file_loads_into_a_cpu_model():
    model = _make_model()
    for param in model.parameters():
        param.fill_(0.625)          # fill_ is CPU-only; move afterwards
    model.to("cuda")
    path = _checkpoint_path("cuda_to_cpu")
    try:
        ortho.save_state_dict(model, path)
        destination = _make_model()
        ortho.load_state_dict(destination, path)
    finally:
        path.unlink(missing_ok=True)
    assert all(p.device.is_cpu() for p in destination.parameters())
    _assert_model_states_bit_identical(model, destination)


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_map_location_moves_the_model_and_the_optimizer_state():
    model = _make_model()
    optimizer = Adam(model.parameters(), lr=0.01)
    _train_steps(model, optimizer, 2)
    path = _checkpoint_path("maploc_cuda")
    try:
        ortho.save_checkpoint(path, model=model, optimizer=optimizer)
        destination = _make_model()               # starts on the CPU
        destination_optimizer = Adam(destination.parameters(), lr=1.0)
        ortho.load_checkpoint(path, model=destination,
                           optimizer=destination_optimizer,
                           map_location="cuda")
    finally:
        path.unlink(missing_ok=True)
    assert all(p.device.is_cuda() for p in destination.parameters())
    assert all(m.device.is_cuda() for m in destination_optimizer._m)
    assert all(v.device.is_cuda() for v in destination_optimizer._v)
    _assert_model_states_bit_identical(model, destination)


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_map_location_accepts_an_exact_device_object():
    model = _make_model()
    path = _checkpoint_path("maploc_obj")
    device = ortho.Device(ortho.DeviceType.CUDA, 0)
    try:
        ortho.save_state_dict(model, path)
        destination = _make_model()
        ortho.load_state_dict(destination, path, map_location=device)
    finally:
        path.unlink(missing_ok=True)
    assert all(str(p.device) == "device('cuda:0')"
               for p in destination.parameters())


try:                                        # pragma: no cover - probe only
    import torch as _torch
    _N_GPUS = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
except Exception:                           # pragma: no cover
    _N_GPUS = 0


@pytest.mark.skipif(_N_GPUS < 2, reason="needs >= 2 GPUs; this machine has %d"
                                        % _N_GPUS)
def test_map_location_to_a_second_gpu():
    """Skips honestly on single-GPU hardware rather than pretending cuda:1."""
    model = _make_model()
    path = _checkpoint_path("maploc_gpu1")
    device = ortho.Device(ortho.DeviceType.CUDA, 1)
    try:
        ortho.save_state_dict(model, path)
        destination = _make_model()
        ortho.load_state_dict(destination, path, map_location=device)
    finally:
        path.unlink(missing_ok=True)
    assert all(str(p.device) == "device('cuda:1')"
               for p in destination.parameters())


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_checkpoint_cuda_model_load_preserves_existing_device():
    model = _make_model()
    for param in model.parameters():
        param.fill_(0.5)
    model.to("cuda")

    path = _checkpoint_path("cuda")
    try:
        ortho.save_checkpoint(path, model=model)

        restored = _make_model().to("cuda")
        ortho.load_checkpoint(path, model=restored)
    finally:
        path.unlink(missing_ok=True)

    _assert_model_states_close(model, restored)
    assert all(param.device.is_cuda() for param in restored.parameters())


@pytest.mark.skipif(not CUDA, reason="CUDA not available")
def test_checkpoint_amp_optimizer_and_scaler_resume_matches_continuous_path():
    model = _make_model().to("cuda").half()
    optimizer = MixedPrecisionOptimizer(model, SGD, lr=0.01)
    scaler = GradScaler(init_scale=8.0, growth_interval=2)
    _train_amp_steps(model, optimizer, scaler, 2)

    path = _checkpoint_path("amp_resume")
    try:
        ortho.save_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            metadata={"scaler": scaler.state_dict(), "step": 2},
        )

        _train_amp_steps(model, optimizer, scaler, 2)

        resumed_model = _make_model().to("cuda").half()
        resumed_optimizer = MixedPrecisionOptimizer(resumed_model, SGD, lr=1.0)
        resumed_scaler = GradScaler(init_scale=1.0)
        metadata = ortho.load_checkpoint(
            path,
            model=resumed_model,
            optimizer=resumed_optimizer,
        )
        resumed_scaler.load_state_dict(metadata["scaler"])
        _train_amp_steps(resumed_model, resumed_optimizer, resumed_scaler, 2)
    finally:
        path.unlink(missing_ok=True)

    assert metadata["step"] == 2
    assert resumed_scaler.scale_factor == pytest.approx(scaler.scale_factor)
    _assert_model_states_close(model, resumed_model)
