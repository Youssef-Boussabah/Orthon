"""BatchNorm persistent running statistics.

Scope is BatchNorm *model state*: registration, enumeration, state_dict,
load_state_dict, device/dtype movement and checkpoint round-trips.

These tests cover model state only. Numerical and autograd correctness is
asserted separately against PyTorch in test_batchnorm_parity.py.
"""
import os
import tempfile

import numpy as np
import pytest

import orthon as ortho
from orthon.nn import BatchNorm1d, BatchNorm2d, Linear, Sequential
from orthon.nn.module import Module

CUDA = ortho.cuda_enabled()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

X1 = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.5
X2 = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) * 0.3


def _t(arr):
    return ortho.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _np(t):
    return np.array(t.tolist(), dtype=np.float64).reshape(tuple(t.shape))


class TinyNet(Module):
    """Model mixing trainable parameters with BatchNorm buffers."""

    def __init__(self):
        super().__init__()
        self.fc = Linear(3, 3)
        self.bn = BatchNorm1d(3)

    def forward(self, x):
        return self.bn(self.fc(x))


# ---- buffer registration ----

@pytest.mark.parametrize("factory", [lambda: BatchNorm1d(3), lambda: BatchNorm2d(2)])
def test_running_stats_are_registered_buffers(factory):
    bn = factory()
    assert "running_mean" in bn._buffers
    assert "running_var" in bn._buffers


@pytest.mark.parametrize("factory", [lambda: BatchNorm1d(3), lambda: BatchNorm2d(2)])
def test_running_stats_are_tensors(factory):
    bn = factory()
    assert isinstance(bn.running_mean, ortho.Tensor)
    assert isinstance(bn.running_var, ortho.Tensor)


def test_running_stats_initial_values():
    bn = BatchNorm1d(3)
    assert bn.running_mean.tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert bn.running_var.tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert list(bn.running_mean.shape) == [3]
    assert list(bn.running_var.shape) == [3]


def test_running_stats_are_persistent():
    bn = BatchNorm1d(3)
    assert bn._non_persistent_buffers == set()
    assert "running_mean" in bn.state_dict()
    assert "running_var" in bn.state_dict()


def test_running_stats_are_not_parameters():
    bn = BatchNorm1d(3)
    assert len(list(bn.parameters())) == 2          # weight, bias only
    names = [n for n, _ in bn.named_parameters()]
    assert names == ["weight", "bias"]
    ids = {id(p) for p in bn.parameters()}
    assert id(bn.running_mean) not in ids
    assert id(bn.running_var) not in ids


def test_running_stats_do_not_require_grad():
    bn = BatchNorm1d(3)
    assert bn.running_mean.requires_grad is False
    assert bn.running_var.requires_grad is False


def test_running_stats_in_named_buffers():
    bn = BatchNorm1d(3)
    assert [n for n, _ in bn.named_buffers()] == ["running_mean", "running_var"]
    assert len(list(bn.buffers())) == 2


def test_optimizer_does_not_receive_running_stats():
    from orthon.optim import SGD
    bn = BatchNorm1d(3)
    opt = SGD(bn.parameters(), lr=0.1)
    assert len(opt._params) == 2


# ---- no num_batches_tracked, no unrelated state ----

@pytest.mark.parametrize("factory", [lambda: BatchNorm1d(3), lambda: BatchNorm2d(2)])
def test_no_num_batches_tracked_buffer(factory):
    bn = factory()
    assert "num_batches_tracked" not in bn._buffers
    assert "num_batches_tracked" not in bn.state_dict()


def test_state_dict_has_no_unexpected_entries():
    assert sorted(BatchNorm1d(3).state_dict().keys()) == [
        "bias", "running_mean", "running_var", "weight",
    ]


def test_training_does_not_add_state():
    bn = BatchNorm1d(3)
    before = sorted(bn.state_dict().keys())
    bn.train()
    for _ in range(3):
        bn(_t(X1))
    assert sorted(bn.state_dict().keys()) == before
    assert [n for n, _ in bn.named_buffers()] == ["running_mean", "running_var"]


# ---- state_dict ----

def test_state_dict_keys_affine_true():
    assert sorted(BatchNorm1d(3).state_dict().keys()) == [
        "bias", "running_mean", "running_var", "weight",
    ]


def test_state_dict_order_matches_module_contract():
    """Parameters first, then buffers, in registration order."""
    assert list(BatchNorm1d(3).state_dict().keys()) == [
        "weight", "bias", "running_mean", "running_var",
    ]


def test_state_dict_keys_affine_false():
    bn = BatchNorm1d(3, affine=False)
    assert sorted(bn.state_dict().keys()) == ["running_mean", "running_var"]
    assert len(list(bn.parameters())) == 0


def test_affine_false_still_persists_running_stats():
    bn = BatchNorm2d(2, affine=False)
    bn.train()
    bn(_t(X2))
    sd = bn.state_dict()
    assert set(sd.keys()) == {"running_mean", "running_var"}
    assert all(isinstance(v, ortho.Tensor) for v in sd.values())


def test_state_dict_values_are_tensors():
    for v in BatchNorm1d(3).state_dict().values():
        assert isinstance(v, ortho.Tensor)


def test_nested_state_dict_keys():
    assert sorted(TinyNet().state_dict().keys()) == [
        "bn.bias", "bn.running_mean", "bn.running_var", "bn.weight",
        "fc.bias", "fc.weight",
    ]


def test_sequential_state_dict_keys():
    model = Sequential(Linear(3, 3), BatchNorm1d(3))
    assert sorted(model.state_dict().keys()) == [
        "0.bias", "0.weight",
        "1.bias", "1.running_mean", "1.running_var", "1.weight",
    ]


# ---- training keeps buffers registered and on-device ----

def test_training_updates_running_stats():
    bn = BatchNorm1d(3)
    bn.train()
    bn(_t(X1))
    # momentum 0.1 against batch mean of X1 columns
    expected_mean = 0.1 * X1.mean(axis=0)
    assert bn.running_mean.tolist() == pytest.approx(expected_mean.tolist(), abs=1e-6)


def test_training_update_keeps_buffer_registered():
    bn = BatchNorm1d(3)
    bn.train()
    bn(_t(X1))
    assert "running_mean" in bn._buffers
    assert isinstance(bn.running_mean, ortho.Tensor)
    assert "running_mean" in bn.state_dict()
    assert bn._non_persistent_buffers == set()


def test_training_update_does_not_create_parameters():
    bn = BatchNorm1d(3)
    bn.train()
    bn(_t(X1))
    assert len(list(bn.parameters())) == 2
    assert bn.running_mean.requires_grad is False


def test_repeated_training_accumulates_momentum():
    bn = BatchNorm1d(3)
    bn.train()
    m = X1.mean(axis=0)
    bn(_t(X1))
    bn(_t(X1))
    expected = 0.1 * m + 0.9 * (0.1 * m)
    assert bn.running_mean.tolist() == pytest.approx(expected.tolist(), abs=1e-6)


# ---- eval uses registered state ----

def test_eval_uses_registered_running_stats():
    bn = BatchNorm1d(3, affine=False)
    bn.running_mean = _t(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    bn.running_var = _t(np.array([4.0, 9.0, 16.0], dtype=np.float32))
    bn.eval()
    y = _np(bn(_t(X1)))
    expected = (X1 - np.array([1.0, 2.0, 3.0])) / np.sqrt(
        np.array([4.0, 9.0, 16.0]) + 1e-5)
    assert y.ravel() == pytest.approx(expected.ravel(), abs=1e-5)


def test_eval_after_training_uses_updated_stats():
    bn = BatchNorm1d(3, affine=False)
    bn.train()
    bn(_t(X1))
    rm = np.array(bn.running_mean.tolist(), dtype=np.float64)
    rv = np.array(bn.running_var.tolist(), dtype=np.float64)
    bn.eval()
    y = _np(bn(_t(X1)))
    expected = (X1 - rm) / np.sqrt(rv + 1e-5)
    assert y.ravel() == pytest.approx(expected.ravel(), abs=1e-5)


# ---- device and dtype ----

def test_to_cpu_keeps_running_stats():
    bn = BatchNorm1d(3).to("cpu")
    assert bn.running_mean.device.is_cpu()
    assert bn.running_var.device.is_cpu()
    assert isinstance(bn.running_mean, ortho.Tensor)


def test_cast_casts_running_stats():
    bn = BatchNorm1d(3).cast("float16")
    assert bn.running_mean.dtype == "float16"
    assert bn.running_var.dtype == "float16"


def test_half_and_bfloat16_cast_running_stats():
    assert BatchNorm1d(3).half().running_mean.dtype == "float16"
    assert BatchNorm1d(3).bfloat16().running_var.dtype == "bfloat16"


def test_cast_keeps_running_stats_registered():
    bn = BatchNorm1d(3).half()
    assert "running_mean" in bn._buffers
    assert "running_mean" in bn.state_dict()


@requires_cuda
def test_to_cuda_moves_running_stats():
    bn = BatchNorm1d(3).to("cuda")
    assert bn.running_mean.device.is_cuda()
    assert bn.running_var.device.is_cuda()


@requires_cuda
def test_cuda_roundtrip_preserves_running_stats():
    bn = BatchNorm1d(3)
    bn.train()
    bn(_t(X1))
    before_mean = bn.running_mean.tolist()
    before_var = bn.running_var.tolist()
    bn.to("cuda")
    assert bn.running_mean.device.is_cuda()
    bn.to("cpu")
    assert bn.running_mean.device.is_cpu()
    assert bn.running_mean.tolist() == pytest.approx(before_mean, abs=1e-6)
    assert bn.running_var.tolist() == pytest.approx(before_var, abs=1e-6)


@requires_cuda
def test_cuda_training_keeps_stats_on_device():
    bn = BatchNorm1d(3).to("cuda")
    bn.train()
    bn(_t(X1).to("cuda"))
    assert bn.running_mean.device.is_cuda()
    assert bn.running_var.device.is_cuda()
    assert "running_mean" in bn.state_dict()


@requires_cuda
def test_cuda_eval_matches_cpu_eval():
    cpu = BatchNorm1d(3, affine=False)
    cpu.train()
    cpu(_t(X1))
    cpu.eval()
    expected = _np(cpu(_t(X1)))

    gpu = BatchNorm1d(3, affine=False)
    gpu.train()
    gpu(_t(X1))
    gpu.eval()
    gpu.to("cuda")
    got = _np(gpu(_t(X1).to("cuda")).to("cpu"))
    assert got.ravel() == pytest.approx(expected.ravel(), abs=1e-5)


@requires_cuda
def test_cuda_cast_moves_and_casts_running_stats():
    bn = BatchNorm1d(3).to("cuda").half()
    assert bn.running_mean.device.is_cuda()
    assert bn.running_mean.dtype == "float16"


# ---- load_state_dict ----

def test_load_state_dict_restores_running_stats():
    src, dst = BatchNorm1d(3), BatchNorm1d(3)
    src.train()
    src(_t(X1))
    dst.load_state_dict(src.state_dict())
    assert dst.running_mean.tolist() == pytest.approx(src.running_mean.tolist())
    assert dst.running_var.tolist() == pytest.approx(src.running_var.tolist())


def test_load_state_dict_buffers_are_independent():
    src, dst = BatchNorm1d(3), BatchNorm1d(3)
    src.train()
    src(_t(X1))
    dst.load_state_dict(src.state_dict())
    assert dst.running_mean is not src.running_mean
    assert dst.running_var is not src.running_var


def test_mutating_destination_stats_does_not_affect_source():
    src, dst = BatchNorm1d(3), BatchNorm1d(3)
    src.train()
    src(_t(X1))
    original = src.running_mean.tolist()
    dst.load_state_dict(src.state_dict())
    dst.running_mean.fill_(99.0)
    assert src.running_mean.tolist() == pytest.approx(original)


def test_further_training_of_source_does_not_affect_destination():
    src, dst = BatchNorm1d(3), BatchNorm1d(3)
    src.train()
    src(_t(X1))
    dst.load_state_dict(src.state_dict())
    loaded = dst.running_mean.tolist()
    for _ in range(3):
        src(_t(X1 + 5.0))
    assert dst.running_mean.tolist() == pytest.approx(loaded)


def test_load_state_dict_keeps_registration_and_persistence():
    src, dst = BatchNorm1d(3), BatchNorm1d(3)
    dst.load_state_dict(src.state_dict())
    assert "running_mean" in dst._buffers
    assert "running_var" in dst._buffers
    assert dst._non_persistent_buffers == set()
    assert [n for n, _ in dst.named_buffers()] == ["running_mean", "running_var"]


def test_load_state_dict_preserves_destination_dtype():
    src = BatchNorm1d(3)
    src.train()
    src(_t(X1))
    dst = BatchNorm1d(3).half()
    dst.load_state_dict(src.state_dict())
    assert dst.running_mean.dtype == "float16"
    assert dst.running_mean.to("cpu").tolist() == pytest.approx(
        src.running_mean.tolist(), abs=1e-2)


def test_load_state_dict_affine_false_roundtrip():
    src, dst = BatchNorm1d(3, affine=False), BatchNorm1d(3, affine=False)
    src.train()
    src(_t(X1))
    dst.load_state_dict(src.state_dict())
    assert dst.running_mean.tolist() == pytest.approx(src.running_mean.tolist())


def test_load_state_dict_strict_detects_missing_running_stats():
    bn = BatchNorm1d(3)
    sd = bn.state_dict()
    del sd["running_mean"]
    with pytest.raises(KeyError):
        bn.load_state_dict(sd, strict=True)


@requires_cuda
def test_load_state_dict_preserves_destination_device():
    """affine=False isolates the buffer path.

    With affine=True this same CPU->CUDA load is blocked before it reaches the
    buffers by a separate, pre-existing defect in the float32 *parameter* path
    of Module.load_state_dict (in-place add_ without device reconciliation).
    """
    src = BatchNorm1d(3, affine=False)
    src.train()
    src(_t(X1))
    dst = BatchNorm1d(3, affine=False).to("cuda")
    dst.load_state_dict(src.state_dict())
    assert dst.running_mean.device.is_cuda()
    assert dst.running_var.device.is_cuda()
    assert dst.running_mean.to("cpu").tolist() == pytest.approx(
        src.running_mean.tolist(), abs=1e-6)


# ---- checkpoint round trip ----

def _trained_net():
    net = TinyNet()
    net.train()
    for _ in range(3):
        net(_t(X1))
    return net


def test_checkpoint_roundtrip_preserves_running_stats():
    net = _trained_net()
    mean_before = net.bn.running_mean.tolist()
    var_before = net.bn.running_var.tolist()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.ckpt")
        ortho.save_checkpoint(path, model=net)
        restored = TinyNet()
        ortho.load_checkpoint(path, model=restored)
    assert restored.bn.running_mean.tolist() == pytest.approx(mean_before, abs=1e-6)
    assert restored.bn.running_var.tolist() == pytest.approx(var_before, abs=1e-6)


def test_checkpoint_roundtrip_preserves_eval_output():
    net = _trained_net()
    net.eval()
    expected = _np(net(_t(X1)))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.ckpt")
        ortho.save_checkpoint(path, model=net)
        restored = TinyNet()
        ortho.load_checkpoint(path, model=restored)
    restored.eval()
    assert _np(restored(_t(X1))).ravel() == pytest.approx(expected.ravel(), abs=1e-6)


def test_checkpoint_roundtrip_affine_false():
    bn = BatchNorm1d(3, affine=False)
    bn.train()
    bn(_t(X1))
    before = bn.running_mean.tolist()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.ckpt")
        ortho.save_checkpoint(path, model=bn)
        restored = BatchNorm1d(3, affine=False)
        ortho.load_checkpoint(path, model=restored)
    assert restored.running_mean.tolist() == pytest.approx(before, abs=1e-6)


@requires_cuda
def test_checkpoint_roundtrip_map_location_cuda():
    net = _trained_net()
    mean_before = net.bn.running_mean.tolist()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ck.ckpt")
        ortho.save_checkpoint(path, model=net)
        restored = TinyNet()
        ortho.load_checkpoint(path, model=restored, map_location="cuda")
    assert restored.bn.running_mean.device.is_cuda()
    assert restored.bn.running_mean.to("cpu").tolist() == pytest.approx(
        mean_before, abs=1e-6)


# ---- save_state_dict / load_state_dict round trip ----

def test_legacy_save_load_state_dict_preserves_running_stats():
    """save_state_dict/load_state_dict is float32-only by format; that suffices."""
    net = _trained_net()
    mean_before = net.bn.running_mean.tolist()
    var_before = net.bn.running_var.tolist()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sd.ckpt")
        ortho.save_state_dict(net, path)
        restored = TinyNet()
        ortho.load_state_dict(restored, path)
    assert restored.bn.running_mean.tolist() == pytest.approx(mean_before, abs=1e-6)
    assert restored.bn.running_var.tolist() == pytest.approx(var_before, abs=1e-6)


def test_legacy_save_load_preserves_eval_output():
    net = _trained_net()
    net.eval()
    expected = _np(net(_t(X1)))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sd.ckpt")
        ortho.save_state_dict(net, path)
        restored = TinyNet()
        ortho.load_state_dict(restored, path)
    restored.eval()
    assert _np(restored(_t(X1))).ravel() == pytest.approx(expected.ravel(), abs=1e-6)


# ---- Numerical freeze ----

def test_bn1d_training_forward_and_running_stats():
    """Normalisation uses biased variance; running_var tracks the unbiased one.

    See test_running_var_split below for the same split stated directly.
    """
    bn = BatchNorm1d(3)
    bn.train()
    y = _np(bn(_t(X1)))
    expected_mean = X1.mean(axis=0)
    biased_var = X1.var(axis=0)                    # ddof=0, used to normalise
    unbiased_var = X1.var(axis=0, ddof=1)          # ddof=1, used for running_var
    expected = (X1 - expected_mean) / np.sqrt(biased_var + 1e-5)
    assert y.ravel() == pytest.approx(expected.ravel(), abs=1e-5)
    assert bn.running_mean.tolist() == pytest.approx(
        (0.1 * expected_mean).tolist(), abs=1e-6)
    assert bn.running_var.tolist() == pytest.approx(
        (0.9 + 0.1 * unbiased_var).tolist(), abs=1e-6)


def test_bn2d_training_forward_and_running_stats():
    bn = BatchNorm2d(2)
    bn.train()
    y = _np(bn(_t(X2)))
    axes = (0, 2, 3)
    m = X2.mean(axis=axes)
    biased_var = X2.var(axis=axes)
    unbiased_var = X2.var(axis=axes, ddof=1)
    expected = (X2 - m.reshape(1, 2, 1, 1)) / np.sqrt(
        biased_var.reshape(1, 2, 1, 1) + 1e-5)
    assert y.ravel() == pytest.approx(expected.ravel(), abs=1e-5)
    assert bn.running_mean.tolist() == pytest.approx((0.1 * m).tolist(), abs=1e-6)
    assert bn.running_var.tolist() == pytest.approx(
        (0.9 + 0.1 * unbiased_var).tolist(), abs=1e-6)


def test_running_var_split_biased_normalisation_unbiased_running():
    """PyTorch's split: biased variance normalises, unbiased is tracked.

    Using the biased variance for the running statistic too would be a silent
    divergence from PyTorch.
    """
    bn = BatchNorm1d(3)
    bn.train()
    y = _np(bn(_t(X1)))
    biased = X1.var(axis=0)
    unbiased = X1.var(axis=0, ddof=1)
    assert biased != pytest.approx(unbiased)       # the two genuinely differ
    # normalisation divides by sqrt(biased + eps)
    assert y.ravel() == pytest.approx(
        ((X1 - X1.mean(axis=0)) / np.sqrt(biased + 1e-5)).ravel(), abs=1e-5)
    # running_var accumulates the unbiased estimate
    assert bn.running_var.tolist() == pytest.approx(
        (0.9 + 0.1 * unbiased).tolist(), abs=1e-6)
    assert bn.running_var.tolist() != pytest.approx(
        (0.9 + 0.1 * biased).tolist(), abs=1e-6)


def test_backward_populates_grads():
    """Gradients exist. Numerical correctness is asserted in the parity tests.

    The training path is graph-connected and verified against PyTorch in
    tests/python/test_batchnorm_parity.py.
    """
    x = _t(X1)
    x.requires_grad_(True)
    bn = BatchNorm1d(3)
    bn.train()
    ortho.sum(bn(x)).backward()
    assert x.grad is not None
    assert bn.weight.grad is not None
    assert bn.bias.grad is not None
