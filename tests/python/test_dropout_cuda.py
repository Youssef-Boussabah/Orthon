from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest

import orthon
from orthon import nn




requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


@requires_cuda
def test_cuda_dropout_output_stays_cuda_and_scales():
    x = orthon.ones([4096]).to("cuda")
    drop = nn.Dropout(0.25, seed=123)
    y = drop(x)
    values = y.to("cpu").tolist()
    dropped = sum(1 for v in values if v == 0.0)
    kept = len(values) - dropped

    assert y.device.is_cuda()
    assert 0.68 <= kept / len(values) <= 0.82
    assert max(values) == pytest.approx(1.0 / 0.75)


@requires_cuda
def test_cuda_dropout_backward_masks_and_scales_gradients():
    x = orthon.ones([2048]).to("cuda")
    x.requires_grad_(True)
    drop = nn.Dropout(0.4, seed=99)
    y = drop(x)
    y.backward()

    values = y.to("cpu").tolist()
    grads = x.grad.to("cpu").tolist()
    assert x.grad.device.is_cuda()
    assert grads == pytest.approx(values)
    assert set(round(v, 6) for v in grads).issubset({0.0, round(1.0 / 0.6, 6)})


@requires_cuda
def test_cuda_dropout_fixed_seed_reproducible_first_call():
    x = orthon.ones([512]).to("cuda")
    a = nn.Dropout(0.35, seed=7)(x).to("cpu").tolist()
    b = nn.Dropout(0.35, seed=7)(x).to("cpu").tolist()
    assert a == pytest.approx(b)


@requires_cuda
def test_cuda_dropout_repeated_calls_advance_masks():
    x = orthon.ones([512]).to("cuda")
    drop = nn.Dropout(0.35, seed=7)
    first = drop(x).to("cpu").tolist()
    second = drop(x).to("cpu").tolist()
    assert first != pytest.approx(second)


@requires_cuda
def test_cuda_dropout_eval_is_identity():
    x = orthon.ones([16]).to("cuda")
    x.requires_grad_(True)
    drop = nn.Dropout(0.9, seed=1).eval()
    y = drop(x)
    assert y is x


@requires_cuda
def test_cuda_dropout_large_tensor_kept_fraction():
    x = orthon.ones([1024, 1024]).to("cuda")
    p = 0.3
    values = nn.Dropout(p, seed=42)(x).to("cpu").tolist()
    kept = sum(1 for v in values if v != 0.0)
    assert abs(kept / len(values) - (1.0 - p)) < 0.02


@requires_cuda
def test_cuda_dropout_forward_does_not_materialize_host_values(monkeypatch, capfd):
    monkeypatch.setenv("ORTHON_DEBUG_DEVICE", "1")
    x = orthon.ones([1024]).to("cuda")
    y = nn.Dropout(0.5, seed=11)(x)
    orthon.cuda_synchronize()
    assert y.device.is_cuda()
    captured = capfd.readouterr()
    assert "Tensor.tolist" not in captured.err + captured.out
    assert "Tensor.item" not in captured.err + captured.out
