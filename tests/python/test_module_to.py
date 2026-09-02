"""Tests for Module.to(device)."""
import pytest
import orthon
from orthon.nn import Linear, ReLU, Sequential


def requires_cuda(fn):
    """Decorator: skip if CUDA is not compiled in."""
    return pytest.mark.skipif(
        not orthon.cuda_enabled(),
        reason="CUDA not available"
    )(fn)


# CPU round-trip without an actual device change.

def test_to_cpu_returns_self():
    model = Linear(4, 2)
    result = model.to("cpu")
    assert result is model


def test_to_cpu_preserves_params_on_cpu():
    model = Linear(4, 2)
    model.to("cpu")
    for p in model.parameters():
        assert p.device.is_cpu()


def test_to_cpu_params_still_require_grad():
    model = Linear(4, 2)
    model.to("cpu")
    for p in model.parameters():
        assert p.requires_grad


def test_sequential_to_cpu_moves_all_params():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    model.to("cpu")
    for p in model.parameters():
        assert p.device.is_cpu()
        assert p.requires_grad


def test_to_preserves_param_count():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    before = len(list(model.parameters()))
    model.to("cpu")
    after = len(list(model.parameters()))
    assert before == after


def test_to_preserves_forward_output_shape():
    model = Linear(4, 2)
    model.to("cpu")
    x = orthon.full([3, 4], 1.0)
    assert model(x).shape == (3, 2)


# CUDA tests, skipped without a CUDA build.

@requires_cuda
def test_to_cuda_moves_params():
    model = Linear(4, 2)
    model.to("cuda")
    for p in model.parameters():
        assert p.device.is_cuda()
        assert p.requires_grad


@requires_cuda
def test_sequential_to_cuda_and_back():
    model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    model.to("cuda").to("cpu")
    for p in model.parameters():
        assert p.device.is_cpu()
        assert p.requires_grad
