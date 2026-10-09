from tests.support.capabilities import orthon_cuda_available
import numpy as np
import pytest

import orthon
from orthon import nn

torch = pytest.importorskip("torch")


def _tensor(values, shape, device="cpu", requires_grad=False):
    t = orthon.from_numpy(np.array(values, dtype=np.float32).reshape(shape))
    if device == "cuda":
        if not orthon_cuda_available():
            pytest.skip("CUDA not enabled")
        t = t.to("cuda")
    if requires_grad:
        t.requires_grad_(True)
    return t


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fused_linear_relu_forward_backward_matches_torch(device):
    if device == "cuda" and not orthon_cuda_available():
        pytest.skip("CUDA not enabled")

    x_vals = [[-0.2, 0.4, 1.0], [1.5, -0.5, 0.25]]
    w_vals = [[0.3, -0.7], [0.2, 0.5], [-0.4, 0.1]]
    b_vals = [0.05, -0.1]
    x = _tensor(x_vals, (2, 3), device, True)
    w = _tensor(w_vals, (3, 2), device, True)
    b = _tensor(b_vals, (2,), device, True)

    y = orthon.fused_linear_relu(x, w, b)
    assert y.requires_grad
    assert y._has_grad_fn
    y.backward()

    torch_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    tx = torch.tensor(x_vals, dtype=torch.float32, device=torch_device, requires_grad=True)
    tw = torch.tensor(w_vals, dtype=torch.float32, device=torch_device, requires_grad=True)
    tb = torch.tensor(b_vals, dtype=torch.float32, device=torch_device, requires_grad=True)
    ty = torch.relu(tx @ tw + tb)
    ty.sum().backward()

    assert y.to("cpu").tolist() == pytest.approx(ty.detach().cpu().flatten().tolist(), abs=1e-5)
    assert x.grad.to("cpu").tolist() == pytest.approx(tx.grad.cpu().flatten().tolist(), abs=1e-5)
    assert w.grad.to("cpu").tolist() == pytest.approx(tw.grad.cpu().flatten().tolist(), abs=1e-5)
    assert b.grad.to("cpu").tolist() == pytest.approx(tb.grad.cpu().flatten().tolist(), abs=1e-5)


def test_fused_linear_relu_module_trains_in_small_mlp():
    model = nn.Sequential(nn.FusedLinearReLU(3, 4), nn.Linear(4, 2))
    x = _tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.1]], (2, 3), requires_grad=True)
    out = model(x)
    assert out.requires_grad
    out.backward()
    for p in model.parameters():
        assert p.grad is not None


def _r2_fused_inputs(device, tracked=True):
    return (
        _tensor([[-.2, .4, 1.], [1.5, -.5, .25]], (2, 3), device, tracked),
        _tensor([[.3, -.7], [.2, .5], [-.4, .1]], (3, 2), device, tracked),
        _tensor([.05, -.1], (2,), device, tracked),
    )


def _r2_fused_reference(inputs):
    x, w, b = [np.asarray(t.to("cpu").tolist()).reshape(t.shape) for t in inputs]
    mask = (x @ w + b > 0).astype(np.float64)
    return mask @ w.T, x.T @ mask, mask.sum(axis=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_r2_fused_released_root_remains_nonleaf(device):
    inputs = _r2_fused_inputs(device)
    expected = _r2_fused_reference(inputs)
    output = orthon.fused_linear_relu(*inputs)
    assert output.requires_grad and not output.is_leaf and output._has_grad_fn
    output.backward()
    assert not output.is_leaf and not output._has_grad_fn and output.grad is None
    for value, reference in zip(inputs, expected):
        np.testing.assert_allclose(value.grad.to("cpu").tolist(), reference.ravel(), atol=2e-6, rtol=2e-5)
    snapshots = [value.grad.to("cpu").tolist() for value in inputs]
    with pytest.raises(RuntimeError, match="freed graph"):
        output.backward()
    assert output.grad is None
    assert [value.grad.to("cpu").tolist() for value in inputs] == snapshots


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_r2_fused_retained_root_accumulates_then_releases(device):
    inputs = _r2_fused_inputs(device)
    output = orthon.fused_linear_relu(*inputs)
    for _ in range(2):
        output.backward(retain_graph=True)
        assert output._has_grad_fn and not output.is_leaf
    for value, reference in zip(inputs, _r2_fused_reference(inputs)):
        np.testing.assert_allclose(value.grad.to("cpu").tolist(), (2 * reference).ravel(), atol=2e-6, rtol=2e-5)
    output.backward()
    assert not output._has_grad_fn and output.grad is None
    with pytest.raises(RuntimeError, match="freed graph"):
        output.backward(retain_graph=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_r2_shared_fused_branches_match_unfused_gradients(device):
    fused_inputs = _r2_fused_inputs(device)
    plain_inputs = _r2_fused_inputs(device)
    fused = orthon.fused_linear_relu(*fused_inputs)
    x, w, b = plain_inputs
    plain = orthon.relu(orthon.matmul(x, w) + b)
    for output in [fused, plain]:
        (output + output * 2).sum().backward()
    for actual, reference in zip(fused_inputs, plain_inputs):
        np.testing.assert_allclose(actual.grad.to("cpu").tolist(), reference.grad.to("cpu").tolist(), atol=2e-6, rtol=2e-5)
    assert fused.grad is None and not fused._has_grad_fn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("operand", [0, 1, 2])
def test_r2_fused_saved_operand_mutation_preserves_history_on_error(device, operand):
    inputs = _r2_fused_inputs(device)
    output = orthon.fused_linear_relu(*inputs)
    inputs[operand].mul_(2)
    with pytest.raises(RuntimeError, match="modified in-place"):
        output.backward()
    assert output._has_grad_fn and not output.is_leaf and output.grad is None
    assert all(value.grad is None for value in inputs)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_r2_fused_untracked_outputs_are_plain_leaves(device):
    tracked = _r2_fused_inputs(device)
    with orthon.no_grad():
        output = orthon.fused_linear_relu(*tracked)
    assert not output.requires_grad and output.is_leaf and not output._has_grad_fn
    output = orthon.fused_linear_relu(*_r2_fused_inputs(device, False))
    assert not output.requires_grad and output.is_leaf and not output._has_grad_fn


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("operand", [0, 1, 2])
def test_r2_fused_tracks_each_required_operand_independently(device, operand):
    inputs = _r2_fused_inputs(device, False)
    inputs[operand].requires_grad_(True)
    output = orthon.fused_linear_relu(*inputs)
    assert output.requires_grad and not output.is_leaf and output._has_grad_fn
    output.backward()
    expected = _r2_fused_reference(inputs)[operand]
    np.testing.assert_allclose(inputs[operand].grad.to("cpu").tolist(), expected.ravel(), atol=2e-6, rtol=2e-5)
    assert all(value.grad is None for i, value in enumerate(inputs) if i != operand)
