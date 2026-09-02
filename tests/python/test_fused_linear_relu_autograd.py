import numpy as np
import pytest

import orthon
from orthon import nn

torch = pytest.importorskip("torch")


def _tensor(values, shape, device="cpu", requires_grad=False):
    t = orthon.from_numpy(np.array(values, dtype=np.float32).reshape(shape))
    if device == "cuda":
        if not orthon.cuda_enabled():
            pytest.skip("CUDA not enabled")
        t = t.to("cuda")
    if requires_grad:
        t.requires_grad_(True)
    return t


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fused_linear_relu_forward_backward_matches_torch(device):
    if device == "cuda" and not orthon.cuda_enabled():
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
