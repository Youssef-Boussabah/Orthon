from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest

import orthon as ortho
import orthon.functional as F
from orthon import nn

torch = pytest.importorskip("torch")
torch_F = pytest.importorskip("torch.nn.functional")




requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")


def _tensor(values, shape=None, device="cpu", requires_grad=False):
    if shape is None:
        shape = [len(values)]
    x = ortho.zeros(shape)
    flat = list(values)
    if len(shape) == 1:
        for i, value in enumerate(flat):
            x[i] = float(value)
    elif len(shape) == 2:
        cols = shape[1]
        for i, value in enumerate(flat):
            x[[i // cols, i % cols]] = float(value)
    else:
        raise ValueError("test helper only supports 1D/2D")
    if device == "cuda":
        x = x.to("cuda")
    x.requires_grad_(requires_grad)
    return x


ACTIVATIONS = [
    ("sigmoid", lambda x: ortho.sigmoid(x), torch.sigmoid, nn.Sigmoid),
    ("tanh", lambda x: ortho.tanh(x), torch.tanh, nn.Tanh),
    (
        "leaky_relu",
        lambda x: ortho.leaky_relu(x, 0.2),
        lambda x: torch_F.leaky_relu(x, negative_slope=0.2),
        lambda: nn.LeakyReLU(0.2),
    ),
    (
        "gelu",
        lambda x: ortho.gelu(x),
        lambda x: torch_F.gelu(x, approximate="none"),
        nn.GELU,
    ),
]


@pytest.mark.parametrize("name,ortho_op,torch_op,module_factory", ACTIVATIONS)
def test_activation_cpu_forward_backward_matches_torch(
    name, ortho_op, torch_op, module_factory
):
    vals = [-3.0, -1.0, -0.25, 0.0, 0.5, 2.0]
    x = _tensor(vals, requires_grad=True)
    y = ortho_op(x)
    y.backward()

    tx = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
    ty = torch_op(tx)
    ty.sum().backward()

    assert y.tolist() == pytest.approx(ty.detach().tolist(), rel=1e-5, abs=1e-5)
    assert x.grad.tolist() == pytest.approx(tx.grad.tolist(), rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("name,ortho_op,torch_op,module_factory", ACTIVATIONS)
@requires_cuda
def test_activation_cuda_forward_backward_matches_torch(
    name, ortho_op, torch_op, module_factory
):
    vals = [-3.0, -1.0, -0.25, 0.0, 0.5, 2.0]
    x = _tensor(vals, device="cuda", requires_grad=True)
    y = ortho_op(x)
    y.backward()

    tx = torch.tensor(vals, dtype=torch.float32, requires_grad=True)
    ty = torch_op(tx)
    ty.sum().backward()

    assert y.to("cpu").tolist() == pytest.approx(
        ty.detach().tolist(), rel=1e-5, abs=1e-5
    )
    assert x.grad.device.is_cuda()
    assert x.grad.to("cpu").tolist() == pytest.approx(
        tx.grad.tolist(), rel=1e-5, abs=1e-5
    )


@pytest.mark.parametrize("name,ortho_op,torch_op,module_factory", ACTIVATIONS)
def test_activation_modules_and_functional_api(name, ortho_op, torch_op, module_factory):
    vals = [-1.0, 0.0, 2.0]
    x = _tensor(vals)
    module = module_factory()

    module_result = module(x).tolist()
    functional = getattr(F, name)
    if name == "leaky_relu":
        functional_result = functional(x, negative_slope=0.2).tolist()
    else:
        functional_result = functional(x).tolist()

    assert module_result == pytest.approx(ortho_op(x).tolist())
    assert functional_result == pytest.approx(ortho_op(x).tolist())


def test_activation_no_grad_outputs_are_plain_tensors():
    x = _tensor([1.0, -1.0], requires_grad=True)
    with ortho.no_grad():
        y = ortho.sigmoid(x)
    assert not y.requires_grad
    assert not y._has_grad_fn


def test_activation_cpu_non_contiguous_forward_backward():
    base = _tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], shape=[2, 3])
    view = base.transpose(0, 1)
    view.requires_grad_(True)
    y = ortho.tanh(view)
    y.backward()

    tx = torch.tensor(
        [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=torch.float32
    ).t()
    tx = tx.detach().clone().requires_grad_(True)
    torch.tanh(tx).sum().backward()

    assert y.tolist() == pytest.approx(torch.tanh(tx).detach().flatten().tolist())
    # Explicit float32 tolerance: tanh'(x) = 1 - tanh(x)^2 is ~2.5e-05 at
    # x = -6, where approx's bare rel=1e-6 demands sub-float32-ulp agreement.
    # Different libm tanhf roundings (glibc vs MSVC vs SLEEF) differ by ~1 ulp,
    # measured at 3e-08 absolute on Linux/GCC 13.3 against the torch reference.
    assert view.grad.tolist() == pytest.approx(
        tx.grad.flatten().tolist(), rel=1e-5, abs=1e-6)
