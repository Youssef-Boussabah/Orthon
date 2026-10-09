from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest

import orthon as ortho
import orthon.functional as F
import orthon.nn as nn

torch = pytest.importorskip("torch")




requires_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and _cuda_available()), reason="CUDA not available")


def _tensor_from_values(values, shape, device="cpu", requires_grad=False):
    x = ortho.zeros(list(shape))
    ndim = len(shape)
    strides = [1] * ndim
    for dim in range(ndim - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]

    for flat, value in enumerate(values):
        rem = flat
        idx = []
        for stride in strides:
            idx.append(rem // stride)
            rem %= stride
        x[idx] = float(value)

    if device == "cuda":
        x = x.to("cuda")
    x.requires_grad_(requires_grad)
    return x


def _torch_tensor(values, shape, device="cpu", requires_grad=False):
    x = torch.tensor(values, dtype=torch.float32, device=device).reshape(*shape)
    if requires_grad:
        x = x.clone().detach().requires_grad_(True)
    return x


def _assert_tensor_matches_torch(actual, expected, rel=1e-4, abs=1e-4):
    if actual.device.is_cuda():
        actual = actual.to("cpu")
    assert list(actual.shape) == list(expected.shape)
    assert actual.tolist() == pytest.approx(
        expected.detach().cpu().flatten().tolist(), rel=rel, abs=abs
    )


@pytest.mark.parametrize(
    "shape, normalized_shape",
    [
        ((4, 8), 8),
        ((2, 3, 4), 4),
        ((2, 3, 4), (3, 4)),
    ],
)
@pytest.mark.parametrize("affine", [True, False])
def test_layernorm_cpu_forward_matches_torch(shape, normalized_shape, affine):
    values = [float(i - 7) / 5.0 for i in range(torch.tensor(shape).prod().item())]
    x = _tensor_from_values(values, shape)
    layer = nn.LayerNorm(normalized_shape, elementwise_affine=affine)
    y = layer(x)

    tx = _torch_tensor(values, shape)
    tlayer = torch.nn.LayerNorm(normalized_shape, elementwise_affine=affine)
    ty = tlayer(tx)

    _assert_tensor_matches_torch(y, ty)


@pytest.mark.parametrize("normalized_shape", [8, (3, 4)])
def test_layernorm_cpu_backward_matches_torch(normalized_shape):
    shape = (4, 8) if normalized_shape == 8 else (2, 3, 4)
    values = [float(i - 11) / 7.0 for i in range(torch.tensor(shape).prod().item())]
    x = _tensor_from_values(values, shape, requires_grad=True)
    layer = nn.LayerNorm(normalized_shape)
    layer(x).sum().backward()

    tx = _torch_tensor(values, shape, requires_grad=True)
    tlayer = torch.nn.LayerNorm(normalized_shape)
    tlayer(tx).sum().backward()

    assert x.grad.tolist() == pytest.approx(
        tx.grad.detach().flatten().tolist(), rel=1e-4, abs=1e-4
    )
    assert layer.weight.grad.tolist() == pytest.approx(
        tlayer.weight.grad.detach().flatten().tolist(), rel=1e-4, abs=1e-4
    )
    assert layer.bias.grad.tolist() == pytest.approx(
        tlayer.bias.grad.detach().flatten().tolist(), rel=1e-4, abs=1e-4
    )


def test_functional_layer_norm_affine_false_and_eps_matches_torch():
    shape = (2, 3, 4)
    values = [float(i % 5 - 2) for i in range(24)]
    x = _tensor_from_values(values, shape)
    y = F.layer_norm(x, (3, 4), eps=1e-3)

    tx = _torch_tensor(values, shape)
    ty = torch.nn.functional.layer_norm(tx, (3, 4), eps=1e-3)

    _assert_tensor_matches_torch(y, ty)


def test_layernorm_module_parameters_and_modes():
    affine = nn.LayerNorm((3, 4))
    assert len(list(affine.parameters())) == 2

    no_affine = nn.LayerNorm((3, 4), elementwise_affine=False)
    assert list(no_affine.parameters()) == []

    x = ortho.ones([2, 3, 4])
    train_out = affine.train()(x)
    eval_out = affine.eval()(x)
    assert train_out.tolist() == pytest.approx(eval_out.tolist())


def test_layernorm_non_contiguous_forward_matches_torch():
    values = [float(i + 1) / 3.0 for i in range(24)]
    base = _tensor_from_values(values, (2, 3, 4))
    view = base.transpose(1, 2)
    y = nn.LayerNorm(3, elementwise_affine=False)(view)

    tx = torch.tensor(values, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    ty = torch.nn.LayerNorm(3, elementwise_affine=False)(tx)

    assert not view.is_contiguous
    _assert_tensor_matches_torch(y, ty)


def test_layernorm_invalid_shape_raises():
    layer = nn.LayerNorm((2, 4))
    with pytest.raises(ValueError):
        layer(ortho.ones([2, 3, 4]))


@requires_cuda
@pytest.mark.parametrize("normalized_shape", [8, (3, 4)])
@pytest.mark.parametrize("affine", [True, False])
def test_layernorm_cuda_forward_backward_matches_torch(normalized_shape, affine):
    shape = (4, 8) if normalized_shape == 8 else (2, 3, 4)
    values = [float(i - 5) / 9.0 for i in range(torch.tensor(shape).prod().item())]
    x = _tensor_from_values(values, shape, device="cuda", requires_grad=True)
    layer = nn.LayerNorm(normalized_shape, elementwise_affine=affine).to("cuda")
    y = layer(x)
    y.sum().backward()

    tx = _torch_tensor(values, shape, device="cuda", requires_grad=True)
    tlayer = torch.nn.LayerNorm(normalized_shape, elementwise_affine=affine).cuda()
    ty = tlayer(tx)
    ty.sum().backward()

    assert y.device.is_cuda()
    assert x.grad.device.is_cuda()
    _assert_tensor_matches_torch(y, ty)
    assert x.grad.to("cpu").tolist() == pytest.approx(
        tx.grad.detach().cpu().flatten().tolist(), rel=1e-4, abs=1e-4
    )
    if affine:
        assert layer.weight.device.is_cuda()
        assert layer.bias.device.is_cuda()
        assert layer.weight.grad.device.is_cuda()
        assert layer.bias.grad.device.is_cuda()
        assert layer.weight.grad.to("cpu").tolist() == pytest.approx(
            tlayer.weight.grad.detach().cpu().flatten().tolist(), rel=1e-4, abs=1e-4
        )
        assert layer.bias.grad.to("cpu").tolist() == pytest.approx(
            tlayer.bias.grad.detach().cpu().flatten().tolist(), rel=1e-4, abs=1e-4
        )


@requires_cuda
def test_layernorm_module_to_cuda_moves_parameters():
    layer = nn.LayerNorm(4).to("cuda")
    assert layer.weight.device.is_cuda()
    assert layer.bias.device.is_cuda()

    x = ortho.full([2, 4], 1.0).to("cuda")
    x.requires_grad_(True)
    y = layer(x)
    y.sum().backward()
    assert y.device.is_cuda()
    assert x.grad.device.is_cuda()
