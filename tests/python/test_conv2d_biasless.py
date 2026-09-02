"""Conv2d(bias=False) correctness on CPU and CUDA.

The biasless path synthesises a zero bias, which must land on the same device
as the real operands. Building it with ortho.zeros([C_out]) alone gives a CPU
float32 tensor, so after .to("cuda") the call is CUDA input, CUDA weight and a
CPU bias, and native validation rejects it with:

    RuntimeError: conv2d: input, weight, and bias must be on the same device

Weights are set explicitly rather than taken from the random initialiser, and
backward uses a fixed non-uniform upstream so the gradients are non-trivial.
"""
import numpy as np
import pytest

import orthon as ortho
from orthon.nn import Conv2d

torch = pytest.importorskip("torch")

CUDA = ortho.cuda_enabled()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)

# (C_in, C_out, kernel, stride, padding, input shape)
CONFIGS = [
    (2, 3, (3, 3), (1, 1), (0, 0), (2, 2, 5, 5)),   # simple 3x3
    (2, 3, (3, 3), (1, 1), (1, 1), (2, 2, 5, 5)),   # padding
    (2, 3, (3, 3), (2, 2), (1, 1), (2, 2, 6, 6)),   # stride > 1
    (2, 3, (2, 3), (1, 1), (0, 1), (2, 2, 5, 6)),   # non-square kernel
    (1, 1, (1, 1), (1, 1), (0, 0), (1, 1, 4, 4)),   # 1x1
]
IDS = ["simple3x3", "padding", "stride2", "nonsquare", "1x1"]


def _vals(shape, scale, phase):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.float64)
    return (np.cos(i * phase) * scale + np.sin(i * 0.37) * 0.4).reshape(shape)


def _t(arr, device="cpu", dtype="float32"):
    t = ortho.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
    if device != "cpu":
        t = t.to(device)
    if dtype != "float32":
        t = ortho.cast(t, dtype)
    return t


def _npy(t):
    return np.array(t.tolist(), dtype=np.float64)


def _set_weight(layer, values):
    layer.weight.mul_(0.0)
    layer.weight.add_(_t(values), alpha=1.0)


def _build(cfg, device, dtype, w_np):
    C_in, C_out, k, s, p, _ = cfg
    layer = Conv2d(C_in, C_out, k, stride=s, padding=p, bias=False)
    _set_weight(layer, w_np)          # set before moving/casting
    if device != "cpu":
        layer.to(device)
    if dtype != "float32":
        layer.cast(dtype)
    return layer


def _parity(cfg, device="cpu", dtype="float32", tdtype=torch.float32, tol=1e-4):
    C_in, C_out, k, s, p, xshape = cfg
    w_np = _vals((C_out, C_in, k[0], k[1]), 0.6, 1.3)
    x_np = _vals(xshape, 1.2, 0.7)

    layer = _build(cfg, device, dtype, w_np)
    x = _t(x_np, device, dtype)
    x.requires_grad_(True)
    y = layer(x)

    up_np = _vals(tuple(int(v) for v in y.shape), 0.9, 2.1)
    up = _t(up_np, device, dtype)
    ortho.sum(y * up).backward()

    tdev = "cuda" if device != "cpu" else "cpu"
    tm = torch.nn.Conv2d(C_in, C_out, k, stride=s, padding=p, bias=False)
    with torch.no_grad():
        tm.weight.copy_(torch.tensor(w_np, dtype=torch.float32))
    tm = tm.to(tdev).to(tdtype)
    xt = torch.tensor(x_np, dtype=tdtype, device=tdev, requires_grad=True)
    upt = torch.tensor(up_np, dtype=tdtype, device=tdev)
    yt = tm(xt)
    (yt * upt).sum().backward()

    def tn(t):
        return t.detach().float().cpu().numpy().ravel().astype(np.float64)

    assert list(y.shape) == list(yt.shape), "output shape"
    assert _npy(y) == pytest.approx(tn(yt), abs=tol), "forward output"
    assert _npy(x.grad) == pytest.approx(tn(xt.grad), abs=tol), "input gradient"
    assert _npy(layer.weight.grad) == pytest.approx(
        tn(tm.weight.grad), abs=tol), "weight gradient"
    assert tm.bias is None
    return layer, x, y


# ---- CPU float32 parity ----

@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_cpu_biasless_parity(cfg):
    _parity(cfg)


def test_cpu_input_gradient_is_substantial():
    """Guards the oracle: a vanishing gradient would make parity vacuous."""
    _, x, _ = _parity(CONFIGS[0])
    assert np.abs(_npy(x.grad)).max() > 0.01


# ---- CUDA float32 correctness (the synthetic-bias path) ----

@requires_cuda
def test_cuda_biasless_forward_succeeds():
    """A bias=False Conv2d forward runs on CUDA and keeps its output there."""
    layer = Conv2d(1, 2, kernel_size=3, bias=False).to("cuda")
    y = layer(ortho.ones([1, 1, 5, 5]).to("cuda"))
    assert list(y.shape) == [1, 2, 3, 3]
    assert y.device.is_cuda()


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("cfg", CONFIGS, ids=IDS)
def test_cuda_biasless_parity(cfg):
    _parity(cfg, device="cuda")


@requires_cuda
def test_cuda_biasless_gradients_stay_on_device():
    cfg = CONFIGS[1]
    w_np = _vals((cfg[1], cfg[0], cfg[2][0], cfg[2][1]), 0.6, 1.3)
    layer = _build(cfg, "cuda", "float32", w_np)
    x = _t(_vals(cfg[5], 1.2, 0.7), "cuda")
    x.requires_grad_(True)
    y = layer(x)
    ortho.sum(y).backward()
    assert y.device.is_cuda()
    assert x.grad.device.is_cuda()
    assert layer.weight.grad.device.is_cuda()


# ---- CUDA low precision ----

@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("cfg", [CONFIGS[0], CONFIGS[1]], ids=["simple3x3", "padding"])
def test_cuda_biasless_float16_parity(cfg):
    _parity(cfg, device="cuda", dtype="float16", tdtype=torch.float16, tol=5e-2)


@requires_cuda
@requires_torch_cuda
@pytest.mark.parametrize("cfg", [CONFIGS[0], CONFIGS[1]], ids=["simple3x3", "padding"])
def test_cuda_biasless_bfloat16_parity(cfg):
    _parity(cfg, device="cuda", dtype="bfloat16", tdtype=torch.bfloat16, tol=3e-1)


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_biasless_low_precision_dtype_follows_input(dtype):
    cfg = CONFIGS[0]
    w_np = _vals((cfg[1], cfg[0], cfg[2][0], cfg[2][1]), 0.6, 1.3)
    layer = _build(cfg, "cuda", dtype, w_np)
    x = _t(_vals(cfg[5], 1.2, 0.7), "cuda", dtype)
    x.requires_grad_(True)
    y = layer(x)
    ortho.sum(y).backward()
    assert y.dtype == dtype
    assert y.device.is_cuda()
    assert x.grad is not None and x.grad.dtype == dtype
    assert layer.weight.grad is not None


# ---- module contract ----

def test_biasless_has_no_bias_parameter():
    layer = Conv2d(2, 3, 3, bias=False)
    assert [n for n, _ in layer.named_parameters()] == ["weight"]
    assert len(list(layer.parameters())) == 1
    assert not hasattr(layer, "bias")


def test_biasless_state_dict_is_weight_only():
    assert sorted(Conv2d(2, 3, 3, bias=False).state_dict().keys()) == ["weight"]


def test_biasless_registers_no_buffers():
    layer = Conv2d(2, 3, 3, bias=False)
    assert list(layer.named_buffers()) == []
    assert layer._buffers == {}


def test_synthetic_bias_is_not_retained_between_forwards():
    layer = Conv2d(2, 3, 3, bias=False)
    x = _t(_vals((1, 2, 5, 5), 1.0, 0.5))
    layer(x)
    layer(x)
    assert not hasattr(layer, "bias")
    assert layer._buffers == {}
    assert sorted(layer.state_dict().keys()) == ["weight"]


def test_biasless_optimizer_sees_only_weight():
    from orthon.optim import SGD
    layer = Conv2d(2, 3, 3, bias=False)
    assert len(SGD(layer.parameters(), lr=0.1)._params) == 1


def test_biasless_survives_module_transforms_cpu():
    layer = Conv2d(2, 3, 3, bias=False).to("cpu")
    assert sorted(layer.state_dict().keys()) == ["weight"]
    assert list(layer.named_buffers()) == []
    assert not hasattr(layer, "bias")


@requires_cuda
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_biasless_survives_to_and_cast(dtype):
    layer = Conv2d(2, 3, 3, bias=False).to("cuda")
    layer.cast(dtype)
    assert sorted(layer.state_dict().keys()) == ["weight"]
    assert list(layer.named_buffers()) == []
    assert not hasattr(layer, "bias")
    assert layer.weight.dtype == dtype
    assert layer.weight.device.is_cuda()


# ---- genuine errors are still reported ----

@requires_cuda
def test_device_mismatch_between_input_and_weight_still_rejected():
    layer = Conv2d(1, 1, 2, bias=False)          # weight stays on CPU
    with pytest.raises(RuntimeError, match="same device"):
        layer(ortho.ones([1, 1, 3, 3]).to("cuda"))


def test_dtype_mismatch_between_input_and_weight_still_rejected():
    layer = Conv2d(1, 1, 2, bias=False)          # float32 weight
    with pytest.raises(RuntimeError):
        layer(ortho.cast(ortho.ones([1, 1, 3, 3]), "float64"))


def test_channel_mismatch_still_rejected():
    layer = Conv2d(2, 3, 3, bias=False)
    with pytest.raises(RuntimeError):
        layer(ortho.ones([1, 5, 6, 6]))


# ---- exact device propagation ----

def test_conv2d_source_does_not_hardcode_cuda_string():
    import inspect
    import orthon.nn.conv2d as mod
    src = inspect.getsource(mod)
    assert '.to("cuda")' not in src
    assert ".to('cuda')" not in src


@requires_cuda
def test_output_lands_on_the_inputs_exact_device():
    layer = Conv2d(1, 2, 3, bias=False).to("cuda")
    x = ortho.ones([1, 1, 5, 5]).to("cuda")
    y = layer(x)
    assert y.device == x.device


# ---- bias=True is unchanged ----

@pytest.mark.parametrize("cfg", [CONFIGS[0], CONFIGS[1]], ids=["simple3x3", "padding"])
def test_cpu_bias_true_parity_unchanged(cfg):
    C_in, C_out, k, s, p, xshape = cfg
    w_np = _vals((C_out, C_in, k[0], k[1]), 0.6, 1.3)
    b_np = _vals((C_out,), 0.5, 1.1)
    x_np = _vals(xshape, 1.2, 0.7)

    layer = Conv2d(C_in, C_out, k, stride=s, padding=p, bias=True)
    _set_weight(layer, w_np)
    layer.bias.mul_(0.0)
    layer.bias.add_(_t(b_np), alpha=1.0)
    x = _t(x_np)
    x.requires_grad_(True)
    y = layer(x)
    up_np = _vals(tuple(int(v) for v in y.shape), 0.9, 2.1)
    ortho.sum(y * _t(up_np)).backward()

    tm = torch.nn.Conv2d(C_in, C_out, k, stride=s, padding=p, bias=True)
    with torch.no_grad():
        tm.weight.copy_(torch.tensor(w_np, dtype=torch.float32))
        tm.bias.copy_(torch.tensor(b_np, dtype=torch.float32))
    xt = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    yt = tm(xt)
    (yt * torch.tensor(up_np, dtype=torch.float32)).sum().backward()

    def tn(t):
        return t.detach().float().numpy().ravel().astype(np.float64)

    assert _npy(y) == pytest.approx(tn(yt), abs=1e-4)
    assert _npy(x.grad) == pytest.approx(tn(xt.grad), abs=1e-4)
    assert _npy(layer.weight.grad) == pytest.approx(tn(tm.weight.grad), abs=1e-4)
    assert _npy(layer.bias.grad) == pytest.approx(tn(tm.bias.grad), abs=1e-4)


def test_bias_true_module_contract_unchanged():
    layer = Conv2d(2, 3, 3, bias=True)
    assert [n for n, _ in layer.named_parameters()] == ["weight", "bias"]
    assert sorted(layer.state_dict().keys()) == ["bias", "weight"]
    assert list(layer.named_buffers()) == []


@requires_cuda
@requires_torch_cuda
def test_cuda_bias_true_still_works():
    layer = Conv2d(1, 2, 3, bias=True).to("cuda")
    y = layer(ortho.ones([1, 1, 5, 5]).to("cuda"))
    assert list(y.shape) == [1, 2, 3, 3]
    assert y.device.is_cuda()
