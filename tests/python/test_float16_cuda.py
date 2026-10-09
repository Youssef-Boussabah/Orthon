"""Float16 CUDA forward ops compared against PyTorch float16 on CUDA."""
import orthon._diagnostics as diag
from tests.support.capabilities import orthon_cuda_available as _cuda_available
import numpy as np
import pytest
import orthon as ortho

torch = pytest.importorskip("torch")




cuda_available = _cuda_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA not available in Orthon build")

# Tests that compare against a PyTorch CUDA reference also need the *reference*
# build to have CUDA. A Orthon CUDA build paired with a CPU-only PyTorch wheel
# must skip those rather than fail. Tests exercising only Orthon CUDA are left
# unmarked so they still run in that configuration.
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)

ATOL = 5e-3  # float16 has ~3 decimal digits of precision


def _da16(arr: np.ndarray) -> ortho.Tensor:
    return ortho.from_numpy(arr.astype(np.float16)).to("cuda")


def _pt16(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float16, device="cuda")


def _np16(t: ortho.Tensor) -> np.ndarray:
    """Move float16 tensor to CPU and convert to float32 numpy array."""
    return np.array(t.to("cpu").tolist(), dtype=np.float32).reshape(t.shape)


A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[0.5, -1.0], [2.0, 0.1]])
X = np.array([0.5, 1.0, 2.0, 0.1])
X_neg = np.array([-1.0, 0.5, 2.0, -0.1])


# Dtype tag.

def test_dtype_and_device_tag():
    t = _da16(X)
    assert t.dtype == "float16"
    assert "cuda" in str(t.device)


# Binary forward.

@requires_torch_cuda
def test_add_f16_forward():
    out = ortho.add(_da16(A), _da16(B))
    ref = _pt16(A) + _pt16(B)
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_mul_f16_forward():
    out = ortho.mul(_da16(A), _da16(B))
    ref = _pt16(A) * _pt16(B)
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_add_f16_broadcast():
    a = np.array([[1.0, 2.0, 3.0]])   # (1, 3)
    b = np.array([[10.0], [20.0]])     # (2, 1)
    out = ortho.add(_da16(a), _da16(b))
    ref = _pt16(a) + _pt16(b)
    assert list(out.shape) == [2, 3]
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


# Unary forward.

def _unary_fwd(da_fn, pt_fn, x=X):
    out = da_fn(_da16(x))
    assert out.dtype == "float16"
    ref = pt_fn(_pt16(x))
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_neg_f16():
    _unary_fwd(ortho.neg, lambda x: -x)


@requires_torch_cuda
def test_relu_f16():
    _unary_fwd(ortho.relu, torch.relu, x=X_neg)


@requires_torch_cuda
def test_sigmoid_f16():
    _unary_fwd(ortho.sigmoid, torch.sigmoid)


@requires_torch_cuda
def test_tanh_f16():
    _unary_fwd(ortho.tanh, torch.tanh)


@requires_torch_cuda
def test_exp_f16():
    # Use small values to avoid float16 overflow (max ~65504).
    _unary_fwd(ortho.exp, torch.exp, x=np.array([0.1, 0.5, 1.0, 0.2]))


@requires_torch_cuda
def test_sqrt_f16():
    _unary_fwd(ortho.sqrt, torch.sqrt)


@requires_torch_cuda
def test_reciprocal_f16():
    _unary_fwd(ortho.reciprocal, torch.reciprocal)


@requires_torch_cuda
def test_gelu_f16():
    _unary_fwd(ortho.gelu, lambda x: torch.nn.functional.gelu(x))


@requires_torch_cuda
def test_leaky_relu_f16():
    slope = 0.1
    _unary_fwd(
        lambda t: ortho.leaky_relu(t, slope),
        lambda x: torch.nn.functional.leaky_relu(x, slope),
        x=X_neg,
    )


# Sum and mean.

MAT = np.arange(6, dtype=np.float32).reshape(2, 3) + 1.0  # [[1,2,3],[4,5,6]]


@requires_torch_cuda
def test_sum_f16_all():
    out = ortho.sum(_da16(MAT))
    ref = _pt16(MAT).sum()
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out).item(), ref.cpu().item(), atol=ATOL)


@requires_torch_cuda
def test_sum_f16_dim0():
    out = ortho.sum(_da16(MAT), dim=0)
    ref = _pt16(MAT).sum(0)
    assert list(out.shape) == [3]
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sum_f16_dim1():
    out = ortho.sum(_da16(MAT), dim=1)
    ref = _pt16(MAT).sum(1)
    assert list(out.shape) == [2]
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_mean_f16_all():
    out = ortho.mean(_da16(MAT))
    ref = _pt16(MAT).mean()
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out).item(), ref.cpu().item(), atol=ATOL)


@requires_torch_cuda
def test_mean_f16_dim1():
    out = ortho.mean(_da16(MAT), dim=1)
    ref = _pt16(MAT).mean(1)
    assert list(out.shape) == [2]
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


# Max and argmax.

@requires_torch_cuda
def test_max_f16_all():
    out = ortho.max(_da16(MAT))
    ref = _pt16(MAT).max()
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out).item(), ref.cpu().item(), atol=ATOL)


@requires_torch_cuda
def test_max_f16_dim1():
    out = ortho.max(_da16(MAT), dim=1)
    ref = _pt16(MAT).max(1).values
    assert list(out.shape) == [2]
    assert out.dtype == "float16"
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_argmax_f16_dim1():
    out = ortho.argmax(_da16(MAT), dim=1)
    ref = _pt16(MAT).argmax(1)
    assert out.tolist() == ref.cpu().tolist()


# Float32 CUDA regression.

def test_float32_cuda_add_unaffected():
    a = ortho.from_numpy(A.astype(np.float32)).to("cuda")
    b = ortho.from_numpy(B.astype(np.float32)).to("cuda")
    out = ortho.add(a, b)
    assert out.dtype == "float32"
    result = np.array(out.to("cpu").tolist(), dtype=np.float32).reshape(out.shape)
    np.testing.assert_allclose(result, A + B, atol=1e-5)


@requires_torch_cuda
def test_float32_cuda_relu_unaffected():
    x = ortho.from_numpy(X_neg.astype(np.float32)).to("cuda")
    out = ortho.relu(x)
    assert out.dtype == "float32"
    ref = torch.relu(torch.tensor(X_neg, dtype=torch.float32, device="cuda"))
    result = np.array(out.to("cpu").tolist(), dtype=np.float32)
    np.testing.assert_allclose(result, ref.cpu().numpy(), atol=1e-5)


# Float16 backward.

def _da16_grad(arr: np.ndarray) -> ortho.Tensor:
    t = ortho.from_numpy(arr.astype(np.float16)).to("cuda")
    t.requires_grad_(True)
    return t


def _pt16_grad(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr.astype(np.float32), dtype=torch.float16,
                        device="cuda", requires_grad=True)


def _get_grad_da(t: ortho.Tensor) -> np.ndarray:
    return np.array(t.grad.to("cpu").tolist(), dtype=np.float32)


def _get_grad_pt(t: torch.Tensor) -> np.ndarray:
    return t.grad.cpu().float().numpy()


def _unary_bwd(da_fn, pt_fn, x=X):
    x_da = _da16_grad(x)
    ortho.sum(da_fn(x_da)).backward()

    x_pt = _pt16_grad(x)
    pt_fn(x_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_add_f16_backward():
    a_da = _da16_grad(A)
    b_da = _da16_grad(B)
    ortho.sum(ortho.add(a_da, b_da)).backward()

    a_pt = _pt16_grad(A)
    b_pt = _pt16_grad(B)
    (a_pt + b_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(a_da).flatten(),
                                _get_grad_pt(a_pt).flatten(), atol=ATOL)
    np.testing.assert_allclose(_get_grad_da(b_da).flatten(),
                                _get_grad_pt(b_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_mul_f16_backward():
    a_da = _da16_grad(A)
    b_da = _da16_grad(B)
    ortho.sum(ortho.mul(a_da, b_da)).backward()

    a_pt = _pt16_grad(A)
    b_pt = _pt16_grad(B)
    (a_pt * b_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(a_da).flatten(),
                                _get_grad_pt(a_pt).flatten(), atol=ATOL)
    np.testing.assert_allclose(_get_grad_da(b_da).flatten(),
                                _get_grad_pt(b_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_relu_f16_backward():
    _unary_bwd(ortho.relu, torch.relu, x=X_neg)


@requires_torch_cuda
def test_sigmoid_f16_backward():
    _unary_bwd(ortho.sigmoid, torch.sigmoid)


@requires_torch_cuda
def test_tanh_f16_backward():
    _unary_bwd(ortho.tanh, torch.tanh)


@requires_torch_cuda
def test_leaky_relu_f16_backward():
    slope = 0.1
    _unary_bwd(
        lambda t: ortho.leaky_relu(t, slope),
        lambda x: torch.nn.functional.leaky_relu(x, slope),
        x=X_neg,
    )


@requires_torch_cuda
def test_gelu_f16_backward():
    _unary_bwd(ortho.gelu, torch.nn.functional.gelu)


@requires_torch_cuda
def test_exp_f16_backward():
    _unary_bwd(ortho.exp, torch.exp, x=np.array([0.1, 0.5, 0.2, 0.3]))


@requires_torch_cuda
def test_sqrt_f16_backward():
    _unary_bwd(ortho.sqrt, torch.sqrt)


@requires_torch_cuda
def test_reciprocal_f16_backward():
    # avoid x=0.1 where -1/x^2=-100 exceeds float16 precision; use larger values
    _unary_bwd(ortho.reciprocal, torch.reciprocal,
               x=np.array([0.5, 1.0, 2.0, 0.5]))


# Reduction backward.

@requires_torch_cuda
def test_sum_f16_backward_all():
    x_da = _da16_grad(MAT)
    ortho.sum(x_da).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_sum_f16_backward_dim0():
    x_da = _da16_grad(MAT)
    ortho.sum(ortho.sum(x_da, dim=0)).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.sum(0).sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_sum_f16_backward_dim1():
    x_da = _da16_grad(MAT)
    ortho.sum(ortho.sum(x_da, dim=1)).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.sum(1).sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_mean_f16_backward_all():
    x_da = _da16_grad(MAT)
    ortho.sum(ortho.mean(x_da)).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.mean().sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_mean_f16_backward_dim1():
    x_da = _da16_grad(MAT)
    ortho.sum(ortho.mean(x_da, dim=1)).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.mean(1).sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_max_f16_backward_all():
    x_da = _da16_grad(MAT)
    ortho.max(x_da).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.max().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_max_f16_backward_dim1():
    x_da = _da16_grad(MAT)
    ortho.sum(ortho.max(x_da, dim=1)).backward()

    x_pt = _pt16_grad(MAT)
    x_pt.max(1).values.sum().backward()

    np.testing.assert_allclose(_get_grad_da(x_da).flatten(),
                                _get_grad_pt(x_pt).flatten(), atol=ATOL)


# Matmul forward and backward.

_A2 = np.array([[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]], dtype=np.float32)   # (2, 3)
_B2 = np.array([[0.5, -1.0],
                [2.0,  0.1],
                [1.0,  0.5]], dtype=np.float32)         # (3, 2)


@requires_torch_cuda
def test_matmul_f16_forward_2d():
    out = ortho.matmul(_da16(_A2), _da16(_B2))
    ref = _pt16(_A2) @ _pt16(_B2)
    assert out.dtype == "float16"
    assert list(out.shape) == [2, 2]
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


def test_matmul_f16_matches_fp32_tensor_core_shape():
    a = (np.arange(32 * 48, dtype=np.float32).reshape(32, 48) % 17 - 8.0) * 0.01
    b = (np.arange(48 * 64, dtype=np.float32).reshape(48, 64) % 13 - 6.0) * 0.01
    out = ortho.matmul(_da16(a), _da16(b))
    ref = a @ b

    assert out.dtype == "float16"
    assert list(out.shape) == [32, 64]
    np.testing.assert_allclose(_np16(out), ref, rtol=2e-2, atol=2e-2)


def test_matmul_f16_matches_fp32_non_multiple_shape():
    a = (np.arange(17 * 19, dtype=np.float32).reshape(17, 19) % 11 - 5.0) * 0.01
    b = (np.arange(19 * 23, dtype=np.float32).reshape(19, 23) % 7 - 3.0) * 0.01
    out = ortho.matmul(_da16(a), _da16(b))
    ref = a @ b

    assert out.dtype == "float16"
    assert list(out.shape) == [17, 23]
    np.testing.assert_allclose(_np16(out), ref, rtol=2e-2, atol=2e-2)


@requires_torch_cuda
def test_matmul_f16_forward_batched():
    a = np.stack([_A2, _A2 * 0.5])   # (2, 2, 3)
    b = np.stack([_B2, _B2 * 2.0])   # (2, 3, 2)
    out = ortho.matmul(_da16(a), _da16(b))
    ref = _pt16(a) @ _pt16(b)
    assert out.dtype == "float16"
    assert list(out.shape) == [2, 2, 2]
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_matmul_f16_batched_tensor_core_shape_matches_torch():
    a = np.random.randn(4, 32, 64).astype(np.float32) * 0.05
    b = np.random.randn(4, 64, 48).astype(np.float32) * 0.05
    out = ortho.matmul(_da16(a), _da16(b))
    ref = _pt16(a) @ _pt16(b)
    assert out.dtype == "float16"
    assert list(out.shape) == [4, 32, 48]
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), rtol=1e-2, atol=1e-2)


@requires_torch_cuda
def test_matmul_f16_batched_non_multiple_shape_matches_torch():
    a = np.random.randn(3, 17, 19).astype(np.float32) * 0.05
    b = np.random.randn(3, 19, 23).astype(np.float32) * 0.05
    out = ortho.matmul(_da16(a), _da16(b))
    ref = _pt16(a) @ _pt16(b)
    assert out.dtype == "float16"
    assert list(out.shape) == [3, 17, 23]
    np.testing.assert_allclose(_np16(out), ref.cpu().float().numpy(), rtol=1e-2, atol=1e-2)


def test_matmul_f16_batched_dispatch_matches_custom_kernel():
    custom = getattr(diag, "batched_matmul_custom", None)
    if custom is None:
        pytest.skip("custom batched matmul benchmark helper unavailable")
    a = np.random.randn(2, 13, 15).astype(np.float32) * 0.05
    b = np.random.randn(2, 15, 11).astype(np.float32) * 0.05
    a_da = _da16(a)
    b_da = _da16(b)
    out = ortho.matmul(a_da, b_da)
    ref = custom(a_da, b_da)
    assert out.dtype == "float16"
    assert ref.dtype == "float16"
    np.testing.assert_allclose(_np16(out), _np16(ref), rtol=1e-2, atol=1e-2)


@requires_torch_cuda
def test_matmul_f16_backward_2d():
    a_da = _da16_grad(_A2)
    b_da = _da16_grad(_B2)
    ortho.sum(ortho.matmul(a_da, b_da)).backward()

    a_pt = _pt16_grad(_A2)
    b_pt = _pt16_grad(_B2)
    (a_pt @ b_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(a_da).flatten(),
                                _get_grad_pt(a_pt).flatten(), atol=ATOL)
    np.testing.assert_allclose(_get_grad_da(b_da).flatten(),
                                _get_grad_pt(b_pt).flatten(), atol=ATOL)


# Broadcast backward through unbroadcast.

# a: (1, 3), b: (2, 1)  → out: (2, 3)  grad_a sums dim0, grad_b sums dim1
_Ab = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
_Bb = np.array([[10.0], [20.0]], dtype=np.float32)

# a: (2, 2), b: (1, 2)  → out: (2, 2)  grad_b sums dim0
_Ac = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
_Bc = np.array([[0.5, -1.0]], dtype=np.float32)


@requires_torch_cuda
def test_add_f16_broadcast_backward():
    a_da = _da16_grad(_Ab)
    b_da = _da16_grad(_Bb)
    ortho.sum(ortho.add(a_da, b_da)).backward()

    a_pt = _pt16_grad(_Ab)
    b_pt = _pt16_grad(_Bb)
    (a_pt + b_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(a_da).flatten(),
                                _get_grad_pt(a_pt).flatten(), atol=ATOL)
    np.testing.assert_allclose(_get_grad_da(b_da).flatten(),
                                _get_grad_pt(b_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_mul_f16_broadcast_backward():
    a_da = _da16_grad(_Ac)
    b_da = _da16_grad(_Bc)
    ortho.sum(ortho.mul(a_da, b_da)).backward()

    a_pt = _pt16_grad(_Ac)
    b_pt = _pt16_grad(_Bc)
    (a_pt * b_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(a_da).flatten(),
                                _get_grad_pt(a_pt).flatten(), atol=ATOL)
    np.testing.assert_allclose(_get_grad_da(b_da).flatten(),
                                _get_grad_pt(b_pt).flatten(), atol=ATOL)


@requires_torch_cuda
def test_add_f16_scalar_broadcast_backward():
    # scalar (shape=[1]) broadcast against vector (shape=[4])
    a_da = _da16_grad(np.array([2.0], dtype=np.float32))
    b_da = _da16_grad(X.astype(np.float32))
    ortho.sum(ortho.add(a_da, b_da)).backward()

    a_pt = _pt16_grad(np.array([2.0], dtype=np.float32))
    b_pt = _pt16_grad(X.astype(np.float32))
    (a_pt + b_pt).sum().backward()

    np.testing.assert_allclose(_get_grad_da(a_da).flatten(),
                                _get_grad_pt(a_pt).flatten(), atol=ATOL)
    np.testing.assert_allclose(_get_grad_da(b_da).flatten(),
                                _get_grad_pt(b_pt).flatten(), atol=ATOL)


# Conv2d.
# Conv2d accumulates K_h * K_w * C_in products per output element.
# A 3x3 conv with 2 input channels accumulates 18 float16 products, so we
# use atol=0.1 (much looser than pointwise ops).

CONV_ATOL = 0.1

rng = np.random.default_rng(42)

def _make_conv16(N, C_in, H, W, C_out, K):
    inp = rng.standard_normal((N, C_in, H, W)).astype(np.float32)
    wt  = rng.standard_normal((C_out, C_in, K, K)).astype(np.float32) * 0.1
    bias = rng.standard_normal(C_out).astype(np.float32) * 0.1
    return inp, wt, bias


def _da_conv16(inp, wt, bias, stride=1, padding=0, requires_grad=False):
    t_inp  = ortho.from_numpy(inp.astype(np.float16)).to("cuda")
    t_wt   = ortho.from_numpy(wt.astype(np.float16)).to("cuda")
    t_bias = ortho.from_numpy(bias.astype(np.float16)).to("cuda")
    if requires_grad:
        t_inp.requires_grad_(True)
        t_wt.requires_grad_(True)
        t_bias.requires_grad_(True)
    return t_inp, t_wt, t_bias


def _pt_conv16(inp, wt, bias, stride=1, padding=0, requires_grad=False):
    t_inp  = torch.tensor(inp, dtype=torch.float16, device="cuda")
    t_wt   = torch.tensor(wt,  dtype=torch.float16, device="cuda")
    t_bias = torch.tensor(bias, dtype=torch.float16, device="cuda")
    if requires_grad:
        t_inp  = t_inp.requires_grad_(True)
        t_wt   = t_wt.requires_grad_(True)
        t_bias = t_bias.requires_grad_(True)
    return t_inp, t_wt, t_bias


@requires_torch_cuda
def test_conv2d_f16_forward_1x1():
    inp, wt, bias = _make_conv16(2, 3, 8, 8, 4, 1)
    da_inp, da_wt, da_bias = _da_conv16(inp, wt, bias)
    pt_inp, pt_wt, pt_bias = _pt_conv16(inp, wt, bias)

    out_da = ortho.conv2d(da_inp, da_wt, da_bias, 1, 1, 0, 0)
    out_pt = torch.nn.functional.conv2d(pt_inp, pt_wt, pt_bias, stride=1, padding=0)

    np.testing.assert_allclose(
        _np16(out_da), out_pt.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_f16_forward_3x3():
    inp, wt, bias = _make_conv16(2, 2, 8, 8, 4, 3)
    da_inp, da_wt, da_bias = _da_conv16(inp, wt, bias)
    pt_inp, pt_wt, pt_bias = _pt_conv16(inp, wt, bias)

    out_da = ortho.conv2d(da_inp, da_wt, da_bias, 1, 1, 1, 1)
    out_pt = torch.nn.functional.conv2d(pt_inp, pt_wt, pt_bias, stride=1, padding=1)

    np.testing.assert_allclose(
        _np16(out_da), out_pt.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_f16_forward_stride2():
    inp, wt, bias = _make_conv16(1, 2, 10, 10, 3, 3)
    da_inp, da_wt, da_bias = _da_conv16(inp, wt, bias)
    pt_inp, pt_wt, pt_bias = _pt_conv16(inp, wt, bias)

    out_da = ortho.conv2d(da_inp, da_wt, da_bias, 2, 2, 0, 0)
    out_pt = torch.nn.functional.conv2d(pt_inp, pt_wt, pt_bias, stride=2, padding=0)

    np.testing.assert_allclose(
        _np16(out_da), out_pt.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_f16_backward_weight_grad():
    inp, wt, bias = _make_conv16(2, 2, 6, 6, 3, 3)
    da_inp, da_wt, da_bias = _da_conv16(inp, wt, bias, requires_grad=True)
    pt_inp, pt_wt, pt_bias = _pt_conv16(inp, wt, bias, requires_grad=True)

    ortho.sum(ortho.conv2d(da_inp, da_wt, da_bias, 1, 1, 1, 1)).backward()
    torch.nn.functional.conv2d(pt_inp, pt_wt, pt_bias, stride=1, padding=1).sum().backward()

    np.testing.assert_allclose(
        _get_grad_da(da_wt).flatten(), pt_wt.grad.cpu().float().numpy().flatten(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_f16_backward_input_grad():
    inp, wt, bias = _make_conv16(2, 2, 6, 6, 3, 3)
    da_inp, da_wt, da_bias = _da_conv16(inp, wt, bias, requires_grad=True)
    pt_inp, pt_wt, pt_bias = _pt_conv16(inp, wt, bias, requires_grad=True)

    ortho.sum(ortho.conv2d(da_inp, da_wt, da_bias, 1, 1, 1, 1)).backward()
    torch.nn.functional.conv2d(pt_inp, pt_wt, pt_bias, stride=1, padding=1).sum().backward()

    np.testing.assert_allclose(
        _get_grad_da(da_inp).flatten(), pt_inp.grad.cpu().float().numpy().flatten(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_f16_backward_bias_grad():
    inp, wt, bias = _make_conv16(2, 2, 6, 6, 3, 3)
    da_inp, da_wt, da_bias = _da_conv16(inp, wt, bias, requires_grad=True)
    pt_inp, pt_wt, pt_bias = _pt_conv16(inp, wt, bias, requires_grad=True)

    ortho.sum(ortho.conv2d(da_inp, da_wt, da_bias, 1, 1, 1, 1)).backward()
    torch.nn.functional.conv2d(pt_inp, pt_wt, pt_bias, stride=1, padding=1).sum().backward()

    np.testing.assert_allclose(
        _get_grad_da(da_bias), pt_bias.grad.cpu().float().numpy(), atol=CONV_ATOL)
