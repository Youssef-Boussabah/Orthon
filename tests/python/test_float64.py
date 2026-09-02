"""Float64 CPU arithmetic, forward and backward compared against PyTorch."""
import numpy as np
import pytest
import orthon as ortho

torch = pytest.importorskip("torch")


def _np(t: ortho.Tensor) -> np.ndarray:
    return np.array(t.tolist(), dtype=np.float64).reshape(t.shape)


def _from(arr: np.ndarray) -> ortho.Tensor:
    return ortho.from_numpy(arr)


def _da64(arr: np.ndarray, grad: bool = False) -> ortho.Tensor:
    t = _from(arr)
    if grad:
        t.requires_grad_(True)
    return t


def _pt64(arr: np.ndarray, grad: bool = False) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float64, requires_grad=grad)


# Helpers.

A2 = np.array([[1.0, 2.0], [3.0, 4.0]])
B2 = np.array([[0.5, -1.0], [2.0, 0.1]])
ATOL = 1e-12


# Dtype tag.

def test_empty_float64_dtype():
    t = ortho.empty([3, 4], dtype="float64")
    assert t.dtype == "float64"


# Binary ops.

def test_add_float64_forward():
    a = _da64(A2); b = _da64(B2)
    out = ortho.add(a, b)
    assert out.dtype == "float64"
    np.testing.assert_allclose(_np(out), A2 + B2, atol=ATOL)


def test_add_float64_backward():
    a = _da64(A2, grad=True); b = _da64(B2, grad=True)
    a_pt = _pt64(A2, grad=True); b_pt = _pt64(B2, grad=True)
    ortho.sum(ortho.add(a, b)).backward()
    (a_pt + b_pt).sum().backward()
    np.testing.assert_allclose(_np(a.grad), a_pt.grad.numpy(), atol=ATOL)
    np.testing.assert_allclose(_np(b.grad), b_pt.grad.numpy(), atol=ATOL)


def test_mul_float64_forward():
    a = _da64(A2); b = _da64(B2)
    out = ortho.mul(a, b)
    assert out.dtype == "float64"
    np.testing.assert_allclose(_np(out), A2 * B2, atol=ATOL)


def test_mul_float64_backward():
    a = _da64(A2, grad=True); b = _da64(B2, grad=True)
    a_pt = _pt64(A2, grad=True); b_pt = _pt64(B2, grad=True)
    ortho.sum(ortho.mul(a, b)).backward()
    (a_pt * b_pt).sum().backward()
    np.testing.assert_allclose(_np(a.grad), a_pt.grad.numpy(), atol=ATOL)
    np.testing.assert_allclose(_np(b.grad), b_pt.grad.numpy(), atol=ATOL)


def test_add_float64_broadcast():
    a = np.array([[1.0, 2.0, 3.0]])   # (1, 3)
    b = np.array([[10.0], [20.0]])     # (2, 1) -> broadcast to (2, 3)
    out = ortho.add(_da64(a), _da64(b))
    ref = torch.tensor(a, dtype=torch.float64) + torch.tensor(b, dtype=torch.float64)
    np.testing.assert_allclose(_np(out), ref.numpy(), atol=ATOL)


# Unary ops.

X = np.array([0.5, 1.0, 2.0, 0.1])


def _unary_fwd(da_fn, pt_fn, x=X):
    out = da_fn(_da64(x))
    assert out.dtype == "float64"
    ref = pt_fn(torch.tensor(x, dtype=torch.float64))
    np.testing.assert_allclose(_np(out), ref.numpy(), atol=1e-11)


def _unary_bwd(da_fn, pt_fn, x=X):
    t = _da64(x, grad=True)
    t_pt = _pt64(x, grad=True)
    ortho.sum(da_fn(t)).backward()
    pt_fn(t_pt).sum().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=1e-11)


def test_neg_float64():       _unary_fwd(ortho.neg,       lambda x: -x)
def test_relu_float64_fwd():  _unary_fwd(ortho.relu,      torch.relu)
def test_relu_float64_bwd():  _unary_bwd(ortho.relu,      torch.relu,
                                          x=np.array([-1.0, 0.5, 2.0, -0.1]))
def test_sigmoid_float64_fwd(): _unary_fwd(ortho.sigmoid, torch.sigmoid)
def test_sigmoid_float64_bwd(): _unary_bwd(ortho.sigmoid, torch.sigmoid)
def test_tanh_float64_fwd():  _unary_fwd(ortho.tanh,      torch.tanh)
def test_tanh_float64_bwd():  _unary_bwd(ortho.tanh,      torch.tanh)
def test_exp_float64_fwd():   _unary_fwd(ortho.exp,       torch.exp)
def test_exp_float64_bwd():   _unary_bwd(ortho.exp,       torch.exp)
def test_sqrt_float64_fwd():  _unary_fwd(ortho.sqrt,      torch.sqrt)
def test_sqrt_float64_bwd():  _unary_bwd(ortho.sqrt,      torch.sqrt)
def test_gelu_float64_fwd():
    _unary_fwd(ortho.gelu, lambda x: torch.nn.functional.gelu(x))
def test_gelu_float64_bwd():
    _unary_bwd(ortho.gelu, lambda x: torch.nn.functional.gelu(x))
def test_leaky_relu_float64_fwd():
    slope = 0.1
    _unary_fwd(lambda t: ortho.leaky_relu(t, slope),
               lambda x: torch.nn.functional.leaky_relu(x, slope),
               x=np.array([-1.0, 0.5, 2.0, -0.1]))
def test_leaky_relu_float64_bwd():
    slope = 0.1
    _unary_bwd(lambda t: ortho.leaky_relu(t, slope),
               lambda x: torch.nn.functional.leaky_relu(x, slope),
               x=np.array([-1.0, 0.5, 2.0, -0.1]))


# Sum and mean.

MAT = np.arange(6, dtype=np.float64).reshape(2, 3) + 1.0


def test_sum_float64_all():
    t = _da64(MAT, grad=True)
    t_pt = _pt64(MAT, grad=True)
    ortho.sum(t).backward()
    t_pt.sum().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=ATOL)


def test_sum_float64_dim0():
    t = _da64(MAT, grad=True)
    t_pt = _pt64(MAT, grad=True)
    ortho.sum(ortho.sum(t, dim=0)).backward()
    t_pt.sum(0).sum().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=ATOL)


def test_sum_float64_keepdim():
    t = _da64(MAT)
    ref = torch.tensor(MAT, dtype=torch.float64).sum(1, keepdim=True)
    out = ortho.sum(t, dim=1, keepdim=True)
    assert out.dtype == "float64"
    np.testing.assert_allclose(_np(out), ref.numpy(), atol=ATOL)


def test_mean_float64_all():
    t = _da64(MAT, grad=True)
    t_pt = _pt64(MAT, grad=True)
    ortho.mean(t).backward()
    t_pt.mean().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=1e-10)


def test_mean_float64_dim1():
    t = _da64(MAT, grad=True)
    t_pt = _pt64(MAT, grad=True)
    ortho.sum(ortho.mean(t, dim=1)).backward()
    t_pt.mean(1).sum().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=1e-10)


# Max.

def test_max_float64_all():
    t = _da64(MAT, grad=True)
    t_pt = _pt64(MAT, grad=True)
    ortho.max(t).backward()
    t_pt.max().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=ATOL)


def test_max_float64_dim():
    t = _da64(MAT, grad=True)
    t_pt = _pt64(MAT, grad=True)
    ortho.sum(ortho.max(t, dim=1)).backward()
    t_pt.max(1).values.sum().backward()
    np.testing.assert_allclose(_np(t.grad), t_pt.grad.numpy(), atol=ATOL)


# Matmul.

def test_matmul_float64_forward():
    a = np.random.default_rng(0).standard_normal((3, 4))
    b = np.random.default_rng(1).standard_normal((4, 5))
    out = ortho.matmul(_da64(a), _da64(b))
    assert out.dtype == "float64"
    ref = torch.tensor(a, dtype=torch.float64) @ torch.tensor(b, dtype=torch.float64)
    np.testing.assert_allclose(_np(out), ref.numpy(), atol=1e-11)


def test_matmul_float64_backward():
    a = np.random.default_rng(2).standard_normal((3, 4))
    b = np.random.default_rng(3).standard_normal((4, 5))
    a_da = _da64(a, grad=True); b_da = _da64(b, grad=True)
    a_pt = _pt64(a, grad=True); b_pt = _pt64(b, grad=True)
    ortho.sum(ortho.matmul(a_da, b_da)).backward()
    (a_pt @ b_pt).sum().backward()
    np.testing.assert_allclose(_np(a_da.grad), a_pt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(_np(b_da.grad), b_pt.grad.numpy(), atol=1e-10)


# Dtype rejection for CUDA-only paths.

def test_float32_still_works():
    a = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    b = ortho.from_numpy(np.array([3.0, 4.0], dtype=np.float32))
    out = ortho.add(a, b)
    assert out.dtype == "float32"
    np.testing.assert_allclose(out.tolist(), [4.0, 6.0])


def test_dtype_mismatch_rejected():
    a = _da64(np.array([1.0, 2.0]))
    b = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(RuntimeError):
        ortho.add(a, b)
