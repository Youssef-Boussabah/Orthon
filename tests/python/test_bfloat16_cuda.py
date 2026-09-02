"""BFloat16 CUDA unary and binary ops compared against PyTorch bfloat16 on CUDA."""
import numpy as np
import pytest
import orthon as ortho

torch = pytest.importorskip("torch")


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


cuda_available = _cuda_available()
pytestmark = pytest.mark.skipif(
    not cuda_available, reason="CUDA not available in Orthon build"
)

# Tests that compare against a PyTorch CUDA reference also need the *reference*
# build to have CUDA. A Orthon CUDA build paired with a CPU-only PyTorch wheel
# must skip those rather than fail. Tests exercising only Orthon CUDA are left
# unmarked so they still run in that configuration.
requires_torch_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="PyTorch reference build has no CUDA support",
)

ATOL = 0.02  # BF16 eps ~0.0078; 0.02 gives ~2.5x buffer for transcendentals


def _da_bf16(arr: np.ndarray) -> ortho.Tensor:
    return ortho.from_numpy(arr.astype(np.float32), dtype="bfloat16").to("cuda")


def _pt_bf16(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.bfloat16, device="cuda")


def _np_bf16(t: ortho.Tensor) -> np.ndarray:
    return np.array(t.to("cpu").tolist(), dtype=np.float32).reshape(t.shape)


A  = np.array([[1.0, 2.0], [3.0, 4.0]])
B  = np.array([[0.5, -1.0], [2.0, 0.1]])
X  = np.array([0.5, 1.0, 2.0, 0.1])
XP = np.array([0.5, 1.0, 2.0, 0.3])   # all positive for sqrt/reciprocal


# from_numpy round-trip.

def test_from_numpy_dtype_tag():
    t = _da_bf16(X)
    assert t.dtype == "bfloat16"
    assert "cuda" in str(t.device)


def test_from_numpy_values():
    # The PyTorch reference here is a CPU bfloat16 tensor, so this test does not
    # need a CUDA-capable PyTorch build.
    arr = np.array([1.0, -2.0, 0.5, 4.0], dtype=np.float32)
    t = _da_bf16(arr)
    ref = torch.tensor(arr, dtype=torch.bfloat16).float().numpy()
    np.testing.assert_allclose(_np_bf16(t), ref, atol=ATOL)


def test_from_numpy_rejects_non_float32():
    with pytest.raises((TypeError, RuntimeError)):
        ortho.from_numpy(np.array([1, 2], dtype=np.int32), dtype="bfloat16")


# neg.

@requires_torch_cuda
def test_neg_bf16_forward():
    out = ortho.neg(_da_bf16(A))
    ref = -_pt_bf16(A)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


# relu.

@requires_torch_cuda
def test_relu_bf16_forward():
    inp = np.array([-1.0, 0.5, -0.1, 2.0])
    out = ortho.relu(_da_bf16(inp))
    ref = torch.relu(_pt_bf16(inp))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_relu_bf16_backward():
    inp_np = np.array([-1.0, 0.5, -0.1, 2.0], dtype=np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    out = ortho.relu(x)
    out.backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.relu(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# sigmoid.

@requires_torch_cuda
def test_sigmoid_bf16_forward():
    out = ortho.sigmoid(_da_bf16(X))
    ref = torch.sigmoid(_pt_bf16(X))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sigmoid_bf16_backward():
    inp_np = X.astype(np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.sigmoid(x).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.sigmoid(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# tanh.

@requires_torch_cuda
def test_tanh_bf16_forward():
    out = ortho.tanh(_da_bf16(X))
    ref = torch.tanh(_pt_bf16(X))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_tanh_bf16_backward():
    inp_np = X.astype(np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.tanh(x).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.tanh(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# leaky_relu.

@requires_torch_cuda
def test_leaky_relu_bf16_forward():
    inp = np.array([-1.0, 0.5, -0.1, 2.0])
    out = ortho.leaky_relu(_da_bf16(inp), negative_slope=0.1)
    ref = torch.nn.functional.leaky_relu(_pt_bf16(inp), negative_slope=0.1)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_leaky_relu_bf16_backward():
    inp_np = np.array([-1.0, 0.5, -0.1, 2.0], dtype=np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.leaky_relu(x, negative_slope=0.1).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.nn.functional.leaky_relu(ref_x, negative_slope=0.1).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# gelu.

@requires_torch_cuda
def test_gelu_bf16_forward():
    out = ortho.gelu(_da_bf16(X))
    ref = torch.nn.functional.gelu(_pt_bf16(X))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_gelu_bf16_backward():
    inp_np = X.astype(np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.gelu(x).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.nn.functional.gelu(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# exp.

@requires_torch_cuda
def test_exp_bf16_forward():
    out = ortho.exp(_da_bf16(X))
    ref = torch.exp(_pt_bf16(X))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_exp_bf16_backward():
    inp_np = X.astype(np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.exp(x).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.exp(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# sqrt.

@requires_torch_cuda
def test_sqrt_bf16_forward():
    out = ortho.sqrt(_da_bf16(XP))
    ref = torch.sqrt(_pt_bf16(XP))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sqrt_bf16_backward():
    inp_np = XP.astype(np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.sqrt(x).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.sqrt(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# reciprocal.

@requires_torch_cuda
def test_reciprocal_bf16_forward():
    out = ortho.reciprocal(_da_bf16(XP))
    ref = torch.reciprocal(_pt_bf16(XP))
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_reciprocal_bf16_backward():
    inp_np = XP.astype(np.float32)
    x = _da_bf16(inp_np)
    x.requires_grad_(True)
    ortho.reciprocal(x).backward()
    ref_x = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    torch.reciprocal(ref_x).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(x.grad), ref_x.grad.cpu().float().numpy(), atol=ATOL
    )


# add forward.

@requires_torch_cuda
def test_add_bf16_forward_same_shape():
    out = ortho.add(_da_bf16(A), _da_bf16(B))
    ref = _pt_bf16(A) + _pt_bf16(B)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_add_bf16_forward_broadcast():
    a_np = np.array([[1.0, 2.0, 3.0]])    # shape (1, 3)
    b_np = np.array([[0.5], [1.5]])        # shape (2, 1)
    out = ortho.add(_da_bf16(a_np), _da_bf16(b_np))
    ref = _pt_bf16(a_np) + _pt_bf16(b_np)
    assert tuple(out.shape) == (2, 3)
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


# mul forward.

@requires_torch_cuda
def test_mul_bf16_forward_same_shape():
    out = ortho.mul(_da_bf16(A), _da_bf16(B))
    ref = _pt_bf16(A) * _pt_bf16(B)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_mul_bf16_forward_broadcast():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]])   # shape (2, 2)
    b_np = np.array([[0.5, 2.0]])                 # shape (1, 2)
    out = ortho.mul(_da_bf16(a_np), _da_bf16(b_np))
    ref = _pt_bf16(a_np) * _pt_bf16(b_np)
    assert tuple(out.shape) == (2, 2)
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


# add backward for same-shape tensors.

@requires_torch_cuda
def test_add_bf16_backward_same_shape():
    a_np = A.astype(np.float32)
    b_np = B.astype(np.float32)
    xa = _da_bf16(a_np); xa.requires_grad_(True)
    xb = _da_bf16(b_np); xb.requires_grad_(True)
    ortho.add(xa, xb).backward()
    ra = torch.tensor(a_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rb = torch.tensor(b_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    (ra + rb).sum().backward()
    np.testing.assert_allclose(_np_bf16(xa.grad), ra.grad.cpu().float().numpy(), atol=ATOL)
    np.testing.assert_allclose(_np_bf16(xb.grad), rb.grad.cpu().float().numpy(), atol=ATOL)


# mul backward for same-shape tensors.

@requires_torch_cuda
def test_mul_bf16_backward_same_shape():
    a_np = A.astype(np.float32)
    b_np = B.astype(np.float32)
    xa = _da_bf16(a_np); xa.requires_grad_(True)
    xb = _da_bf16(b_np); xb.requires_grad_(True)
    ortho.mul(xa, xb).backward()
    ra = torch.tensor(a_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rb = torch.tensor(b_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    (ra * rb).sum().backward()
    np.testing.assert_allclose(_np_bf16(xa.grad), ra.grad.cpu().float().numpy(), atol=ATOL)
    np.testing.assert_allclose(_np_bf16(xb.grad), rb.grad.cpu().float().numpy(), atol=ATOL)


# Broadcast backward.

@requires_torch_cuda
def test_add_bf16_broadcast_backward():
    a_np = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)   # (1, 3)
    b_np = np.array([[0.5], [1.5]], dtype=np.float32)       # (2, 1)
    xa = _da_bf16(a_np); xa.requires_grad_(True)
    xb = _da_bf16(b_np); xb.requires_grad_(True)
    ortho.add(xa, xb).backward()
    ra = torch.tensor(a_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rb = torch.tensor(b_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    (ra + rb).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(xa.grad), ra.grad.cpu().float().numpy(), atol=ATOL
    )
    np.testing.assert_allclose(
        _np_bf16(xb.grad), rb.grad.cpu().float().numpy(), atol=ATOL
    )


@requires_torch_cuda
def test_mul_bf16_broadcast_backward():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # (2, 2)
    b_np = np.array([[0.5, 2.0]], dtype=np.float32)               # (1, 2)
    xa = _da_bf16(a_np); xa.requires_grad_(True)
    xb = _da_bf16(b_np); xb.requires_grad_(True)
    ortho.mul(xa, xb).backward()
    ra = torch.tensor(a_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rb = torch.tensor(b_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    (ra * rb).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(xa.grad), ra.grad.cpu().float().numpy(), atol=ATOL
    )
    np.testing.assert_allclose(
        _np_bf16(xb.grad), rb.grad.cpu().float().numpy(), atol=ATOL
    )


@requires_torch_cuda
def test_add_bf16_scalar_broadcast_backward():
    a_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)    # (3,)
    b_np = np.array([0.5], dtype=np.float32)               # (1,), scalar-like
    xa = _da_bf16(a_np); xa.requires_grad_(True)
    xb = _da_bf16(b_np); xb.requires_grad_(True)
    ortho.add(xa, xb).backward()
    ra = torch.tensor(a_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rb = torch.tensor(b_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    (ra + rb).sum().backward()
    np.testing.assert_allclose(
        _np_bf16(xa.grad), ra.grad.cpu().float().numpy(), atol=ATOL
    )
    np.testing.assert_allclose(
        _np_bf16(xb.grad), rb.grad.cpu().float().numpy(), atol=ATOL
    )


# sum.

@requires_torch_cuda
def test_sum_bf16_all():
    out = ortho.sum(_da_bf16(A))
    ref = _pt_bf16(A).sum()
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sum_bf16_dim0():
    out = ortho.sum(_da_bf16(A), dim=0)
    ref = _pt_bf16(A).sum(dim=0)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sum_bf16_dim1():
    out = ortho.sum(_da_bf16(A), dim=1)
    ref = _pt_bf16(A).sum(dim=1)
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sum_bf16_backward_all():
    inp_np = A.astype(np.float32)
    x = _da_bf16(inp_np); x.requires_grad_(True)
    ortho.sum(x).backward()
    rx = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rx.sum().backward()
    np.testing.assert_allclose(_np_bf16(x.grad), rx.grad.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_sum_bf16_backward_dim1():
    inp_np = A.astype(np.float32)
    x = _da_bf16(inp_np); x.requires_grad_(True)
    ortho.sum(x, dim=1).backward()
    rx = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rx.sum(dim=1).sum().backward()
    np.testing.assert_allclose(_np_bf16(x.grad), rx.grad.cpu().float().numpy(), atol=ATOL)


# mean.

@requires_torch_cuda
def test_mean_bf16_all():
    out = ortho.mean(_da_bf16(A))
    ref = _pt_bf16(A).mean()
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_mean_bf16_dim1():
    out = ortho.mean(_da_bf16(A), dim=1)
    ref = _pt_bf16(A).mean(dim=1)
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_mean_bf16_backward_all():
    inp_np = A.astype(np.float32)
    x = _da_bf16(inp_np); x.requires_grad_(True)
    ortho.mean(x).backward()
    rx = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rx.mean().backward()
    np.testing.assert_allclose(_np_bf16(x.grad), rx.grad.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_mean_bf16_backward_dim1():
    inp_np = A.astype(np.float32)
    x = _da_bf16(inp_np); x.requires_grad_(True)
    ortho.mean(x, dim=1).backward()
    rx = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rx.mean(dim=1).sum().backward()
    np.testing.assert_allclose(_np_bf16(x.grad), rx.grad.cpu().float().numpy(), atol=ATOL)


# max.

@requires_torch_cuda
def test_max_bf16_all():
    out = ortho.max(_da_bf16(A))
    ref = _pt_bf16(A).max()
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_max_bf16_dim1():
    out = ortho.max(_da_bf16(A), dim=1)
    ref = _pt_bf16(A).max(dim=1).values
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_max_bf16_backward_all():
    inp_np = A.astype(np.float32)
    x = _da_bf16(inp_np); x.requires_grad_(True)
    ortho.max(x).backward()
    rx = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rx.max().backward()
    np.testing.assert_allclose(_np_bf16(x.grad), rx.grad.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_max_bf16_backward_dim1():
    inp_np = A.astype(np.float32)
    x = _da_bf16(inp_np); x.requires_grad_(True)
    ortho.max(x, dim=1).backward()
    rx = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rx.max(dim=1).values.sum().backward()
    np.testing.assert_allclose(_np_bf16(x.grad), rx.grad.cpu().float().numpy(), atol=ATOL)


# argmax.

@requires_torch_cuda
def test_argmax_bf16_all():
    idx = ortho.argmax(_da_bf16(A))
    ref = _pt_bf16(A).argmax()
    assert int(idx.item()) == int(ref.item())


@requires_torch_cuda
def test_argmax_bf16_dim1():
    idx = ortho.argmax(_da_bf16(A), dim=1)
    ref = _pt_bf16(A).argmax(dim=1)
    np.testing.assert_array_equal(
        np.array(idx.tolist(), dtype=np.int64),
        ref.cpu().numpy()
    )


# matmul.

@requires_torch_cuda
def test_matmul_bf16_2d_forward():
    a_np = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0]], dtype=np.float32)   # (2, 3)
    b_np = np.array([[1.0, 0.0],
                     [0.0, 1.0],
                     [1.0, 1.0]], dtype=np.float32)         # (3, 2)
    out = ortho.matmul(_da_bf16(a_np), _da_bf16(b_np))
    ref = _pt_bf16(a_np) @ _pt_bf16(b_np)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


def test_matmul_bf16_2d_tensor_core_shape_matches_fp32():
    a_np = (np.arange(256 * 256, dtype=np.float32).reshape(256, 256) % 23 - 11.0) * 0.01
    b_np = (np.arange(256 * 256, dtype=np.float32).reshape(256, 256) % 19 - 9.0) * 0.01
    out = ortho.matmul(_da_bf16(a_np), _da_bf16(b_np))
    ref = a_np @ b_np
    assert out.dtype == "bfloat16"
    assert list(out.shape) == [256, 256]
    np.testing.assert_allclose(_np_bf16(out), ref, rtol=4e-2, atol=4e-2)


@requires_torch_cuda
def test_matmul_bf16_2d_backward():
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    xa = _da_bf16(a_np); xa.requires_grad_(True)
    xb = _da_bf16(b_np); xb.requires_grad_(True)
    ortho.matmul(xa, xb).backward()
    ra = torch.tensor(a_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    rb = torch.tensor(b_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    (ra @ rb).sum().backward()
    np.testing.assert_allclose(_np_bf16(xa.grad), ra.grad.cpu().float().numpy(), atol=ATOL)
    np.testing.assert_allclose(_np_bf16(xb.grad), rb.grad.cpu().float().numpy(), atol=ATOL)


@requires_torch_cuda
def test_matmul_bf16_batched_forward():
    a_np = np.random.randn(2, 3, 4).astype(np.float32)
    b_np = np.random.randn(2, 4, 5).astype(np.float32)
    out = ortho.matmul(_da_bf16(a_np), _da_bf16(b_np))
    ref = _pt_bf16(a_np) @ _pt_bf16(b_np)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=ATOL)


def test_matmul_bf16_batched_tensor_core_shape_matches_fp32():
    a_np = (np.arange(8 * 64 * 128, dtype=np.float32).reshape(8, 64, 128) % 29 - 14.0) * 0.01
    b_np = (np.arange(8 * 128 * 256, dtype=np.float32).reshape(8, 128, 256) % 31 - 15.0) * 0.01
    out = ortho.matmul(_da_bf16(a_np), _da_bf16(b_np))
    ref = a_np @ b_np
    assert out.dtype == "bfloat16"
    assert list(out.shape) == [8, 64, 256]
    np.testing.assert_allclose(_np_bf16(out), ref, rtol=4e-2, atol=4e-2)


def test_matmul_bf16_dispatch_matches_custom_kernel():
    custom_2d = getattr(ortho, "_matmul_tiled_custom", None)
    custom_batched = getattr(ortho, "_matmul_batched_tiled_custom", None)
    if custom_2d is None or custom_batched is None:
        pytest.skip("custom matmul benchmark helpers unavailable")

    a2 = _da_bf16(np.random.randn(33, 35).astype(np.float32) * 0.05)
    b2 = _da_bf16(np.random.randn(35, 31).astype(np.float32) * 0.05)
    out2 = ortho.matmul(a2, b2)
    ref2 = custom_2d(a2, b2)
    np.testing.assert_allclose(_np_bf16(out2), _np_bf16(ref2), rtol=4e-2, atol=4e-2)

    a3 = _da_bf16(np.random.randn(2, 17, 19).astype(np.float32) * 0.05)
    b3 = _da_bf16(np.random.randn(2, 19, 23).astype(np.float32) * 0.05)
    out3 = ortho.matmul(a3, b3)
    ref3 = custom_batched(a3, b3)
    np.testing.assert_allclose(_np_bf16(out3), _np_bf16(ref3), rtol=4e-2, atol=4e-2)


# Conv2d.

CONV_ATOL = 0.1  # conv accumulates many muls; BF16 needs more headroom


def _conv2d_da(inp_np, wt_np, bias_np, stride=1, padding=0):
    inp = _da_bf16(inp_np)
    wt  = _da_bf16(wt_np)
    b   = _da_bf16(bias_np)
    return ortho.conv2d(inp, wt, b, stride, stride, padding, padding)


def _conv2d_pt(inp_np, wt_np, bias_np, stride=1, padding=0):
    import torch.nn.functional as F
    inp = _pt_bf16(inp_np)
    wt  = _pt_bf16(wt_np)
    b   = torch.tensor(bias_np, dtype=torch.bfloat16, device="cuda")
    return F.conv2d(inp, wt, b, stride=stride, padding=padding)


@requires_torch_cuda
def test_conv2d_bf16_forward_1x1():
    np.random.seed(0)
    N, C_in, H, W, C_out = 1, 3, 8, 8, 4
    inp_np = np.random.randn(N, C_in, H, W).astype(np.float32)
    wt_np  = np.random.randn(C_out, C_in, 1, 1).astype(np.float32)
    bias_np = np.zeros(C_out, dtype=np.float32)
    out = _conv2d_da(inp_np, wt_np, bias_np)
    ref = _conv2d_pt(inp_np, wt_np, bias_np)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_bf16_forward_3x3():
    np.random.seed(1)
    N, C_in, H, W, C_out = 1, 2, 6, 6, 4
    inp_np = np.random.randn(N, C_in, H, W).astype(np.float32)
    wt_np  = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias_np = np.random.randn(C_out).astype(np.float32) * 0.1
    out = _conv2d_da(inp_np, wt_np, bias_np, padding=1)
    ref = _conv2d_pt(inp_np, wt_np, bias_np, padding=1)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_bf16_forward_stride2():
    np.random.seed(2)
    N, C_in, H, W, C_out = 1, 2, 8, 8, 4
    inp_np = np.random.randn(N, C_in, H, W).astype(np.float32)
    wt_np  = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias_np = np.zeros(C_out, dtype=np.float32)
    out = _conv2d_da(inp_np, wt_np, bias_np, stride=2, padding=1)
    ref = _conv2d_pt(inp_np, wt_np, bias_np, stride=2, padding=1)
    assert out.dtype == "bfloat16"
    np.testing.assert_allclose(_np_bf16(out), ref.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_bf16_backward_weight():
    np.random.seed(3)
    N, C_in, H, W, C_out = 1, 2, 6, 6, 3
    inp_np  = np.random.randn(N, C_in, H, W).astype(np.float32)
    wt_np   = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias_np = np.zeros(C_out, dtype=np.float32)

    inp_da = _da_bf16(inp_np); inp_da.requires_grad_(True)
    wt_da  = _da_bf16(wt_np);  wt_da.requires_grad_(True)
    b_da   = _da_bf16(bias_np)
    ortho.conv2d(inp_da, wt_da, b_da, 1, 1, 1, 1).backward()

    import torch.nn.functional as F
    inp_pt = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    wt_pt  = torch.tensor(wt_np,  dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b_pt   = torch.tensor(bias_np, dtype=torch.bfloat16, device="cuda")
    F.conv2d(inp_pt, wt_pt, b_pt, padding=1).sum().backward()

    np.testing.assert_allclose(
        _np_bf16(wt_da.grad), wt_pt.grad.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_bf16_backward_input():
    np.random.seed(4)
    N, C_in, H, W, C_out = 1, 2, 6, 6, 3
    inp_np  = np.random.randn(N, C_in, H, W).astype(np.float32)
    wt_np   = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias_np = np.zeros(C_out, dtype=np.float32)

    inp_da = _da_bf16(inp_np); inp_da.requires_grad_(True)
    wt_da  = _da_bf16(wt_np);  wt_da.requires_grad_(True)
    b_da   = _da_bf16(bias_np)
    ortho.conv2d(inp_da, wt_da, b_da, 1, 1, 1, 1).backward()

    import torch.nn.functional as F
    inp_pt = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    wt_pt  = torch.tensor(wt_np,  dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b_pt   = torch.tensor(bias_np, dtype=torch.bfloat16, device="cuda")
    F.conv2d(inp_pt, wt_pt, b_pt, padding=1).sum().backward()

    np.testing.assert_allclose(
        _np_bf16(inp_da.grad), inp_pt.grad.cpu().float().numpy(), atol=CONV_ATOL)


@requires_torch_cuda
def test_conv2d_bf16_backward_bias():
    np.random.seed(5)
    N, C_in, H, W, C_out = 1, 2, 6, 6, 3
    inp_np  = np.random.randn(N, C_in, H, W).astype(np.float32)
    wt_np   = np.random.randn(C_out, C_in, 3, 3).astype(np.float32)
    bias_np = np.random.randn(C_out).astype(np.float32) * 0.1

    inp_da = _da_bf16(inp_np)
    wt_da  = _da_bf16(wt_np);  wt_da.requires_grad_(True)
    b_da   = _da_bf16(bias_np); b_da.requires_grad_(True)
    ortho.conv2d(inp_da, wt_da, b_da, 1, 1, 1, 1).backward()

    import torch.nn.functional as F
    inp_pt = torch.tensor(inp_np, dtype=torch.bfloat16, device="cuda")
    wt_pt  = torch.tensor(wt_np,  dtype=torch.bfloat16, device="cuda", requires_grad=True)
    b_pt   = torch.tensor(bias_np, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    F.conv2d(inp_pt, wt_pt, b_pt, padding=1).sum().backward()

    np.testing.assert_allclose(
        _np_bf16(b_da.grad), b_pt.grad.cpu().float().numpy(), atol=CONV_ATOL)
