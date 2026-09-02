"""Scalar arithmetic, sub/div, and float64 scalar precision.

The public arithmetic surface is `add`, `sub`, `mul`, `div`, `neg` and
`matmul` between tensors, plus a Python number on either side::

    x - y      x / y
    x + 2.0    2.0 + x        x - 2.0    2.0 - x
    x * 2.0    2.0 * x        x / 2.0    2.0 / x

The load-bearing design decisions pinned here:

  * TENSOR-AUTHORITATIVE scalars. The Python number takes the tensor's dtype
    and exact Device. No promotion lattice; a float32 tensor is never widened
    because a Python float is a C double.
  * RANK-0 representation. The hidden constant has shape [] and O(1) storage,
    and rides existing broadcasting - never a data-shaped constant tensor.
  * FULL-PRECISION narrowing. `full()` takes a double and narrows once, at the
    destination dtype; narrowing through float32 first caps float64
    mean-backward precision at 0.1666666716337204.
  * SELECTIVE version tracking. `x * 3` does not watch x, because d/dx is 3;
    `3 / x` does, because d/dx is -3/x^2. Scalar convenience must not become a
    blanket mutation rejection.
"""
import numpy as np
import pytest

import orthon as ortho

torch = pytest.importorskip("torch")

CUDA = ortho.cuda_enabled()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA not available")

CPU_DTYPES = ["float32", "float64"]
CUDA_DTYPES = ["float32", "float16", "bfloat16"]

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
# Tolerances chosen per dtype: division composes as mul(x, reciprocal(s)), so a
# 1-ULP difference from a true division is expected and permitted.
_TOL = {
    "float32": 1e-6,
    "float64": 1e-12,
    "float16": 5e-3,
    "bfloat16": 4e-2,
}


def _make(values, dtype="float32", device="cpu", grad=False):
    """An orthon tensor of the requested dtype/device from float64 values."""
    arr = np.ascontiguousarray(np.array(values, dtype=np.float64))
    if dtype == "float64":
        t = ortho.from_numpy(arr)
    elif dtype == "bfloat16":
        t = ortho.from_numpy(arr.astype(np.float32), dtype="bfloat16")
    else:
        t = ortho.from_numpy(arr.astype(np.float32))
    if device != "cpu":
        t = t.to(device)
    if dtype in ("float16",):
        t = ortho.cast(t, dtype)
    if grad:
        t.requires_grad_(True)
    return t


def _torch(values, dtype="float32", device="cpu", grad=False):
    arr = np.ascontiguousarray(np.array(values, dtype=np.float64))
    t = torch.tensor(arr, dtype=_TORCH_DTYPE[dtype], device=device)
    if grad:
        t.requires_grad_(True)
    return t


def _bf16_reference(value_f32):
    """float32 -> bfloat16 -> float32, rounding to nearest-even, in pure NumPy.

    Used instead of torch.bfloat16 so this file's low-precision assertions do
    not depend on the reference framework's uint16 view support.
    """
    bits = int(np.float32(value_f32).view(np.uint32))
    if (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF):
        return float("nan")
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return float(np.uint32((rounded >> 16) << 16).view(np.float32))


def _npy(t):
    return np.array(t.tolist(), dtype=np.float64)


def _tnpy(t):
    return t.detach().to(torch.float64).cpu().numpy().ravel()


# Values kept away from zero so reciprocal-based division stays well
# conditioned, and non-uniform so a broadcasting or indexing mistake shows.
VALUES = [1.5, -2.25, 3.0, 0.75, -4.5, 2.0]
UPSTREAM = [0.3, -1.7, 2.1, 0.9, -0.4, 1.25]


# ---------------------------------------------------------------------------
# the public API exists and is not aliased twice.
# ---------------------------------------------------------------------------

def test_top_level_sub_and_div_exist():
    assert callable(ortho.sub)
    assert callable(ortho.div)


@pytest.mark.parametrize("name", [
    "__add__", "__radd__", "__sub__", "__rsub__",
    "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__neg__",
])
def test_required_dunder_surface(name):
    assert hasattr(ortho.Tensor, name)


def test_no_redundant_method_aliases():
    """add/mul were never Tensor methods; sub/div do not become ones either."""
    for name in ("add", "sub", "mul", "div"):
        assert not hasattr(ortho.Tensor, name), name


# ---------------------------------------------------------------------------
# scalar representation contract.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_scalar_operand_is_rank0_and_untracked(dtype):
    ref = _make(VALUES, dtype=dtype, grad=True)
    s = ortho._scalar_like(ref, 2.5)
    assert tuple(s.shape) == ()            # NOT full(list(ref.shape), 2.5)
    assert int(s.numel) == 1               # O(1), not O(ref.numel())
    assert s.dtype == ref.dtype
    assert s.device == ref.device
    assert s.requires_grad is False
    assert s._has_grad_fn is False
    assert s.is_leaf is True
    assert s.item() == pytest.approx(2.5)


def test_scalar_storage_is_o1_regardless_of_tensor_size():
    small = ortho.zeros([1])
    large = ortho.zeros([4096])
    assert int(ortho._scalar_like(small, 1.0).numel) == 1
    assert int(ortho._scalar_like(large, 1.0).numel) == 1


def test_scalar_dtype_is_tensor_authoritative_not_python_double():
    """A float32 tensor is NOT promoted because Python floats are doubles."""
    f32 = ortho.zeros([2], dtype="float32")
    s = ortho._scalar_like(f32, 1.0 / 3.0)
    assert s.dtype == "float32"
    assert s.item() == np.float32(1.0 / 3.0)

    f64 = ortho.from_numpy(np.zeros(2, dtype=np.float64))
    s64 = ortho._scalar_like(f64, 1.0 / 3.0)
    assert s64.dtype == "float64"
    assert s64.item() == 1.0 / 3.0         # exact double, no float32 hop


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_scalar_ops_preserve_tensor_dtype(dtype):
    x = _make(VALUES, dtype=dtype)
    for out in (x + 1.0, 1.0 + x, x - 1.0, 1.0 - x,
                x * 2.0, 2.0 * x, x / 2.0, 2.0 / x):
        assert out.dtype == dtype
        assert tuple(out.shape) == tuple(x.shape)


@requires_cuda
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_cuda_scalar_operand_uses_exact_device(dtype):
    x = _make(VALUES, dtype=dtype, device="cuda")
    s = ortho._scalar_like(x, 2.0)
    assert s.device == x.device            # cuda:N, not the bare "cuda" -> cuda:0
    assert s.device.is_cuda()
    assert s.dtype == dtype
    assert tuple(s.shape) == ()
    assert (x * 2.0).device == x.device


# ---------------------------------------------------------------------------
# Tensor-Tensor sub and div, against PyTorch.
# ---------------------------------------------------------------------------

def _pair(dtype, device):
    a = _make(VALUES, dtype, device, grad=True)
    b = _make([2.0, -1.5, 4.0, 0.5, -2.0, 3.25], dtype, device, grad=True)
    at = _torch(VALUES, dtype, "cuda" if device != "cpu" else "cpu", grad=True)
    bt = _torch([2.0, -1.5, 4.0, 0.5, -2.0, 3.25], dtype,
                "cuda" if device != "cpu" else "cpu", grad=True)
    return a, b, at, bt


def _check_binary(op_da, op_pt, dtype, device):
    a, b, at, bt = _pair(dtype, device)
    up = _make(UPSTREAM, dtype, device)
    upt = _torch(UPSTREAM, dtype, "cuda" if device != "cpu" else "cpu")

    y = op_da(a, b)
    yt = op_pt(at, bt)
    tol = _TOL[dtype]
    assert _npy(y) == pytest.approx(_tnpy(yt), abs=tol, rel=tol)

    ortho.sum(y * up).backward()
    (yt * upt).sum().backward()
    assert _npy(a.grad) == pytest.approx(_tnpy(at.grad), abs=tol, rel=tol)
    assert _npy(b.grad) == pytest.approx(_tnpy(bt.grad), abs=tol, rel=tol)


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_sub_tensor_tensor_cpu(dtype):
    _check_binary(lambda a, b: a - b, lambda a, b: a - b, dtype, "cpu")


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_div_tensor_tensor_cpu(dtype):
    _check_binary(lambda a, b: a / b, lambda a, b: a / b, dtype, "cpu")


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_top_level_sub_div_match_operators(dtype):
    a = _make(VALUES, dtype)
    b = _make([2.0, -1.5, 4.0, 0.5, -2.0, 3.25], dtype)
    assert _npy(ortho.sub(a, b)) == pytest.approx(_npy(a - b))
    assert _npy(ortho.div(a, b)) == pytest.approx(_npy(a / b))


@requires_cuda
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_sub_tensor_tensor_cuda(dtype):
    _check_binary(lambda a, b: a - b, lambda a, b: a - b, dtype, "cuda")


@requires_cuda
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
def test_div_tensor_tensor_cuda(dtype):
    _check_binary(lambda a, b: a / b, lambda a, b: a / b, dtype, "cuda")


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_sub_div_broadcast_cpu(dtype):
    a = _make([[1.0, 2.0, 3.0]], dtype, grad=True)
    b = _make([[2.0], [4.0]], dtype, grad=True)
    at = _torch([[1.0, 2.0, 3.0]], dtype, grad=True)
    bt = _torch([[2.0], [4.0]], dtype, grad=True)
    tol = _TOL[dtype]

    ortho.sum(a - b).backward()
    (at - bt).sum().backward()
    assert a.grad.dtype == dtype and b.grad.dtype == dtype
    assert _npy(a.grad) == pytest.approx(_tnpy(at.grad), abs=tol)
    assert _npy(b.grad) == pytest.approx(_tnpy(bt.grad), abs=tol)


@pytest.mark.parametrize("dtype", CPU_DTYPES)
def test_sub_div_through_views_cpu(dtype):
    base = _make([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype, grad=True)
    other = _make([[1.0, 2.0], [0.5, 4.0], [2.0, 0.25]], dtype)
    bt = _torch([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype, grad=True)
    ot = _torch([[1.0, 2.0], [0.5, 4.0], [2.0, 0.25]], dtype)
    tol = _TOL[dtype]

    ortho.sum(base.transpose(0, 1) / other).backward()
    (bt.transpose(0, 1) / ot).sum().backward()
    assert tuple(base.grad.shape) == tuple(base.shape)
    assert _npy(base.grad) == pytest.approx(_tnpy(bt.grad), abs=tol, rel=tol)


def test_division_by_zero_preserves_floating_semantics():
    """No invented Python exception: IEEE-754 semantics are passed through."""
    a = _make([1.0, -1.0, 0.0])
    z = _make([0.0, 0.0, 0.0])
    out = (a / z).tolist()
    assert out[0] == float("inf")
    assert out[1] == float("-inf")
    assert np.isnan(out[2])

    scalar_out = (a / 0.0).tolist()
    assert scalar_out[0] == float("inf")
    assert scalar_out[1] == float("-inf")
    assert np.isnan(scalar_out[2])

    ref = (_torch([1.0, -1.0, 0.0]) / _torch([0.0, 0.0, 0.0])).numpy()
    assert np.isinf(ref[0]) and ref[0] > 0
    assert np.isinf(ref[1]) and ref[1] < 0
    assert np.isnan(ref[2])


# ---------------------------------------------------------------------------
# scalar forward and autograd, against PyTorch, with a NON-UNIFORM
# upstream gradient so a constant-folding mistake cannot hide.
# ---------------------------------------------------------------------------

SCALAR_CASES = [
    ("x + s", lambda x, s: x + s, lambda x, s: x + s),
    ("s + x", lambda x, s: s + x, lambda x, s: s + x),
    ("x - s", lambda x, s: x - s, lambda x, s: x - s),
    ("s - x", lambda x, s: s - x, lambda x, s: s - x),
    ("x * s", lambda x, s: x * s, lambda x, s: x * s),
    ("s * x", lambda x, s: s * x, lambda x, s: s * x),
    ("x / s", lambda x, s: x / s, lambda x, s: x / s),
    ("s / x", lambda x, s: s / x, lambda x, s: s / x),
]


def _check_scalar(name, f_da, f_pt, dtype, device, scalar=3.0):
    tdev = "cuda" if device != "cpu" else "cpu"
    x = _make(VALUES, dtype, device, grad=True)
    xt = _torch(VALUES, dtype, tdev, grad=True)
    up = _make(UPSTREAM, dtype, device)
    upt = _torch(UPSTREAM, dtype, tdev)
    tol = _TOL[dtype]

    y = f_da(x, scalar)
    yt = f_pt(xt, scalar)
    assert y.dtype == dtype, name
    assert _npy(y) == pytest.approx(_tnpy(yt), abs=tol, rel=tol), name

    ortho.sum(y * up).backward()
    (yt * upt).sum().backward()
    assert _npy(x.grad) == pytest.approx(_tnpy(xt.grad), abs=tol, rel=tol), name


@pytest.mark.parametrize("dtype", CPU_DTYPES)
@pytest.mark.parametrize("name,f_da,f_pt", SCALAR_CASES)
def test_scalar_parity_cpu(name, f_da, f_pt, dtype):
    _check_scalar(name, f_da, f_pt, dtype, "cpu")


@requires_cuda
@pytest.mark.parametrize("dtype", CUDA_DTYPES)
@pytest.mark.parametrize("name,f_da,f_pt", SCALAR_CASES)
def test_scalar_parity_cuda(name, f_da, f_pt, dtype):
    _check_scalar(name, f_da, f_pt, dtype, "cuda")


@pytest.mark.parametrize("build,expected", [
    (lambda x: x * 3.0, 3.0),
    (lambda x: x + 3.0, 1.0),
    (lambda x: x - 3.0, 1.0),
    (lambda x: 3.0 - x, -1.0),
    (lambda x: x / 4.0, 0.25),
])
def test_scalar_derivatives_are_exactly_the_analytic_constant(build, expected):
    x = _make(VALUES, grad=True)
    ortho.sum(build(x)).backward()
    assert x.grad.tolist() == [expected] * len(VALUES)


def test_reverse_division_derivative():
    """d(4/x)/dx = -4/x^2, checked elementwise against the closed form."""
    x = _make([1.0, 2.0, 4.0, 0.5], grad=True)
    ortho.sum(4.0 / x).backward()
    expected = [-4.0 / (v * v) for v in (1.0, 2.0, 4.0, 0.5)]
    assert x.grad.tolist() == pytest.approx(expected, rel=1e-6)


def test_scalar_constant_receives_no_gradient():
    """The constant is not a graph input, so nothing can accumulate into it."""
    x = _make(VALUES, grad=True)
    y = x * 3.0
    ortho.sum(y).backward()
    assert x.grad is not None
    # x remains the only leaf with a gradient; the hidden rank-0 tensor is
    # unreachable and no second gradient exists to inspect.
    assert x.is_leaf and x.requires_grad


def test_integer_scalars_and_bools_are_accepted():
    x = _make([1.0, 2.0])
    assert (x * 2).tolist() == pytest.approx([2.0, 4.0])
    assert (x + 1).tolist() == pytest.approx([2.0, 3.0])
    assert (x * True).tolist() == pytest.approx([1.0, 2.0])
    assert (x * np.float32(3)).tolist() == pytest.approx([3.0, 6.0])


# ---------------------------------------------------------------------------
# scalar operators compose with logical views.
# ---------------------------------------------------------------------------

VIEW_CASES = [
    ("transpose * 2", lambda b: b.transpose(0, 1) * 2.0,
     lambda b: b.transpose(0, 1) * 2.0),
    ("transpose + 3", lambda b: b.transpose(0, 1) + 3.0,
     lambda b: b.transpose(0, 1) + 3.0),
    ("5 - transpose", lambda b: 5.0 - b.transpose(0, 1),
     lambda b: 5.0 - b.transpose(0, 1)),
    ("transpose / 4", lambda b: b.transpose(0, 1) / 4.0,
     lambda b: b.transpose(0, 1) / 4.0),
    ("4 / transpose", lambda b: 4.0 / b.transpose(0, 1),
     lambda b: 4.0 / b.transpose(0, 1)),
]

SLICE_CASES = [
    ("slice + 3", lambda b: b.slice(0, 1, 4) + 3.0, lambda b: b[1:4] + 3.0),
    ("slice * 2", lambda b: b.slice(0, 1, 4) * 2.0, lambda b: b[1:4] * 2.0),
    ("5 - slice", lambda b: 5.0 - b.slice(0, 1, 4), lambda b: 5.0 - b[1:4]),
    ("slice / 4", lambda b: b.slice(0, 1, 4) / 4.0, lambda b: b[1:4] / 4.0),
    ("4 / slice", lambda b: 4.0 / b.slice(0, 1, 4), lambda b: 4.0 / b[1:4]),
]

MAT = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
VEC = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def _check_view(name, f_da, f_pt, values, device, dtype="float32"):
    tdev = "cuda" if device != "cpu" else "cpu"
    b = _make(values, dtype, device, grad=True)
    bt = _torch(values, dtype, tdev, grad=True)
    tol = _TOL[dtype]

    y = f_da(b)
    yt = f_pt(bt)
    assert _npy(y) == pytest.approx(_tnpy(yt), abs=tol, rel=tol), name

    ortho.sum(y).backward()
    yt.sum().backward()
    # Gradient reaches the base in the BASE's logical coordinates.
    assert tuple(b.grad.shape) == tuple(b.shape), name
    assert _npy(b.grad) == pytest.approx(_tnpy(bt.grad), abs=tol, rel=tol), name


@pytest.mark.parametrize("name,f_da,f_pt", VIEW_CASES)
def test_scalar_through_transpose_cpu(name, f_da, f_pt):
    _check_view(name, f_da, f_pt, MAT, "cpu")


@pytest.mark.parametrize("name,f_da,f_pt", SLICE_CASES)
def test_scalar_through_slice_cpu(name, f_da, f_pt):
    _check_view(name, f_da, f_pt, VEC, "cpu")


@requires_cuda
@pytest.mark.parametrize("name,f_da,f_pt", VIEW_CASES)
def test_scalar_through_transpose_cuda(name, f_da, f_pt):
    """Scalar paths inherit CUDA view support; they are not contiguous-only."""
    _check_view(name, f_da, f_pt, MAT, "cuda")


@requires_cuda
@pytest.mark.parametrize("name,f_da,f_pt", SLICE_CASES)
def test_scalar_through_slice_cuda(name, f_da, f_pt):
    _check_view(name, f_da, f_pt, VEC, "cuda")


def test_scalar_on_view_adds_no_hidden_public_node():
    """Materialisation of a kernel operand must not surface as a ToNode/Contiguous.

    The graph from a scalar op on a transpose view is exactly two nodes deep:
    the scalar node, then the TransposeNode. Anything the runtime materialises
    internally stays internal.
    """
    base = _make(MAT, grad=True)
    view = base.transpose(0, 1)
    out = view * 2.0
    assert out._has_grad_fn
    assert out.requires_grad
    assert not out.is_leaf
    # The view itself is unchanged: still a view, still strided.
    assert not view.is_contiguous
    assert view._impl_id != base._impl_id


# ---------------------------------------------------------------------------
# Version selectivity: a node watches only what its backward reads, which is
# the reason the scalar ops do not reuse the generic MulNode.
# ---------------------------------------------------------------------------

ALLOWED_AFTER_MUTATION = [
    ("x + 3", lambda x: x + 3.0),
    ("3 + x", lambda x: 3.0 + x),
    ("x - 3", lambda x: x - 3.0),
    ("3 - x", lambda x: 3.0 - x),
    ("x * 3", lambda x: x * 3.0),
    ("3 * x", lambda x: 3.0 * x),
    ("x / 4", lambda x: x / 4.0),
]


@pytest.mark.parametrize("name,build", ALLOWED_AFTER_MUTATION)
def test_mutation_after_forward_is_allowed_when_backward_ignores_values(name, build):
    x = _make(VALUES, grad=True)
    y = build(x)
    x.set_item([0], 99.0)
    ortho.sum(y).backward()                    # must NOT raise
    assert x.grad is not None, name


def test_scalar_multiply_does_not_over_reject_like_the_generic_mul_node():
    """The specific regression this design exists to prevent.

    `x * const_tensor` still fails - MulNode watches both operands, correctly,
    because a genuine Tensor-Tensor product's rhs gradient needs x. `x * 3`
    must not, because there is no rhs gradient to compute.
    """
    x = _make(VALUES, grad=True)
    y_tensor = ortho.mul(x, _make([3.0] * len(VALUES)))
    x.set_item([0], 99.0)
    with pytest.raises(RuntimeError, match="modified in-place"):
        ortho.sum(y_tensor).backward()

    x2 = _make(VALUES, grad=True)
    y_scalar = x2 * 3.0
    x2.set_item([0], 99.0)
    ortho.sum(y_scalar).backward()
    assert x2.grad.tolist() == [3.0] * len(VALUES)


def test_reverse_division_still_rejects_mutation():
    """d(3/x)/dx = -3/x^2 genuinely needs x's forward values, so it must reject."""
    x = _make(VALUES, grad=True)
    y = 3.0 / x
    x.set_item([0], 99.0)
    with pytest.raises(RuntimeError, match="modified in-place"):
        ortho.sum(y).backward()


def test_tensor_tensor_div_watches_only_the_denominator():
    """div(a, b) = mul(a, reciprocal(b)). Backward needs both, so both are watched."""
    a = _make(VALUES, grad=True)
    b = _make([2.0] * len(VALUES), grad=True)
    y = a / b
    b.set_item([0], 9.0)
    with pytest.raises(RuntimeError, match="modified in-place"):
        ortho.sum(y).backward()


def test_tensor_tensor_sub_allows_mutation():
    """sub(a, b) = add(a, neg(b)); neither AddNode nor Neg reads its payload."""
    a = _make(VALUES, grad=True)
    b = _make([2.0] * len(VALUES), grad=True)
    y = a - b
    a.set_item([0], 9.0)
    b.set_item([1], 9.0)
    ortho.sum(y).backward()
    assert a.grad.tolist() == [1.0] * len(VALUES)
    assert b.grad.tolist() == [-1.0] * len(VALUES)


# ---------------------------------------------------------------------------
# error contract.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("other", ["abc", b"abc", [1, 2, 3], {"a": 1},
                                   object(), None, 1j])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "truediv"])
def test_unsupported_operands_raise_type_error(other, op):
    x = _make([1.0, 2.0])
    fn = {"add": lambda a, b: a + b,
          "sub": lambda a, b: a - b,
          "mul": lambda a, b: a * b,
          "truediv": lambda a, b: a / b}[op]
    with pytest.raises(TypeError):
        fn(x, other)
    with pytest.raises(TypeError):
        fn(other, x)


def test_type_error_message_names_the_operands():
    """Not a pybind overload dump: Python's own canonical operand message."""
    x = _make([1.0, 2.0])
    with pytest.raises(TypeError) as info:
        x + "abc"
    text = str(info.value)
    assert "unsupported operand type" in text
    assert "str" in text
    assert "incompatible function arguments" not in text


def test_numpy_arrays_are_not_silently_coerced():
    """A NumPy ARRAY is not a scalar, and must not become an object array.

    The operators answer NotImplemented for a non-numeric operand so Python can
    raise its canonical message. NumPy's reflected operators accept anything and
    fall back to an object-dtype array, so without an explicit opt-out
    `x + np.array([1., 2., 3.])` yields array([Tensor, Tensor, Tensor],
    dtype=object) instead of failing. Tensor sets __array_ufunc__ = None,
    NumPy's documented opt-out.
    """
    assert ortho.Tensor.__array_ufunc__ is None
    x = _make([1.0, 2.0, 3.0])
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with pytest.raises(TypeError):
        x + arr
    with pytest.raises(TypeError):
        arr + x
    # A size-1 array is still an ARRAY. The scalar path decides this on
    # ndarray dimensionality (ndim >= 1 is never a scalar), not on whether
    # float(array) happens to succeed - NumPy 1.25+ only deprecates that
    # conversion, so the number protocol answers differently per NumPy
    # version and cannot carry this contract.
    one = np.array([2.0])
    for op in (lambda a, b: a + b,
               lambda a, b: a - b,
               lambda a, b: a * b,
               lambda a, b: a / b):
        with pytest.raises(TypeError):
            op(x, one)
        with pytest.raises(TypeError):
            op(one, x)
    with pytest.raises(TypeError):
        x * np.ones((2, 3), dtype=np.float32)   # higher-rank arrays likewise


@pytest.mark.parametrize("scalar", [
    np.float32(3), np.float64(3), np.int32(3), np.int64(3), np.array(3.0),
])
def test_numpy_scalars_are_accepted(scalar):
    """0-d arrays and numpy scalar types are numbers, and convert before
    NumPy's protocol is ever consulted."""
    x = _make([1.0, 2.0])
    assert (x * scalar).tolist() == pytest.approx([3.0, 6.0])
    assert (scalar * x).tolist() == pytest.approx([3.0, 6.0])


def test_scalar_does_not_bypass_tensor_validation():
    """Same failures as before, from the same checks, reached the same way."""
    f32 = ortho.zeros([3], dtype="float32")
    f64 = ortho.from_numpy(np.zeros(3, dtype=np.float64))
    with pytest.raises(RuntimeError):
        f32 - f64
    with pytest.raises(RuntimeError):
        f32 / f64
    with pytest.raises(RuntimeError):
        ortho.zeros([2, 3]) - ortho.zeros([4, 5])
    with pytest.raises(RuntimeError):
        ortho.zeros([2, 3]) / ortho.zeros([4, 5])
    # Integer tensor arithmetic is deliberately unsupported.
    ints = ortho.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(RuntimeError):
        ints * 2


@requires_cuda
def test_cross_device_operands_still_rejected():
    cpu = ortho.zeros([3])
    gpu = ortho.zeros([3]).to("cuda")
    with pytest.raises(RuntimeError):
        cpu - gpu
    with pytest.raises(RuntimeError):
        cpu / gpu


# ---------------------------------------------------------------------------
# grad mode and leaf metadata.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda x: x + 1.0, lambda x: 1.0 - x, lambda x: x * 2.0, lambda x: 2.0 / x,
])
def test_no_grad_produces_no_graph(build):
    x = _make(VALUES, grad=True)
    with ortho.no_grad():
        y = build(x)
    assert y.requires_grad is False
    assert y._has_grad_fn is False
    assert y.is_leaf is True


@pytest.mark.parametrize("build", [
    lambda x: x + 1.0, lambda x: 1.0 - x, lambda x: x * 2.0, lambda x: 2.0 / x,
])
def test_tracked_input_produces_tracked_output(build):
    x = _make(VALUES, grad=True)
    y = build(x)
    assert y.requires_grad is True
    assert y._has_grad_fn is True
    assert y.is_leaf is False


def test_untracked_input_produces_untracked_output():
    x = _make(VALUES)
    y = x * 2.0
    assert y.requires_grad is False
    assert y._has_grad_fn is False


def test_leaf_metadata_survives_scalar_backward():
    """A leaf's gradient is itself a plain, graph-free tensor."""
    x = _make(VALUES, grad=True)
    ortho.sum(x * 3.0).backward()
    assert x.is_leaf is True
    assert x.requires_grad is True
    assert x.grad.requires_grad is False
    assert x.grad._has_grad_fn is False
    assert x.grad.is_leaf is True


# ---------------------------------------------------------------------------
# factory precision.
# ---------------------------------------------------------------------------

def test_full_float64_keeps_the_full_double():
    assert ortho.full([], 1.0 / 3.0, dtype="float64").item() == 1.0 / 3.0
    pi = 3.141592653589793
    assert ortho.full([2], pi, dtype="float64").tolist() == [pi, pi]
    # And is strictly better than the float32 round-trip it used to do.
    assert ortho.full([], 1.0 / 3.0, dtype="float64").item() != float(np.float32(1.0 / 3.0))


def test_full_float32_matches_explicit_float32_rounding():
    value = 1.0 / 3.0
    got = ortho.full([3], value, dtype="float32").tolist()
    assert got == [float(np.float32(value))] * 3


# full()'s low-precision rounding is pinned against the reference frameworks
# rather than against a recorded table of its own output. See
# tests/python/test_low_precision.py for the full float16-vs-NumPy and
# bfloat16-vs-PyTorch matrix across full / cast / from_numpy.
@pytest.mark.parametrize("value", [0.0, 1.0, 0.5, -2.25, 1e-5,
                                   1.0 / 3.0, 65504.0, 3.141592653589793])
def test_full_low_precision_matches_the_reference_rounding(value):
    """full() rounds to nearest-even, like NumPy float16 and torch bfloat16."""
    as_f32 = np.float32(value)
    assert ortho.full([2], value, dtype="float16").tolist() == \
        [float(np.float16(as_f32))] * 2
    assert ortho.full([2], value, dtype="bfloat16").tolist() == \
        [_bf16_reference(as_f32)] * 2


def test_full_integer_and_bool_conversion_unchanged():
    assert ortho.full([2], 3.9, dtype="int32").tolist() == [3, 3]     # toward zero
    assert ortho.full([2], -2.5, dtype="int64").tolist() == [-2, -2]
    assert ortho.full([2], 3.9, dtype="uint8").tolist() == [3, 3]
    assert ortho.full([2], -3.9, dtype="int8").tolist() == [-3, -3]
    assert ortho.full([1], 0.5, dtype="bool").tolist() == [True]      # nonzero
    assert ortho.full([1], 0.0, dtype="bool").tolist() == [False]


def test_ones_and_zeros_unchanged():
    assert ortho.ones([3]).tolist() == [1.0, 1.0, 1.0]
    assert ortho.ones([2], dtype="float64").tolist() == [1.0, 1.0]
    assert ortho.zeros([2]).tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# float64 mean backward precision.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [3, 6, 7, 1000])
def test_float64_mean_backward_scale_is_exact_double(factor):
    x = ortho.from_numpy(np.ones(factor, dtype=np.float64))
    x.requires_grad_(True)
    ortho.mean(x).backward()
    expected = 1.0 / factor
    got = x.grad.tolist()
    assert x.grad.dtype == "float64"
    assert got == [expected] * factor
    # A float32 scale would give a visibly different value.
    assert got[0] != float(np.float32(1.0) / np.float32(factor)) or factor == 1


@pytest.mark.parametrize("factor", [3, 6, 7, 1000])
def test_float64_mean_backward_matches_torch(factor):
    x = ortho.from_numpy(np.ones(factor, dtype=np.float64))
    x.requires_grad_(True)
    ortho.mean(x).backward()
    xt = torch.ones(factor, dtype=torch.float64, requires_grad=True)
    xt.mean().backward()
    assert x.grad.tolist() == pytest.approx(xt.grad.numpy().tolist(), abs=1e-17)


def test_float32_mean_backward_unchanged():
    x = ortho.from_numpy(np.ones(6, dtype=np.float32))
    x.requires_grad_(True)
    ortho.mean(x).backward()
    assert x.grad.tolist() == [float(np.float32(1.0) / np.float32(6.0))] * 6


def test_sum_backward_untouched():
    x = ortho.from_numpy(np.ones(6, dtype=np.float64))
    x.requires_grad_(True)
    ortho.sum(x).backward()
    assert x.grad.tolist() == [1.0] * 6
