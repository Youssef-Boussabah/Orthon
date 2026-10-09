"""Dtype infrastructure plumbing tests.

Covers creation, from_numpy, tolist/item, CPU/CUDA round-trips, and
non-contiguous contiguous() copies for all newly-supported dtypes.
No arithmetic ops are tested here.
"""
from tests.support.capabilities import orthon_cuda_available as _cuda_available
import pytest
import numpy as np
import orthon as ortho




cuda_available = _cuda_available()
requires_cuda = pytest.mark.skipif(not cuda_available, reason="CUDA not available")


# from_numpy dtype detection and round-trip shape.

def test_from_numpy_float32():
    t = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    assert t.dtype == "float32"
    assert t.shape == (2,)


def test_from_numpy_float64():
    t = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float64))
    assert t.dtype == "float64"
    assert t.shape == (2,)


def test_from_numpy_float16():
    t = ortho.from_numpy(np.array([1.0, 2.0], dtype=np.float16))
    assert t.dtype == "float16"
    assert t.shape == (2,)


def test_from_numpy_int32():
    t = ortho.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    assert t.dtype == "int32"


def test_from_numpy_int64():
    t = ortho.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    assert t.dtype == "int64"


def test_from_numpy_uint8():
    t = ortho.from_numpy(np.array([0, 127, 255], dtype=np.uint8))
    assert t.dtype == "uint8"
    assert t.shape == (3,)


def test_from_numpy_int8():
    t = ortho.from_numpy(np.array([-128, 0, 127], dtype=np.int8))
    assert t.dtype == "int8"
    assert t.shape == (3,)


def test_from_numpy_bool():
    t = ortho.from_numpy(np.array([True, False, True], dtype=np.bool_))
    assert t.dtype == "bool"
    assert t.shape == (3,)


def test_from_numpy_unsupported_raises():
    with pytest.raises((TypeError, Exception)):
        ortho.from_numpy(np.array([1+2j], dtype=np.complex64))


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("order", ["=", "<", ">"])
def test_r3_numpy_byte_order_preserves_values_bits_and_input(dtype, order):
    if dtype.startswith("float"):
        native = np.array([1.5, -2.25, 0., -0., np.inf, -np.inf, np.nan, .125], dtype=dtype)
    else:
        bounds = np.iinfo(dtype)
        native = np.array([bounds.min, -257, -1, 0, 1, 257, bounds.max], dtype=dtype)
    source = native.astype(native.dtype.newbyteorder(order))
    original = source.tobytes()
    result = ortho.from_numpy(source)
    assert result.dtype == dtype and result.shape == source.shape
    assert ortho._tensor_to_bytes(result) == native.tobytes()
    assert source.tobytes() == original and source.dtype.byteorder == native.dtype.newbyteorder(order).byteorder
    np.testing.assert_equal(result.tolist(), native.tolist())
    source.fill(0)
    assert ortho._tensor_to_bytes(result) == native.tobytes()


@pytest.mark.parametrize("dtype,values", [
    ("int8", [-128, -1, 0, 127]), ("uint8", [0, 1, 127, 255]),
    ("bool", [True, False, True, False]),
])
def test_r3_numpy_byte_order_independent_types(dtype, values):
    source = np.array(values, dtype=dtype)
    assert source.dtype.byteorder == "|"
    result = ortho.from_numpy(source)
    assert result.dtype == dtype and result.tolist() == values
    assert ortho._tensor_to_bytes(result) == source.tobytes()


@pytest.mark.parametrize("layout", ["scalar", "empty", "matrix", "offset"])
def test_r3_foreign_numpy_layout_and_lifetime(layout):
    native = np.arange(12, dtype=np.float32).reshape(3, 4)
    if layout == "scalar":
        native = np.array(-0., dtype=np.float32)
    elif layout == "empty":
        native = native[:0]
    elif layout == "offset":
        native = native[1:]
    foreign = np.dtype("f4").newbyteorder("S")
    # Build the offset after conversion so the supplied foreign array is a view.
    source = (np.arange(12, dtype=np.float32).astype(foreign).reshape(3, 4)[1:]
              if layout == "offset" else native.astype(foreign))
    assert source.flags.c_contiguous
    result = ortho.from_numpy(source)
    assert result.shape == native.shape
    assert ortho._tensor_to_bytes(result) == native.tobytes()
    del source
    assert ortho._tensor_to_bytes(result) == native.tobytes()


@pytest.mark.parametrize("order", ["=", "<", ">"])
def test_r3_bfloat16_endian_rounding_and_nonfinite_bits(order):
    words = np.array([0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000,
                      0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345], dtype=np.uint32)
    native = words.view(np.float32)
    source = native.astype(native.dtype.newbyteorder(order))
    original = source.tobytes()
    result = ortho.from_numpy(source, dtype="bfloat16")
    expected = np.array([0x3f80, 0x3f82, 0xbf80, 0xbf82, 0, 0x8000,
                         0x7f80, 0xff80, 0x7fc1], dtype=np.uint16).tobytes()
    assert ortho._tensor_to_bytes(result) == expected
    assert ortho._tensor_to_bytes(ortho.from_numpy(native, dtype="bfloat16")) == expected
    assert source.tobytes() == original


@pytest.mark.parametrize("foreign", [False, True])
def test_r3_bfloat16_legal_unaligned_contiguous_input(foreign):
    dtype = np.dtype("f4").newbyteorder("S" if foreign else "=")
    backing = bytearray(13)
    source = np.ndarray((3,), dtype=dtype, buffer=backing, offset=1)
    source[:] = [1., -2., .5]
    assert source.flags.c_contiguous and not source.flags.aligned
    original = bytes(backing)
    assert ortho.from_numpy(source, dtype="bfloat16").tolist() == [1., -2., .5]
    assert bytes(backing) == original


@pytest.mark.parametrize("override", [None, "bfloat16"])
def test_r3_foreign_numpy_preserves_layout_rejection(override):
    source = np.arange(12, dtype=np.float32).astype(np.dtype("f4").newbyteorder("S")).reshape(3, 4).T
    with pytest.raises(ValueError, match="from_numpy: expected a C-contiguous array"):
        ortho.from_numpy(source, dtype=override)


@pytest.mark.parametrize("dtype", ["complex64", "object", "uint32"])
def test_r3_numpy_unsupported_dtype_stays_rejected(dtype):
    with pytest.raises(TypeError, match="from_numpy: unsupported array dtype"):
        ortho.from_numpy(np.array([1], dtype=dtype))


def test_r3_numpy_override_does_not_add_implicit_casts():
    with pytest.raises(TypeError, match="explicit dtype= override only supports"):
        ortho.from_numpy(np.array([1.], dtype=np.float64), dtype="float32")
    with pytest.raises(TypeError, match="requires a float32 input array"):
        ortho.from_numpy(np.array([1.], dtype=np.float64), dtype="bfloat16")
    with pytest.raises(TypeError, match="unsupported array dtype"):
        ortho.from_numpy(np.zeros(1, dtype=[("value", ">f4")]))


@requires_cuda
def test_r3_normalized_numpy_values_survive_cuda_transfer():
    native = np.array([[1.5, -2.25], [-0., .125]], dtype=np.float32)
    source = native.astype(native.dtype.newbyteorder("S"))
    result = ortho.from_numpy(source).to("cuda")
    assert result.dtype == "float32"
    assert ortho._tensor_to_bytes(result.to("cpu")) == native.tobytes()


# fill ops with dtype keyword.

def test_zeros_default_float32():
    t = ortho.zeros([2, 3])
    assert t.dtype == "float32"
    assert t.shape == (2, 3)


def test_zeros_float64():
    t = ortho.zeros([2, 3], dtype="float64")
    assert t.dtype == "float64"
    assert t.tolist() == pytest.approx([0.0] * 6)


def test_zeros_float16():
    t = ortho.zeros([4], dtype="float16")
    assert t.dtype == "float16"
    lst = t.tolist()
    assert all(v == pytest.approx(0.0) for v in lst)


def test_zeros_bfloat16():
    t = ortho.zeros([3], dtype="bfloat16")
    assert t.dtype == "bfloat16"
    lst = t.tolist()
    assert all(v == pytest.approx(0.0) for v in lst)


def test_zeros_int32():
    t = ortho.zeros([3], dtype="int32")
    assert t.dtype == "int32"
    assert t.tolist() == [0, 0, 0]


def test_zeros_int64():
    t = ortho.zeros([3], dtype="int64")
    assert t.dtype == "int64"
    assert t.tolist() == [0, 0, 0]


def test_zeros_uint8():
    t = ortho.zeros([3], dtype="uint8")
    assert t.dtype == "uint8"
    assert t.tolist() == [0, 0, 0]


def test_zeros_int8():
    t = ortho.zeros([3], dtype="int8")
    assert t.dtype == "int8"
    assert t.tolist() == [0, 0, 0]


def test_zeros_bool():
    t = ortho.zeros([3], dtype="bool")
    assert t.dtype == "bool"
    assert t.tolist() == [False, False, False]


def test_ones_float64():
    t = ortho.ones([2], dtype="float64")
    assert t.dtype == "float64"
    assert t.tolist() == pytest.approx([1.0, 1.0])


def test_ones_int32():
    t = ortho.ones([3], dtype="int32")
    assert t.dtype == "int32"
    assert t.tolist() == [1, 1, 1]


def test_ones_uint8():
    t = ortho.ones([2], dtype="uint8")
    assert t.dtype == "uint8"
    assert t.tolist() == [1, 1]


def test_ones_bool():
    t = ortho.ones([2], dtype="bool")
    assert t.dtype == "bool"
    assert t.tolist() == [True, True]


def test_full_uint8():
    t = ortho.full([4], 200.0, dtype="uint8")
    assert t.dtype == "uint8"
    assert t.tolist() == [200, 200, 200, 200]


def test_full_int8():
    t = ortho.full([2], -5.0, dtype="int8")
    assert t.dtype == "int8"
    assert t.tolist() == [-5, -5]


def test_full_float64():
    t = ortho.full([3], 3.14, dtype="float64")
    assert t.dtype == "float64"
    assert t.tolist() == pytest.approx([3.14, 3.14, 3.14], rel=1e-6)


def test_full_float16():
    t = ortho.full([2], 1.0, dtype="float16")
    assert t.dtype == "float16"
    lst = t.tolist()
    assert all(abs(v - 1.0) < 0.01 for v in lst)


def test_empty_shape_preserved():
    for dt in ("float32", "float64", "float16", "bfloat16",
               "int32", "int64", "uint8", "int8", "bool"):
        t = ortho.empty([2, 3], dtype=dt)
        assert t.dtype == dt
        assert t.shape == (2, 3)


# tolist and item for supported dtypes.

def test_tolist_float64():
    arr = np.array([1.0, 2.5, 3.125], dtype=np.float64)
    t = ortho.from_numpy(arr)
    assert t.tolist() == pytest.approx([1.0, 2.5, 3.125], rel=1e-12)


def test_tolist_float16():
    arr = np.array([1.0, 2.0, 0.5], dtype=np.float16)
    t = ortho.from_numpy(arr)
    lst = t.tolist()
    assert len(lst) == 3
    assert abs(lst[0] - 1.0) < 0.01
    assert abs(lst[1] - 2.0) < 0.01
    assert abs(lst[2] - 0.5) < 0.01


def test_tolist_uint8():
    arr = np.array([0, 100, 200, 255], dtype=np.uint8)
    t = ortho.from_numpy(arr)
    assert t.tolist() == [0, 100, 200, 255]


def test_tolist_int8():
    arr = np.array([-128, -1, 0, 127], dtype=np.int8)
    t = ortho.from_numpy(arr)
    assert t.tolist() == [-128, -1, 0, 127]


def test_tolist_bool():
    arr = np.array([True, False, True, False], dtype=np.bool_)
    t = ortho.from_numpy(arr)
    assert t.tolist() == [True, False, True, False]


def test_item_float64():
    arr = np.array([3.141592653589793], dtype=np.float64)
    t = ortho.from_numpy(arr)
    assert abs(t.item() - 3.141592653589793) < 1e-12


def test_item_float16_one():
    arr = np.array([1.0], dtype=np.float16)
    t = ortho.from_numpy(arr)
    assert abs(t.item() - 1.0) < 0.01


def test_item_float16_zero():
    arr = np.array([0.0], dtype=np.float16)
    t = ortho.from_numpy(arr)
    assert t.item() == pytest.approx(0.0)


def test_item_uint8_boundary():
    for val in (0, 127, 255):
        arr = np.array([val], dtype=np.uint8)
        t = ortho.from_numpy(arr)
        assert t.item() == val


def test_item_int8_boundary():
    for val in (-128, 0, 127):
        arr = np.array([val], dtype=np.int8)
        t = ortho.from_numpy(arr)
        assert t.item() == val


def test_item_bool_true():
    t = ortho.from_numpy(np.array([True], dtype=np.bool_))
    assert t.item() is True


def test_item_bool_false():
    t = ortho.from_numpy(np.array([False], dtype=np.bool_))
    assert t.item() is False


# CPU to CUDA to CPU round-trips.

@requires_cuda
def test_cuda_roundtrip_float64():
    arr = np.array([1.5, -2.5, 3.0], dtype=np.float64)
    cpu = ortho.from_numpy(arr)
    gpu = cpu.to("cuda")
    assert gpu.dtype == "float64"
    back = gpu.to("cpu")
    assert back.dtype == "float64"
    assert back.tolist() == pytest.approx([1.5, -2.5, 3.0], rel=1e-12)


@requires_cuda
def test_cuda_roundtrip_float16():
    arr = np.array([1.0, 2.0, 0.5], dtype=np.float16)
    cpu = ortho.from_numpy(arr)
    gpu = cpu.to("cuda")
    assert gpu.dtype == "float16"
    back = gpu.to("cpu")
    assert back.dtype == "float16"
    lst = back.tolist()
    assert abs(lst[0] - 1.0) < 0.01
    assert abs(lst[1] - 2.0) < 0.01


@requires_cuda
def test_cuda_roundtrip_uint8():
    arr = np.array([0, 128, 255], dtype=np.uint8)
    cpu = ortho.from_numpy(arr)
    gpu = cpu.to("cuda")
    back = gpu.to("cpu")
    assert back.dtype == "uint8"
    assert back.tolist() == [0, 128, 255]


@requires_cuda
def test_cuda_roundtrip_int8():
    arr = np.array([-128, 0, 127], dtype=np.int8)
    cpu = ortho.from_numpy(arr)
    gpu = cpu.to("cuda")
    back = gpu.to("cpu")
    assert back.dtype == "int8"
    assert back.tolist() == [-128, 0, 127]


@requires_cuda
def test_cuda_roundtrip_bool():
    arr = np.array([True, False, True, False], dtype=np.bool_)
    cpu = ortho.from_numpy(arr)
    gpu = cpu.to("cuda")
    back = gpu.to("cpu")
    assert back.dtype == "bool"
    assert back.tolist() == [True, False, True, False]


# Non-contiguous contiguous() copies for non-float dtypes.

def test_noncontiguous_contiguous_int32():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    t = ortho.from_numpy(arr)
    t_T = t.transpose(0, 1)        # shape (3, 2), non-contiguous
    assert not t_T.is_contiguous
    t_c = t_T.contiguous()
    assert t_c.is_contiguous
    assert t_c.shape == (3, 2)
    # arr.T flattened in C order: col 0 = [0,3], col 1 = [1,4], col 2 = [2,5]
    assert t_c.tolist() == [0, 3, 1, 4, 2, 5]


def test_noncontiguous_contiguous_int64():
    arr = np.arange(4, dtype=np.int64).reshape(2, 2)
    t = ortho.from_numpy(arr)
    t_T = t.transpose(0, 1)
    t_c = t_T.contiguous()
    assert t_c.tolist() == [0, 2, 1, 3]


def test_noncontiguous_contiguous_float64():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    t = ortho.from_numpy(arr)
    t_T = t.transpose(0, 1)
    t_c = t_T.contiguous()
    assert t_c.tolist() == pytest.approx([1.0, 3.0, 2.0, 4.0])


def test_noncontiguous_contiguous_uint8():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    t = ortho.from_numpy(arr)
    t_T = t.transpose(0, 1)
    t_c = t_T.contiguous()
    assert t_c.shape == (3, 2)
    assert t_c.tolist() == [1, 4, 2, 5, 3, 6]


def test_noncontiguous_contiguous_bool():
    arr = np.array([[True, False], [False, True]], dtype=np.bool_)
    t = ortho.from_numpy(arr)
    t_T = t.transpose(0, 1)
    t_c = t_T.contiguous()
    assert t_c.tolist() == [True, False, False, True]


def test_noncontiguous_contiguous_float16():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
    t = ortho.from_numpy(arr)
    t_T = t.transpose(0, 1)
    t_c = t_T.contiguous()
    lst = t_c.tolist()
    assert abs(lst[0] - 1.0) < 0.01
    assert abs(lst[1] - 3.0) < 0.1
    assert abs(lst[2] - 2.0) < 0.01
    assert abs(lst[3] - 4.0) < 0.1


# DType enum accessibility from Python.

def test_dtype_enum_values():
    assert ortho.DType.Float32  is not None
    assert ortho.DType.Float64  is not None
    assert ortho.DType.Float16  is not None
    assert ortho.DType.BFloat16 is not None
    assert ortho.DType.Int32    is not None
    assert ortho.DType.Int64    is not None
    assert ortho.DType.UInt8    is not None
    assert ortho.DType.Int8     is not None
    assert ortho.DType.Bool     is not None
