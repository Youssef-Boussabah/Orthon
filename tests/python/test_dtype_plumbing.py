"""Dtype infrastructure plumbing tests.

Covers creation, from_numpy, tolist/item, CPU/CUDA round-trips, and
non-contiguous contiguous() copies for all newly-supported dtypes.
No arithmetic ops are tested here.
"""
import pytest
import numpy as np
import orthon as ortho


def _cuda_available():
    try:
        ortho.zeros([1]).to("cuda")
        return True
    except RuntimeError:
        return False


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
