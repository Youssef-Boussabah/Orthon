import pytest
import numpy as np
import orthon as ortho


def test_zeros_shape():
    t = ortho.zeros([2, 3])
    assert t.shape == (2, 3)


def test_ones_numel():
    t = ortho.ones([3, 4])
    assert t.numel == 12


def test_full_tolist():
    t = ortho.full([2, 3], 5.0)
    assert t.tolist() == pytest.approx([5.0] * 6)


def test_empty_shape():
    t = ortho.empty([4, 5])
    assert t.shape == (4, 5)


def test_is_contiguous():
    t = ortho.zeros([2, 3])
    assert t.is_contiguous is True


def test_ndim():
    t = ortho.zeros([2, 3, 4])
    assert t.ndim == 3


def test_storage_offset_default():
    t = ortho.zeros([2, 3])
    assert t.storage_offset == 0


def test_item():
    t = ortho.full([1], 7.0)
    assert t.item() == pytest.approx(7.0)


def test_strides_contiguous():
    t = ortho.zeros([2, 3])
    assert t.strides == (3, 1)


def test_repr():
    t = ortho.zeros([2, 3])
    r = repr(t)
    assert "Tensor" in r
    assert "2" in r
    assert "3" in r


# fill_.

def test_fill_basic():
    t = ortho.zeros([2, 3])
    t.fill_(7.0)
    assert t.tolist() == pytest.approx([7.0] * 6)


def test_fill_returns_self():
    t = ortho.zeros([3])
    result = t.fill_(5.0)
    assert result.tolist() == pytest.approx([5.0] * 3)


def test_fill_module_level():
    t = ortho.zeros([2, 2])
    ortho.fill_(t, 3.0)
    assert t.tolist() == pytest.approx([3.0] * 4)


def test_fill_non_contiguous():
    # fill_ a transposed (non-contiguous) view, then read back via original
    t = ortho.zeros([2, 3])
    tr = t.transpose(0, 1)       # shape (3,2), non-contiguous
    tr.fill_(9.0)
    # original storage is shared; all 6 elements must now be 9.0
    assert t.tolist() == pytest.approx([9.0] * 6)


# __getitem__.

def test_getitem_1d_shape():
    t = ortho.ones([5])
    elem = t[0]
    assert elem.shape == ()       # 0-d view
    assert elem.numel == 1


def test_getitem_1d_value():
    t = ortho.zeros([5])
    t.fill_(4.0)
    assert t[2].item() == pytest.approx(4.0)


def test_getitem_negative_index():
    t = ortho.zeros([5])
    t.fill_(4.0)
    assert t[-1].item() == pytest.approx(4.0)


def test_getitem_2d_returns_row():
    t = ortho.zeros([3, 4])
    row = t[1]
    assert row.shape == (4,)
    assert row.numel == 4


def test_getitem_out_of_range():
    t = ortho.zeros([3])
    with pytest.raises((IndexError, RuntimeError)):
        _ = t[99]


# tolist on non-contiguous tensors.

def test_tolist_transposed():
    # [1,2,3,4,5,6] stored row-major as [[1,2,3],[4,5,6]]
    # after transpose shape is (3,2), logical order is [[1,4],[2,5],[3,6]]
    t = ortho.zeros([2, 3])
    t.fill_(1.0)
    tr = t.transpose(0, 1)
    vals = tr.tolist()
    assert len(vals) == 6
    assert vals == pytest.approx([1.0] * 6)


# Error messages.

def test_reshape_error_contains_shapes():
    t = ortho.zeros([2, 3])
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ortho.reshape(t, [2, 4])
    msg = str(exc.value)
    assert "2" in msg and "3" in msg   # original shape present
    assert "2" in msg and "4" in msg   # target shape present


def test_broadcast_error_contains_shapes():
    a = ortho.zeros([2, 3])
    b = ortho.zeros([4])
    with pytest.raises(RuntimeError) as exc:
        _ = a + b
    msg = str(exc.value)
    assert "3" in msg or "4" in msg    # conflicting dimensions present


# from_numpy.

def test_from_numpy_preserves_shape():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = ortho.from_numpy(arr)
    assert t.shape == (2, 3)


def test_from_numpy_preserves_values():
    arr = np.array([[1.0, 2.5], [-3.0, 4.0]], dtype=np.float32)
    t = ortho.from_numpy(arr)
    assert t.tolist() == pytest.approx(arr.flatten().tolist())


def test_from_numpy_copies_data_independently():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = ortho.from_numpy(arr)
    arr[0] = 99.0
    assert t.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_from_numpy_float64_roundtrip():
    arr = np.array([1.0, 2.0], dtype=np.float64)
    t = ortho.from_numpy(arr)
    assert t.dtype == "float64"
    assert t.tolist() == pytest.approx(arr.tolist(), rel=1e-12)


def test_from_numpy_rejects_non_contiguous_input():
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    with pytest.raises(ValueError):
        ortho.from_numpy(arr)
