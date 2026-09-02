import pytest
import orthon as ortho


# get_item.

def test_get_item_contiguous():
    t = ortho.full([2, 3], 5.0)
    assert t.get_item([0, 0]) == pytest.approx(5.0)
    assert t.get_item([1, 2]) == pytest.approx(5.0)


def test_get_item_after_set():
    t = ortho.zeros([3, 3])
    t.set_item([1, 2], 7.0)
    assert t.get_item([1, 2]) == pytest.approx(7.0)
    assert t.get_item([0, 0]) == pytest.approx(0.0)


def test_get_item_negative_index():
    t = ortho.zeros([3, 4])
    t.set_item([2, 3], 9.0)
    assert t.get_item([-1, -1]) == pytest.approx(9.0)


def test_get_item_transposed():
    t = ortho.zeros([2, 3])
    t.set_item([0, 1], 42.0)
    tr = t.transpose(0, 1)   # shape (3, 2)
    assert tr.get_item([1, 0]) == pytest.approx(42.0)


def test_get_item_wrong_ndim():
    t = ortho.zeros([2, 3])
    with pytest.raises((ValueError, RuntimeError)):
        t.get_item([0])         # 1 index for 2-d tensor


# set_item.

def test_set_item_basic():
    t = ortho.zeros([4])
    t.set_item([2], 3.5)
    assert t.get_item([2]) == pytest.approx(3.5)
    assert t.get_item([0]) == pytest.approx(0.0)


def test_set_item_all_elements():
    t = ortho.zeros([2, 2])
    for i in range(2):
        for j in range(2):
            t.set_item([i, j], float(i * 2 + j))
    assert t.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_set_item_transposed():
    t = ortho.zeros([2, 3])
    tr = t.transpose(0, 1)    # shape (3, 2), shares storage
    tr.set_item([1, 0], 11.0)
    assert t.get_item([0, 1]) == pytest.approx(11.0)


def test_set_item_out_of_range():
    t = ortho.zeros([3])
    with pytest.raises((IndexError, RuntimeError)):
        t.set_item([99], 1.0)


# __getitem__ tuple/list syntax.

def test_getitem_tuple_2d():
    t = ortho.zeros([3, 4])
    t.set_item([2, 1], 6.0)
    elem = t[2, 1]
    assert elem.item() == pytest.approx(6.0)


def test_getitem_partial_returns_row():
    t = ortho.zeros([3, 4])
    t.set_item([1, 0], 1.0)
    t.set_item([1, 1], 2.0)
    row = t[1]
    assert row.shape == (4,)
    assert row[0].item() == pytest.approx(1.0)
    assert row[1].item() == pytest.approx(2.0)


def test_getitem_negative_tuple():
    t = ortho.zeros([2, 3])
    t.set_item([1, 2], 5.0)
    assert t[-1, -1].item() == pytest.approx(5.0)


def test_getitem_too_many_indices():
    t = ortho.zeros([2, 3])
    with pytest.raises((IndexError, RuntimeError)):
        _ = t[0, 1, 2]         # 3 indices for a 2-d tensor


# __setitem__ operator syntax.

def test_setitem_scalar():
    t = ortho.zeros([3, 3])
    t[1, 2] = 8.0
    assert t.get_item([1, 2]) == pytest.approx(8.0)


def test_setitem_partial_fills_row():
    t = ortho.zeros([3, 4])
    t[1] = 7.0                 # partial index → fill entire row 1
    assert t.get_item([1, 0]) == pytest.approx(7.0)
    assert t.get_item([1, 3]) == pytest.approx(7.0)
    assert t.get_item([0, 0]) == pytest.approx(0.0)


def test_setitem_partial_fills_column_via_transposed_view():
    t = ortho.zeros([3, 4])
    tr = t.transpose(0, 1)    # shape (4, 3)
    tr[2] = 5.0               # sets column 2 of original
    for r in range(3):
        assert t.get_item([r, 2]) == pytest.approx(5.0)
    assert t.get_item([0, 0]) == pytest.approx(0.0)


def test_setitem_negative_index():
    t = ortho.zeros([3, 3])
    t[-1, -1] = 3.0
    assert t.get_item([2, 2]) == pytest.approx(3.0)


def test_setitem_out_of_range():
    t = ortho.zeros([2, 2])
    with pytest.raises((IndexError, RuntimeError)):
        t[5] = 1.0
