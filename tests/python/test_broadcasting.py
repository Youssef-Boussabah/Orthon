import pytest
import orthon as ortho


def test_broadcast_2x3_plus_3():
    a = ortho.full([2, 3], 10.0)
    b = ortho.full([3], 2.0)
    c = a + b
    assert c.shape == (2, 3)
    assert c.tolist() == pytest.approx([12.0] * 6)


def test_broadcast_2x3_plus_1x3():
    a = ortho.full([2, 3], 4.0)
    b = ortho.full([1, 3], 2.0)
    c = a + b
    assert c.shape == (2, 3)
    assert c.tolist() == pytest.approx([6.0] * 6)


def test_broadcast_2x3_plus_2x1():
    a = ortho.full([2, 3], 4.0)
    b = ortho.full([2, 1], 6.0)
    c = a + b
    assert c.shape == (2, 3)
    assert c.tolist() == pytest.approx([10.0] * 6)


def test_broadcast_mul_2x3_times_3():
    a = ortho.full([2, 3], 3.0)
    b = ortho.full([3], 4.0)
    c = a * b
    assert c.shape == (2, 3)
    assert c.tolist() == pytest.approx([12.0] * 6)


def test_broadcast_mismatch_raises():
    a = ortho.full([2, 3], 1.0)
    b = ortho.full([4], 1.0)
    with pytest.raises(RuntimeError):
        _ = a + b
