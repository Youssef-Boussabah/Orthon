import pytest
import orthon as ortho


def test_add():
    a = ortho.full([2, 3], 3.0)
    b = ortho.full([2, 3], 4.0)
    c = ortho.add(a, b)
    assert c.shape == (2, 3)
    assert c.tolist() == pytest.approx([7.0] * 6)


def test_mul():
    a = ortho.full([2, 3], 3.0)
    b = ortho.full([2, 3], 4.0)
    c = ortho.mul(a, b)
    assert c.tolist() == pytest.approx([12.0] * 6)


def test_add_operator():
    a = ortho.full([2, 3], 1.0)
    b = ortho.full([2, 3], 2.0)
    c = a + b
    assert c.tolist() == pytest.approx([3.0] * 6)


def test_mul_operator():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)
    c = a * b
    assert c.tolist() == pytest.approx([6.0] * 6)


def test_reshape():
    t = ortho.ones([2, 3])
    r = ortho.reshape(t, [6])
    assert r.shape == (6,)
    assert r.numel == 6


def test_reshape_method():
    t = ortho.ones([2, 3])
    r = t.reshape([3, 2])
    assert r.shape == (3, 2)


def test_transpose_shape():
    t = ortho.zeros([2, 3])
    tr = ortho.transpose(t, 0, 1)
    assert tr.shape == (3, 2)


def test_transpose_strides():
    t = ortho.zeros([2, 3])
    tr = ortho.transpose(t, 0, 1)
    assert tr.strides == (1, 3)


def test_transpose_method():
    t = ortho.zeros([2, 3])
    tr = t.transpose(0, 1)
    assert tr.shape == (3, 2)
