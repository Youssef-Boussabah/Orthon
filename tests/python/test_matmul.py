import pytest
import orthon as ortho


def test_square_ones():
    # ones([2,2]) @ ones([2,2]) = [[2,2],[2,2]]
    a = ortho.ones([2, 2])
    b = ortho.ones([2, 2])
    c = ortho.matmul(a, b)
    assert c.shape == (2, 2)
    assert c.tolist() == pytest.approx([2.0, 2.0, 2.0, 2.0])


def test_rect_2x3_by_3x2():
    # ones([2,3]) @ ones([3,2]) = [[3,3],[3,3]]
    a = ortho.ones([2, 3])
    b = ortho.ones([3, 2])
    c = ortho.matmul(a, b)
    assert c.shape == (2, 2)
    assert c.tolist() == pytest.approx([3.0] * 4)


def test_scalar_result():
    # full([1,2], 3.0) @ full([2,1], 4.0) = [[24.0]]
    a = ortho.full([1, 2], 3.0)
    b = ortho.full([2, 1], 4.0)
    c = ortho.matmul(a, b)
    assert c.shape == (1, 1)
    assert c.tolist() == pytest.approx([24.0])


def test_method_matmul():
    a = ortho.ones([2, 3])
    b = ortho.ones([3, 4])
    c = a.matmul(b)
    assert c.shape == (2, 4)
    assert c.tolist() == pytest.approx([3.0] * 8)


def test_dunder_matmul():
    a = ortho.ones([2, 3])
    b = ortho.ones([3, 2])
    c = a @ b
    assert c.shape == (2, 2)
    assert c.tolist() == pytest.approx([3.0] * 4)


def test_noncontiguous_input():
    # transpose(ones([2,3])) has shape [3,2], is non-contiguous
    # [3,2] @ ones([2,4]) = ones([3,4]) * 2
    base = ortho.ones([2, 3])
    tr = ortho.transpose(base, 0, 1)
    assert tr.is_contiguous is False
    other = ortho.ones([2, 4])
    c = ortho.matmul(tr, other)
    assert c.is_contiguous is True
    assert c.shape == (3, 4)
    assert c.tolist() == pytest.approx([2.0] * 12)


def test_shape_mismatch_raises():
    a = ortho.zeros([2, 3])
    b = ortho.zeros([4, 2])
    with pytest.raises(RuntimeError):
        ortho.matmul(a, b)


def test_output_shape():
    a = ortho.ones([3, 4])
    b = ortho.ones([4, 5])
    c = ortho.matmul(a, b)
    assert c.shape == (3, 5)


def test_output_is_contiguous():
    a = ortho.ones([3, 3])
    b = ortho.ones([3, 3])
    c = ortho.matmul(a, b)
    assert c.is_contiguous is True


def test_full_values():
    # full([2,2], 2.0) @ full([2,2], 3.0) = [[12,12],[12,12]]
    a = ortho.full([2, 2], 2.0)
    b = ortho.full([2, 2], 3.0)
    c = ortho.matmul(a, b)
    assert c.tolist() == pytest.approx([12.0] * 4)
