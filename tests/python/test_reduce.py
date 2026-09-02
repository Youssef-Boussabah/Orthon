import pytest
import orthon as ortho


# Tensor.sum() method.

def test_sum_zeros():
    t = ortho.zeros([3, 4])
    assert t.sum().item() == pytest.approx(0.0)


def test_sum_ones():
    t = ortho.ones([3, 4])
    assert t.sum().item() == pytest.approx(12.0)


def test_sum_full():
    t = ortho.full([5], 2.5)
    assert t.sum().item() == pytest.approx(12.5)


def test_sum_contiguous_2d():
    t = ortho.full([2, 3], 1.0)
    assert t.sum().item() == pytest.approx(6.0)


def test_sum_non_contiguous():
    t = ortho.ones([2, 3])
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous
    assert tr.sum().item() == pytest.approx(6.0)


def test_sum_non_contiguous_mixed():
    t = ortho.zeros([2, 3])
    for i in range(2):
        for j in range(3):
            t.set_item([i, j], float(i * 3 + j + 1))  # 1..6
    tr = t.transpose(0, 1)
    assert tr.sum().item() == pytest.approx(21.0)


def test_sum_empty_raises():
    t = ortho.empty([0])
    with pytest.raises(RuntimeError, match="empty"):
        t.sum()


# Tensor.mean() method.

def test_mean_full():
    t = ortho.full([4], 3.0)
    assert t.mean().item() == pytest.approx(3.0)


def test_mean_ones():
    t = ortho.ones([2, 5])
    assert t.mean().item() == pytest.approx(1.0)


def test_mean_mixed():
    # [0, 1, 2, 3] → mean = 1.5
    t = ortho.zeros([4])
    for i in range(4):
        t.set_item([i], float(i))
    assert t.mean().item() == pytest.approx(1.5)


def test_mean_non_contiguous():
    t = ortho.full([3, 3], 6.0)
    tr = t.transpose(0, 1)
    assert not tr.is_contiguous
    assert tr.mean().item() == pytest.approx(6.0)


def test_mean_empty_raises():
    t = ortho.empty([0])
    with pytest.raises(RuntimeError, match="empty"):
        t.mean()


# Module-level ortho.sum() and ortho.mean().

def test_module_sum():
    t = ortho.full([3], 4.0)
    assert ortho.sum(t).item() == pytest.approx(12.0)


def test_module_mean():
    t = ortho.full([3], 4.0)
    assert ortho.mean(t).item() == pytest.approx(4.0)


def test_module_sum_non_contiguous():
    t = ortho.ones([2, 3])
    tr = t.transpose(0, 1)
    assert ortho.sum(tr).item() == pytest.approx(6.0)


def test_module_mean_non_contiguous():
    t = ortho.full([2, 4], 5.0)
    tr = t.transpose(0, 1)
    assert ortho.mean(tr).item() == pytest.approx(5.0)
