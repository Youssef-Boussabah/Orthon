import orthon as ortho
import pytest


def test_backward_mul_both_require_grad():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    a.requires_grad_(True)
    b.requires_grad_(True)

    c = a * b
    c.backward()

    assert a.grad.tolist() == pytest.approx([3.0] * 6)
    assert b.grad.tolist() == pytest.approx([2.0] * 6)


def test_backward_mul_left_requires_grad():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    a.requires_grad_(True)

    c = a * b
    c.backward()

    assert a.grad.tolist() == pytest.approx([3.0] * 6)
    assert b.grad is None


def test_backward_mul_right_requires_grad():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    b.requires_grad_(True)

    c = a * b
    c.backward()

    assert a.grad is None
    assert b.grad.tolist() == pytest.approx([2.0] * 6)


def test_backward_mul_accumulates():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    a.requires_grad_(True)

    c = a * b
    c.backward(retain_graph=True)
    c.backward()

    assert a.grad.tolist() == pytest.approx([6.0] * 6)
