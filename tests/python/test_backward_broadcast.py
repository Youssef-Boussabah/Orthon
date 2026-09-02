import orthon as ortho
import pytest


def test_add_backward_broadcast_vector():
    a = ortho.ones([2, 3])
    b = ortho.ones([3])
    b.requires_grad_(True)

    c = a + b
    c.backward()

    assert b.grad.shape == (3,)
    assert b.grad.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_add_backward_broadcast_row():
    a = ortho.ones([2, 3])
    b = ortho.ones([1, 3])
    b.requires_grad_(True)

    c = a + b
    c.backward()

    assert b.grad.shape == (1, 3)
    assert b.grad.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_add_backward_broadcast_col():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 1])
    b.requires_grad_(True)

    c = a + b
    c.backward()

    assert b.grad.shape == (2, 1)
    assert b.grad.tolist() == pytest.approx([3.0, 3.0])


def test_mul_backward_broadcast_vector():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3], 3.0)
    b.requires_grad_(True)

    c = a * b
    c.backward()

    # dc/db = a, summed over rows → [4,4,4]
    assert b.grad.shape == (3,)
    assert b.grad.tolist() == pytest.approx([4.0, 4.0, 4.0])


def test_mul_backward_broadcast_col():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 1], 3.0)
    b.requires_grad_(True)

    c = a * b
    c.backward()

    # dc/db = a, summed over columns → [6,6]
    assert b.grad.shape == (2, 1)
    assert b.grad.tolist() == pytest.approx([6.0, 6.0])


def test_broadcast_both_inputs_require_grad():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3], 3.0)

    a.requires_grad_(True)
    b.requires_grad_(True)

    c = a * b
    c.backward()

    assert a.grad.shape == (2, 3)
    assert a.grad.tolist() == pytest.approx([3.0] * 6)

    assert b.grad.shape == (3,)
    assert b.grad.tolist() == pytest.approx([4.0, 4.0, 4.0])
