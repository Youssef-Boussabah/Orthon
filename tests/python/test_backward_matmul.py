import orthon as ortho
import pytest


def test_backward_matmul_both_require_grad():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3, 4], 3.0)

    a.requires_grad_(True)
    b.requires_grad_(True)

    c = a @ b
    c.backward()

    # grad_output is ones [2,4]
    # dA = ones[2,4] @ B.T[4,3] → each row sums B's columns: 3*4 = 12
    assert a.grad.shape == (2, 3)
    assert a.grad.tolist() == pytest.approx([12.0] * 6)

    # dB = A.T[3,2] @ ones[2,4] → each entry = sum over A column: 2*2 = 4
    assert b.grad.shape == (3, 4)
    assert b.grad.tolist() == pytest.approx([4.0] * 12)


def test_backward_matmul_left_only():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3, 4], 3.0)

    a.requires_grad_(True)

    c = a @ b
    c.backward()

    assert a.grad is not None
    assert b.grad is None


def test_backward_matmul_right_only():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3, 4], 3.0)

    b.requires_grad_(True)

    c = a @ b
    c.backward()

    assert a.grad is None
    assert b.grad is not None


def test_backward_matmul_accumulates():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([3, 4], 3.0)

    a.requires_grad_(True)

    c = a @ b
    c.backward(retain_graph=True)
    c.backward()

    assert a.grad.tolist() == pytest.approx([24.0] * 6)
