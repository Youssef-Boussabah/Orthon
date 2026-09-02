import orthon as ortho
import pytest


def test_chain_add_mul():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)
    c = ortho.full([2, 3], 4.0)

    a.requires_grad_(True)
    b.requires_grad_(True)
    c.requires_grad_(True)

    z = (a * b) + c
    z.backward()

    assert a.grad.tolist() == pytest.approx([3.0] * 6)
    assert b.grad.tolist() == pytest.approx([2.0] * 6)
    assert c.grad.tolist() == pytest.approx([1.0] * 6)


def test_chain_with_relu():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    a.requires_grad_(True)
    b.requires_grad_(True)

    z = ortho.relu(a * b)
    z.backward()

    # a*b = 6 > 0, so relu passes gradient through unchanged
    assert a.grad.tolist() == pytest.approx([3.0] * 6)
    assert b.grad.tolist() == pytest.approx([2.0] * 6)


def test_branching_graph():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    a.requires_grad_(True)
    b.requires_grad_(True)

    x = a * b
    y = a + b
    z = x + y

    z.backward()

    # dz/da = b + 1 = 4,  dz/db = a + 1 = 3
    assert a.grad.tolist() == pytest.approx([4.0] * 6)
    assert b.grad.tolist() == pytest.approx([3.0] * 6)


def test_accumulation_across_paths():
    a = ortho.full([2, 3], 2.0)
    b = ortho.full([2, 3], 3.0)

    a.requires_grad_(True)
    b.requires_grad_(True)

    z = (a * b) + (a * b)
    z.backward()

    # two identical paths → gradients double
    assert a.grad.tolist() == pytest.approx([6.0] * 6)
    assert b.grad.tolist() == pytest.approx([4.0] * 6)
