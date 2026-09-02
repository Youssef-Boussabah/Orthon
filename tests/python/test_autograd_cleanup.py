import orthon as ortho
import pytest


def test_backward_clears_graph_by_default():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    z = a + b
    assert z._has_grad_fn is True

    z.backward()

    assert z._has_grad_fn is False
    assert a.grad is not None


def test_retain_graph_keeps_graph():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    z = a + b
    z.backward(retain_graph=True)

    assert z._has_grad_fn is True
    assert a.grad.tolist() == pytest.approx([1.0] * 6)


def test_retain_graph_allows_repeated_backward():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    z = a + b
    z.backward(retain_graph=True)
    z.backward(retain_graph=True)

    assert a.grad.tolist() == pytest.approx([2.0] * 6)


def test_backward_twice_without_retain_graph_raises():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    z = a + b
    z.backward()

    with pytest.raises(RuntimeError):
        z.backward()


def test_leaf_grad_survives_cleanup():
    a = ortho.ones([2, 3])
    b = ortho.ones([2, 3])
    a.requires_grad_(True)

    z = (a + b) * b
    z.backward()

    assert a.grad is not None
    assert a.grad.tolist() == pytest.approx([1.0] * 6)
