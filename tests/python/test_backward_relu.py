import orthon as ortho
import pytest


def test_relu_forward():
    x = ortho.full([2, 3], -1.0)
    x[0, 0] = 2.0

    y = ortho.relu(x)

    expected = [2.0] + [0.0] * 5
    assert y.tolist() == pytest.approx(expected)


def test_relu_backward():
    x = ortho.full([2, 3], -1.0)
    x[0, 0] = 2.0

    x.requires_grad_(True)

    y = ortho.relu(x)
    y.backward()

    expected = [1.0] + [0.0] * 5
    assert x.grad.tolist() == pytest.approx(expected)


def test_relu_all_positive():
    x = ortho.full([2, 3], 5.0)
    x.requires_grad_(True)

    y = ortho.relu(x)
    y.backward()

    assert x.grad.tolist() == pytest.approx([1.0] * 6)


def test_relu_all_negative():
    x = ortho.full([2, 3], -5.0)
    x.requires_grad_(True)

    y = ortho.relu(x)
    y.backward()

    assert x.grad.tolist() == pytest.approx([0.0] * 6)


def test_relu_accumulates():
    x = ortho.full([2, 3], 2.0)
    x.requires_grad_(True)

    y = ortho.relu(x)
    y.backward(retain_graph=True)
    y.backward()

    assert x.grad.tolist() == pytest.approx([2.0] * 6)
